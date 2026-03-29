"""
LangGraph agentic query router.
Dynamically selects retrieval strategy based on query classification.
Includes sufficiency evaluation and retry loop.
"""

from typing import Literal, TypedDict, List

from langgraph.graph import END, StateGraph

from graphnotebook.llm.embeddings import EmbeddingEngine
from graphnotebook.llm.gateway import LLMGateway
from graphnotebook.retrieval.context_builder import ContextBuilder
from graphnotebook.retrieval.global_search import GlobalSearcher
from graphnotebook.retrieval.local_search import LocalSearcher
from graphnotebook.retrieval.reranker import Reranker, RetrievedChunk
from graphnotebook.retrieval.text2cypher import Text2CypherRetriever


class QueryState(TypedDict):
    query: str
    query_embedding: List[float]
    notebook_id: str
    search_mode: str
    conversation_history: list
    iterations: int
    stream: bool
    retrieved_chunks: List[RetrievedChunk]
    community_summaries: List[dict]
    context: str
    answer: str
    sources: List[dict]
    full_synthesis_prompt: str
    system_prompt: str


# ── Singletons (initialized at module level) ────────
llm = LLMGateway("routing")
synthesis_llm = LLMGateway("synthesis")
reranker = Reranker()
context_builder = ContextBuilder()

# ── Dynamic Agent Builder ───────────────────────────


def build_query_agent(neo4j_client):
    """
    Builds the state graph using the provided neo4j client.
    Closes over searcher instances.
    """
    embedding_engine = EmbeddingEngine()
    local_searcher = LocalSearcher(neo4j_client, embedding_engine=embedding_engine)
    global_searcher = GlobalSearcher(neo4j_client, llm_gateway=synthesis_llm)
    text2cypher_retriever = Text2CypherRetriever(neo4j_client, llm_gateway=synthesis_llm)

    def classify_query(state: QueryState) -> QueryState:
        """Determine retrieval strategy via LLM classification."""
        if state.get("search_mode") != "auto":
            return state

        classification = llm.invoke_json(
            prompt=f"""Classify this knowledge base query:
"{state["query"]}"

- "local": asks about a specific entity, fact, or detail
- "global": asks for themes, summaries, overviews, or cross-document patterns
- "hybrid": needs both specific facts and broader context

Respond: {{"mode": "local|global|hybrid"}}"""
        )
        state["search_mode"] = classification.get("mode", "hybrid")
        return state

    def execute_retrieval(state: QueryState) -> QueryState:
        """Execute retrieval based on classification."""
        mode = state.get("search_mode", "hybrid")
        notebook_id = state.get("notebook_id")

        if "retrieved_chunks" not in state:
            state["retrieved_chunks"] = []
        if "community_summaries" not in state:
            state["community_summaries"] = []

        if mode in ("local", "hybrid"):
            raw_chunks = local_searcher.hybrid_search(
                query_text=state["query"],
                query_embedding=state.get("query_embedding"),
                top_k=20,
                notebook_id=notebook_id,
            )
            state["retrieved_chunks"] = reranker.rerank(
                state["query"], raw_chunks, top_k=8
            )

        if mode in ("global", "hybrid"):
            summaries = global_searcher.community_manager.get_relevant_summaries(
                state["query_embedding"], top_n=5, notebook_id=notebook_id
            )
            state["community_summaries"] = summaries

        return state

    def evaluate_sufficiency(state: QueryState) -> Literal["synthesize", "retry"]:
        """Check if retrieved context is sufficient."""
        total = len(state.get("retrieved_chunks", []))
        total += len(state.get("community_summaries", []))
        if total == 0 and state.get("iterations", 0) < 2:
            return "retry"
        return "synthesize"

    def retry_broader(state: QueryState) -> QueryState:
        """Widen search: try text2cypher fallback."""
        state["iterations"] = state.get("iterations", 0) + 1
        cypher_results = text2cypher_retriever.query(state["query"])

        chunks = state.get("retrieved_chunks", [])
        for i, row in enumerate(cypher_results):
            text_repr = "\\n".join(f"{k}: {v}" for k, v in row.items())
            chunks.append(
                RetrievedChunk(
                    id=f"cypher_{i}",
                    text=f"Cypher Result {i + 1}: {text_repr}",
                    score=1.0,
                    source="text2cypher",
                    metadata={"page_number": None, "entities": [], "relationships": []},
                )
            )

        state["retrieved_chunks"] = chunks
        return state

    def synthesize(state: QueryState) -> QueryState:
        """Generate final answer with source attribution."""
        notebook_id = state.get("notebook_id")

        if (
            state.get("search_mode") == "global"
            and not state.get("retrieved_chunks")
            and state.get("community_summaries")
        ):
            # Map reduce bypass
            answer = global_searcher.search(
                query=state["query"],
                query_embedding=state["query_embedding"],
                top_communities=5,
                notebook_id=notebook_id,
            )
            state["context"] = ""
            state["answer"] = answer
            state["sources"] = [
                {"type": "community", "title": c.get("title", "Community " + str(c.get("id")))}
                for c in state["community_summaries"]
            ]
            return state

        # Normal synthesis assembly
        context = context_builder.build(
            chunks=state.get("retrieved_chunks", []),
            community_summaries=state.get("community_summaries", []),
        )
        
        history_text = ""
        if state.get("conversation_history"):
            history = state["conversation_history"][-6:]
            history_parts = []
            for m in history:
                role = "User" if m["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {m['content']}")
            history_text = "\n".join(history_parts)

        prompt = f"Question: {state['query']}\n"
        if history_text:
            prompt = (
                f"Conversation History:\n{history_text}\n\n"
                f"New Question: {state['query']}\n"
            )

        state["context"] = context
        state["full_synthesis_prompt"] = f"{prompt}\n\nContext:\n{context}"
        state["system_prompt"] = (
            "Answer based ONLY on the provided context. "
            "Cite sources as [Source: filename]. "
            "If context is insufficient, say so clearly."
        )

        if not state.get("stream", False):
            state["answer"] = synthesis_llm.invoke(
                prompt=state["full_synthesis_prompt"], system=state["system_prompt"]
            )
        
        sources = []
        for c in state.get("retrieved_chunks", []):
            sources.append(getattr(c, "source", "unknown"))

        state["sources"] = [{"type": "chunk", "source": s} for s in set(sources)]
        return state

    # ── Build Agentic Graph ─────────────────────────────
    query_workflow = StateGraph(QueryState)
    query_workflow.add_node("classify", classify_query)
    query_workflow.add_node("retrieve", execute_retrieval)
    query_workflow.add_node("retry", retry_broader)
    query_workflow.add_node("synthesize", synthesize)

    query_workflow.set_entry_point("classify")
    query_workflow.add_edge("classify", "retrieve")
    query_workflow.add_conditional_edges(
        "retrieve",
        evaluate_sufficiency,
        {"synthesize": "synthesize", "retry": "retry"},
    )
    query_workflow.add_edge("retry", "synthesize")
    query_workflow.add_edge("synthesize", END)

    return query_workflow.compile()
