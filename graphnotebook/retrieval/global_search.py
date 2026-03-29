"""
Global search: Map-reduce over community summaries.
"""

from graphnotebook.graph.communities import CommunityManager
from graphnotebook.llm.gateway import LLMGateway


class GlobalSearcher:
    """Implement global map-reduce search over community summaries."""

    def __init__(
        self,
        neo4j_client,
        notebook_id: str = "",
        llm_gateway: LLMGateway = None,
        community_manager: CommunityManager = None,
    ):
        self.community_manager = community_manager or CommunityManager(
            neo4j_client, notebook_id, llm_gateway
        )
        self.notebook_id = notebook_id
        self.llm = llm_gateway or LLMGateway("synthesis")

    def search(
        self, query: str, query_embedding: list, top_n: int = 5, notebook_id: str = None
    ) -> dict:
        """
        Execute map-reduce global search.
        1. Find relevant communities
        2. Map: score and extract partial answers per community
        3. Reduce: synthesize into final answer
        """
        # 1. Fetch relevant lazy summaries
        summaries = self.community_manager.get_relevant_summaries(
            query_embedding, top_n=top_n, notebook_id=notebook_id
        )

        if not summaries:
            return {
                "answer": "No relevant community info found to answer this global query.",
                "context": "",
            }

        # 2. Map Phase: Extract partial answers
        partial_answers = []
        for c in summaries:
            # We use invoke_json to structure the map phase score
            map_response = self.llm.invoke_json(
                prompt=f"""Given the following community summary, extract information
to answer the user's question. If the community is irrelevant, score it 0.

Community Title: {c.get("title")}
Community Summary: {c.get("summary")}
Question: {query}

Respond in JSON format:
{{
    "score": 0-10,
    "answer": "partial answer or explanation"
}}""",
                system="You are an expert knowledge extractor.",
            )

            score = map_response.get("score", 0)
            if score > 0:
                partial_answers.append({
                    "title": c.get("title"),
                    "score": score,
                    "rank": c.get("rank", 0),  # Preserve rank for tie-breaking
                    "answer": map_response.get("answer", "")
                })

        if not partial_answers:
            return {"answer": "No relevant info in communities.", "context": ""}

        # 3. Sort by LLM score, then community rank
        partial_answers = sorted(
            partial_answers, key=lambda x: (x["score"], x.get("rank", 0)), reverse=True
        )

        if not partial_answers:
            return {
                "answer": "Communities did not contain relevant information for this query.",
                "context": "",
            }

        # 3. Reduce Phase: Synthesize final answer
        context_parts = []
        for i, pa in enumerate(partial_answers):
            context_parts.append(
                f"### [Community: {pa['title']}] (Score: {pa['score']})\\n"
                f"{pa['answer']}\\n"
            )

        final_context = "\n".join(context_parts)

        final_answer = self.llm.invoke(
            prompt=f"""Question: {query}

Partial Evidence from Communities:
{final_context}

Synthesize a comprehensive final answer based ONLY on the evidence above. 
Cite the community titles when referencing evidence.""",
            system="You are an expert synthesizer. Provide comprehensive overviews.",
        )

        return {"answer": final_answer, "context": final_context}
