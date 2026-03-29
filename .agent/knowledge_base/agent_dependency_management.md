---
title: Agentic Dependency Management in LangGraph Workflows
category: architecture
tags: [langgraph, orchestration, state, dependency-injection]
---

# Agentic Dependency Management in LangGraph Workflows

## Context
When orchestrating multi-step graph execution flows (e.g., Document Ingestion, Hybrid Retrieval), certain heavy dependencies like the `neo4j.Driver` connection pool or the `SentenceTransformer` local embedding engine must be utilized cleanly without loading them multiple times or passing them deeply through class instantiation trees.

## Problem
Traditional object-oriented classes or global variables often result in cluttered syntax, difficult unit testing, and memory bloat when executing LangGraph `StateGraph` nodes asynchronously. Furthermore, the `uv run pytest` execution requires strict control over test boundaries to mock database connections.

## Solution
GraphNotebook employs two distinct and highly effective structural patterns for managing dependencies across LangGraph states depending on the persistence required:

### Pattern A: State-Payload Injection (As used in `pipeline.py`)
Heavy, runtime-defined clients are attached natively to the LangGraph `TypedDict` State, propagating dynamically through each isolated functional node.

```python
class IngestionState(TypedDict):
    neo4j_client: Any         # Injected during initial graph.invoke()
    parsed_doc: ParsedDoc
    status: str

async def process_step(state: IngestionState) -> IngestionState:
    neo4j = state.get("neo4j_client")
    if not neo4j:
        raise ValueError("neo4j_client missing from execution state.")
    
    # Execute query
    result = neo4j.query("...")
    return state
```

### Pattern B: Closures and Module Singletons (As used in `router.py`)
When creating interactive loops that don't need heavy database mutations but rather constant LLM pinging across a single session, the pipeline wraps the node functions inside a compiler block and leverages module level singletons (e.g., `LLMGateway("routing")`) coupled with closure parameters.

```python
llm = LLMGateway("routing") # Module singleton

def build_query_agent(neo4j_client):
    # Node function runs inside closure, capturing 'neo4j_client' natively
    def execute_retrieval(state: QueryState) -> QueryState:
        # neo4j_client is directly accessible without cluttering 'state' keys
        res = neo4j_client.query("...")
        return state
        
    workflow = StateGraph(QueryState)
    workflow.add_node("retrieve", execute_retrieval)
    return workflow.compile()
```

## Gotchas
* **Testing Friction (Pattern A)**: Developers MUST remember to mock and append the `neo4j_client` key to the initial dictionary passed into `pipeline.invoke()` in the `pytest` suite, otherwise nodes will fast-fail with `NoneType` errors.
* **Testing Friction (Pattern B)**: Singletons like `LLMGateway` instantiated natively at the module level are harder to monkeypatch correctly. You must patch `graphnotebook.retrieval.router.llm` before compiling the agent.
* Never instantiate a new `neo4j.Driver` or `SentenceTransformer` inside a node. Always pipe them from `main.py` entrypoint down to these compilation factories.
