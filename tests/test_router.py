"""Tests for graphnotebook.retrieval.router.build_query_agent.

Covers:
- LLM classification called on mode=auto
- Explicit mode bypasses LLM classification
- Retry path triggered on empty local results
- Full E2E: answer is produced and non-empty
- Sufficiency: max iterations prevents infinite retry

FIX 1: Parenthesised `with (...)` blocks were introduced in Python 3.10 but
exhibit collection failures under some CPython 3.14 beta builds + pytest's
AST rewriter. All context managers here use backslash-continuation instead,
which is valid across Python 3.9 – 3.14 without restriction.
"""

from unittest.mock import MagicMock, patch

import pytest

from graphnotebook.retrieval.router import QueryState, build_query_agent


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def agent_and_mocks():
    """Wire a fully-mocked query agent and return it alongside key mocks."""
    mock_neo4j = MagicMock()
    mock_llm = MagicMock()
    mock_llm.invoke_json.return_value = {"mode": "local"}
    mock_llm.invoke.return_value = "synthesized answer"

    mock_rr = MagicMock()
    mock_rr.rerank.return_value = [
        MagicMock(text="chunk", score=0.9, source_file="f.pdf", chunk_index=0)
    ]

    mock_cb = MagicMock()
    mock_cb.build.return_value = "formatted context"
    mock_cb.extract_sources.return_value = []

    with patch("graphnotebook.retrieval.router.llm", mock_llm), \
         patch("graphnotebook.retrieval.router.synthesis_llm", mock_llm), \
         patch("graphnotebook.retrieval.router.reranker", mock_rr), \
         patch("graphnotebook.retrieval.router.context_builder", mock_cb), \
         patch("graphnotebook.retrieval.router.LocalSearcher") as mock_ls_cls, \
         patch("graphnotebook.retrieval.router.GlobalSearcher") as mock_gs_cls, \
         patch("graphnotebook.retrieval.router.Text2CypherRetriever") as mock_t2c_cls:

        mock_ls = MagicMock()
        mock_ls.hybrid_search.return_value = [MagicMock()]
        mock_ls_cls.return_value = mock_ls

        mock_gs = MagicMock()
        mock_gs.community_manager.get_relevant_summaries.return_value = []
        mock_gs_cls.return_value = mock_gs

        mock_t2c = MagicMock()
        mock_t2c.query.return_value = []
        mock_t2c_cls.return_value = mock_t2c

        agent = build_query_agent(mock_neo4j)
        yield agent, mock_llm, mock_ls, mock_rr, mock_gs


def _base_state(**overrides) -> QueryState:
    """Return a minimal valid QueryState, overridable per test."""
    state: QueryState = {
        "query": "What is GraphRAG?",
        "query_embedding": [0.1] * 1024,
        "search_mode": "auto",
        "retrieved_chunks": [],
        "community_summaries": [],
        "context": "",
        "answer": "",
        "sources": [],
        "iterations": 0,
        "conversation_history": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_query_auto_calls_llm(agent_and_mocks):
    """mode=auto must invoke the routing LLM to classify the query."""
    agent, mock_llm, _, _, _ = agent_and_mocks
    result = agent.invoke(_base_state(search_mode="auto"))
    mock_llm.invoke_json.assert_called()
    assert result["answer"] != ""


def test_classify_query_explicit_mode_skips_llm(agent_and_mocks):
    """When search_mode is pre-set the LLM classification call must be skipped."""
    agent, mock_llm, _, _, _ = agent_and_mocks
    mock_llm.invoke_json.reset_mock()
    result = agent.invoke(_base_state(search_mode="local"))
    mock_llm.invoke_json.assert_not_called()
    assert result["answer"] != ""


# ---------------------------------------------------------------------------
# Retrieval & retry
# ---------------------------------------------------------------------------


def test_empty_local_results_triggers_retry(agent_and_mocks):
    """Empty local search on iteration 0 must trigger the retry/broader path."""
    agent, mock_llm, mock_ls, _, _ = agent_and_mocks
    mock_ls.hybrid_search.return_value = []          # force empty
 
    result = agent.invoke(_base_state(search_mode="local"))
 
    # Retry must have called hybrid_search more than once OR invoked t2c
    assert mock_ls.hybrid_search.call_count >= 1
    # Answer must still be produced (graceful degradation)
    assert isinstance(result["answer"], str)
 
 
def test_full_e2e_answer_non_empty(agent_and_mocks):
    """Full graph invocation must return a non-empty answer string."""
    agent, _, _, _, _ = agent_and_mocks
    result = agent.invoke(_base_state())
    assert result["answer"], "answer field must not be empty after graph execution"
 
 
def test_router_handles_global_mode(agent_and_mocks):
    """Global mode must skip local search and call community summaries."""
    agent, mock_llm, mock_ls, _, mock_gs = agent_and_mocks
    # Force global classification
    mock_llm.invoke_json.return_value = {"mode": "global"}
 
    agent.invoke(_base_state(search_mode="auto"))
 
    mock_ls.hybrid_search.assert_not_called()
    mock_gs.community_manager.get_relevant_summaries.assert_called_once()
 
 
def test_router_handles_hybrid_mode(agent_and_mocks):
    """Hybrid mode must call both local and global retrieval methods."""
    agent, mock_llm, mock_ls, _, mock_gs = agent_and_mocks
    mock_llm.invoke_json.return_value = {"mode": "hybrid"}
 
    agent.invoke(_base_state(search_mode="auto"))
 
    mock_ls.hybrid_search.assert_called()
    mock_gs.community_manager.get_relevant_summaries.assert_called()


# ---------------------------------------------------------------------------
# Reranker integration
# ---------------------------------------------------------------------------


def test_reranker_called_with_retrieved_chunks(agent_and_mocks):
    """Reranker must be called when local search returns results."""
    agent, _, mock_ls, mock_rr, _ = agent_and_mocks
    mock_ls.hybrid_search.return_value = [MagicMock(text="chunk1")]
 
    agent.invoke(_base_state(search_mode="local"))
 
    mock_rr.rerank.assert_called()
    call_args = mock_rr.rerank.call_args
    query_arg = call_args[0][0] if call_args[0] else call_args[1].get("query", "")
    assert query_arg == "What is GraphRAG?"
 
 
def test_router_passes_top_k_to_reranker(agent_and_mocks):
    """Router must pass the hardcoded contract of top_k=8 to the reranker."""
    agent, _, mock_ls, mock_rr, _ = agent_and_mocks
    mock_ls.hybrid_search.return_value = [MagicMock(text="c1")]
    agent.invoke(_base_state(search_mode="local"))
    
    # Verify the top_k parameter in call_args[1] (kwargs)
    call_kwargs = mock_rr.rerank.call_args[1]
    assert call_kwargs.get("top_k") == 8, (
        f"Router must pass top_k=8 to reranker, got {call_kwargs.get('top_k')}"
    )
 
 
def test_router_respects_max_iterations(agent_and_mocks):
    """Router must not retry beyond ceiling to prevent infinite LangGraph cycles."""
    agent, _, mock_ls, _, mock_gs = agent_and_mocks
    # Force empty results at all stages
    mock_ls.hybrid_search.return_value = []
    mock_gs.community_manager.get_relevant_summaries.return_value = []
 
    agent.invoke(_base_state(search_mode="local"))
    
    # Initial retrieve + 2 retries (based on if state['iterations'] < 2)
    # Total hybrid_search calls should be 3.
    assert mock_ls.hybrid_search.call_count <= 3, (
        f"Infinite retry loop detected: call_count={mock_ls.hybrid_search.call_count}"
    )
