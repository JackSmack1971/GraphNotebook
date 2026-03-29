"""Tests for graphnotebook.retrieval.router.build_query_agent."""

from unittest.mock import MagicMock, patch

import pytest

from graphnotebook.retrieval.router import QueryState, build_query_agent


@pytest.fixture()
def agent_and_mocks():
    mock_neo4j = MagicMock()
    with (
        patch("graphnotebook.retrieval.router.llm") as mock_llm,
        patch("graphnotebook.retrieval.router.synthesis_llm") as mock_synthesis_llm,
        patch("graphnotebook.retrieval.router.reranker") as mock_rr,
        patch("graphnotebook.retrieval.router.context_builder") as mock_cb,
        patch("graphnotebook.retrieval.router.LocalSearcher") as mock_ls_cls,
        patch("graphnotebook.retrieval.router.GlobalSearcher") as mock_gs_cls,
        patch("graphnotebook.retrieval.router.Text2CypherRetriever") as mock_t2c_cls,
    ):
        mock_llm.invoke_json.return_value = {"mode": "local"}
        mock_llm.invoke.return_value = "synthesized answer"
        mock_synthesis_llm.invoke.return_value = "synthesized answer"

        mock_rr.rerank.return_value = [
            MagicMock(text="chunk", score=0.9, source_file="f.pdf", chunk_index=0)
        ]

        mock_ls = MagicMock()
        mock_ls.hybrid_search.return_value = [MagicMock()]
        mock_ls_cls.return_value = mock_ls

        mock_gs = MagicMock()
        mock_gs.community_manager.get_relevant_summaries.return_value = []
        mock_gs_cls.return_value = mock_gs

        mock_t2c = MagicMock()
        mock_t2c.query.return_value = []
        mock_t2c_cls.return_value = mock_t2c

        mock_cb.build.return_value = ("formatted context", [])

        agent = build_query_agent(mock_neo4j)
        yield agent, mock_llm, mock_ls, mock_rr


def test_classify_query_auto_calls_llm(agent_and_mocks):
    agent, mock_llm, mock_ls, _ = agent_and_mocks
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
    result = agent.invoke(state)
    mock_llm.invoke_json.assert_called()
    assert result["answer"] != ""


def test_classify_query_explicit_mode_skips_llm(agent_and_mocks):
    """When search_mode is pre-set, the LLM classification call must be skipped."""
    agent, mock_llm, _, _ = agent_and_mocks
    state: QueryState = {
        "query": "test",
        "query_embedding": [0.0] * 1024,
        "search_mode": "local",  # explicit → no LLM classify
        "retrieved_chunks": [],
        "community_summaries": [],
        "context": "",
        "answer": "",
        "sources": [],
        "iterations": 0,
        "conversation_history": [],
    }
    agent.invoke(state)
    # invoke_json used for classification — should NOT be called here
    for c in mock_llm.invoke_json.call_args_list:
        assert "Classify" not in str(c)


def test_evaluate_sufficiency_triggers_retry_on_empty():
    """Empty retrieval on iter=0 must route to retry_broader."""
    from graphnotebook.retrieval.router import build_query_agent

    mock_neo4j = MagicMock()
    with (
        patch("graphnotebook.retrieval.router.llm"),
        patch("graphnotebook.retrieval.router.synthesis_llm"),
        patch("graphnotebook.retrieval.router.reranker"),
        patch("graphnotebook.retrieval.router.context_builder"),
        patch("graphnotebook.retrieval.router.LocalSearcher") as mock_ls_cls,
        patch("graphnotebook.retrieval.router.GlobalSearcher") as mock_gs_cls,
        patch("graphnotebook.retrieval.router.Text2CypherRetriever") as mock_t2c_cls,
    ):
        mock_ls = MagicMock()
        mock_ls.search.return_value = []  # Changed from hybrid_search to search
        mock_ls_cls.return_value = mock_ls

        mock_gs = MagicMock()
        mock_gs.community_manager.get_relevant_summaries.return_value = []
        mock_gs_cls.return_value = mock_gs

        mock_t2c = MagicMock()
        mock_t2c.query.return_value = [{"result": "fallback data"}]
        mock_t2c_cls.return_value = mock_t2c

        agent = build_query_agent(mock_neo4j)
        state: QueryState = {
            "query": "obscure fact",
            "query_embedding": [0.0] * 1024,
            "search_mode": "local",
            "retrieved_chunks": [],
            "community_summaries": [],
            "context": "",
            "answer": "",
            "sources": [],
            "iterations": 0,
            "conversation_history": [],
        }
        result = agent.invoke(state)
        # Text2Cypher fallback must have been called
        mock_t2c.query.assert_called_once_with("obscure fact")


def test_retry_broader_wraps_cypher_results_as_chunks(agent_and_mocks):
    """Cypher dicts from text2cypher must be wrapped into RetrievedChunk objects."""
    agent, _, mock_ls, _ = agent_and_mocks
    # Force empty local search → retry path
    mock_ls.hybrid_search.return_value = []

    with (
        patch("graphnotebook.retrieval.router.LocalSearcher") as mock_ls_cls,
        patch("graphnotebook.retrieval.router.GlobalSearcher") as mock_gs_cls,
        patch("graphnotebook.retrieval.router.Text2CypherRetriever") as mock_t2c_cls,
        patch("graphnotebook.retrieval.router.llm") as mock_llm,
        patch("graphnotebook.retrieval.router.synthesis_llm") as mock_synth_llm,
        patch("graphnotebook.retrieval.router.reranker") as mock_rr,
    ):
        mock_ls = MagicMock()
        mock_ls.search.return_value = []
        mock_ls_cls.return_value = mock_ls

        mock_gs = MagicMock()
        mock_gs.community_manager.get_relevant_summaries.return_value = []
        mock_gs_cls.return_value = mock_gs

        mock_t2c = MagicMock()
        mock_t2c.query.return_value = [{"name": "Test Entity", "type": "Person"}]
        mock_t2c_cls.return_value = mock_t2c
        
        mock_rr.rerank.return_value = [] # Ensure rerank returns empty to trigger retry

        from graphnotebook.retrieval.router import build_query_agent
        agent = build_query_agent(MagicMock())

        state: QueryState = {
            "query": "test",
            "query_embedding": [0.0] * 1024,
            "search_mode": "local",
            "retrieved_chunks": [],
            "community_summaries": [],
            "context": "",
            "answer": "",
            "sources": [],
            "iterations": 0,
            "conversation_history": [],
            "notebook_id": "test_nb",
        }
        result = agent.invoke(state)

        # Verify text2cypher results are wrapped as chunks
        assert len(result["retrieved_chunks"]) >= 1
        assert any("Test Entity" in c.text for c in result["retrieved_chunks"])
        assert any(c.source == "text2cypher" for c in result["retrieved_chunks"])
