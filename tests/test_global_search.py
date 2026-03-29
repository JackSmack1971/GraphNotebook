"""Tests for graphnotebook.retrieval.global_search.GlobalSearcher.

Phase 2 scaffolding — covers:
- search() assembles community summaries into context string
- Empty summaries → empty/minimal context, no crash
- top_n parameter is forwarded to community_manager
- Returned object contains 'context' key with string value
"""

from unittest.mock import MagicMock

import pytest

from graphnotebook.retrieval.global_search import GlobalSearcher


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def global_searcher(mock_neo4j, mock_llm):
    """GlobalSearcher wired with mock CommunityManager internals."""
    # Pre-mock invoke_json to return a valid dict
    mock_llm.invoke_json.return_value = {"score": 5, "answer": "Partial answer."}
    searcher = GlobalSearcher(neo4j_client=mock_neo4j, llm_gateway=mock_llm)
    # Patch the internal community_manager so no real Neo4j calls are made
    searcher.community_manager = MagicMock()
    return searcher


def _make_summary(title: str, body: str, rank: int = 1) -> dict:
    return {"title": title, "summary": body, "rank": rank, "key_findings": []}


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------


def test_search_returns_context_string(global_searcher, mock_llm):
    """search() must return a non-empty context string when summaries exist."""
    global_searcher.community_manager.get_relevant_summaries.return_value = [
        _make_summary("Cluster A", "Entities in cluster A discuss topic X."),
        _make_summary("Cluster B", "Cluster B contains related concepts."),
    ]
    mock_llm.invoke_json.return_value = {"score": 5, "answer": "Topic X info."}
    result = global_searcher.search("what is topic X?", query_embedding=[0.1] * 1024)

    assert "context" in result, "search() must return a dict with a 'context' key"
    assert "answer" in result
    assert isinstance(result["context"], str)
    assert len(result["context"]) > 0, "context must not be empty when summaries exist"


def test_search_empty_summaries_returns_empty_context(global_searcher):
    """No relevant summaries → empty context string, no exception."""
    global_searcher.community_manager.get_relevant_summaries.return_value = []
    result = global_searcher.search("no match", query_embedding=[0.0] * 1024)

    assert "context" in result
    # Empty or minimal context is acceptable — must not raise
    assert isinstance(result["context"], str)


def test_search_includes_summary_titles_in_context(global_searcher, mock_llm):
    """Community titles must appear in the assembled context."""
    global_searcher.community_manager.get_relevant_summaries.return_value = [
        _make_summary("UniqueClusterTitle", "Body text about the cluster."),
    ]
    mock_llm.invoke_json.return_value = {"score": 8, "answer": "Unique info."}
    result = global_searcher.search("q", query_embedding=[0.1] * 1024)
    assert "UniqueClusterTitle" in result["context"], (
        "Community title 'UniqueClusterTitle' must be present in assembled context"
    )


def test_search_forwards_top_n_to_community_manager(global_searcher):
    """top_n must be forwarded to get_relevant_summaries."""
    global_searcher.community_manager.get_relevant_summaries.return_value = []
    global_searcher.search("q", query_embedding=[0.1] * 1024, top_n=3)

    call_kwargs = global_searcher.community_manager.get_relevant_summaries.call_args
    # Accept positional or keyword argument
    passed_top_n = (
        call_kwargs[1].get("top_n")
        if call_kwargs[1]
        else (call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
    )
    assert passed_top_n == 3, (
        f"Expected top_n=3 forwarded to community_manager, got {passed_top_n}"
    )


def test_search_higher_ranked_summaries_appear_first(global_searcher):
    """Higher-rank summaries must appear before lower-rank in context."""
    global_searcher.community_manager.get_relevant_summaries.return_value = [
        _make_summary("LowRank", "Low importance.", rank=1),
        _make_summary("HighRank", "High importance.", rank=5),
    ]
    result = global_searcher.search("q", query_embedding=[0.1] * 1024)
    ctx = result["context"]
    # REMOVE the conditional guard — assert unconditionally:
    assert "HighRank" in ctx, "HighRank title must appear in assembled context"
    assert "LowRank" in ctx, "LowRank title must appear in assembled context"
    high_pos = ctx.find("HighRank")
    low_pos = ctx.find("LowRank")
    assert high_pos < low_pos, (
        "Higher-rank community summaries must appear before lower-rank ones"
    )
