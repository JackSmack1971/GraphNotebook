"""Tests for graphnotebook.graph.communities.CommunityManager.
conftest.py patches the diskcache issue at collection time.
"""

from unittest.mock import MagicMock

import pytest

from graphnotebook.graph.communities import CommunityManager


@pytest.fixture()
def community_manager(mock_neo4j, mock_llm):
    return CommunityManager(neo4j_client=mock_neo4j, llm_gateway=mock_llm)


def test_init(community_manager, mock_neo4j, mock_llm):
    assert community_manager.neo4j is mock_neo4j
    assert community_manager.llm is mock_llm


def test_detect_communities_calls_gds_in_order(community_manager, mock_neo4j):
    mock_neo4j.query.return_value = [{"communityCount": 5, "modularity": 0.4}]
    community_manager.detect_communities()
    calls = [c.args[0] for c in mock_neo4j.query.call_args_list]
    drop_idx = next(i for i, c in enumerate(calls) if "gds.graph.drop" in c)
    project_idx = next(i for i, c in enumerate(calls) if "gds.graph.project" in c)
    leiden_idx = next(i for i, c in enumerate(calls) if "gds.leiden" in c)
    assert drop_idx < project_idx < leiden_idx


def test_detect_communities_drops_projection_on_exception(
    community_manager, mock_neo4j
):
    """GDS projection must be dropped in finally even when Leiden raises."""
    call_count = [0]

    def side_effect(query, *args, **kwargs):
        call_count[0] += 1
        if "gds.leiden" in query:
            raise RuntimeError("GDS unavailable")
        return []

    mock_neo4j.query.side_effect = side_effect
    with pytest.raises(RuntimeError):
        community_manager.detect_communities()
    # At least one drop call must have occurred after the exception
    drop_calls = [
        c for c in mock_neo4j.query.call_args_list if "gds.graph.drop" in str(c)
    ]
    assert len(drop_calls) >= 1


def test_get_summary_returns_cached(community_manager, mock_neo4j):
    mock_neo4j.query.return_value = [
        {
            "title": "Cached Title",
            "summary": "Cached body.",
            "key_findings": ["f1"],
            "rank": 2,
        }
    ]
    result = community_manager.get_summary("cid_123")
    assert result["title"] == "Cached Title"
    assert result["summary"] == "Cached body."
    # LLM must NOT be called on cache hit
    community_manager.llm.invoke_json.assert_not_called()


def test_get_summary_generates_when_cache_miss(community_manager, mock_neo4j, mock_llm):
    mock_neo4j.query.side_effect = [
        [],  # cache miss
        [
            {
                "entities": [{"name": "E1", "type": "Concept", "desc": "x"}],
                "relationships": [],
            }
        ],
        None,  # _cache_summary write
    ]
    mock_llm.invoke_json.return_value = {
        "title": "Generated",
        "summary": "Gen body.",
        "key_findings": [],
        "rank": 1,
    }
    result = community_manager.get_summary("cid_miss")
    assert result["title"] == "Generated"
    mock_llm.invoke_json.assert_called_once()


def test_get_relevant_summaries_returns_cached_directly(community_manager, mock_neo4j):
    mock_neo4j.query.return_value = [
        {
            "community_id": "c1",
            "cached_summary": "Existing summary.",
            "title": "Topic A",
            "match_count": 3,
            "avg_score": 0.88,
        }
    ]
    results = community_manager.get_relevant_summaries([0.1] * 1024, top_n=1)
    assert len(results) == 1
    assert results[0]["summary"] == "Existing summary."
    # get_summary (lazy gen) must NOT be called when cache is present
    community_manager.llm.invoke_json.assert_not_called()


def test_get_relevant_summaries_lazy_generates_on_miss(
    community_manager, mock_neo4j, mock_llm
):
    mock_neo4j.query.side_effect = [
        [  # vector query result — no cached_summary
            {
                "community_id": "c2",
                "cached_summary": None,
                "title": "Topic B",
                "match_count": 2,
                "avg_score": 0.75,
            }
        ],
        [],  # _get_cached (cache miss)
        [{"entities": [], "relationships": []}],  # context fetch
        None,  # _cache_summary
    ]
    mock_llm.invoke_json.return_value = {
        "title": "Lazy Title",
        "summary": "Lazy body.",
        "key_findings": [],
        "rank": 0,
    }
    results = community_manager.get_relevant_summaries([0.1] * 1024, top_n=1)
    assert results[0]["summary"] == "Lazy body."
    mock_llm.invoke_json.assert_called_once()


def test_cache_summary_writes_all_fields(community_manager, mock_neo4j):
    mock_neo4j.query.return_value = []
    summary = {"title": "T", "summary": "S", "rank": 3, "key_findings": ["f1", "f2"]}
    community_manager._cache_summary("cid_x", summary)
    call_args = mock_neo4j.query.call_args
    params = (
        call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
    )
    assert params.get("title") == "T"
    assert params.get("rank") == 3


def test_cache_summary_uses_zero_rank_when_key_absent(community_manager, mock_neo4j):
    mock_neo4j.query.return_value = []
    summary = {"title": "T", "summary": "S", "key_findings": []}
    # No "rank" key — default must be 0
    community_manager._cache_summary("cid_z", summary)
    call_args = mock_neo4j.query.call_args
    params = (
        call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
    )
    assert params.get("rank") == 0, (
        "Missing 'rank' key must default to 0, not any other value"
    )


def test_detect_communities_returns_quality_metrics(community_manager, mock_neo4j):
    """detect_communities must return Leiden metrics as a non-empty result list."""
    mock_neo4j.query.return_value = [{"communityCount": 5, "modularity": 0.4}]
    result = community_manager.detect_communities()

    # Guard: method must return the Leiden result list, not None
    assert result is not None, (
        "detect_communities returned None — source method must 'return result'"
    )
    assert len(result) > 0, "Leiden result list must not be empty"

    # Pin the quality metrics from result[0] (the list row from neo4j.query)
    metrics = result[0]
    assert metrics["communityCount"] == 5
    assert metrics["modularity"] == pytest.approx(0.4, abs=1e-6)
