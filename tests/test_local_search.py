"""Tests for graphnotebook.retrieval.local_search.LocalSearcher.

Phase 2 scaffolding — covers:
- hybrid_search result → RetrievedChunk mapping
- Empty Neo4j result → empty list (no crash)
- notebook_id scoping (query param must include notebook_id)
- Embedding is called with correct text
"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from graphnotebook.retrieval.local_search import LocalSearcher


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_searcher(mock_neo4j, mock_embedding_engine):
    return LocalSearcher(
        neo4j_client=mock_neo4j,
        embedding_engine=mock_embedding_engine,
        notebook_id="nb_test",
    )


# ---------------------------------------------------------------------------
# hybrid_search — result mapping
# ---------------------------------------------------------------------------


def test_hybrid_search_returns_retrieved_chunks(local_searcher, mock_neo4j):
    """Neo4j rows must be mapped to RetrievedChunk objects."""
    mock_neo4j.query.return_value = [
        {
            "text": "This is chunk text.",
            "score": 0.85,
            "source_file": "doc.pdf",
            "chunk_index": 2,
            "entities": [],
            "relationships": [],
        }
    ]
    results = local_searcher.hybrid_search("test query", top_k=5)

    assert len(results) == 1
    chunk = results[0]
    assert chunk.text == "This is chunk text."
    assert chunk.score == pytest.approx(0.85)
    assert chunk.source_file == "doc.pdf"
    assert chunk.chunk_index == 2


def test_hybrid_search_empty_result(local_searcher, mock_neo4j):
    """Empty Neo4j result must return an empty list without raising."""
    mock_neo4j.query.return_value = []
    results = local_searcher.hybrid_search("nothing matches", top_k=5)
    assert results == []


def test_hybrid_search_multiple_chunks_sorted_by_score(local_searcher, mock_neo4j):
    """Results must be returned in score-descending order."""
    mock_neo4j.query.return_value = [
        {"text": "low", "score": 0.3, "source_file": "a.pdf",
         "chunk_index": 0, "entities": [], "relationships": []},
        {"text": "high", "score": 0.9, "source_file": "b.pdf",
         "chunk_index": 1, "entities": [], "relationships": []},
        {"text": "mid", "score": 0.6, "source_file": "c.pdf",
         "chunk_index": 2, "entities": [], "relationships": []},
    ]
    results = local_searcher.hybrid_search("query", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), (
        "hybrid_search results must be sorted highest score first"
    )


# ---------------------------------------------------------------------------
# Notebook-ID scoping
# ---------------------------------------------------------------------------


def test_hybrid_search_passes_notebook_id(local_searcher, mock_neo4j):
    """The notebook_id must appear in the Neo4j query parameters."""
    mock_neo4j.query.return_value = []
    local_searcher.hybrid_search("test", top_k=5)

    all_calls = mock_neo4j.query.call_args_list
    param_values: list[str] = []
    for call_ in all_calls:
        args = call_.args
        kwargs = call_.kwargs
        if len(args) > 1 and isinstance(args[1], dict):
            param_values.extend(str(v) for v in args[1].values())
        for kw in ("parameters", "params"):
            if kw in kwargs and isinstance(kwargs[kw], dict):
                param_values.extend(str(v) for v in kwargs[kw].values())

    assert any("nb_test" in v for v in param_values), (
        "notebook_id 'nb_test' was not passed as a parameter to the Neo4j query. "
        "All retrieval queries must be scoped to the current notebook."
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_hybrid_search_embeds_query_text(local_searcher, mock_embedding_engine):
    """The query text must be embedded before the vector search."""
    local_searcher.hybrid_search("embed this query", top_k=3)
    mock_embedding_engine.embed_single.assert_called_once_with("embed this query")
