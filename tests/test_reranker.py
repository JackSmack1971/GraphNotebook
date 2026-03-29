"""Tests for graphnotebook.retrieval.reranker.Reranker."""

from unittest.mock import MagicMock, patch

import pytest

from graphnotebook.retrieval.reranker import Reranker, RetrievedChunk


def _make_chunk(text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        source_file="test.pdf",
        chunk_index=0,
        entities=[],
        relationships=[],
    )


@pytest.fixture()
def reranker():
    with patch("graphnotebook.retrieval.reranker.CrossEncoder") as mock_ce_cls:
        instance = MagicMock()
        instance.predict.return_value = [0.9, 0.2, 0.6]
        mock_ce_cls.return_value = instance
        yield Reranker(), instance


def test_rerank_orders_by_score(reranker):
    rr, mock_ce = reranker
    chunks = [
        _make_chunk("low relevance"),
        _make_chunk("zero relevance"),
        _make_chunk("medium relevance"),
    ]
    result = rr.rerank("query", chunks, top_k=3)
    # mock scores: 0.9, 0.2, 0.6 → sorted desc: 0.9, 0.6, 0.2
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.6)
    assert result[2].score == pytest.approx(0.2)


def test_rerank_respects_top_k(reranker):
    rr, mock_ce = reranker
    mock_ce.predict.return_value = [0.9, 0.2, 0.6]
    chunks = [_make_chunk(f"chunk {i}") for i in range(3)]
    result = rr.rerank("q", chunks, top_k=2)
    assert len(result) == 2
    assert result[0].score >= result[1].score


def test_rerank_empty_input_returns_empty(reranker):
    rr, _ = reranker
    result = rr.rerank("query", [], top_k=5)
    assert result == []


def test_rerank_cross_encoder_called_with_pairs(reranker):
    rr, mock_ce = reranker
    mock_ce.predict.return_value = [0.5]
    chunk = _make_chunk("relevant text")
    rr.rerank("my query", [chunk], top_k=1)
    pairs_arg = mock_ce.predict.call_args[0][0]
    assert pairs_arg == [("my query", "relevant text")]
