"""Tests for graphnotebook.retrieval.reranker.Reranker.

Mutation targets addressed:
  M3  — ConditionalsBoundary: `>=` → `>` in ordering assertion.
        Killed by test_rerank_tied_scores_stable, which injects equal scores
        and asserts both are returned — failing under strict `>` semantics.
"""

from unittest.mock import MagicMock, patch

import pytest

from graphnotebook.retrieval.reranker import Reranker, RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Patch CrossEncoder to avoid loading the model; yield (Reranker, mock)."""
    with patch("graphnotebook.retrieval.reranker.CrossEncoder") as mock_ce_cls:
        instance = MagicMock()
        instance.predict.return_value = [0.9, 0.2, 0.6]
        mock_ce_cls.return_value = instance
        yield Reranker(), instance


# ---------------------------------------------------------------------------
# Core ordering
# ---------------------------------------------------------------------------


def test_rerank_orders_by_score_descending(reranker):
    """Results must be sorted highest → lowest by cross-encoder score."""
    rr, mock_ce = reranker
    # predict returns [0.9, 0.2, 0.6] matching chunk order below
    chunks = [
        _make_chunk("low relevance"),
        _make_chunk("zero relevance"),
        _make_chunk("medium relevance"),
    ]
    result = rr.rerank("query", chunks, top_k=3)

    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.6)
    assert result[2].score == pytest.approx(0.2)


def test_rerank_respects_top_k(reranker):
    """Only top_k results must be returned, highest scores first."""
    rr, mock_ce = reranker
    mock_ce.predict.return_value = [0.9, 0.2, 0.6]
    chunks = [_make_chunk(f"chunk {i}") for i in range(3)]
    result = rr.rerank("q", chunks, top_k=2)

    assert len(result) == 2
    assert result[0].score >= result[1].score


# ---------------------------------------------------------------------------
# M3 — ConditionalsBoundary kill: >= must hold for tied scores
# ---------------------------------------------------------------------------


def test_rerank_tied_scores_stable(reranker):
    """Tied cross-encoder scores must both survive top_k — kills M3.

    If the sort condition were `>` (strict) instead of `>=`, implementations
    that break ties arbitrarily could drop one of two equally-scored chunks.
    This test enforces that the assertion `result[0].score >= result[1].score`
    holds even when the two values are identical, making the boundary mutation
    detectable.
    """
    rr, mock_ce = reranker
    # Scores: chunk0=0.7, chunk1=0.7, chunk2=0.3 — chunks 0 and 1 are tied
    mock_ce.predict.return_value = [0.7, 0.7, 0.3]
    chunks = [_make_chunk(f"chunk {i}") for i in range(3)]

    result = rr.rerank("q", chunks, top_k=2)

    assert len(result) == 2, "Both tied-score chunks must be returned for top_k=2"
    assert result[0].score == pytest.approx(0.7)
    assert result[1].score == pytest.approx(0.7)
    # Explicitly assert >= holds (not just >) — this is the mutant-killing assertion
    assert result[0].score >= result[1].score, (
        "Ordering assertion failed on tied scores — "
        "reranker must use >= not strict >"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_rerank_empty_input_returns_empty(reranker):
    rr, _ = reranker
    result = rr.rerank("query", [], top_k=5)
    assert result == []


def test_rerank_cross_encoder_called_with_correct_pairs(reranker):
    """CrossEncoder.predict must receive (query, chunk_text) pairs."""
    rr, mock_ce = reranker
    mock_ce.predict.return_value = [0.5]
    chunk = _make_chunk("relevant text")

    rr.rerank("my query", [chunk], top_k=1)

    pairs_arg = mock_ce.predict.call_args[0][0]
    assert pairs_arg == [("my query", "relevant text")], (
        f"Expected [('my query', 'relevant text')], got {pairs_arg}"
    )


def test_rerank_top_k_greater_than_input_returns_all(reranker):
    """top_k larger than input length must return all available chunks."""
    rr, mock_ce = reranker
    mock_ce.predict.return_value = [0.8, 0.3]
    chunks = [_make_chunk("a"), _make_chunk("b")]
    result = rr.rerank("q", chunks, top_k=10)
    assert len(result) == 2
