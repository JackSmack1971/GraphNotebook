"""Tests for graphnotebook.retrieval.context_builder.ContextBuilder.

Phase 2 scaffolding — covers:
- build() assembles chunk texts into a formatted context string
- Source attribution: each chunk's source_file and chunk_index must appear
- extract_sources() returns deduplicated source_file list
- Empty chunks → empty context, no crash
- Community summaries are incorporated when provided
"""

from unittest.mock import MagicMock

import pytest

from graphnotebook.retrieval.context_builder import ContextBuilder
from graphnotebook.retrieval.reranker import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(text: str, source: str = "doc.pdf", idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=0.8,
        source_file=source,
        chunk_index=idx,
        page_number=0,
        entities=[],
        relationships=[],
    )


@pytest.fixture()
def builder():
    return ContextBuilder()


# ---------------------------------------------------------------------------
# build() — context assembly
# ---------------------------------------------------------------------------


def test_build_returns_string(builder):
    chunks = [_chunk("Some relevant text.")]
    result = builder.build(chunks, community_summaries=[])
    assert isinstance(result, str)


def test_build_includes_chunk_text(builder):
    """The chunk's text must appear verbatim (or close) in the context."""
    chunks = [_chunk("Unique chunk content for test.")]
    result = builder.build(chunks, community_summaries=[])
    assert "Unique chunk content for test." in result, (
        "build() must include the chunk text in the assembled context"
    )


def test_build_includes_source_attribution(builder):
    """Source file name must appear in the context for traceability."""
    chunks = [_chunk("text", source="research_paper.pdf", idx=3)]
    result = builder.build(chunks, community_summaries=[])
    assert "research_paper.pdf" in result, (
        "Source file name must be present in the context for attribution"
    )


def test_build_empty_chunks_returns_string(builder):
    """No chunks + no summaries must return a string (empty or placeholder)."""
    result = builder.build([], community_summaries=[])
    assert isinstance(result, str)


def test_build_multiple_chunks_all_included(builder):
    """All chunks must contribute to the context, not just the first."""
    chunks = [
        _chunk("First chunk text.", source="a.pdf", idx=0),
        _chunk("Second chunk text.", source="b.pdf", idx=1),
        _chunk("Third chunk text.", source="c.pdf", idx=2),
    ]
    result = builder.build(chunks, community_summaries=[])
    assert "First chunk text." in result
    assert "Second chunk text." in result
    assert "Third chunk text." in result


def test_build_includes_community_summary_when_provided(builder):
    """Community summaries must be incorporated when passed."""
    chunks = [_chunk("chunk text")]
    summaries = [{"title": "Key Theme", "summary": "Community insight here."}]
    result = builder.build(chunks, community_summaries=summaries)
    assert "Community insight here." in result or "Key Theme" in result, (
        "Community summary content must appear in the assembled context"
    )


# ---------------------------------------------------------------------------
# extract_sources()
# ---------------------------------------------------------------------------


def test_extract_sources_returns_list(builder):
    chunks = [_chunk("t", source="doc.pdf")]
    sources = builder.extract_sources(chunks)
    assert isinstance(sources, list)


def test_extract_sources_deduplicates(builder):
    """The same source_file referenced by multiple chunks must appear once."""
    chunks = [
        _chunk("text1", source="same.pdf", idx=0),
        _chunk("text2", source="same.pdf", idx=1),
        _chunk("text3", source="other.pdf", idx=0),
    ]
    sources = builder.extract_sources(chunks)
    assert any("same.pdf (Page 0)" in s for s in sources)
    # Deduplication check: if source appeared multiple times, check if any unique (source + page) is duplicated
    # same.pdf (Page 0) should appear once, other.pdf (Page 0) should appear once
    assert len(sources) == 2, f"Expected 2 unique source/page entries, got {len(sources)}: {sources}"
    assert any("other.pdf (Page 0)" in s for s in sources)


def test_extract_sources_empty_chunks(builder):
    sources = builder.extract_sources([])
    assert sources == []
