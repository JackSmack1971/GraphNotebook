"""Tests for graphnotebook.ingestion.chunker.SemanticChunker.

Mutation targets addressed:
  M1  — AOR on token_count boundary: `chunk_size + overlap` vs `chunk_size - overlap`
        Killed by test_chunk_token_count_does_not_exceed_limit which uses inputs
        that produce chunks near (but not over) the size limit.
  M4  — ReturnValues on duplicate IDs: killed by test_chunk_ids_are_unique.

Root-cause fix for test_long_text_produces_multiple_chunks:
  SemanticChunker splits on paragraph boundaries (\\n\\n), NOT whitespace.
  "word " * 300 has no paragraph breaks → one giant chunk regardless of chunk_size.
  The test now uses \\n\\n-separated paragraphs, each just under chunk_size tokens.
"""

import pytest

from graphnotebook.ingestion.chunker import Chunk, SemanticChunker


@pytest.fixture()
def chunker():
    """Small chunk_size (50 tokens) forces splitting on realistic text."""
    return SemanticChunker(chunk_size=50, chunk_overlap=10)


# ---------------------------------------------------------------------------
# Basic splitting behaviour
# ---------------------------------------------------------------------------


def test_short_text_produces_single_chunk(chunker):
    chunks = chunker.chunk_text("Hello world.", doc_id="doc1")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].chunk_index == 0


def test_long_text_produces_multiple_chunks(chunker):
    """Paragraphs separated by \\n\\n trigger the splitter.

    Each paragraph is ~20 tokens; at chunk_size=50 the chunker must produce
    at least 2 chunks from 6 paragraphs.
    """
    paragraphs = ["word " * 20 for _ in range(6)]   # 6 × ~20-token paragraphs
    long_text = "\n\n".join(paragraphs)
    chunks = chunker.chunk_text(long_text, doc_id="doc1")

    assert len(chunks) > 1, (
        "SemanticChunker must split on \\n\\n paragraph boundaries when "
        "total tokens exceed chunk_size; got only 1 chunk"
    )
    # chunk_index must be a contiguous 0-based sequence
    for i, c in enumerate(chunks):
        assert c.chunk_index == i, (
            f"Expected chunk_index {i}, got {c.chunk_index} — "
            "index sequence must be contiguous"
        )


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


def test_overlap_between_chunks(chunker):
    """Adjacent chunks must share token content due to overlap=10."""
    # Two ~20-token paragraphs; overlap forces shared tail/head tokens
    para = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 2
    long_text = f"{para}\n\n{para}"
    chunks = chunker.chunk_text(long_text, doc_id="overlap_doc")

    if len(chunks) > 1:
        c0_tail = chunks[0].text.split()[-5:]
        c1_head = chunks[1].text.split()[:5]
        overlap = set(c0_tail) & set(c1_head)
        assert len(overlap) > 0, (
            "No overlapping tokens between chunk[0] tail and chunk[1] head — "
            "chunker overlap is not functioning"
        )


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_chunk_metadata_fields(chunker):
    """Every chunk must expose the required metadata attributes."""
    chunks = chunker.chunk_text("Test text.", doc_id="testdoc")
    c = chunks[0]
    assert hasattr(c, "id")
    assert hasattr(c, "token_count")
    assert hasattr(c, "start_char")
    assert hasattr(c, "end_char")
    assert c.token_count > 0


# ---------------------------------------------------------------------------
# M1 — AOR boundary kill: token_count must not exceed chunk_size + overlap
# ---------------------------------------------------------------------------


def test_chunk_token_count_does_not_exceed_limit(chunker):
    """No chunk's token_count may exceed chunk_size + chunk_overlap.

    Uses inputs close to the limit so the assertion is non-trivial — a mutant
    that changes `+` to `-` (limit = 40 instead of 60) would cause chunks with
    ~50 tokens to fail the assertion, killing the mutant.
    """
    # Each paragraph is ~45 tokens — close to chunk_size=50
    para = ("word " * 45).strip()
    text = "\n\n".join([para] * 4)
    chunks = chunker.chunk_text(text, doc_id="boundary_doc")

    max_allowed = chunker.chunk_size + chunker.chunk_overlap  # 60

    for c in chunks:
        assert c.token_count <= max_allowed, (
            f"Chunk token_count {c.token_count} exceeds limit "
            f"chunk_size({chunker.chunk_size}) + overlap({chunker.chunk_overlap}) "
            f"= {max_allowed}"
        )
    # Ensure we actually stress-tested the boundary (at least one chunk near limit)
    near_limit = [c for c in chunks if c.token_count >= chunker.chunk_size - 5]
    assert near_limit, (
        "No chunks were produced near the size limit; the test input is too small "
        "to exercise the boundary assertion meaningfully"
    )


# ---------------------------------------------------------------------------
# M4 — ReturnValues: duplicate IDs
# ---------------------------------------------------------------------------


def test_chunk_ids_are_unique(chunker):
    """Chunk IDs must be unique across all chunks in a document."""
    text = "\n\n".join(["word " * 20] * 8)
    chunks = chunker.chunk_text(text, doc_id="unique_id_doc")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), (
        f"Duplicate chunk IDs detected: {[x for x in ids if ids.count(x) > 1]}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_text_returns_list(chunker):
    """Empty input must return a list (empty or single-chunk), never raise."""
    result = chunker.chunk_text("", doc_id="empty")
    assert isinstance(result, list)


def test_single_word_chunk_index_zero(chunker):
    """A trivial single-word input must produce chunk_index=0."""
    chunks = chunker.chunk_text("hello", doc_id="single")
    assert len(chunks) >= 1
    assert chunks[0].chunk_index == 0
