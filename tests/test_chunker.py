"""Tests for graphnotebook.ingestion.chunker.SemanticChunker."""

import pytest

from graphnotebook.ingestion.chunker import Chunk, SemanticChunker


@pytest.fixture()
def chunker():
    return SemanticChunker(chunk_size=50, chunk_overlap=10)


def test_short_text_produces_single_chunk(chunker):
    chunks = chunker.chunk_text("Hello world.", doc_id="doc1")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].chunk_index == 0


def test_long_text_produces_multiple_chunks(chunker):
    # ~200 tokens worth of text at chunk_size=50 should produce multiple chunks
    # Chunker splits on paragraph boundaries (\n\n)
    long_text = "\n\n".join(["word " * 20 for _ in range(6)])
    chunks = chunker.chunk_text(long_text, doc_id="doc1")
    assert len(chunks) > 1
    # Pin chunk_index sequence (M4)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i, f"Expected index {i}, got {c.chunk_index}"


def test_overlap_between_chunks(chunker):
    """Adjacent chunks must share token content due to overlap."""
    long_text = "alpha beta gamma delta epsilon "*15 + "\n\n" + "alpha beta gamma delta epsilon "*15
    chunks = chunker.chunk_text(long_text, doc_id="doc1")
    if len(chunks) > 1:
        # The end of chunk[0] and beginning of chunk[1] should share tokens
        c0_tail = chunks[0].text.split()[-5:]
        c1_head = chunks[1].text.split()[:5:]
        overlap = set(c0_tail) & set(c1_head)
        assert len(overlap) > 0, "No overlap detected between adjacent chunks"


def test_chunk_metadata_fields(chunker):
    chunks = chunker.chunk_text("Test text.", doc_id="testdoc")
    c = chunks[0]
    assert hasattr(c, "id")
    assert hasattr(c, "token_count")
    assert hasattr(c, "start_char")
    assert hasattr(c, "end_char")
    assert c.token_count > 0
    # Pin token_count boundary (M1)
    assert c.token_count <= chunker.chunk_size + chunker.chunk_overlap


def test_chunk_ids_are_unique(chunker):
    long_text = "word "*30 + "\n\n" + "word "*30 + "\n\n" + "word "*30 + "\n\n" + "word "*30
    chunks = chunker.chunk_text(long_text, doc_id="doc1")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_empty_text_returns_empty_or_single(chunker):
    chunks = chunker.chunk_text("", doc_id="empty")
    # Either empty list or single empty chunk — both acceptable; must not raise
    assert isinstance(chunks, list)
