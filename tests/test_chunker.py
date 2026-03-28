import pytest
from graphnotebook.ingestion.chunker import SemanticChunker

def test_semantic_chunker():
    chunker = SemanticChunker(chunk_size=10, chunk_overlap=2)
    
    # A long text with paragraphs
    text = "Paragraph one is short.\n\nParagraph two is slightly longer but still a paragraph.\n\nParagraph three."  # noqa: E501
    chunks = chunker.chunk_text(text, doc_id="testdoc")
    
    assert len(chunks) > 0
    assert chunks[0].id.startswith("testdoc_chunk_")
    
    # Check that chunks overlap correctly or are bound by paragraphs
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char > 0
    assert chunks[0].token_count > 0
    
    # All text must be covered
    full_chunked_text = " ".join([c.text for c in chunks])
    assert "Paragraph one" in full_chunked_text
    assert "Paragraph three." in full_chunked_text
