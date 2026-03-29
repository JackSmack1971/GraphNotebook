import pytest
from unittest.mock import MagicMock
from graphnotebook.ingestion.pipeline import parse_step
from graphnotebook.ingestion.parsers import ParsedDocument
import graphnotebook.ingestion.pipeline as pipeline


@pytest.mark.asyncio
async def test_incremental_skip_exact_duplicate():
    """Test that exact file + schema match skips everything."""
    mock_neo4j = MagicMock()
    # Mock result showing same file hash and same schema hash
    mock_neo4j.query.return_value = [{"schema_hash": "hash_v1", "id": "doc123"}]

    state = {
        "file_path": "test.pdf",
        "notebook_id": "nb1",
        "notebook_schema_hash": "hash_v1",
        "neo4j_client": mock_neo4j,
        "status": "started",
    }

    original_parse = pipeline.parse_document
    pipeline.parse_document = lambda path: ParsedDocument(
        filename="test.pdf",
        file_type="pdf",
        raw_text="test",
        file_hash="same_hash",
        raw_text_length=4,
        pages=[],
    )

    try:
        new_state = await parse_step(state)
        assert new_state["status"] == "complete"
        assert new_state["skip_chunking"] is True
        assert new_state["skip_extraction"] is True
    finally:
        pipeline.parse_document = original_parse


@pytest.mark.asyncio
async def test_incremental_reextract_on_schema_change():
    """Test that file match but schema change triggered re-extraction."""
    mock_neo4j = MagicMock()
    # Mock result showing same file hash but OLD schema hash
    mock_neo4j.query.return_value = [{"schema_hash": "hash_v1", "id": "doc123"}]

    state = {
        "file_path": "test.pdf",
        "notebook_id": "nb1",
        "notebook_schema_hash": "hash_v2", # NEW Schema
        "neo4j_client": mock_neo4j,
        "status": "started",
    }

    original_parse = pipeline.parse_document
    pipeline.parse_document = lambda path: ParsedDocument(
        filename="test.pdf",
        file_type="pdf",
        raw_text="test",
        file_hash="same_hash",
        raw_text_length=4,
        pages=[],
    )

    try:
        new_state = await parse_step(state)
        assert new_state["status"] == "re-extracting"
        assert new_state["skip_chunking"] is True
        assert new_state["skip_extraction"] is False
    finally:
        pipeline.parse_document = original_parse


@pytest.mark.asyncio
async def test_incremental_full_ingest_on_new_file():
    """Test that a new file runs the full pipeline."""
    mock_neo4j = MagicMock()
    # Mock result showing NO file match
    mock_neo4j.query.side_effect = [
        [], # No hash match
        [], # No filename match
    ]

    state = {
        "file_path": "new.pdf",
        "notebook_id": "nb1",
        "notebook_schema_hash": "hash_v1",
        "neo4j_client": mock_neo4j,
        "status": "started",
    }

    original_parse = pipeline.parse_document
    pipeline.parse_document = lambda path: ParsedDocument(
        filename="new.pdf",
        file_type="pdf",
        raw_text="test",
        file_hash="new_hash",
        raw_text_length=4,
        pages=[],
    )

    try:
        new_state = await parse_step(state)
        assert new_state["status"] == "parsed"
        assert new_state.get("skip_chunking") is not True
        assert new_state.get("skip_extraction") is not True
    finally:
        pipeline.parse_document = original_parse
