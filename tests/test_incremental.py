import pytest
from unittest.mock import MagicMock
from graphnotebook.ingestion.pipeline import ingestion_pipeline

@pytest.mark.asyncio
async def test_incremental_skip_doc():
    mock_neo4j = MagicMock()
    # Mock CHECK_DOC_HASH result
    mock_neo4j.query.return_value = [{"exists": True, "doc_id": "123"}]
    
    state = {
        "file_path": "test.pdf",
        "notebook_id": "nb1",
        "neo4j_client": mock_neo4j,
        "status": "started"
    }
    
    # We only run the 'parse' step for this test
    # (Simplified: in a real test we'd invoke the whole graph but stub out 
    # the parse_document function to control the hash)
    from graphnotebook.ingestion.pipeline import parse_step
    from graphnotebook.ingestion.parsers import ParsedDocument
    
    # Monkeypatch parse_document
    import graphnotebook.ingestion.pipeline as pipeline
    original_parse = pipeline.parse_document
    pipeline.parse_document = lambda path: ParsedDocument(
        filename="test.pdf", file_type="pdf", raw_text="test", 
        file_hash="same_hash", raw_text_length=4, pages=[]
    )
    
    try:
        new_state = await parse_step(state)
        assert new_state["status"] == "failed"
        assert "already indexed" in new_state["error"]
    finally:
        pipeline.parse_document = original_parse

@pytest.mark.asyncio
async def test_incremental_reingest_filename_collision():
    mock_neo4j = MagicMock()
    # Mock CHECK_DOC_HASH result (hash is different)
    # Mock filename match result
    mock_neo4j.query.side_effect = [
        [{"exists": False}], # Hash doesn't match
        [{"id": "old_id"}],  # Filename matches
        None,                # DELETE_DOC_CASCADE call
        [{"count": 0}]       # Entity counting query
    ]
    
    state = {
        "file_path": "test.pdf",
        "notebook_id": "nb1",
        "neo4j_client": mock_neo4j,
        "status": "started"
    }
    
    from graphnotebook.ingestion.pipeline import parse_step
    from graphnotebook.ingestion.parsers import ParsedDocument
    
    import graphnotebook.ingestion.pipeline as pipeline
    original_parse = pipeline.parse_document
    pipeline.parse_document = lambda path: ParsedDocument(
        filename="test.pdf", file_type="pdf", raw_text="test", 
        file_hash="new_hash", raw_text_length=4, pages=[]
    )
    
    try:
        new_state = await parse_step(state)
        assert new_state["status"] == "parsed"
        # Verify DELETE_DOC_CASCADE was called
        calls = mock_neo4j.query.call_args_list
        assert any("DELETE_DOC_CASCADE" in str(c) for c in calls)
    finally:
        pipeline.parse_document = original_parse
