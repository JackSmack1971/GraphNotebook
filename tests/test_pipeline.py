"""Tests for graphnotebook.ingestion.pipeline step functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphnotebook.ingestion.pipeline import (
    chunk_step,
    detect_communities_step,
    embed_and_store_step,
    extract_kg_step,
    resolve_entities_step,
    should_chunk,
    should_resolve,
)
from graphnotebook.ingestion.parsers import PageContent, ParsedDocument

_SAMPLE_DOC = ParsedDocument(
    filename="test.pdf",
    file_type="pdf",
    file_hash="deadbeef",
    pages=[PageContent(page_number=1, text="Sample text.")],
    raw_text="Sample text.",
    raw_text_length=12,
    metadata={"page_count": 1},
)


@pytest.mark.asyncio
async def test_chunk_step_happy_path():
    state = {
        "parsed_doc": _SAMPLE_DOC,
        "chunks": [],
        "status": "parsed",
    }
    with patch("graphnotebook.ingestion.pipeline.SemanticChunker") as mock_chunker_cls:
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = [MagicMock()]
        mock_chunker_cls.return_value = mock_chunker
        result = await chunk_step(state)
    assert result["status"] == "chunked"
    assert len(result["chunks"]) == 1


@pytest.mark.asyncio
async def test_chunk_step_passthrough_on_error():
    state = {"error": "upstream failed", "status": "failed"}
    result = await chunk_step(state)
    assert result["error"] == "upstream failed"


@pytest.mark.asyncio
async def test_embed_and_store_step_sets_flag(mock_neo4j, mock_embedding_engine):
    state = {
        "parsed_doc": _SAMPLE_DOC,
        "chunks": [MagicMock(text="chunk text")],
        "notebook_id": "nb1",
        "neo4j_client": mock_neo4j,
        "embedding_engine": mock_embedding_engine,
        "status": "chunked",
    }
    with patch(
        "graphnotebook.ingestion.pipeline.EmbeddingEngine",
        return_value=mock_embedding_engine,
    ):
        result = await embed_and_store_step(state)
    assert result["embeddings_stored"] is True
    assert result["status"] == "embedded"


@pytest.mark.asyncio
async def test_extract_kg_step_counts_entities(mock_neo4j):
    mock_neo4j.query.return_value = [{"count": 7}]
    state = {
        "parsed_doc": _SAMPLE_DOC,
        "notebook_id": "nb1",
        "neo4j_client": mock_neo4j,
        "status": "embedded",
        "config": MagicMock(),
        "llm_gateway": MagicMock(),
    }
    with patch("graphnotebook.ingestion.pipeline.KGConstructor") as mock_kg_cls:
        mock_kg = MagicMock()
        mock_kg.ingest_text = AsyncMock(return_value={"nodes": 5})
        mock_kg_cls.return_value = mock_kg
        result = await extract_kg_step(state)
    assert result["kg_built"] is True
    assert result["entity_count"] == 7
    assert result["status"] == "extracted"


@pytest.mark.asyncio
async def test_extract_kg_step_count_failure_graceful(mock_neo4j):
    """Entity count failure must not abort the pipeline."""
    mock_neo4j.query.side_effect = Exception("count query failed")
    state = {
        "parsed_doc": _SAMPLE_DOC,
        "neo4j_client": mock_neo4j,
        "status": "embedded",
        "config": MagicMock(),
        "llm_gateway": MagicMock(),
    }
    with patch("graphnotebook.ingestion.pipeline.KGConstructor") as mock_kg_cls:
        mock_kg = MagicMock()
        mock_kg.ingest_text = AsyncMock(return_value={})
        mock_kg_cls.return_value = mock_kg
        result = await extract_kg_step(state)
    assert result["entity_count"] == 0
    assert result["kg_built"] is True


@pytest.mark.asyncio
async def test_resolve_entities_step_calls_resolver(mock_neo4j):
    state = {"neo4j_client": mock_neo4j, "status": "extracted"}
    with patch("graphnotebook.ingestion.pipeline.EntityResolver") as mock_resolver_cls:
        mock_resolver = MagicMock()
        mock_resolver_cls.return_value = mock_resolver
        result = await resolve_entities_step(state)
    mock_resolver.resolve_all.assert_called_once()
    assert result["status"] == "resolved"


@pytest.mark.asyncio
async def test_detect_communities_step_calls_manager(mock_neo4j, mock_llm):
    state = {
        "neo4j_client": mock_neo4j,
        "llm_gateway": mock_llm,
        "status": "resolved",
    }
    with patch("graphnotebook.ingestion.pipeline.CommunityManager") as mock_cm_cls:
        mock_cm = MagicMock()
        mock_cm_cls.return_value = mock_cm
        result = await detect_communities_step(state)
    mock_cm.detect_communities.assert_called_once()


# ---------------------------------------------------------------------------
# Priority 1: Routing Functions
# ---------------------------------------------------------------------------


def test_should_chunk_returns_end_on_error():
    assert should_chunk({"error": "boom", "status": "failed"}) == "end"


def test_should_chunk_returns_chunk_on_clean_state():
    assert (
        should_chunk({"parsed_doc": object(), "status": "parsed"}) == "chunk"
    )


def test_should_chunk_skips_to_extract_kg_when_skip_chunking():
    assert (
        should_chunk({"skip_chunking": True, "status": "parsed"}) == "extract_kg"
    )


def test_should_resolve_returns_end_on_error():
    assert should_resolve({"error": "fail", "status": "failed"}) == "end"


def test_should_resolve_returns_resolve_on_clean():
    assert should_resolve({"status": "extracted"}) == "resolve"


# ---------------------------------------------------------------------------
# Priority 2: Error Passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_communities_step_passthrough_on_error():
    state = {"error": "upstream boom", "status": "failed"}
    result = await detect_communities_step(state)
    assert result["error"] == "upstream boom"
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_embed_and_store_step_passthrough_on_error():
    state = {"error": "parse failed", "status": "failed"}
    result = await embed_and_store_step(state)
    assert result["error"] == "parse failed"


@pytest.mark.asyncio
async def test_extract_kg_step_passthrough_on_error():
    state = {"error": "embed failed", "status": "failed"}
    result = await extract_kg_step(state)
    assert result["error"] == "embed failed"
