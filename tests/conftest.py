"""
Shared fixtures and import-time patches for GraphNotebook test suite.

CRITICAL: The `litellm.Cache(type="disk")` call at gateway.py module-scope
crashes test collection when `diskcache` is absent. We patch it here via
sys.modules injection BEFORE any graphnotebook.* import occurs.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ── Patch litellm disk cache at import time ──────────────────────────────────

# Build a thin fake `diskcache` module so litellm.Cache(type="disk") succeeds

_fake_diskcache = ModuleType("diskcache")
_fake_diskcache.Cache = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
sys.modules.setdefault("diskcache", _fake_diskcache)

# Also stub the litellm module-level cache assignment so gateway.py passes

import litellm  # noqa: E402 (must come after sys.modules patch)

litellm.cache = MagicMock()

# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def mock_neo4j():
    """Reusable mock Neo4j client with chainable .query side_effect."""
    client = MagicMock()
    client.query.return_value = []
    return client


@pytest.fixture()
def mock_llm():
    """Reusable mock LLMGateway."""
    llm = MagicMock()
    llm.invoke.return_value = "mocked response"
    llm.invoke_json.return_value = {"mode": "local"}
    llm.invoke_stream.return_value = iter(["chunk1", "chunk2"])
    return llm


@pytest.fixture()
def mock_embedding_engine():
    """Mock EmbeddingEngine returning fixed-size float list."""
    import numpy as np

    engine = MagicMock()
    engine.embed.return_value = np.ones((1, 1024), dtype="float32")
    engine.embed_single.return_value = [0.1] * 1024
    engine.dimensions = 1024
    return engine


@pytest.fixture()
def sample_parsed_doc():
    """Minimal ParsedDocument for pipeline tests."""
    from graphnotebook.ingestion.parsers import PageContent, ParsedDocument

    return ParsedDocument(
        filename="sample.pdf",
        file_type="pdf",
        file_hash="abc123deadbeef",
        pages=[PageContent(page_number=1, text="Hello world.")],
        raw_text="Hello world.",
        raw_text_length=12,
        metadata={"page_count": 1},
    )
