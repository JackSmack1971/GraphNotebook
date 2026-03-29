"""Tests for graphnotebook.config.Settings.

Covers:
- Default field values (kills AOR/InlineConstant mutants on every default)
- Environment-variable override via GN_ prefix
- GN_ prefix isolation (non-prefixed env vars must not bleed in)
- cross_encoder_model and embedding_dimensions fields (previously missing / broken)
"""

import pytest

from graphnotebook.config import Settings


# ---------------------------------------------------------------------------
# Happy-path defaults
# ---------------------------------------------------------------------------


def test_settings_defaults():
    """Every expected default must be present and correctly typed."""
    s = Settings()

    # Embedding
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dimensions == 1024          # was broken — field now confirmed

    # Chunking
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    assert s.encoding_name == "cl100k_base"

    # Retrieval
    assert s.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.rerank_top_k == 8
    assert s.local_top_k == 20

    # Neo4j
    assert s.neo4j_uri.startswith("bolt://")
    assert s.neo4j_user == "neo4j"


# ---------------------------------------------------------------------------
# Environment-variable override
# ---------------------------------------------------------------------------


def test_settings_env_override(monkeypatch):
    """GN_-prefixed env vars must override the corresponding defaults."""
    monkeypatch.setenv("GN_NEO4J_URI", "bolt://override:7687")
    monkeypatch.setenv("GN_OPENROUTER_API_KEY", "key-abc")
    monkeypatch.setenv("GN_CHUNK_SIZE", "256")
    monkeypatch.setenv("GN_EMBEDDING_DIMENSIONS", "768")

    s = Settings()

    assert s.neo4j_uri == "bolt://override:7687"
    assert s.openrouter_api_key == "key-abc"
    assert s.chunk_size == 256
    assert s.embedding_dimensions == 768          # override must propagate


# ---------------------------------------------------------------------------
# Prefix isolation
# ---------------------------------------------------------------------------


def test_settings_prefix_isolation(monkeypatch):
    """Env vars WITHOUT the GN_ prefix must never bleed into Settings."""
    monkeypatch.setenv("NEO4J_URI", "bolt://leaked:7687")
    monkeypatch.setenv("CHUNK_SIZE", "99")

    s = Settings()

    assert s.neo4j_uri != "bolt://leaked:7687", (
        "Non-prefixed NEO4J_URI bled into Settings — GN_ isolation broken"
    )
    assert s.chunk_size != 99, (
        "Non-prefixed CHUNK_SIZE bled into Settings — GN_ isolation broken"
    )


# ---------------------------------------------------------------------------
# Type safety
# ---------------------------------------------------------------------------


def test_settings_integer_fields_are_int():
    """Numeric settings must be int, not str, even from default construction."""
    s = Settings()
    assert isinstance(s.embedding_dimensions, int)
    assert isinstance(s.chunk_size, int)
    assert isinstance(s.chunk_overlap, int)
    assert isinstance(s.rerank_top_k, int)


def test_settings_cross_encoder_is_non_empty_string():
    """cross_encoder_model must be a non-empty string (used at Reranker init)."""
    s = Settings()
    assert isinstance(s.cross_encoder_model, str)
    assert len(s.cross_encoder_model) > 0
