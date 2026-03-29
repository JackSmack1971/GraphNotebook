"""Tests for graphnotebook.config.Settings."""

from graphnotebook.config import Settings


def test_settings_defaults():
    """Verify all expected defaults are present without env override."""
    s = Settings()
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    assert s.neo4j_uri.startswith("bolt://")


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("GN_NEO4J_URI", "bolt://override:7687")
    monkeypatch.setenv("GN_OPENROUTER_API_KEY", "key-abc")
    s = Settings()
    assert s.neo4j_uri == "bolt://override:7687"
    assert s.openrouter_api_key == "key-abc"


def test_settings_prefix_isolation(monkeypatch):
    """Env vars WITHOUT GN_ prefix must not bleed into settings."""
    monkeypatch.setenv("NEO4J_URI", "bolt://leaked:7687")
    s = Settings()
    assert s.neo4j_uri != "bolt://leaked:7687"
