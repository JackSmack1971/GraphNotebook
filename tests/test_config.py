import os
from graphnotebook.config import Settings

def test_settings_load(monkeypatch):
    """Test environment variable overrides."""
    monkeypatch.setenv("GN_NEO4J_URI", "bolt://testuser:7687")
    monkeypatch.setenv("GN_OPENROUTER_API_KEY", "test-key-123")
    
    settings = Settings()
    
    assert settings.neo4j_uri == "bolt://testuser:7687"
    assert settings.openrouter_api_key == "test-key-123"
    # Defaults should remain
    assert settings.embedding_model == "BAAI/bge-m3"
