"""Tests for graphnotebook.llm.embeddings.EmbeddingEngine.

FIX 7 — Singleton enforcement:
  test_single_instance_not_reloaded now asserts mock_cls.call_count == 1,
  not >= 1. The source has been updated to use a module-level _MODEL_CACHE
  dict so that EmbeddingEngine("BAAI/bge-m3") called twice only loads once.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixture: patch SentenceTransformer for every test
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_sentence_transformer():
    """Patch SentenceTransformer AND reset the module-level cache so each
    test starts from a clean state."""
    with patch("graphnotebook.llm.embeddings.SentenceTransformer") as mock_cls, \
         patch("graphnotebook.llm.embeddings._MODEL_CACHE", {}):
        instance = MagicMock()
        instance.get_sentence_embedding_dimension.return_value = 1024
        instance.encode.return_value = np.ones((1, 1024), dtype="float32")
        mock_cls.return_value = instance
        yield mock_cls, instance


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_embedding_engine_init(mock_sentence_transformer):
    from graphnotebook.llm.embeddings import EmbeddingEngine

    mock_cls, instance = mock_sentence_transformer
    engine = EmbeddingEngine("BAAI/bge-m3")
    mock_cls.assert_called_once_with("BAAI/bge-m3")
    assert engine.dimensions == 1024
    assert engine.model_name == "BAAI/bge-m3"


# ---------------------------------------------------------------------------
# embed (batch)
# ---------------------------------------------------------------------------


def test_embed_batch_shape(mock_sentence_transformer):
    from graphnotebook.llm.embeddings import EmbeddingEngine

    _, instance = mock_sentence_transformer
    instance.encode.return_value = np.ones((3, 1024), dtype="float32")
    engine = EmbeddingEngine()
    result = engine.embed(["a", "b", "c"], batch_size=2)
    assert result.shape == (3, 1024)


def test_embed_batch_uses_normalize(mock_sentence_transformer):
    """normalize_embeddings=True must always be passed."""
    from graphnotebook.llm.embeddings import EmbeddingEngine

    _, instance = mock_sentence_transformer
    instance.encode.return_value = np.ones((2, 1024), dtype="float32")
    engine = EmbeddingEngine()
    engine.embed(["x", "y"])
    call_kwargs = instance.encode.call_args[1]
    assert call_kwargs.get("normalize_embeddings") is True, (
        "normalize_embeddings=True must be passed to batch encode"
    )


# ---------------------------------------------------------------------------
# embed_single
# ---------------------------------------------------------------------------


def test_embed_single_returns_list(mock_sentence_transformer):
    from graphnotebook.llm.embeddings import EmbeddingEngine

    _, instance = mock_sentence_transformer
    instance.encode.return_value = np.array([0.5] * 1024)
    engine = EmbeddingEngine()
    result = engine.embed_single("hello")
    assert isinstance(result, list)
    assert len(result) == 1024


def test_embed_single_uses_normalize(mock_sentence_transformer):
    from graphnotebook.llm.embeddings import EmbeddingEngine

    _, instance = mock_sentence_transformer
    instance.encode.return_value = np.array([0.1] * 1024)
    engine = EmbeddingEngine()
    engine.embed_single("test")
    call_kwargs = instance.encode.call_args[1]
    assert call_kwargs.get("normalize_embeddings") is True


# ---------------------------------------------------------------------------
# FIX 7 — Singleton: SentenceTransformer loaded exactly once per model name
# ---------------------------------------------------------------------------


def test_single_instance_not_reloaded(mock_sentence_transformer):
    """Second EmbeddingEngine with same model_name must reuse the cached model.

    Upgraded from `>= 1` to `== 1` — enforces the singleton contract required
    by GEMINI.md: 'Load SentenceTransformer (BGE-M3, 1.1GB) once at startup.'
    """
    from graphnotebook.llm.embeddings import EmbeddingEngine

    mock_cls, _ = mock_sentence_transformer
    EmbeddingEngine("BAAI/bge-m3")
    EmbeddingEngine("BAAI/bge-m3")   # second call — must hit cache, not reload

    assert mock_cls.call_count == 1, (
        f"SentenceTransformer was instantiated {mock_cls.call_count} times; "
        "expected exactly 1 (singleton violation — 1.1 GB model reloaded)"
    )


def test_different_model_names_load_separately(mock_sentence_transformer):
    """Two different model names must each trigger one load (separate entries)."""
    from graphnotebook.llm.embeddings import EmbeddingEngine

    mock_cls, instance = mock_sentence_transformer
    # Second model returns different dimension to distinguish instances
    instance2 = MagicMock()
    instance2.get_sentence_embedding_dimension.return_value = 768
    instance2.encode.return_value = np.ones((1, 768), dtype="float32")
    mock_cls.side_effect = [instance, instance2]

    e1 = EmbeddingEngine("BAAI/bge-m3")
    e2 = EmbeddingEngine("BAAI/bge-base-en-v1.5")

    assert mock_cls.call_count == 2, (
        "Different model names must each trigger a separate SentenceTransformer load"
    )
    assert e1.dimensions == 1024
    assert e2.dimensions == 768
