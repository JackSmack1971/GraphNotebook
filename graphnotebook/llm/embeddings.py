"""
Local embedding engine using sentence-transformers.
Zero API cost, zero rate limits.

Singleton guarantee: SentenceTransformer (BGE-M3, ~1.1 GB) is loaded once
per model name per process. Subsequent EmbeddingEngine(same_model) calls reuse
the cached instance. This matches the GEMINI.md/CLAUDE.md constraint:
    "Load SentenceTransformer (BGE-M3, 1.1GB) once at startup."
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

# Module-level cache: model_name → SentenceTransformer instance
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def _get_or_load(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer, loading once on first call."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class EmbeddingEngine:
    """BGE-M3 embedder with batch encoding and process-level model singleton."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self.model = _get_or_load(model_name)
        self.dimensions: int = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch-encode texts into normalised embedding matrix."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

    def embed_single(self, text: str) -> List[float]:
        """Encode a single text and return as a plain float list (Neo4j-safe)."""
        return self.model.encode(text, normalize_embeddings=True).tolist()
