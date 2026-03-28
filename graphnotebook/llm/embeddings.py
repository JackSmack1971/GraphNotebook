"""
Local embedding engine using sentence-transformers.
Zero API cost, zero rate limits.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """BGE-M3 embedder with batch encoding."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch encode texts to embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

    def embed_single(self, text: str) -> List[float]:
        """Encode a single text, return as list for Neo4j."""
        return self.model.encode(text, normalize_embeddings=True).tolist()
