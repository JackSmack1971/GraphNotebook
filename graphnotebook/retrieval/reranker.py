"""
Cross-encoder reranker for retrieval precision.
Pattern: retrieve broadly (top-20) → rerank precisely (top-5).
"""

from dataclasses import dataclass, field
from typing import List

from sentence_transformers import CrossEncoder


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source_file: str
    chunk_index: int
    entities: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    community_context: str = ""


class Reranker:
    """Cross-encoder reranker using ms-marco model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # This will download the model locally upon first instantiation.
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: List[RetrievedChunk], top_k: int = 5
    ) -> List[RetrievedChunk]:
        """Rerank chunks by cross-encoder relevance score."""
        if not chunks:
            return []
            
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # update objects directly
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
            
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
