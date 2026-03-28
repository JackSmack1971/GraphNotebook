import pytest
from unittest.mock import MagicMock
from graphnotebook.retrieval.reranker import Reranker, RetrievedChunk

class MockCrossEncoder:
    def predict(self, pairs):
        # We'll assign higher mock scores if the text contains specific keywords from the query  # noqa: E501
        return [float("paris" in pair[1].lower()) for pair in pairs]

def test_reranker(monkeypatch):
    monkeypatch.setattr("graphnotebook.retrieval.reranker.CrossEncoder", lambda model_name: MockCrossEncoder())  # noqa: E501
    
    reranker = Reranker(model_name="mock-model")
    
    chunks = [
        RetrievedChunk(text="The capital of Spain is Madrid.", score=0.0, source_file="doc1.txt", chunk_index=1),  # noqa: E501
        RetrievedChunk(text="Paris is the capital of France.", score=0.0, source_file="doc1.txt", chunk_index=2),  # noqa: E501
    ]
    
    results = reranker.rerank("What is the capital of France?", chunks, top_k=1)
    
    assert len(results) == 1
    assert "Paris" in results[0].text
    assert "Madrid" not in results[0].text
