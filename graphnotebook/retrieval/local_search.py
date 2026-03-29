"""
Local search: entity-centric retrieval.
Vector similarity → graph traversal → hybrid full-text → rerank.
"""

from typing import List

from .reranker import RetrievedChunk
from graphnotebook.graph import queries


class LocalSearcher:
    """
    Search scoped to specific entities and their immediate graph neighborhood.
    """

    def __init__(self, neo4j_client, notebook_id: str = None):
        self.neo4j = neo4j_client
        self.notebook_id = notebook_id

    def search(
        self, query_embedding: List[float], top_k: int = 20, notebook_id: str = None
    ) -> List[RetrievedChunk]:  # noqa: E501
        """Perform vector + graph traversal search."""
        results = self.neo4j.query(
            queries.LOCAL_SEARCH,
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "notebook_id": notebook_id or self.notebook_id,
            },
        )
        return self._parse_results(results)

    def hybrid_search(
        self, query_embedding: List[float], query_text: str, top_k: int = 20, notebook_id: str = None
    ) -> List[RetrievedChunk]:  # noqa: E501
        """Perform hybrid vector + BM25 + graph traversal search."""
        results = self.neo4j.query(
            queries.HYBRID_SEARCH,
            {
                "query_embedding": query_embedding,
                "query_text": query_text,
                "top_k": top_k,
                "notebook_id": notebook_id or self.notebook_id,
            },
        )
        return self._parse_results(results)

    def _parse_results(self, records: List[dict]) -> List[RetrievedChunk]:
        """Convert Neo4j rows into RetrievedChunk objects."""
        chunks = []
        for row in records:
            # Filter out null entities (from optional matches)
            entities = [
                e for e in row.get("entities", []) 
                if e and e.get("id")
            ]
            relationships = [
                r for r in row.get("relationships", []) 
                if r and r.get("source")
            ]

            chunks.append(
                RetrievedChunk(
                    id=row["chunk_id"],
                    text=row["text"],
                    source=row["source"],
                    score=row.get("vec_score", 0.0),
                    metadata={
                        "entities": entities,
                        "relationships": relationships,
                        "page_number": row.get("page_number")
                    },
                )
            )
        return chunks
