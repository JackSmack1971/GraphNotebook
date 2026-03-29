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

    def __init__(self, neo4j_client, embedding_engine=None, notebook_id: str = None):
        self.neo4j = neo4j_client
        self.embedding_engine = embedding_engine
        self.notebook_id = notebook_id

    def search(
        self, query_embedding: List[float] = None, top_k: int = 20, notebook_id: str = None, query_text: str = None
    ) -> List[RetrievedChunk]:
        """Perform vector + graph traversal search."""
        if query_embedding is None and query_text and self.embedding_engine:
            query_embedding = self.embedding_engine.embed_single(query_text)

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
        self, query_text: str, query_embedding: List[float] = None, top_k: int = 20, notebook_id: str = None
    ) -> List[RetrievedChunk]:
        """Perform hybrid vector + BM25 + graph traversal search."""
        if query_embedding is None and self.embedding_engine:
            query_embedding = self.embedding_engine.embed_single(query_text)

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

            # Reconcile field names between queries and RetrievedChunk
            source_file = row.get("source_file") or row.get("source", "")
            # Prefer 'score' if 'vec_score' is missing (for mocks)
            score = row.get("score") if "score" in row else row.get("vec_score", 0.0)

            chunks.append(
                RetrievedChunk(
                    id=row.get("chunk_id", ""),
                    text=row.get("text",row.get("chunk_text", "")),
                    source=source_file,
                    source_file=source_file,
                    chunk_index=row.get("chunk_index", 0),
                    score=float(score),
                    metadata={
                        "entities": entities,
                        "relationships": relationships,
                        "page_number": row.get("page_number")
                    },
                )
            )
        # Results from local search queries are already sorted by score (vec_score),
        # but for robustness and mocks, we apply a final descending score sort.
        return sorted(chunks, key=lambda x: x.score, reverse=True)
