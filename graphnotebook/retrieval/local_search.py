"""
Local search: entity-centric retrieval.
Vector similarity → graph traversal → hybrid full-text → rerank.
"""

from typing import List

from .reranker import RetrievedChunk

# ── Local Search ─────────────
LOCAL_SEARCH_CYPHER = """
// 1. Vector search for relevant chunks
CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
YIELD node AS chunk, score AS vec_score

// 2. Get document source
MATCH (chunk)<-[:HAS_CHUNK]-(doc:Document)

// 3. Find entities mentioned in those chunks (neo4j-graphrag extracts specific labels)
// In neo4j-graphrag, chunk nodes are linked to extract entities via MENTIONS
// To keep it standard, assuming chunks MENTION entities. 
// If not, fallback to document.
OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity)
WHERE labels(entity)[0] IN [
    'Person', 'Organization', 'Concept', 'Technology', 
    'Location', 'Event', 'Metric'
]

// 4. Traverse 2-hop entity relationships
OPTIONAL MATCH (entity)-[r]-(neighbor)
WHERE labels(neighbor)[0] IN [
    'Person', 'Organization', 'Concept', 'Technology', 
    'Location', 'Event', 'Metric'
]

// 5. Get community context (Optional phase 2 addition)
OPTIONAL MATCH (entity)-[:BELONGS_TO]->(community:Community)

// 6. Return enriched context
RETURN chunk.text AS chunk_text,
       chunk.id AS chunk_id,
       vec_score,
       doc.filename AS source_file,
       chunk.chunk_index AS chunk_index,
       chunk.page_number AS page_number,
       collect(DISTINCT {
         name: entity.id,
         type: labels(entity)[0]
       }) AS entities,
       collect(DISTINCT {
         source: entity.id,
         rel: type(r),
         target: neighbor.id
       }) AS relationships,
       community.title AS community_title
ORDER BY vec_score DESC
LIMIT $top_k
"""

# ── Hybrid: combine vector + full-text ──────────────
HYBRID_SEARCH_CYPHER = """
// Vector results
CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
YIELD node AS chunk, score AS vec_score
WITH collect({node: chunk, score: vec_score, source: 'vector'}) AS vec_results

// Full-text results
CALL db.index.fulltext.queryNodes('chunk_fulltext', $query_text, {limit: $top_k})
YIELD node AS ft_chunk, score AS ft_score
WITH vec_results,
     collect({node: ft_chunk, score: ft_score, source: 'fulltext'}) AS ft_results

// Merge and deduplicate
WITH vec_results + ft_results AS all_results
UNWIND all_results AS result
WITH result.node AS chunk,
     max(result.score) AS best_score
     
MATCH (chunk)<-[:HAS_CHUNK]-(doc:Document)

OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity)
WHERE labels(entity)[0] IN [
    'Person', 'Organization', 'Concept', 'Technology', 
    'Location', 'Event', 'Metric'
]

OPTIONAL MATCH (entity)-[r]-(neighbor)
WHERE labels(neighbor)[0] IN [
    'Person', 'Organization', 'Concept', 'Technology', 
    'Location', 'Event', 'Metric'
]

OPTIONAL MATCH (entity)-[:BELONGS_TO]->(community:Community)

RETURN chunk.text AS chunk_text,
       chunk.id AS chunk_id,
       best_score,
       doc.filename AS source_file,
       chunk.chunk_index AS chunk_index,
       chunk.page_number AS page_number,
       collect(DISTINCT {name: entity.id, type: labels(entity)[0]}) AS entities,
       collect(DISTINCT {
         source: entity.id, rel: type(r), target: neighbor.id
       }) AS relationships,
       community.title AS community_title
ORDER BY best_score DESC
LIMIT $top_k
"""


class LocalSearcher:
    """Execute complex retrieval queries returning structured chunks."""

    def __init__(self, neo4j_client):
        self.neo4j = neo4j_client

    def search(
        self, query_embedding: List[float], top_k: int = 20
    ) -> List[RetrievedChunk]:  # noqa: E501
        """Perform vector + graph traversal search."""
        results = self.neo4j.query(
            LOCAL_SEARCH_CYPHER, {"query_embedding": query_embedding, "top_k": top_k}
        )
        return self._parse_results(results)

    def hybrid_search(
        self, query_embedding: List[float], query_text: str, top_k: int = 20
    ) -> List[RetrievedChunk]:  # noqa: E501
        """Perform hybrid vector + BM25 + graph traversal search."""
        results = self.neo4j.query(
            HYBRID_SEARCH_CYPHER,
            {
                "query_embedding": query_embedding,
                "query_text": query_text,
                "top_k": top_k,
            },
        )
        return self._parse_results(results)

    def _parse_results(self, records: List[dict]) -> List[RetrievedChunk]:
        """Convert Neo4j rows into RetrievedChunk objects."""
        chunks = []
        for row in records:
            # Filter out null entities (from optional matches)
            entities = [e for e in row.get("entities", []) if e and e.get("name")]
            relationships = [
                r for r in row.get("relationships", []) if r and r.get("source")
            ]  # noqa: E501

            chunks.append(
                RetrievedChunk(
                    text=row["chunk_text"],
                    score=row.get("best_score", row.get("vec_score", 0.0)),
                    source_file=row["source_file"],
                    chunk_index=row["chunk_index"],
                    entities=entities,
                    relationships=relationships,
                    community_context=row.get("community_title") or "",
                )
            )
        return chunks
