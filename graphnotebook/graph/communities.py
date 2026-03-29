"""
Community detection via Neo4j GDS Leiden algorithm.
Lazy summarization: generate summaries on-demand, cache in Neo4j.
"""

from graphnotebook.llm.gateway import LLMGateway


class CommunityManager:
    """Detect communities and manage lazy summaries."""

    def __init__(self, neo4j_client, notebook_id: str = "", llm_gateway: LLMGateway = None):
        self.neo4j = neo4j_client
        self.notebook_id = notebook_id
        self.llm = llm_gateway or LLMGateway("summarization")

    # ── Detection (runs at ingest time) ─────────────

    def detect_communities(self):
        """Run Leiden community detection on the entity graph."""
        # 1. Drop existing projection if it exists
        self.neo4j.query("""
            CALL gds.graph.exists('entity_graph') YIELD exists
            WITH exists WHERE exists
            CALL gds.graph.drop('entity_graph') YIELD graphName RETURN graphName
        """)

        # 2. Project entity graph into GDS (Scoped to Notebook)
        # Using a Cypher projection to only include entities owned by this notebook
        self.neo4j.query("""
            CALL gds.graph.project.cypher(
                'entity_graph',
                'MATCH (n:Notebook {id: $nb_id})-[:OWNER_OF]->(e:Entity) RETURN id(e) AS id',
                'MATCH (n:Notebook {id: $nb_id})-[:OWNER_OF]->(s)-[r:RELATES_TO]-(t) 
                 WHERE (n)-[:OWNER_OF]->(t) 
                 RETURN id(s) AS source, id(t) AS target, coalesce(r.weight, 1.0) AS weight',
                {parameters: {nb_id: $nb_id}}
            )
        """, params={"nb_id": self.notebook_id})

        # 3. Run Leiden with hierarchical levels
        try:
            result = self.neo4j.query("""
                CALL gds.leiden.write(
                    'entity_graph',
                    {
                        writeProperty: 'community_id',
                        includeIntermediateCommunities: true,
                        maxLevels: 3,
                        gamma: 1.0,
                        theta: 0.01
                    }
                )
                YIELD communityCount, modularity
                RETURN communityCount, modularity
            """)

            # 4. Clear old BELONGS_TO relationships to prep for new mapping
            self.neo4j.query("MATCH (:Entity)-[r:BELONGS_TO]->(:Community) DELETE r")

            # 5. Create Community nodes and link entities, handle cache invalidation
            self.neo4j.query("""
                MATCH (e:Entity)
                WHERE e.community_id IS NOT NULL
                WITH e.community_id AS cid, collect(e) AS members
                MERGE (c:Community {id: toString(cid)})
                
                // Track membership changes to clear stale summaries
                WITH c, members, size(members) AS new_size
                FOREACH (ignore IN CASE WHEN c.entity_count IS NOT NULL 
                         AND c.entity_count <> new_size THEN [1] ELSE [] END |
                    SET c.summary = null, c.title = null, c.key_findings = null
                )
                
                SET c.entity_count = new_size,
                    c.level = coalesce(c.level, 0),
                    c.created_at = coalesce(c.created_at, datetime())
                
                WITH c, members
                UNWIND members AS member
                MERGE (member)-[:BELONGS_TO]->(c)
            """)

            # 6. Delete orphaned communities
            self.neo4j.query("""
                MATCH (c:Community)
                WHERE NOT ()-[:BELONGS_TO]->(c)
                DETACH DELETE c
            """)

            return result
        finally:
            # Always drop the GDS projection to prevent memory leaks
            self.neo4j.query("CALL gds.graph.drop('entity_graph')")

    # ── Lazy Summarization (runs at query time) ─────

    def get_summary(self, community_id: str, force: bool = False) -> dict:
        """
        Get community summary. Generate if not cached.
        Returns: {"title": str, "summary": str, "key_findings": list, "rank": int}
        """
        if not force:
            cached = self._get_cached(community_id)
            if cached:
                return cached

        # Gather community context
        context = self.neo4j.query(
            """
            MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {id: $cid})
            OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity)-[:BELONGS_TO]->(c)
            RETURN
                collect(DISTINCT {
                    name: e.name, type: e.type, desc: e.description
                }) AS entities,
                collect(DISTINCT {
                    source: e.name, rel: r.type,
                    target: other.name, desc: r.description
                }) AS relationships
        """,
            params={"cid": community_id},
        )

        if not context:
            return {"title": "Empty", "summary": "", "key_findings": [], "rank": 0}

        entities = context[0]["entities"]
        rels = context[0]["relationships"]

        # Generate summary via LLM
        summary = self.llm.invoke_json(
            prompt=f"""Summarize this knowledge graph community.

## Entities
{entities}

## Relationships
{rels}

Respond as JSON:
{{
    "title": "Short descriptive title",
    "summary": "2-4 paragraph comprehensive summary",
    "key_findings": ["finding1", "finding2", "finding3"],
    "rank": 0
}}""",
            system=(
                "You are a knowledge graph analyst. "
                "Be precise and comprehensive. Rank defaults to 0."
            ),
        )

        # Cache in Neo4j
        self._cache_summary(community_id, summary)
        return summary

    def get_relevant_summaries(self, query_embedding: list, top_n: int = 5, notebook_id: str = None) -> list:
        """
        For global search: find communities relevant to query via entity embeddings.
        Only summarize the relevant ones (lazy).
        """
        results = self.neo4j.query(
            """
            CALL db.index.vector.queryNodes(
                'entity_embeddings', $top_k, $query_embedding
            ) YIELD node AS entity, score
            MATCH (n:Notebook {id: $notebook_id})
            MATCH (n)-[:OWNER_OF]->(entity)-[:BELONGS_TO]->(c:Community)
            WITH c, count(entity) AS match_count, avg(score) AS avg_score
            ORDER BY match_count DESC, avg_score DESC
            LIMIT $top_n
            RETURN c.id AS community_id,
                   c.summary AS cached_summary,
                   c.title AS title,
                   match_count, avg_score
            """,
            parameters={
                "query_embedding": query_embedding,
                "top_k": top_n * 3,
                "top_n": top_n,
                "notebook_id": notebook_id or self.notebook_id,
            },
        )

        summaries = []
        for row in results:
            if row.get("cached_summary"):
                summaries.append(
                    {
                        "title": row.get("title", ""),
                        "summary": row.get("cached_summary", ""),
                        "community_id": row["community_id"],
                    }
                )
            else:
                s = self.get_summary(row["community_id"])
                summaries.append(
                    {
                        "title": s.get("title", ""),
                        "summary": s.get("summary", ""),
                        "community_id": row["community_id"],
                    }
                )
        return summaries

    def _get_cached(self, community_id: str) -> dict | None:
        result = self.neo4j.query(
            "MATCH (c:Community {id: $cid}) "
            "WHERE c.summary IS NOT NULL "
            "RETURN c.title AS title, c.summary AS summary, "
            "c.key_findings AS key_findings, c.rank AS rank",
            params={"cid": community_id},
        )
        return result[0] if result else None

    def _cache_summary(self, community_id: str, summary: dict):
        self.neo4j.query(
            """
            MATCH (c:Community {id: $cid})
            SET c.title = $title,
                c.summary = $summary,
                c.rank = $rank,
                c.key_findings = $findings,
                c.summarized_at = datetime()
        """,
            params={
                "cid": community_id,
                "title": summary.get("title", ""),
                "summary": summary.get("summary", ""),
                "rank": summary.get("rank", 0),
                "findings": summary.get("key_findings", []),
            },
        )
