"""
Entity resolution: deduplicate entities across documents.
Phase 1: RapidFuzz string similarity.
"""

from rapidfuzz import fuzz, process


class EntityResolver:
    """Merge duplicate entities in the knowledge graph."""

    def __init__(self, neo4j_client, threshold: float = 85.0):
        self.neo4j = neo4j_client
        self.threshold = threshold

    def resolve_all(self):
        """
        Find and merge duplicate entities.
        1. Group entities by type
        2. Within each type, fuzzy-match names
        3. Merge duplicates (keep highest mention_count as canonical)
        """
        # We group by labels instead of explicitly finding "type"
        # The neo4j-graphrag node label is usually the specific type (e.g., :Person).

        # We need a robust query to find standard entities.
        # We'll fetch all nodes that have an ID and a Name, and then resolve them,
        # grouped by their primary label.

        labels_result = self.neo4j.query("CALL db.labels() YIELD label RETURN label")
        # Filter typical non-entity labels
        non_entities = ["Document", "Chunk", "Notebook", "Community", "Entity"]
        valid_labels = [
            r["label"] for r in labels_result if r["label"] not in non_entities
        ]

        for label in valid_labels:
            self._resolve_type(label)

    def _resolve_type(self, label: str):
        """Resolve duplicates within a single entity type (label)."""
        entities = self.neo4j.query(
            f"MATCH (e:`{label}`) "
            "RETURN e.id AS id, e.name AS name, e.mention_count AS mc "
            "ORDER BY e.mention_count DESC"
        )
        if not entities:
            return

        names = [e["name"] for e in entities if e.get("name")]
        if not names:
            return

        merged = set()

        for i, entity in enumerate(entities):
            if not entity.get("name") or entity["id"] in merged:
                continue

            # Find matching names further down the list
            similar_names = names[i + 1 :]
            if not similar_names:
                break

            matches = process.extract(
                entity["name"],
                similar_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.threshold,
            )

            for match_name, score, idx in matches:
                match_entity = entities[i + 1 + idx]
                if match_entity["id"] not in merged:
                    self._merge_entities(entity["id"], match_entity["id"], label)
                    merged.add(match_entity["id"])

    def _merge_entities(self, keep_id: str, merge_id: str, label: str):
        """Merge merge_id entity into keep_id entity in Neo4j."""
        self.neo4j.query(
            f"""
            MATCH (keep:`{label}` {{id: $keep_id}})
            MATCH (merge:`{label}` {{id: $merge_id}})

            // Transfer all relationships
            CALL {{
                WITH keep, merge
                MATCH (merge)-[r]->(other)
                WHERE other <> keep
                CALL apoc.create.relationship(
                    keep, type(r), properties(r), other
                ) YIELD rel AS r_out
                RETURN r_out
            }}
            
            CALL {{
                WITH keep, merge
                MATCH (other)-[r]->(merge)
                WHERE other <> keep
                CALL apoc.create.relationship(
                    other, type(r), properties(r), keep
                ) YIELD rel AS r_in
                RETURN r_in
            }}

            // Delete merged entity
            DETACH DELETE merge
        """,
            parameters={"keep_id": keep_id, "merge_id": merge_id},
        )
