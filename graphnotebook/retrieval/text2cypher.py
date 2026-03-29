"""
Text-to-Cypher using Neo4j native ai.text2cypher procedure (5.20+ via APOC).
Falls back to LLM-generated Cypher if native procedure unavailable.
"""


class Text2CypherRetriever:
    """Convert natural language to Cypher and execute."""

    def __init__(self, neo4j_client, llm_gateway=None):
        self.neo4j = neo4j_client
        self.llm = llm_gateway
        self._native_available = self._check_native_support()

    def _check_native_support(self) -> bool:
        """Check if ai.text2cypher is available."""
        try:
            result = self.neo4j.query(
                "CALL dbms.procedures() YIELD name "
                "WHERE name = 'ai.text2cypher' RETURN name"
            )
            return len(result) > 0
        except Exception:
            return False

    def query(self, natural_language: str) -> list:
        """Convert NL to Cypher and execute."""
        if self._native_available:
            return self._native_query(natural_language)
        return self._llm_query(natural_language)

    def _native_query(self, nl: str) -> list:
        """Use Neo4j native ai.text2cypher."""
        try:
            result = self.neo4j.query(
                "CALL ai.text2cypher($query, {schema: $schema}) "
                "YIELD cypher RETURN cypher",
                params={"query": nl, "schema": self._get_schema_description()},
            )
            if result and result[0].get("cypher"):
                return self.neo4j.query(result[0]["cypher"])
        except Exception:
            pass
        return []

    def _llm_query(self, nl: str) -> list:
        """Fallback: generate Cypher via LLM."""
        if not self.llm:
            return []
        schema_desc = self._get_schema_description()

        system_prompt = (
            "Generate ONLY a valid Cypher query against neo4j based on "
            "the schema provided. Do NOT wrap in markdown fences."
        )

        cypher = self.llm.invoke(
            prompt=f"Convert to Cypher:\\nQuestion: {nl}\\nSchema: {schema_desc}",
            system=system_prompt,
        )

        cypher = (
            cypher.strip()
            .strip("`")
            .removeprefix("cypher\n")
            .removeprefix("cypher")
            .strip()
        )

        if "\n" in cypher:
            # removing prefix from multi-line strings if LLM returns fences
            lines = cypher.split("\n")
            if "cypher" in lines[0].lower() or "```" in lines[0]:
                lines = lines[1:]
            if lines and "```" in lines[-1]:
                lines = lines[:-1]
            cypher = "\n".join(lines).strip()

        try:
            return self.neo4j.query(cypher)
        except Exception:
            return []

    def _get_schema_description(self) -> str:
        """Get graph schema for text2cypher context."""
        return (
            "Nodes: Notebook, Document, Chunk, Entity, Community\n"
            "Entity properties: id, name, type, description, mention_count\n"
            "Relationships: CONTAINS, HAS_CHUNK, MENTIONS, RELATES_TO, BELONGS_TO\n"
            "RELATES_TO properties: type, description, weight"
        )
