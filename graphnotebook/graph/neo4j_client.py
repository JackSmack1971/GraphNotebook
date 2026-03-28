"""
Neo4j driver context management and query execution.
Connection pool management and basic CRUD wrapper.
"""

from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase


class Neo4jClient:
    """Wrapper around Neo4j driver providing simple query execution."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close driver connection pool."""
        self.driver.close()

    def query(
        self, cypher: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute Cypher query and return list of dictionaries.
        Creates a session, executes query, returns materialized list.
        """
        if parameters is None:
            parameters = {}

        with self.driver.session() as session:
            result = session.run(cypher, parameters)
            # Fetch all records and return as dictionaries
            return [dict(record) for record in result]

    def health_check(self) -> bool:
        """Verify connection to Neo4j is successful."""
        try:
            result = self.query("RETURN 1 AS test")
            return len(result) == 1 and result[0]["test"] == 1
        except Exception as e:
            print(f"Neo4j health check failed: {e}")
            return False
