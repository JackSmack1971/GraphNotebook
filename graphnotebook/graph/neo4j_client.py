"""
Neo4j driver context management and query execution.
Connection pool management and basic CRUD wrapper.
"""

from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase


class Neo4jClient:
    """Wrapper around Neo4j driver providing simple query execution."""

    def __getstate__(self):
        """Exclude non-deepcopyable driver/locks for Gradio gr.State."""
        state = self.__dict__.copy()
        state.pop("driver", None)  # Gradio state doesn't need to pickle the driver
        return state

    def __setstate__(self, state):
        """Recreate driver on unpickle (Gradio session restore)."""
        self.__dict__.update(state)
        if hasattr(self, "uri") and hasattr(self, "auth"):
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
        else:
            self.driver = None

    def __deepcopy__(self, memo):
        """Explicit support for copy.deepcopy used by Gradio State."""
        import copy
        return copy.deepcopy(self.__getstate__(), memo)

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        self.uri = uri
        self.auth = (user, password)
        self.driver = GraphDatabase.driver(self.uri, auth=self.auth)

    def close(self):
        """Close driver connection pool."""
        if hasattr(self, "driver") and self.driver:
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
            if len(result) == 1 and result[0]["test"] == 1:
                print("✅ Neo4j health check passed.")
                return True
            return False
        except Exception:
            print("❌ Neo4j health check failed: Couldn't connect to localhost:7687")
            print("\nThe Neo4j graph database is not currently running.")
            print("GraphNotebook requires Neo4j to store the knowledge graph.")
            print("\n✅ Quick fix (run these commands in your project directory):")
            print("   1. docker compose up -d neo4j")
            print("   2. Wait ~30 seconds, then open http://localhost:7474")
            print("      Login: neo4j / graphnotebook")
            print("   3. Initialize schema (first time only):")
            print("      uv run python -m graphnotebook.graph.schema_init")
            print("\nAfter that, restart the app with:")
            print("   uv run python -m graphnotebook.main")
            print("The UI will then be available at http://localhost:7860")
            self.close()
            return False
