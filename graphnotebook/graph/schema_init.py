"""
Idempotent Neo4j Graph Schema initialization.
Creates constraints, Vector Indexes (1024d), and Fulltext Indexes.
"""

from graphnotebook.config import Settings
from graphnotebook.graph.neo4j_client import Neo4jClient


def initialize_schema(neo4j: Neo4jClient):
    """Run all schema queries to enforce graph structure."""

    # 1. Constraints
    constraints = [
        "CREATE CONSTRAINT notebook_id IF NOT EXISTS FOR (n:Notebook) REQUIRE n.id IS UNIQUE;",  # noqa: E501
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;",  # noqa: E501
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;",  # noqa: E501
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;",  # noqa: E501
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (cm:Community) REQUIRE cm.id IS UNIQUE;",  # noqa: E501,
    ]

    for c in constraints:
        neo4j.query(c)

    # 2. Vector Indexes (1024d for BGE-M3)
    vector_indexes = [
        """
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1024,
            `vector.similarity_function`: 'cosine'
        }};
        """,
        """
        CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
        FOR (e:Entity) ON (e.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1024,
            `vector.similarity_function`: 'cosine'
        }};
        """,
    ]

    for v in vector_indexes:
        # Vector index creation can sometimes be delayed or idempotent blocks
        # Using Cypher supported by neo4j 5.20+
        neo4j.query(v)

    # 3. Full text indexes
    full_text_indexes = [
        """
        CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
        FOR (c:Chunk) ON EACH [c.text]
        OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard'}};
        """
    ]

    for f in full_text_indexes:
        neo4j.query(f)

    print("Graph schema initialization completed successfully.")


if __name__ == "__main__":
    settings = Settings()
    client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    try:
        initialize_schema(client)
    finally:
        client.close()
