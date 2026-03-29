import os

from graphnotebook.config import Settings
from graphnotebook.graph.neo4j_client import Neo4jClient
from graphnotebook.llm.embeddings import EmbeddingEngine
from graphnotebook.llm.gateway import LLMGateway
from graphnotebook.ui.app import build_app


def main():
    # 1. Load config
    settings = Settings()

    # 2. Check essential environment variables
    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key

    # 3. Initialize components
    # Instantiate Neo4j Driver (one instance for the app)
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    # Initialize gateways and embeddings
    llm_gateway = LLMGateway(task="synthesis")
    embedding_engine = EmbeddingEngine(model_name=settings.embedding_model)

    try:
        # Check Neo4j Health
        if not neo4j_client.health_check():
            print("Warning: Neo4j health check failed. Ensure the database is running.")
        else:
            # 5. Run Phase 4 Migration
            print("Running Phase 4 Automatic Migration (Idempotent)...")
            from graphnotebook.graph import queries
            res = neo4j_client.query(queries.MIGRATE_ENTITIES_TO_NOTEBOOKS)
            if res:
                count = res[0].get("count", 0)
                print(f"Migrated {count} orphaned entities to notebooks.")

        # 4. Build and launch Gradio UI
        app = build_app(
            settings=settings,
            neo4j_client=neo4j_client,
            llm_gateway=llm_gateway,
            embedding_engine=embedding_engine,
        )

        # Launch non-blocking (or blocking based on use case, here we block)
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            theme=gr.themes.Soft(),
            share=False,
            show_error=True,
        )
    finally:
        # Graceful shutdown
        print("Shutting down Neo4j client...")
        neo4j_client.close()


if __name__ == "__main__":
    main()
