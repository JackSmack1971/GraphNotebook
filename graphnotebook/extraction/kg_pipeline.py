"""
Schema-enforced KG construction using neo4j-graphrag SimpleKGPipeline.
Receives pre-parsed text from ingestion/parsers.py (never raw PDF bytes).
LLM client is unified through LLMGateway.
"""

from langchain_community.chat_models import ChatLiteLLM
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from graphnotebook.config import Settings
from graphnotebook.extraction.schema import build_default_schema, build_schema_from_json
from graphnotebook.llm.gateway import LLMGateway


class KGConstructor:
    """
    Schema-enforced knowledge graph builder.
    Wraps neo4j-graphrag SimpleKGPipeline.
    """

    def __init__(self, settings: Settings, llm_gateway: LLMGateway):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

        # Derive LangChain-compatible LLM from gateway's model name
        # (neo4j-graphrag requires LangChain LLM interface)
        self.llm = ChatLiteLLM(
            model=llm_gateway.model,
            temperature=0,
        )

        # Local embedder specific to neo4j-graphrag expectations (requires derived class)  # noqa: E501
        self.embedder = SentenceTransformerEmbeddings(model=settings.embedding_model)

        # Default schema
        self.default_schema = build_default_schema()

    def _build_pipeline(self, schema=None) -> SimpleKGPipeline:
        """Create a KG construction pipeline with given schema."""
        return SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embedder,
            text_splitter=FixedSizeSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            ),
            schema=schema or self.default_schema,
            from_pdf=False,  # always receive pre-parsed text
        )

    async def ingest_text(
        self,
        text: str,
        notebook_schema_json: dict = None,
    ) -> dict:
        """
        Ingest pre-parsed text into the knowledge graph.

        Args:
            text: Raw text from parsers.py
            notebook_schema_json: Optional per-notebook schema override
        """
        schema = (
            build_schema_from_json(notebook_schema_json)
            if notebook_schema_json
            else self.default_schema
        )
        pipeline = self._build_pipeline(schema=schema)
        # run_async executes chunks, extraction, and Neo4j ingest
        await pipeline.run_async(text=text)
        return {"documents": 1, "status": "success"}
