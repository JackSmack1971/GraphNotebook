"""Ingestion orchestrator. Coordinates parse → chunk → embed → Neo4j storage."""

from __future__ import annotations  # helps with forward references / Python 3.7+

from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

# ── graphnotebook package imports (absolute – most reliable) ─────────────────
from graphnotebook.config import Settings
from graphnotebook.extraction.kg_pipeline import KGConstructor
from graphnotebook.extraction.resolver import EntityResolver
from graphnotebook.graph import queries
from graphnotebook.graph.communities import CommunityManager
from graphnotebook.graph.neo4j_client import Neo4jClient
from graphnotebook.llm.embeddings import EmbeddingEngine
from graphnotebook.llm.gateway import LLMGateway

# ── Local imports (same directory as pipeline.py) ───────────────────────────
# These assume you have:
#   graphnotebook/
#   └── ingestion/          ← wherever pipeline.py lives
#       ├── __init__.py
#       ├── pipeline.py
#       ├── chunker.py
#       └── parsers.py
from .chunker import Chunk, SemanticChunker
from .parsers import ParsedDocument, parse_document


class IngestionState(TypedDict, total=False):
    file_path: str
    notebook_id: str
    parsed_doc: Optional[ParsedDocument]
    chunks: List[Chunk]
    embeddings_stored: bool
    kg_built: bool
    status: str
    error: Optional[str]
    entity_count: int
    # Dependencies (passed through state for simplicity in this pure functional graph)
    config: Optional[Settings]
    neo4j_client: Optional[Neo4jClient]
    embedding_engine: Optional[EmbeddingEngine]
    llm_gateway: Optional[LLMGateway]


async def parse_step(state: IngestionState) -> IngestionState:
    """Parse uploaded document to raw text + metadata."""
    try:
        parsed = parse_document(state["file_path"])

        neo4j = state.get("neo4j_client")
        if neo4j:
            # Check for duplicate in this notebook

            nb_id = state.get("notebook_id", "default")
            result = neo4j.query(
                queries.CHECK_DOC_HASH,
                {"file_hash": parsed.file_hash, "notebook_id": nb_id},
            )

            if result and result[0]["exists"]:
                state["error"] = "Document already indexed in this notebook."
                state["status"] = "failed"
                return state

            # If filename exists in this notebook but hash is different, delete old one
            old_docs = neo4j.query(
                "MATCH (d:Document {filename: $filename, notebook_id: $notebook_id}) "
                "RETURN d.id AS id",
                {"filename": parsed.filename, "notebook_id": nb_id},
            )
            for old in old_docs:
                neo4j.query(queries.DELETE_DOC_CASCADE, {"doc_id": old["id"]})

        state["parsed_doc"] = parsed
        state["status"] = "parsed"
    except Exception as e:
        state["error"] = str(e)
        state["status"] = "failed"
    return state


async def chunk_step(state: IngestionState) -> IngestionState:
    """Split parsed text into semantic chunks."""
    if state.get("error"):
        return state
    try:
        cfg = state.get("config")
        chunker = SemanticChunker(
            chunk_size=cfg.chunk_size if cfg else 512,
            chunk_overlap=cfg.chunk_overlap if cfg else 64,
            encoding_name=cfg.encoding_name if cfg else "cl100k_base",
        )
        doc = state["parsed_doc"]
        if not doc:
            raise ValueError("Parsed document missing in state")

        state["chunks"] = chunker.chunk_text(doc.raw_text, doc_id=doc.file_hash[:8])
        state["status"] = "chunked"
    except Exception as e:
        state["error"] = f"Chunking failed: {e}"
        state["status"] = "failed"
    return state


async def embed_and_store_step(state: IngestionState) -> IngestionState:
    """Embed chunks and store Document + Chunk nodes in Neo4j."""
    if state.get("error"):
        return state

    try:
        embedder = state.get("embedding_engine")
        neo4j = state.get("neo4j_client")

        if not embedder or not neo4j:
            raise ValueError("Embedding engine and Neo4j client required for ingest")

        # 1. Embed chunks
        chunks = state["chunks"]
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)

        # 2. Store Document
        doc = state["parsed_doc"]
        if not doc:
            raise ValueError("Parsed document missing in state")

        doc_params = {
            "id": doc.file_hash,
            "notebook_id": state.get("notebook_id", "default"),
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_hash": doc.file_hash,
            "raw_text_length": doc.raw_text_length,
            "chunk_count": len(chunks),
        }
        neo4j.query(queries.UPSERT_DOCUMENT, doc_params)

        # 3. Store Chunks
        for c, emb in zip(chunks, embeddings):
            chunk_params = {
                "id": c.id,
                "doc_id": doc.file_hash,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "token_count": c.token_count,
                "page_number": c.page_number,
                "section_header": c.section_header,
                "embedding": emb.tolist(),
            }
            neo4j.query(queries.UPSERT_CHUNK, chunk_params)

        state["embeddings_stored"] = True
        state["status"] = "embedded"
    except Exception as e:
        state["error"] = f"Embed/Store failed: {e}"
        state["status"] = "failed"

    return state


async def extract_kg_step(state: IngestionState) -> IngestionState:
    """Run schema-enforced entity/relationship extraction."""
    if state.get("error"):
        return state

    try:
        cfg = state.get("config")
        llm_gateway = state.get("llm_gateway")

        if not cfg or not llm_gateway:
            # Fallback if not injected properly, though we should inject it
            llm_gateway = LLMGateway("extraction")

        kg = KGConstructor(settings=cfg, llm_gateway=llm_gateway)

        # Ingest parsed text
        doc = state["parsed_doc"]
        if not doc:
            raise ValueError("Parsed document missing in state")

        # Async run now native
        await kg.ingest_text(doc.raw_text)

        # ── Safely count entities linked to this document ─────────────────────
        doc_id = doc.file_hash
        neo4j = state.get("neo4j_client")
        if neo4j:
            try:
                count_res = neo4j.query(
                    "MATCH (d:Document {id: $doc_id}) "
                    "OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e) "
                    "RETURN count(DISTINCT e) AS count",
                    {"doc_id": doc_id},
                )
                # count_res is always a list; first row always exists
                # because of OPTIONAL MATCH
                state["entity_count"] = count_res[0]["count"] if count_res else 0
            except Exception as count_err:
                # Don't fail the whole pipeline just because counting failed
                msg = f"Warning: Could not count entities for doc {doc_id}: {count_err}"
                print(msg)
                state["entity_count"] = 0
        else:
            state["entity_count"] = 0

        state["kg_built"] = True
        state["status"] = "extracted"
    except Exception as e:
        state["error"] = f"Extraction failed: {e}"
        state["status"] = "failed"

    return state


async def resolve_entities_step(state: IngestionState) -> IngestionState:
    """Run entity deduplication via RapidFuzz."""
    if state.get("error"):
        return state

    try:
        neo4j = state.get("neo4j_client")
        if not neo4j:
            raise ValueError("Neo4j client required for entity resolution")

        resolver = EntityResolver(neo4j_client=neo4j, threshold=85.0)
        resolver.resolve_all()

        state["status"] = "resolved"
    except Exception as e:
        state["error"] = f"Entity Resolution failed: {e}"
        state["status"] = "failed"

    return state


async def detect_communities_step(state: IngestionState) -> IngestionState:
    """Run Leiden community detection on the entity graph."""
    # We always run it, wait, we should only run if entities were extracted
    if state.get("error"):
        return state

    try:
        neo4j = state.get("neo4j_client")
        if not neo4j:
            raise ValueError("Neo4j client required for community detection")

        manager = CommunityManager(neo4j_client=neo4j)
        manager.detect_communities()

        state["status"] = "complete"
    except Exception as e:
        state["error"] = f"Community Detection failed: {e}"
        state["status"] = "failed"

    return state


# ── Build Graph ─────────────────────────────────────
ingestion_workflow = StateGraph(IngestionState)
ingestion_workflow.add_node("parse", parse_step)
ingestion_workflow.add_node("chunk", chunk_step)
ingestion_workflow.add_node("embed_store", embed_and_store_step)
ingestion_workflow.add_node("extract_kg", extract_kg_step)
ingestion_workflow.add_node("resolve_entities", resolve_entities_step)
ingestion_workflow.add_node("detect_communities", detect_communities_step)

ingestion_workflow.set_entry_point("parse")
ingestion_workflow.add_edge("parse", "chunk")
ingestion_workflow.add_edge("chunk", "embed_store")
ingestion_workflow.add_edge("embed_store", "extract_kg")
ingestion_workflow.add_edge("extract_kg", "resolve_entities")
ingestion_workflow.add_edge("resolve_entities", "detect_communities")
ingestion_workflow.add_edge("detect_communities", END)

ingestion_pipeline = ingestion_workflow.compile()
