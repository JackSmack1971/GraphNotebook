# GraphNotebook v1.1 — Final Architecture

> *Personal GraphRAG knowledge base. NotebookLM meets knowledge graphs.*
> *Single-user. Zero cost. Full graph intelligence.*
>
> **Status**: Final consolidated architecture (v1.0 + revision delta + review fixes merged)
> **Date**: March 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Tech Stack](#2-tech-stack)
3. [Module Architecture](#3-module-architecture)
4. [Neo4j Graph Schema](#4-neo4j-graph-schema)
5. [LLM Gateway (LiteLLM)](#5-llm-gateway-litellm)
6. [Document Ingestion Pipeline](#6-document-ingestion-pipeline)
7. [Schema-Enforced Knowledge Graph Construction](#7-schema-enforced-knowledge-graph-construction)
8. [Community Detection & Lazy Summarization](#8-community-detection--lazy-summarization)
9. [Retrieval: Agentic Router + Reranker](#9-retrieval-agentic-router--reranker)
10. [Gradio UI](#10-gradio-ui)
11. [Configuration](#11-configuration)
12. [Docker Compose](#12-docker-compose)
13. [Implementation Roadmap](#13-implementation-roadmap)
14. [Architecture Decision Records](#14-architecture-decision-records)
15. [Dependencies](#15-dependencies)
16. [Quick Start](#16-quick-start)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GRADIO UI                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Upload   │  │ Notebook  │  │ Chat /   │  │ Graph Explorer /  │  │
│  │ Panel    │  │ Manager   │  │ Query    │  │ Communities       │  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └───────┬───────────┘  │
└───────┼──────────────┼──────────────┼────────────────┼──────────────┘
        │              │              │                │
┌───────▼──────────────▼──────────────▼────────────────▼──────────────┐
│                     ORCHESTRATION LAYER                              │
│                    (LangGraph State Machines)                        │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Ingestion│  │ KG Builder │  │ Agentic      │  │ Synthesis    │  │
│  │ Pipeline │  │ (Schema-   │  │ Query Router │  │ Engine       │  │
│  │          │  │  Enforced) │  │ + Reranker   │  │              │  │
│  └────┬─────┘  └─────┬──────┘  └──────┬───────┘  └──────┬───────┘  │
└───────┼──────────────┼────────────────┼──────────────────┼──────────┘
        │              │                │                  │
┌───────▼──────────────▼────────────────▼──────────────────▼──────────┐
│                          DATA LAYER                                  │
│  ┌──────────────────┐  ┌────────────────────────────────────────┐   │
│  │   Neo4j 5.x      │  │  Local File Store                      │   │
│  │   + GDS (Leiden)  │  │  ./data/                               │   │
│  │   + APOC          │  │  ├── litellm_cache/  (LLM response $) │   │
│  │   + Vector Index  │  │  └── uploads/        (raw documents)   │   │
│  │   + Fulltext Idx  │  │                                        │   │
│  └──────────────────┘  └────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
        │                          │
┌───────▼──────────────────────────▼─────────────────────────────────┐
│                       EXTERNAL SERVICES                             │
│  ┌───────────────────┐  ┌────────────────────┐  ┌───────────────┐  │
│  │ LiteLLM Gateway   │  │ sentence-trans.    │  │ Ollama        │  │
│  │ → OpenRouter :free │  │ BAAI/bge-m3       │  │ (optional     │  │
│  │ → Ollama fallback  │  │ (local, 1024d)    │  │  local LLM)   │  │
│  └───────────────────┘  └────────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Zero cost by default** — Free OpenRouter models + local embeddings + local cross-encoder. Optional Ollama for fully offline operation.
2. **Schema-first graph quality** — Explicit ontology prevents LLM-invented entity/relationship types. Clean graphs produce clean retrieval.
3. **Lazy over eager** — Community summaries generated on-demand at query time, cached in Neo4j. Free-tier budget preserved for interactive use.
4. **Retrieve broadly, rerank precisely** — Over-fetch from vector index → cross-encoder precision filter → context assembly.
5. **Agentic retrieval** — LangGraph agent dynamically selects retrieval strategy (local, global, hybrid, text2cypher) with sufficiency evaluation and retry loops.

---

## 2. Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **UI** | Gradio 5.x | Familiar stack, rapid prototyping, built-in file upload, streaming chat |
| **Orchestration** | LangGraph | State machines for ingestion + agentic retrieval flows |
| **Graph DB** | Neo4j Community 5.20+ | Real Cypher, native graph traversal, `ai.text2cypher`, vector index |
| **Graph Algorithms** | Neo4j GDS plugin | Leiden community detection (hierarchical), native performance |
| **Graph Utilities** | APOC plugin | `apoc.text.join`, schema introspection, text2cypher support |
| **GraphRAG lib** | `neo4j-graphrag` ≥1.8 | Official KG builder (`SimpleKGPipeline`), schema enforcement, retrievers |
| **LLM Gateway** | LiteLLM SDK | OpenRouter + Ollama unified interface, disk caching, fallbacks, cost tracking |
| **Embeddings** | `sentence-transformers` (local) | `BAAI/bge-m3` — 1024d, multilingual, top-tier MTEB 2026 |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder, zero API cost |
| **Doc Parsing** | PyMuPDF, python-docx, pathlib | PDF, DOCX, MD/TXT with metadata preservation |
| **Entity Resolution** | RapidFuzz + embedding similarity | Fuzzy string matching + cosine clustering |
| **Config** | Pydantic Settings + `.env` | Type-safe, env var override, `GN_` prefix |

---

## 3. Module Architecture

```
graphnotebook/
├── __init__.py
├── config.py                 # Pydantic settings, model registry, paths
├── main.py                   # Entry point — launches Gradio
│
├── ingestion/
│   ├── __init__.py
│   ├── parsers.py            # PDF (PyMuPDF), DOCX (python-docx), MD/TXT
│   ├── chunker.py            # Semantic chunking w/ overlap + metadata
│   └── pipeline.py           # Ingestion orchestrator (LangGraph)
│
├── extraction/
│   ├── __init__.py
│   ├── kg_pipeline.py        # Schema-enforced KG construction (SimpleKGPipeline)
│   ├── schema.py             # Domain ontology: entity types, relationship types
│   └── resolver.py           # Entity deduplication (RapidFuzz + embedding)
│
├── graph/
│   ├── __init__.py
│   ├── neo4j_client.py       # Connection pool, CRUD operations, health check
│   ├── schema_init.py        # Graph schema DDL (constraints, indexes)
│   ├── communities.py        # GDS Leiden detection + lazy summarization
│   └── queries.py            # Reusable Cypher query templates
│
├── retrieval/
│   ├── __init__.py
│   ├── router.py             # LangGraph agentic router with tools
│   ├── local_search.py       # Vector + graph traversal retrieval
│   ├── global_search.py      # Community-summary map-reduce
│   ├── text2cypher.py        # Native ai.text2cypher wrapper
│   ├── reranker.py           # Cross-encoder reranking
│   └── context_builder.py    # Context assembly with source attribution
│
├── llm/
│   ├── __init__.py
│   ├── gateway.py            # LiteLLM unified interface (OpenRouter + Ollama)
│   ├── models.py             # Free model registry + fallback chains
│   └── embeddings.py         # Local sentence-transformers wrapper (BGE-M3)
│
├── notebooks/
│   ├── __init__.py
│   └── manager.py            # Notebook CRUD, per-notebook schema, doc grouping
│
└── ui/
    ├── __init__.py
    ├── app.py                # Gradio Blocks layout
    ├── components.py         # Reusable UI components
    └── callbacks.py          # Event handlers for upload, query, explore
```

---

## 4. Neo4j Graph Schema

### 4.1 Constraints & Node Types

```cypher
// ── Notebook: logical grouping of documents ──────
CREATE CONSTRAINT notebook_id IF NOT EXISTS
  FOR (n:Notebook) REQUIRE n.id IS UNIQUE;

// ── Document: source file metadata ───────────────
CREATE CONSTRAINT doc_id IF NOT EXISTS
  FOR (d:Document) REQUIRE d.id IS UNIQUE;

// ── Chunk: text segment with embedding ───────────
CREATE CONSTRAINT chunk_id IF NOT EXISTS
  FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

// ── Entity: extracted named entity ───────────────
CREATE CONSTRAINT entity_id IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// ── Community: detected graph community ──────────
CREATE CONSTRAINT community_id IF NOT EXISTS
  FOR (cm:Community) REQUIRE cm.id IS UNIQUE;
```

### 4.2 Relationships

```cypher
// (:Notebook)-[:CONTAINS]->(:Document)
// (:Document)-[:HAS_CHUNK]->(:Chunk)
// (:Chunk)-[:MENTIONS]->(:Entity)
// (:Entity)-[:RELATES_TO {type, description, weight}]->(:Entity)
// (:Entity)-[:BELONGS_TO]->(:Community)
// (:Community)-[:PARENT_OF]->(:Community)   // hierarchical
```

### 4.3 Indexes

```cypher
// ── Vector Index (BGE-M3, 1024d, cosine) ─────────
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
  FOR (c:Chunk) ON (c.embedding)
  OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};

CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
  FOR (e:Entity) ON (e.embedding)
  OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }};

// ── Full-Text Index (BM25 hybrid search) ─────────
CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
  FOR (c:Chunk) ON EACH [c.text]
  OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard'}};
```

### 4.4 Node Properties

```
(:Notebook {
    id, name, description, created_at, updated_at,
    schema_json    // per-notebook ontology override (JSON string)
})

(:Document {
    id, notebook_id, filename, file_type, file_hash,
    raw_text_length, chunk_count, ingested_at, status
})

(:Chunk {
    id, doc_id, text, chunk_index, start_char, end_char,
    embedding,      // 1024d BGE-M3 vector
    token_count, page_number, section_header
})

(:Entity {
    id, name, type, description, embedding,
    source_chunk_ids, mention_count
})

(:Community {
    id, level, title, summary, key_findings,
    entity_count, rank,
    summarized_at,    // staleness detection
    created_at
})
```

---

## 5. LLM Gateway (LiteLLM)

Replaces the custom `OpenRouterLLM` class, `rate_limiter.py`, and `cache.py` with a single unified interface.

### 5.1 `llm/gateway.py`

```python
"""
LiteLLM-based LLM gateway.
Provides:
  - Disk caching (survives restarts, zero infra)
  - Automatic fallbacks across free models + Ollama
  - Cost tracking + observability
  - Unified interface for all LLM tasks
"""

import litellm
from litellm import completion
import json
import os

# ── Global config ──────────────────────────────────
litellm.set_verbose = False

# Enable disk caching
litellm.cache = litellm.Cache(
    type="disk",
    disk_cache_dir="./data/litellm_cache"
)


class LLMGateway:
    """Unified LLM interface via LiteLLM."""

    def __init__(self, task: str = "synthesis"):
        from .models import MODEL_REGISTRY
        config = MODEL_REGISTRY[task]
        self.model = config["primary"]
        self.fallbacks = config["fallbacks"]

    def invoke(self, prompt: str, system: str = "", **kwargs) -> str:
        """Call LLM with automatic fallback chain."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = completion(
                model=self.model,
                messages=messages,
                fallbacks=self.fallbacks,
                num_retries=3,
                caching=True,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"All models failed for task: {e}")

    def invoke_json(self, prompt: str, system: str = "") -> dict:
        """Force JSON output with fence stripping."""
        raw = self.invoke(
            prompt,
            system=system + "\nRespond ONLY with valid JSON. No markdown fences.",
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(cleaned)
```

### 5.2 `llm/models.py`

```python
"""
Free model registry with fallback chains.
Each task maps to a primary model + ordered fallbacks.
Fallback order: best free API → alternative free API → local Ollama.
"""

MODEL_REGISTRY = {
    "extraction": {
        "primary": "openrouter/deepseek/deepseek-r1:free",
        "fallbacks": [
            "openrouter/meta-llama/llama-3.3-70b-instruct:free",
            "openrouter/qwen/qwen3-coder-480b:free",
            "ollama/llama3.1:8b",
        ],
    },
    "synthesis": {
        "primary": "openrouter/deepseek/deepseek-r1:free",
        "fallbacks": [
            "openrouter/nvidia/nemotron-3-super:free",
            "openrouter/meta-llama/llama-3.3-70b-instruct:free",
            "ollama/llama3.1:8b",
        ],
    },
    "summarization": {
        "primary": "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "fallbacks": [
            "openrouter/openrouter/free",
            "ollama/llama3.1:8b",
        ],
    },
    "routing": {
        "primary": "openrouter/openrouter/free",
        "fallbacks": [
            "ollama/llama3.1:8b",
        ],
    },
}
```

### 5.3 `llm/embeddings.py`

```python
"""
Local embedding engine using sentence-transformers.
Zero API cost, zero rate limits.
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List
import numpy as np


class EmbeddingEngine:
    """BGE-M3 embedder with batch encoding."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch encode texts to embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

    def embed_single(self, text: str) -> List[float]:
        """Encode a single text, return as list for Neo4j."""
        return self.model.encode(text, normalize_embeddings=True).tolist()
```

---

## 6. Document Ingestion Pipeline

All document types go through the same path: `parsers.py → raw text + metadata → chunker.py → embeddings.py → Neo4j`.

The `SimpleKGPipeline` receives **pre-parsed text only** (never raw PDF bytes). This avoids metadata loss and duplicate parsing.

### 6.1 `ingestion/parsers.py`

```python
"""
Document parsers. Extract raw text + metadata from PDF, DOCX, MD/TXT.
All parsers return a standardized ParsedDocument dataclass.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from docx import Document as DocxDocument


@dataclass
class PageContent:
    page_number: int
    text: str
    section_header: Optional[str] = None


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    file_hash: str
    pages: List[PageContent]
    raw_text: str               # full concatenated text
    raw_text_length: int
    metadata: dict = field(default_factory=dict)


def parse_pdf(file_path: str) -> ParsedDocument:
    """Extract text from PDF with page-level granularity."""
    path = Path(file_path)
    doc = fitz.open(file_path)
    pages = []
    full_text_parts = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(PageContent(page_number=i + 1, text=text))
        full_text_parts.append(text)

    raw_text = "\n\n".join(full_text_parts)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    return ParsedDocument(
        filename=path.name,
        file_type="pdf",
        file_hash=file_hash,
        pages=pages,
        raw_text=raw_text,
        raw_text_length=len(raw_text),
        metadata={"page_count": len(doc)},
    )


def parse_docx(file_path: str) -> ParsedDocument:
    """Extract text from DOCX with paragraph-level structure."""
    path = Path(file_path)
    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    raw_text = "\n\n".join(paragraphs)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    return ParsedDocument(
        filename=path.name,
        file_type="docx",
        file_hash=file_hash,
        pages=[PageContent(page_number=1, text=raw_text)],
        raw_text=raw_text,
        raw_text_length=len(raw_text),
    )


def parse_text(file_path: str) -> ParsedDocument:
    """Parse plain text / markdown files."""
    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8")
    file_hash = hashlib.sha256(raw_text.encode()).hexdigest()

    return ParsedDocument(
        filename=path.name,
        file_type=path.suffix.lstrip("."),
        file_hash=file_hash,
        pages=[PageContent(page_number=1, text=raw_text)],
        raw_text=raw_text,
        raw_text_length=len(raw_text),
    )


PARSER_MAP = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".md": parse_text,
    ".txt": parse_text,
}


def parse_document(file_path: str) -> ParsedDocument:
    """Route to the correct parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    parser = PARSER_MAP.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser(file_path)
```

### 6.2 `ingestion/chunker.py`

```python
"""
Semantic chunking with token-level control.
Preserves paragraph boundaries, attaches metadata.
"""

import tiktoken
from dataclasses import dataclass
from typing import List
import uuid


@dataclass
class Chunk:
    id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    page_number: int = 0
    section_header: str = ""


class SemanticChunker:
    """
    Token-aware chunker that respects paragraph boundaries.
    Uses tiktoken for accurate token counting.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enc = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, text: str, doc_id: str = "") -> List[Chunk]:
        """Split text into overlapping chunks at paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_tokens = []
        current_text_parts = []
        current_start = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.enc.encode(para)

            # If adding this paragraph exceeds chunk_size, flush
            if (
                current_tokens
                and len(current_tokens) + len(para_tokens) > self.chunk_size
            ):
                chunk_text = "\n\n".join(current_text_parts)
                chunks.append(Chunk(
                    id=f"{doc_id}_chunk_{chunk_index:04d}",
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    token_count=len(current_tokens),
                ))
                chunk_index += 1

                # Overlap: keep last N tokens worth of text
                overlap_text = self.enc.decode(
                    current_tokens[-self.chunk_overlap:]
                )
                current_tokens = self.enc.encode(overlap_text)
                current_text_parts = [overlap_text]
                current_start = current_start + len(chunk_text) - len(overlap_text)

            current_tokens.extend(para_tokens)
            current_text_parts.append(para)

        # Flush remaining
        if current_text_parts:
            chunk_text = "\n\n".join(current_text_parts)
            chunks.append(Chunk(
                id=f"{doc_id}_chunk_{chunk_index:04d}",
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                token_count=len(current_tokens),
            ))

        return chunks
```

### 6.3 `ingestion/pipeline.py`

```python
"""
Ingestion orchestrator. Coordinates parsing → chunking → embedding → Neo4j storage.
Runs as a LangGraph state machine for progress tracking and error recovery.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from .parsers import parse_document, ParsedDocument
from .chunker import SemanticChunker, Chunk


class IngestionState(TypedDict):
    file_path: str
    notebook_id: str
    parsed_doc: Optional[ParsedDocument]
    chunks: List[Chunk]
    embeddings_stored: bool
    kg_built: bool
    status: str
    error: Optional[str]


def parse_step(state: IngestionState) -> IngestionState:
    """Parse uploaded document to raw text + metadata."""
    try:
        parsed = parse_document(state["file_path"])
        # Check for duplicate via file_hash
        # (neo4j_client.check_duplicate(parsed.file_hash))
        state["parsed_doc"] = parsed
        state["status"] = "parsed"
    except Exception as e:
        state["error"] = str(e)
        state["status"] = "failed"
    return state


def chunk_step(state: IngestionState) -> IngestionState:
    """Split parsed text into semantic chunks."""
    if state.get("error"):
        return state
    chunker = SemanticChunker()  # uses config defaults
    doc = state["parsed_doc"]
    state["chunks"] = chunker.chunk_text(doc.raw_text, doc_id=doc.file_hash[:8])
    state["status"] = "chunked"
    return state


def embed_and_store_step(state: IngestionState) -> IngestionState:
    """Embed chunks and store Document + Chunk nodes in Neo4j."""
    if state.get("error"):
        return state
    # embedding_engine.embed([c.text for c in state["chunks"]])
    # neo4j_client.store_document(state["parsed_doc"], state["notebook_id"])
    # neo4j_client.store_chunks(state["chunks"], embeddings)
    state["embeddings_stored"] = True
    state["status"] = "embedded"
    return state


def extract_kg_step(state: IngestionState) -> IngestionState:
    """Run schema-enforced entity/relationship extraction."""
    if state.get("error"):
        return state
    # kg_constructor.ingest_text(state["parsed_doc"].raw_text)
    state["kg_built"] = True
    state["status"] = "complete"
    return state


# ── Build Graph ─────────────────────────────────────
ingestion_workflow = StateGraph(IngestionState)
ingestion_workflow.add_node("parse", parse_step)
ingestion_workflow.add_node("chunk", chunk_step)
ingestion_workflow.add_node("embed_store", embed_and_store_step)
ingestion_workflow.add_node("extract_kg", extract_kg_step)

ingestion_workflow.set_entry_point("parse")
ingestion_workflow.add_edge("parse", "chunk")
ingestion_workflow.add_edge("chunk", "embed_store")
ingestion_workflow.add_edge("embed_store", "extract_kg")
ingestion_workflow.add_edge("extract_kg", END)

ingestion_pipeline = ingestion_workflow.compile()
```

---

## 7. Schema-Enforced Knowledge Graph Construction

Uses `neo4j-graphrag` `SimpleKGPipeline` with an explicit `SchemaBuilder`. The LLM can only extract entity/relationship types defined in the schema. Eliminates ontology drift.

### 7.1 `extraction/schema.py`

```python
"""
Domain ontology definition.
Defines allowed entity types and relationship types for KG extraction.
Can be overridden per-notebook via schema_json on the Notebook node.
"""

from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaRelation,
    SchemaProperty,
)

# ── Default Entity Types ────────────────────────────

DEFAULT_ENTITIES = [
    SchemaEntity(
        label="Person",
        description="A named individual",
        properties=[
            SchemaProperty(name="role", type="STRING"),
            SchemaProperty(name="affiliation", type="STRING"),
        ],
    ),
    SchemaEntity(
        label="Organization",
        description="Company, institution, government body, team",
        properties=[
            SchemaProperty(name="industry", type="STRING"),
        ],
    ),
    SchemaEntity(
        label="Concept",
        description="Abstract idea, theory, methodology, framework",
        properties=[
            SchemaProperty(name="domain", type="STRING"),
        ],
    ),
    SchemaEntity(
        label="Technology",
        description="Software, protocol, tool, language, platform",
        properties=[
            SchemaProperty(name="category", type="STRING"),
        ],
    ),
    SchemaEntity(
        label="Location",
        description="Geographic place, region, country, city",
    ),
    SchemaEntity(
        label="Event",
        description="Named event, conference, incident, milestone",
        properties=[
            SchemaProperty(name="date", type="STRING"),
        ],
    ),
    SchemaEntity(
        label="Metric",
        description="Quantitative measurement, KPI, statistic",
        properties=[
            SchemaProperty(name="value", type="STRING"),
            SchemaProperty(name="unit", type="STRING"),
        ],
    ),
]

# ── Default Relationship Types ──────────────────────

DEFAULT_RELATIONS = [
    SchemaRelation(label="WORKS_FOR", description="Person employed by Organization"),
    SchemaRelation(label="FOUNDED", description="Person founded Organization"),
    SchemaRelation(label="USES", description="Entity uses Technology/Concept"),
    SchemaRelation(label="RELATED_TO", description="General semantic relationship"),
    SchemaRelation(label="PART_OF", description="Component or subset relationship"),
    SchemaRelation(label="LOCATED_IN", description="Geographic containment"),
    SchemaRelation(label="CAUSED_BY", description="Causal relationship"),
    SchemaRelation(label="PRECEDED_BY", description="Temporal ordering"),
    SchemaRelation(label="MEASURED_BY", description="Entity measured by Metric"),
    SchemaRelation(label="COMPETES_WITH", description="Competitive relationship"),
    SchemaRelation(label="COLLABORATES_WITH", description="Partnership or collaboration"),
    SchemaRelation(label="INFLUENCES", description="Directional influence"),
]


def build_default_schema():
    """Build the default extraction schema."""
    builder = SchemaBuilder()
    return builder.create_schema(
        entities=DEFAULT_ENTITIES,
        relations=DEFAULT_RELATIONS,
    )


def build_schema_from_json(schema_json: dict):
    """
    Build a schema from per-notebook JSON override.
    Stored on (:Notebook {schema_json: '...'}).
    Falls back to defaults for missing fields.
    """
    builder = SchemaBuilder()
    entities = [
        SchemaEntity(
            label=e["label"],
            description=e.get("description", ""),
            properties=[
                SchemaProperty(name=p["name"], type=p.get("type", "STRING"))
                for p in e.get("properties", [])
            ],
        )
        for e in schema_json.get("entities", [])
    ] or DEFAULT_ENTITIES

    relations = [
        SchemaRelation(
            label=r["label"],
            description=r.get("description", ""),
        )
        for r in schema_json.get("relationships", [])
    ] or DEFAULT_RELATIONS

    return builder.create_schema(entities=entities, relations=relations)
```

### 7.2 `extraction/kg_pipeline.py`

```python
"""
Schema-enforced KG construction using neo4j-graphrag SimpleKGPipeline.
Receives pre-parsed text from ingestion/parsers.py (never raw PDF bytes).
LLM client is unified through LLMGateway.
"""

import asyncio
from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from langchain_community.chat_models import ChatLiteLLM

from ..config import Settings
from ..llm.gateway import LLMGateway
from .schema import build_default_schema, build_schema_from_json


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

        # Local embedder (same as used in ingestion)
        from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
        self.embedder = SentenceTransformerEmbeddings(
            model=settings.embedding_model
        )

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
        result = await pipeline.run_async(text=text)
        return result
```

### 7.3 `extraction/resolver.py`

```python
"""
Entity resolution: deduplicate entities across documents.
Phase 1: RapidFuzz string similarity.
Phase 2+: Add embedding cosine clustering for semantic dedup.
"""

from rapidfuzz import fuzz, process
from typing import List, Tuple


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
        entity_types = self.neo4j.query(
            "MATCH (e:Entity) RETURN DISTINCT e.type AS type"
        )
        for row in entity_types:
            self._resolve_type(row["type"])

    def _resolve_type(self, entity_type: str):
        """Resolve duplicates within a single entity type."""
        entities = self.neo4j.query(
            "MATCH (e:Entity {type: $type}) "
            "RETURN e.id AS id, e.name AS name, e.mention_count AS mc "
            "ORDER BY e.mention_count DESC",
            params={"type": entity_type},
        )
        names = [e["name"] for e in entities]
        merged = set()

        for i, entity in enumerate(entities):
            if entity["id"] in merged:
                continue
            # Find matches for this entity
            matches = process.extract(
                entity["name"],
                names[i + 1:],
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.threshold,
            )
            for match_name, score, idx in matches:
                match_entity = entities[i + 1 + idx]
                if match_entity["id"] not in merged:
                    self._merge_entities(entity["id"], match_entity["id"])
                    merged.add(match_entity["id"])

    def _merge_entities(self, keep_id: str, merge_id: str):
        """Merge merge_id entity into keep_id entity in Neo4j."""
        self.neo4j.query("""
            MATCH (keep:Entity {id: $keep_id})
            MATCH (merge:Entity {id: $merge_id})

            // Transfer all relationships
            CALL {
                WITH keep, merge
                MATCH (merge)-[r:RELATES_TO]-(other)
                WHERE other <> keep
                MERGE (keep)-[:RELATES_TO {
                    type: r.type,
                    description: r.description,
                    weight: r.weight
                }]-(other)
            }

            // Transfer chunk mentions
            CALL {
                WITH keep, merge
                MATCH (c:Chunk)-[m:MENTIONS]->(merge)
                MERGE (c)-[:MENTIONS]->(keep)
            }

            // Update mention count
            SET keep.mention_count = keep.mention_count +
                                     coalesce(merge.mention_count, 1)

            // Delete merged entity
            DETACH DELETE merge
        """, params={"keep_id": keep_id, "merge_id": merge_id})
```

---

## 8. Community Detection & Lazy Summarization

Communities are detected at ingest time (cheap graph algorithm), but summaries are generated **lazily at query time** and cached in Neo4j.

### 8.1 `graph/communities.py`

```python
"""
Community detection via Neo4j GDS Leiden algorithm.
Lazy summarization: generate summaries on-demand, cache in Neo4j.
"""

from ..llm.gateway import LLMGateway


class CommunityManager:
    """Detect communities and manage lazy summaries."""

    def __init__(self, neo4j_client, llm_gateway: LLMGateway = None):
        self.neo4j = neo4j_client
        self.llm = llm_gateway or LLMGateway("summarization")

    # ── Detection (runs at ingest time) ─────────────

    def detect_communities(self):
        """Run Leiden community detection on the entity graph."""
        # Project entity graph into GDS
        self.neo4j.query("""
            CALL gds.graph.project(
                'entity_graph',
                'Entity',
                {
                    RELATES_TO: {
                        orientation: 'UNDIRECTED',
                        properties: ['weight']
                    }
                }
            )
        """)

        # Run Leiden with hierarchical levels
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

        # Create Community nodes and link entities
        self.neo4j.query("""
            MATCH (e:Entity)
            WHERE e.community_id IS NOT NULL
            WITH e.community_id AS cid, collect(e) AS members
            MERGE (c:Community {id: toString(cid)})
            SET c.entity_count = size(members),
                c.level = 0,
                c.created_at = datetime()
            WITH c, members
            UNWIND members AS member
            MERGE (member)-[:BELONGS_TO]->(c)
        """)

        # Drop GDS projection
        self.neo4j.query("CALL gds.graph.drop('entity_graph')")

        return result

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
        context = self.neo4j.query("""
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
        """, params={"cid": community_id})

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
    "rank": 0-10
}}""",
            system="You are a knowledge graph analyst. Be precise and comprehensive.",
        )

        # Cache in Neo4j
        self._cache_summary(community_id, summary)
        return summary

    def get_relevant_summaries(
        self, query_embedding: list, top_n: int = 5
    ) -> list:
        """
        For global search: find communities relevant to query via entity embeddings.
        Only summarize the relevant ones (lazy).
        """
        results = self.neo4j.query("""
            CALL db.index.vector.queryNodes(
                'entity_embeddings', $top_k, $query_embedding
            ) YIELD node AS entity, score
            MATCH (entity)-[:BELONGS_TO]->(c:Community)
            WITH c, count(entity) AS match_count, avg(score) AS avg_score
            ORDER BY match_count DESC, avg_score DESC
            LIMIT $top_n
            RETURN c.id AS community_id,
                   c.summary AS cached_summary,
                   c.title AS title,
                   match_count, avg_score
        """, params={
            "query_embedding": query_embedding,
            "top_k": top_n * 3,  # over-fetch entities to find communities
            "top_n": top_n,
        })

        summaries = []
        for row in results:
            if row["cached_summary"]:
                summaries.append({
                    "title": row["title"],
                    "summary": row["cached_summary"],
                    "community_id": row["community_id"],
                })
            else:
                s = self.get_summary(row["community_id"])
                summaries.append({
                    "title": s["title"],
                    "summary": s["summary"],
                    "community_id": row["community_id"],
                })
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
        self.neo4j.query("""
            MATCH (c:Community {id: $cid})
            SET c.title = $title,
                c.summary = $summary,
                c.rank = $rank,
                c.key_findings = $findings,
                c.summarized_at = datetime()
        """, params={
            "cid": community_id,
            "title": summary.get("title", ""),
            "summary": summary.get("summary", ""),
            "rank": summary.get("rank", 0),
            "findings": summary.get("key_findings", []),
        })
```

---

## 9. Retrieval: Agentic Router + Reranker

### 9.1 `retrieval/reranker.py`

```python
"""
Cross-encoder reranker for retrieval precision.
Pattern: retrieve broadly (top-20) → rerank precisely (top-5).
"""

from sentence_transformers import CrossEncoder
from typing import List
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source_file: str
    chunk_index: int
    entities: list
    relationships: list
    community_context: str = ""


class Reranker:
    """Cross-encoder reranker using ms-marco model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: List[RetrievedChunk], top_k: int = 5
    ) -> List[RetrievedChunk]:
        """Rerank chunks by cross-encoder relevance score."""
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
```

### 9.2 `retrieval/local_search.py`

```python
"""
Local search: entity-centric retrieval.
Vector similarity → graph traversal → hybrid full-text → rerank.
"""

LOCAL_SEARCH_CYPHER = """
// 1. Vector search for relevant chunks
CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
YIELD node AS chunk, score AS vec_score

// 2. Get document source
MATCH (chunk)<-[:HAS_CHUNK]-(doc:Document)

// 3. Find entities mentioned in those chunks
OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)

// 4. Traverse 2-hop entity relationships
OPTIONAL MATCH (entity)-[r:RELATES_TO]-(neighbor:Entity)

// 5. Get community context
OPTIONAL MATCH (entity)-[:BELONGS_TO]->(community:Community)

// 6. Return enriched context
RETURN chunk.text AS chunk_text,
       chunk.id AS chunk_id,
       vec_score,
       doc.filename AS source_file,
       chunk.chunk_index AS chunk_index,
       chunk.page_number AS page_number,
       collect(DISTINCT {
         name: entity.name,
         type: entity.type,
         description: entity.description
       }) AS entities,
       collect(DISTINCT {
         source: entity.name,
         rel: r.type,
         target: neighbor.name,
         desc: r.description
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
CALL db.index.fulltext.queryNodes('chunk_fulltext', $query_text)
YIELD node AS ft_chunk, score AS ft_score

WITH vec_results,
     collect({node: ft_chunk, score: ft_score, source: 'fulltext'}) AS ft_results

// Merge and deduplicate
WITH vec_results + ft_results AS all_results
UNWIND all_results AS result
WITH result.node AS chunk,
     max(result.score) AS best_score
MATCH (chunk)<-[:HAS_CHUNK]-(doc:Document)
OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)
OPTIONAL MATCH (entity)-[r:RELATES_TO]-(neighbor:Entity)
OPTIONAL MATCH (entity)-[:BELONGS_TO]->(community:Community)

RETURN chunk.text AS chunk_text,
       chunk.id AS chunk_id,
       best_score,
       doc.filename AS source_file,
       chunk.chunk_index AS chunk_index,
       collect(DISTINCT {name: entity.name, type: entity.type}) AS entities,
       collect(DISTINCT {
         source: entity.name, rel: r.type, target: neighbor.name
       }) AS relationships,
       community.title AS community_title
ORDER BY best_score DESC
LIMIT $top_k
"""
```

### 9.3 `retrieval/text2cypher.py`

```python
"""
Text-to-Cypher using Neo4j's native ai.text2cypher procedure.
Available in Neo4j 5.20+ with APOC.
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
            self.neo4j.query("CALL dbms.procedures() YIELD name "
                           "WHERE name = 'ai.text2cypher' RETURN name")
            return True
        except Exception:
            return False

    def query(self, natural_language: str) -> list:
        """Convert NL to Cypher and execute."""
        if self._native_available:
            return self._native_query(natural_language)
        return self._llm_query(natural_language)

    def _native_query(self, nl: str) -> list:
        """Use Neo4j native ai.text2cypher."""
        result = self.neo4j.query(
            "CALL ai.text2cypher($query, {schema: $schema})",
            params={"query": nl, "schema": self._get_schema_description()},
        )
        if result and result[0].get("cypher"):
            return self.neo4j.query(result[0]["cypher"])
        return []

    def _llm_query(self, nl: str) -> list:
        """Fallback: generate Cypher via LLM."""
        if not self.llm:
            return []
        schema_desc = self._get_schema_description()
        cypher = self.llm.invoke(
            prompt=f"Convert to Cypher:\nQuestion: {nl}\nSchema: {schema_desc}",
            system="Generate ONLY a valid Cypher query. No explanation.",
        )
        cypher = cypher.strip().strip("`").strip("```cypher").strip("```")
        try:
            return self.neo4j.query(cypher)
        except Exception:
            return []  # fail silently, other retrievers will compensate

    def _get_schema_description(self) -> str:
        """Get graph schema for text2cypher context."""
        return """
        Nodes: Notebook, Document, Chunk, Entity, Community
        Entity properties: name, type, description
        Relationships: CONTAINS, HAS_CHUNK, MENTIONS, RELATES_TO, BELONGS_TO
        RELATES_TO properties: type, description, weight
        """
```

### 9.4 `retrieval/router.py` — Agentic Query Router

```python
"""
LangGraph agentic query router.
Dynamically selects retrieval strategy based on query classification.
Includes sufficiency evaluation and retry loop.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Annotated
from operator import add

from ..llm.gateway import LLMGateway
from .reranker import Reranker, RetrievedChunk
from .context_builder import ContextBuilder


class QueryState(TypedDict):
    query: str
    query_embedding: list
    search_mode: str              # "auto", "local", "global"
    retrieved_chunks: list
    community_summaries: list
    context: str
    answer: str
    sources: list
    iterations: int


# ── Singletons (initialized at app startup) ────────
llm = LLMGateway("routing")
synthesis_llm = LLMGateway("synthesis")
reranker = Reranker()
context_builder = ContextBuilder()


def classify_query(state: QueryState) -> QueryState:
    """Determine retrieval strategy via LLM classification."""
    if state["search_mode"] != "auto":
        return state

    classification = llm.invoke_json(
        prompt=f"""Classify this knowledge base query:
"{state['query']}"

- "local": asks about a specific entity, fact, or detail
- "global": asks for themes, summaries, overviews, or cross-document patterns
- "hybrid": needs both specific facts and broader context

Respond: {{"mode": "local|global|hybrid"}}"""
    )
    state["search_mode"] = classification.get("mode", "hybrid")
    return state


def execute_retrieval(state: QueryState) -> QueryState:
    """Execute retrieval based on classification."""
    mode = state["search_mode"]

    if mode in ("local", "hybrid"):
        # neo4j vector + graph traversal → rerank
        raw_chunks = _vector_graph_search(
            state["query_embedding"], top_k=20
        )
        reranked = reranker.rerank(state["query"], raw_chunks, top_k=8)
        state["retrieved_chunks"] = reranked

    if mode in ("global", "hybrid"):
        # Lazy community summaries
        summaries = _community_search(
            state["query_embedding"], top_n=5
        )
        state["community_summaries"] = summaries

    return state


def evaluate_sufficiency(state: QueryState) -> Literal["synthesize", "retry"]:
    """Check if retrieved context is sufficient."""
    total = len(state.get("retrieved_chunks", []))
    total += len(state.get("community_summaries", []))
    if total == 0 and state.get("iterations", 0) < 2:
        return "retry"
    return "synthesize"


def retry_broader(state: QueryState) -> QueryState:
    """Widen search: try text2cypher + full-text as fallbacks."""
    state["iterations"] = state.get("iterations", 0) + 1
    # text2cypher fallback
    # cypher_results = text2cypher_retriever.query(state["query"])
    # state["retrieved_chunks"].extend(cypher_results)
    return state


def synthesize(state: QueryState) -> QueryState:
    """Generate final answer with source attribution."""
    context = context_builder.build(
        chunks=state.get("retrieved_chunks", []),
        summaries=state.get("community_summaries", []),
    )
    state["context"] = context
    state["answer"] = synthesis_llm.invoke(
        prompt=f"Question: {state['query']}\n\nContext:\n{context}",
        system=(
            "Answer based ONLY on the provided context. "
            "Cite sources as [Source: filename, chunk N]. "
            "If context is insufficient, say so clearly."
        ),
    )
    state["sources"] = context_builder.extract_sources(
        state.get("retrieved_chunks", [])
    )
    return state


# ── Build Agentic Graph ─────────────────────────────

query_workflow = StateGraph(QueryState)
query_workflow.add_node("classify", classify_query)
query_workflow.add_node("retrieve", execute_retrieval)
query_workflow.add_node("retry", retry_broader)
query_workflow.add_node("synthesize", synthesize)

query_workflow.set_entry_point("classify")
query_workflow.add_edge("classify", "retrieve")
query_workflow.add_conditional_edges(
    "retrieve",
    evaluate_sufficiency,
    {"synthesize": "synthesize", "retry": "retry"},
)
query_workflow.add_edge("retry", "retrieve")
query_workflow.add_edge("synthesize", END)

query_agent = query_workflow.compile()
```

### 9.5 `retrieval/context_builder.py`

```python
"""
Context assembly with source attribution.
Builds the final context string for LLM synthesis.
"""

from typing import List


class ContextBuilder:
    """Assemble retrieval results into structured context for LLM."""

    def build(self, chunks: list = None, summaries: list = None) -> str:
        """Build context string from chunks and community summaries."""
        parts = []

        if chunks:
            parts.append("## Relevant Document Passages\n")
            for i, chunk in enumerate(chunks):
                source = getattr(chunk, "source_file", "unknown")
                idx = getattr(chunk, "chunk_index", i)
                parts.append(
                    f"### [Source: {source}, Chunk {idx}]\n"
                    f"{chunk.text}\n"
                )
                # Add entity context if available
                entities = getattr(chunk, "entities", [])
                if entities:
                    entity_strs = [
                        f"  - {e['name']} ({e['type']})"
                        for e in entities if e.get("name")
                    ]
                    if entity_strs:
                        parts.append(
                            "Related entities:\n" + "\n".join(entity_strs) + "\n"
                        )

        if summaries:
            parts.append("\n## Knowledge Graph Community Insights\n")
            for s in summaries:
                title = s.get("title", "Community")
                summary = s.get("summary", "")
                parts.append(f"### {title}\n{summary}\n")

        return "\n".join(parts) if parts else "No relevant context found."

    def extract_sources(self, chunks: list) -> list:
        """Extract source attribution from retrieved chunks."""
        sources = []
        seen = set()
        for chunk in chunks:
            source = getattr(chunk, "source_file", "unknown")
            idx = getattr(chunk, "chunk_index", 0)
            key = f"{source}:{idx}"
            if key not in seen:
                sources.append({
                    "file": source,
                    "chunk_index": idx,
                    "score": getattr(chunk, "score", 0),
                })
                seen.add(key)
        return sources
```

---

## 10. Gradio UI

```python
# ui/app.py

import gradio as gr


def build_app():
    with gr.Blocks(title="GraphNotebook", theme=gr.themes.Soft()) as app:

        # ── Header ──
        gr.Markdown("# 📓 GraphNotebook\n*Personal knowledge graph powered by GraphRAG*")

        with gr.Row():
            # ── Left Panel: Notebook Manager ──
            with gr.Column(scale=1):
                notebook_dropdown = gr.Dropdown(
                    label="Notebook", choices=[], interactive=True
                )
                with gr.Row():
                    new_notebook_btn = gr.Button("+ New", size="sm")
                    delete_notebook_btn = gr.Button("🗑️", size="sm")

                gr.Markdown("### Documents")
                doc_list = gr.Dataframe(
                    headers=["File", "Status", "Entities", "Chunks"],
                    interactive=False,
                )
                upload = gr.File(
                    label="Upload Documents",
                    file_types=[".pdf", ".md", ".txt", ".docx"],
                    file_count="multiple",
                )
                ingest_btn = gr.Button("🔄 Index Documents", variant="primary")
                ingest_progress = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Graph Stats")
                stats_json = gr.JSON(label="Knowledge Graph")

            # ── Right Panel: Chat + Explorer ──
            with gr.Column(scale=2):
                with gr.Tabs():

                    # Tab 1: Chat
                    with gr.Tab("💬 Chat"):
                        chatbot = gr.Chatbot(
                            label="Ask your documents",
                            height=500,
                            show_copy_button=True,
                        )
                        with gr.Row():
                            query_input = gr.Textbox(
                                placeholder="Ask anything about your documents...",
                                show_label=False,
                                scale=4,
                            )
                            search_mode = gr.Radio(
                                ["Auto", "Local", "Global"],
                                value="Auto",
                                label="Search",
                                scale=1,
                            )

                        sources_accordion = gr.Accordion("📎 Sources", open=False)
                        with sources_accordion:
                            source_display = gr.Markdown()

                    # Tab 2: Graph Explorer
                    with gr.Tab("🔗 Graph Explorer"):
                        entity_search = gr.Textbox(placeholder="Search entities...")
                        graph_viz = gr.HTML(label="Knowledge Graph")
                        entity_detail = gr.JSON(label="Entity Details")

                    # Tab 3: Communities
                    with gr.Tab("🏘️ Communities"):
                        community_list = gr.Dataframe(
                            headers=["Community", "Entities", "Rank", "Theme"],
                            interactive=False,
                        )
                        community_summary = gr.Markdown()

                    # Tab 4: Schema Editor (per-notebook)
                    with gr.Tab("⚙️ Schema"):
                        schema_editor = gr.Code(
                            label="Notebook Schema (JSON)",
                            language="json",
                            lines=20,
                        )
                        save_schema_btn = gr.Button("💾 Save Schema")
                        schema_status = gr.Textbox(label="Status", interactive=False)

    return app
```

---

## 11. Configuration

```python
# config.py

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM (LiteLLM gateway) ──
    openrouter_api_key: str = ""
    ollama_host: str = "http://localhost:11434"

    # ── Neo4j ──
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphnotebook"
    neo4j_database: str = "neo4j"

    # ── Embeddings ──
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = 1024
    # Fallback for memory-constrained systems:
    # embedding_model: str = "BAAI/bge-base-en-v1.5"
    # embedding_dimensions: int = 768

    # ── Chunking ──
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ──
    local_top_k: int = 20          # over-fetch for reranker
    rerank_top_k: int = 8          # post-rerank
    global_top_communities: int = 5
    max_context_tokens: int = 4000

    # ── Reranker ──
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Paths ──
    data_dir: str = "./data"
    litellm_cache_dir: str = "./data/litellm_cache"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "GN_"
```

---

## 12. Docker Compose

```yaml
version: "3.9"

services:
  neo4j:
    image: neo4j:5-community
    container_name: graphnotebook-neo4j
    ports:
      - "7474:7474"   # Neo4j Browser
      - "7687:7687"   # Bolt protocol
    environment:
      NEO4J_AUTH: neo4j/graphnotebook
      NEO4J_PLUGINS: '["graph-data-science", "apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: "gds.*,apoc.*"
      NEO4J_dbms_security_procedures_allowlist: "gds.*,apoc.*"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "neo4j", "status"]
      interval: 10s
      timeout: 5s
      retries: 5

  # OPTIONAL: Local LLM via Ollama (zero-cost fallback)
  ollama:
    image: ollama/ollama:latest
    container_name: graphnotebook-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    profiles: ["local-llm"]
    # After first start: docker exec graphnotebook-ollama ollama pull llama3.1:8b

  app:
    build: .
    container_name: graphnotebook-app
    ports:
      - "7860:7860"   # Gradio
    environment:
      GN_NEO4J_URI: bolt://neo4j:7687
      GN_NEO4J_USER: neo4j
      GN_NEO4J_PASSWORD: graphnotebook
      GN_OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
      GN_EMBEDDING_MODEL: "BAAI/bge-m3"
      GN_EMBEDDING_DIMENSIONS: "1024"
      OLLAMA_HOST: http://ollama:11434
    volumes:
      - ./data:/app/data
    depends_on:
      neo4j:
        condition: service_healthy

volumes:
  neo4j_data:
  neo4j_logs:
  ollama_data:
```

---

## 13. Implementation Roadmap

### Phase 1: Foundation (MVP) — *"Upload & Chat"*
- [ ] Project scaffolding (`pyproject.toml`, `config.py`, `main.py`)
- [ ] Docker Compose (Neo4j + app)
- [ ] Document parsers: `parsers.py` (PDF, DOCX, MD/TXT)
- [ ] Semantic chunker: `chunker.py`
- [ ] LiteLLM gateway: `gateway.py` + `models.py` with disk cache
- [ ] BGE-M3 embedding engine: `embeddings.py`
- [ ] Neo4j client: `neo4j_client.py` + `schema_init.py`
- [ ] Basic vector retrieval (no graph traversal yet)
- [ ] Gradio UI: upload panel, chat tab, source display
- [ ] Ingestion LangGraph pipeline: parse → chunk → embed → store
- **Deliverable**: Working vector RAG with file upload and chat

### Phase 2: Knowledge Graph — *"Extract & Connect"*
- [ ] Domain schema definition: `extraction/schema.py`
- [ ] KG construction pipeline: `extraction/kg_pipeline.py` (`SimpleKGPipeline`)
- [ ] Entity resolution: `extraction/resolver.py` (RapidFuzz)
- [ ] VectorCypher retrieval (vector + 2-hop graph traversal)
- [ ] Hybrid search (vector + full-text BM25)
- [ ] Cross-encoder reranker: `retrieval/reranker.py`
- [ ] Graph stats dashboard in Gradio
- **Deliverable**: GraphRAG with schema-enforced extraction + reranking

### Phase 3: Communities + Agentic — *"Understand & Reason"*
- [ ] GDS Leiden community detection: `graph/communities.py`
- [ ] Lazy community summarization (query-time + Neo4j cache)
- [ ] Global search: map-reduce over relevant community summaries
- [ ] LangGraph agentic router: `retrieval/router.py`
- [ ] Text-to-Cypher tool: `retrieval/text2cypher.py` (native `ai.text2cypher`)
- [ ] Sufficiency evaluation + retry loop
- [ ] Community explorer tab in Gradio
- **Deliverable**: Full GraphRAG with agentic retrieval

### Phase 4: Polish — *"Production Personal Tool"*
- [ ] Graph visualization (pyvis / neovis.js embed in Gradio)
- [ ] Notebook management (create, rename, delete, per-notebook schema)
- [ ] Schema editor tab in Gradio
- [ ] HITL entity review on first ingest (approval UI)
- [ ] Incremental re-indexing (file hash + schema diff detection)
- [ ] Streaming chat responses in Gradio
- [ ] Conversation history with follow-up context
- [ ] Export: graph as GraphML/JSON-LD, summaries as markdown
- [ ] Optional Ollama local fallback (`docker compose --profile local-llm up`)
- **Deliverable**: Feature-complete personal knowledge base

---

## 14. Architecture Decision Records

### ADR-001: LiteLLM Gateway over Raw OpenAI SDK
**Context**: Custom `OpenRouterLLM` required manual rate limiting, caching, and fallback logic across 3 modules.
**Decision**: Replace with LiteLLM Python SDK (not proxy server mode).
**Consequence**: Eliminates `openrouter.py`, `rate_limiter.py`, `cache.py`. Gains disk caching, automatic retries/fallbacks, Ollama integration, cost tracking. One additional dependency (~50MB).

### ADR-002: Schema Enforcement over Free-Form Extraction
**Context**: LLM-invented entity/relationship types produce inconsistent ontologies ("Bitcoin"/"BTC"/"bitcoin" as separate entities).
**Decision**: Use `neo4j-graphrag` `SchemaBuilder` with explicit `SchemaEntity`/`SchemaRelation` types. Allow per-notebook schema overrides via JSON.
**Consequence**: Dramatically cleaner graphs. Reduced entity resolution burden. Less "discovery" of unexpected types — mitigated by broad `Concept` catch-all + per-notebook customization.

### ADR-003: Lazy over Eager Community Summarization
**Context**: Free tier (50 req/day) cannot afford summarizing all communities at ingest time. Most communities are never queried.
**Decision**: Generate summaries on-demand at query time, cache in Neo4j with staleness detection.
**Consequence**: First query touching a new community is ~2-3s slower. All subsequent queries use cache. Saves 80%+ of community-related LLM budget.

### ADR-004: BGE-M3 over all-MiniLM-L6-v2
**Context**: MiniLM (384d, 2022) lags modern MTEB benchmarks by ~15%.
**Decision**: Default to BGE-M3 (1024d). Fallback to BGE-base-en-v1.5 (768d) for constrained systems.
**Consequence**: Better retrieval quality, multilingual support. ~3x model memory (1.1GB vs 350MB). Negligible latency impact for single-user.

### ADR-005: Pre-Parsing over SimpleKGPipeline PDF Mode
**Context**: `SimpleKGPipeline(from_pdf=True)` duplicates work already done by `parsers.py` and loses metadata (page numbers, section headers, file hash).
**Decision**: Always parse via `parsers.py` → raw text → `ingest_text()`. Never use `from_pdf=True`.
**Consequence**: Single parsing path. Consistent metadata. Slightly more code, but cleaner separation of concerns.

### ADR-006: Native ai.text2cypher over Custom Prompt
**Context**: Custom "generate Cypher via LLM" requires a full LLM call and risks injection.
**Decision**: Use Neo4j's native `ai.text2cypher` procedure (5.20+). Fall back to LLM-generated Cypher only if unavailable.
**Consequence**: Faster, schema-aware, safer. Requires Neo4j 5.20+ (already targeted).

---

## 15. Dependencies

```toml
[project]
name = "graphnotebook"
version = "1.1.0"
requires-python = ">=3.11"
description = "Personal GraphRAG knowledge base"

dependencies = [
    # ── UI ──
    "gradio>=5.0",

    # ── Graph ──
    "neo4j>=5.20",
    "neo4j-graphrag[openai]>=1.8",

    # ── LLM ──
    "litellm>=1.55",
    "langchain-community>=0.3",

    # ── Embeddings & Reranking ──
    "sentence-transformers>=3.0",

    # ── Orchestration ──
    "langchain-core>=0.3",
    "langgraph>=0.2",

    # ── Document Parsing ──
    "pymupdf>=1.24",
    "python-docx>=1.0",
    "tiktoken>=0.7",

    # ── Entity Resolution ──
    "rapidfuzz>=3.0",

    # ── Visualization ──
    "pyvis>=0.3",

    # ── Config ──
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",
]

[project.scripts]
graphnotebook = "graphnotebook.main:main"
```

---

## 16. Quick Start

```bash
# 1. Clone & install
git clone https://github.com/JackSmack1971/graphnotebook.git
cd graphnotebook
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env → set GN_OPENROUTER_API_KEY=your_key

# 3. Launch Neo4j (+ optional Ollama)
docker compose up -d neo4j
# With local LLM: docker compose --profile local-llm up -d

# 4. Initialize graph schema
python -m graphnotebook.graph.schema_init

# 5. Run
python -m graphnotebook.main
# → http://localhost:7860

# 6. First use
# - Create a notebook
# - Upload PDF/MD/DOCX files
# - Click "Index Documents"
# - Start chatting with your knowledge graph
```

### `.env.example`

```bash
GN_OPENROUTER_API_KEY=sk-or-...
GN_NEO4J_URI=bolt://localhost:7687
GN_NEO4J_USER=neo4j
GN_NEO4J_PASSWORD=graphnotebook
GN_EMBEDDING_MODEL=BAAI/bge-m3
GN_EMBEDDING_DIMENSIONS=1024
GN_LOG_LEVEL=INFO
# Optional: Ollama for local LLM fallback
OLLAMA_HOST=http://localhost:11434
```
