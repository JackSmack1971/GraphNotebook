# GraphNotebook

> **Personal GraphRAG knowledge base. NotebookLM meets knowledge graphs.**
> *Single-user · Zero cost · Full graph intelligence*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Gradio](https://img.shields.io/badge/gradio-5.x-orange?logo=gradio)](https://www.gradio.app/)
[![LiteLLM](https://img.shields.io/badge/litellm-unified_gateway-purple)](https://docs.litellm.ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

GraphNotebook lets you **upload your documents** (PDF, DOCX, Markdown, TXT), automatically builds a **knowledge graph** from them, and then lets you **chat with that graph** — understanding not just individual facts, but how concepts *relate* to each other across your entire document collection.

Think of it as a personal research assistant that reads everything you give it, maps out every connection, and answers questions with graph-level reasoning — completely offline and at zero LLM cost.

---

## Table of Contents

1. [What Is GraphRAG?](#1-what-is-graphrag)
2. [Architecture Overview](#2-architecture-overview)
3. [Feature Highlights](#3-feature-highlights)
4. [Prerequisites](#4-prerequisites)
5. [Quick Start](#5-quick-start)
6. [Configuration Reference](#6-configuration-reference)
7. [Repository Structure](#7-repository-structure)
8. [How It Works: Pipeline Deep Dive](#8-how-it-works-pipeline-deep-dive)
9. [Neo4j Graph Schema](#9-neo4j-graph-schema)
10. [Retrieval Modes](#10-retrieval-modes)
11. [LLM Gateway & Models](#11-llm-gateway--models)
12. [Development Guide](#12-development-guide)
13. [Roadmap](#13-roadmap)
14. [Architecture Decisions](#14-architecture-decisions)
15. [Glossary](#15-glossary)

---

## 1. What Is GraphRAG?

**RAG** stands for *Retrieval-Augmented Generation*. Traditional RAG splits your documents into chunks, embeds them as vectors, and finds the most similar chunks to your question. It works well for finding specific facts.

**GraphRAG** goes further. After chunking, it extracts *entities* (people, concepts, organizations) and *relationships* between them, stores everything in a **knowledge graph**, and then retrieves answers by traversing those connections — not just by similarity.

```
Traditional RAG:   "Find the 5 chunks closest to this question"
GraphRAG:          "Find related chunks PLUS follow the graph 2 hops
                    to discover connected facts you didn't ask for"
```

GraphNotebook implements the full GraphRAG pattern with an additional **community detection** layer: it automatically groups related entities into thematic clusters and can answer *"give me a high-level summary of this topic"* by reasoning over entire communities at once.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                           GRADIO UI  :7860                           │
│  Upload Panel │ Notebook Manager │ Chat/Query │ Graph Explorer       │
└───────────────────────┬───────────────────────────┬──────────────────┘
                        │                           │
┌───────────────────────▼───────────────────────────▼──────────────────┐
│                   LANGGRAPH ORCHESTRATION LAYER                       │
│   Ingestion Pipeline │ KG Builder │ Agentic Router │ Synthesis Engine │
└───────────────────────┬───────────────────────────┬──────────────────┘
                        │                           │
┌───────────────────────▼───────────────────────────▼──────────────────┐
│                         DATA LAYER                                    │
│  Neo4j 5.x (+ GDS + APOC + Vector Index + Fulltext)  │  ./data/      │
└───────────────────────┬───────────────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                                │
│  LiteLLM → OpenRouter (free models)  │  BGE-M3 (local, 1.1 GB)    │
│  Optional: Ollama (fully local LLM)                                │
└────────────────────────────────────────────────────────────────────┘
```

Every layer is loosely coupled. The LiteLLM gateway is the *only* LLM interface — swap models without touching application code.

---

## 3. Feature Highlights

| Feature | Detail |
|---|---|
| **Document formats** | PDF (PyMuPDF), DOCX (python-docx), Markdown, plain text |
| **Semantic chunking** | tiktoken-aware, paragraph-boundary splitting, configurable overlap |
| **Knowledge graph extraction** | Schema-enforced via `neo4j-graphrag` `SimpleKGPipeline` |
| **Entity deduplication** | RapidFuzz fuzzy match + embedding cosine similarity |
| **Community detection** | Neo4j GDS Leiden algorithm (3 hierarchical levels) |
| **Lazy summarization** | Community summaries generated on-demand, cached in Neo4j |
| **Agentic retrieval** | LangGraph router auto-classifies queries as local/global/hybrid |
| **Reranking** | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) post-retrieval refinement |
| **Text-to-Cypher** | Natural language → Cypher fallback via native `ai.text2cypher` |
| **Zero-cost LLMs** | OpenRouter free-tier models with automatic fallback chains |
| **Local-first option** | Full Ollama integration (`--profile local-llm`) |
| **Disk-cached LLM calls** | LiteLLM disk cache survives restarts, eliminates redundant API calls |
| **Multi-notebook** | Isolated document collections with per-notebook schemas |

---

## 4. Prerequisites

Before you begin, you need the following installed on your machine:

| Requirement | Why it's needed | Install guide |
|---|---|---|
| **Python 3.11+** | The application language | [python.org](https://www.python.org/downloads/) |
| **Docker + Docker Compose** | Runs the Neo4j graph database | [docs.docker.com](https://docs.docker.com/get-docker/) |
| **~2 GB free RAM** | BGE-M3 embedding model (1.1 GB) + Neo4j overhead | — |
| **OpenRouter API key** | Free LLM access (no credit card required for free models) | [openrouter.ai](https://openrouter.ai/) |

> **New to Docker?** Docker runs software in isolated containers. You don't need to install Neo4j manually — Docker handles it for you. Just install Docker Desktop and run the command in Step 5.

---

## 5. Quick Start

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/graphnotebook.git
cd graphnotebook
```

### Step 2 — Create your environment file

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` in any text editor and set:

```env
# Required — get a free key at https://openrouter.ai/
GN_OPENROUTER_API_KEY=your_openrouter_api_key_here

# Neo4j — leave these as-is if using Docker Compose
GN_NEO4J_URI=bolt://localhost:7687
GN_NEO4J_USER=neo4j
GN_NEO4J_PASSWORD=graphnotebook
```

### Step 3 — Start Neo4j via Docker

This starts the graph database in the background:

```bash
docker compose up -d
```

Wait about 30 seconds for Neo4j to fully initialize. You can verify it's ready by opening [http://localhost:7474](http://localhost:7474) in your browser (log in with `neo4j` / `graphnotebook`).

### Step 4 — Install the Python package

```bash
pip install -e ".[dev]"
```

> The `-e` flag installs in *editable mode* — changes to the source code take effect immediately without reinstalling.

### Step 5 — Initialize the graph schema

This creates all Neo4j constraints and indexes that the application requires:

```bash
python -m graphnotebook.graph.schema_init
```

You should see confirmation that indexes were created. This only needs to be run once.

### Step 6 — Launch GraphNotebook

```bash
python -m graphnotebook.main
```

Open your browser at **[http://localhost:7860](http://localhost:7860)** — the Gradio interface is ready.

---

### Optional: Run with a fully local LLM (Ollama)

If you want zero internet dependency, start Ollama alongside Neo4j:

```bash
docker compose --profile local-llm up -d
```

---

## 6. Configuration Reference

All configuration is controlled through environment variables prefixed with `GN_`. The `.env` file at the project root is loaded automatically.

| Variable | Default | Description |
|---|---|---|
| `GN_OPENROUTER_API_KEY` | `""` | OpenRouter API key for free LLM access |
| `GN_NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `GN_NEO4J_USER` | `neo4j` | Neo4j username |
| `GN_NEO4J_PASSWORD` | `graphnotebook` | Neo4j password |
| `GN_NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `GN_EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformer model for embeddings |
| `GN_EMBEDDING_DIMENSIONS` | `1024` | Output embedding dimensionality |
| `GN_CHUNK_SIZE` | `512` | Maximum tokens per document chunk |
| `GN_CHUNK_OVERLAP` | `64` | Overlap tokens between adjacent chunks |
| `GN_LOCAL_TOP_K` | `20` | Candidate chunks fetched before reranking |
| `GN_RERANK_TOP_K` | `8` | Final chunks passed to synthesis after reranking |
| `GN_GLOBAL_TOP_COMMUNITIES` | `5` | Community summaries used in global search |
| `GN_MAX_CONTEXT_TOKENS` | `4000` | Maximum tokens assembled for LLM context |
| `GN_DATA_DIR` | `./data` | Root directory for uploads and LLM cache |
| `GN_LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL for local LLM fallback |

> **Memory-constrained machine?** If BGE-M3 (1.1 GB) is too large, switch to the lighter fallback:
> ```env
> GN_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
> GN_EMBEDDING_DIMENSIONS=768
> ```

---

## 7. Repository Structure

```
graphnotebook/
├── config.py              # All settings via Pydantic (GN_ env prefix)
├── main.py                # Entry point — wires deps, launches Gradio
│
├── ingestion/             # Document → Chunks → Vectors pipeline
│   ├── parsers.py         # PDF, DOCX, MD/TXT → raw text + metadata
│   ├── chunker.py         # tiktoken-based semantic splitter with overlap
│   └── pipeline.py        # LangGraph state machine orchestrating all steps
│
├── extraction/            # Text → Knowledge Graph pipeline
│   ├── kg_pipeline.py     # SimpleKGPipeline wrapper (schema-enforced)
│   ├── schema.py          # Entity types + relationship types (ontology)
│   └── resolver.py        # Deduplication: RapidFuzz + embedding cosine
│
├── graph/                 # Neo4j interface layer
│   ├── neo4j_client.py    # Connection pool, CRUD, health check
│   ├── schema_init.py     # One-time DDL: constraints + indexes
│   ├── communities.py     # GDS Leiden detection + lazy summarization
│   └── queries.py         # All Cypher queries (centralized here only)
│
├── retrieval/             # Query → Answer pipeline
│   ├── router.py          # LangGraph agentic router (classify → retrieve → rerank)
│   ├── local_search.py    # Vector similarity + 2-hop graph traversal
│   ├── global_search.py   # Community summary map-reduce
│   ├── text2cypher.py     # Natural language → Cypher fallback
│   ├── reranker.py        # Cross-encoder reranking
│   └── context_builder.py # Assembles final context + source attribution
│
├── llm/                   # LLM + embedding abstraction
│   ├── gateway.py         # LiteLLM wrapper: invoke, invoke_json, invoke_stream
│   ├── models.py          # Task → (primary model, fallback chain) registry
│   └── embeddings.py      # BGE-M3 singleton wrapper
│
├── notebooks/
│   └── manager.py         # Notebook CRUD + per-notebook schema isolation
│
└── ui/
    ├── app.py             # Gradio Blocks layout definition
    ├── components.py      # Reusable UI widgets
    └── callbacks.py       # Event handlers (upload, query, explore)

tests/                     # pytest test suite
docs/                      # Architecture documentation
docker-compose.yml         # Neo4j + app + optional Ollama services
pyproject.toml             # Package metadata + dependencies
.env.example               # Environment variable template
```

> **Key rule:** All Cypher queries live exclusively in `graph/queries.py` and related graph modules. Never embed Cypher strings in business logic.

---

## 8. How It Works: Pipeline Deep Dive

### 8.1 Document Ingestion

When you upload a file, a **LangGraph state machine** processes it through four sequential steps:

```
[Upload] → parse → chunk → embed_store → extract_kg → [Done]
```

**Step 1 — Parse:** The file is routed to the correct parser based on extension.
- `.pdf` → PyMuPDF extracts text + page numbers
- `.docx` → python-docx extracts paragraphs + headings
- `.md` / `.txt` → read directly

A SHA-256 **file hash** is computed to detect and skip duplicate uploads.

**Step 2 — Chunk:** The raw text is split into overlapping windows of up to 512 tokens (measured by tiktoken's `cl100k_base` tokenizer). Splits happen at paragraph boundaries to preserve semantic coherence. A 64-token overlap between adjacent chunks prevents context loss at boundaries.

**Step 3 — Embed & Store:** BGE-M3 (loaded once at startup) converts every chunk into a 1024-dimensional vector. Chunks and their embeddings are stored in Neo4j as `Chunk` nodes, linked to their parent `Document` node.

**Step 4 — Extract KG:** `SimpleKGPipeline` runs LLM-assisted entity and relationship extraction against your configured schema. Extracted `Entity` nodes are deduplicated against existing nodes using RapidFuzz fuzzy matching and embedding cosine similarity before being written to the graph.

### 8.2 Query Processing

When you ask a question:

```
[Query] → classify_query → execute_retrieval → evaluate_sufficiency
              ↓                                         ↓
         local/global/hybrid               synthesize ← retry_broader
```

1. **Classify:** An LLM call determines whether your question needs local facts, global themes, or both.
2. **Retrieve:** Based on classification, the router calls local search (vector + 2-hop graph traversal), global search (community summary map-reduce), or both.
3. **Rerank:** A cross-encoder model (`ms-marco-MiniLM-L-6-v2`) re-scores the retrieved chunks by relevance to the exact question — more precise than pure vector similarity.
4. **Sufficiency check:** If nothing useful was retrieved and we haven't retried yet, the router broadens the search using Text-to-Cypher as a fallback.
5. **Synthesize:** The context is assembled with source attribution and passed to the synthesis LLM for a final, grounded answer.

### 8.3 Community Detection

After every ingestion, the graph runs GDS **Leiden algorithm** to discover thematic clusters of entities across 3 hierarchical levels. Communities are not summarized immediately — summaries are generated **lazily** the first time a query needs them, then cached directly on the `Community` node in Neo4j.

This means: first query against a new community is slightly slower (one LLM call), every subsequent query against that community is instant.

---

## 9. Neo4j Graph Schema

### Node Types

| Label | Key Properties | Purpose |
|---|---|---|
| `Notebook` | `id`, `name`, `schema_json` | Top-level container for a document collection |
| `Document` | `id`, `filename`, `file_hash`, `file_type` | One uploaded file |
| `Chunk` | `id`, `text`, `embedding` (1024-dim), `chunk_index` | Retrievable text segment |
| `Entity` | `name`, `type`, `description`, `embedding` | Extracted knowledge node |
| `Community` | `id`, `title`, `summary`, `key_findings`, `rank`, `level` | Thematic cluster of entities |

### Relationship Types

| Relationship | From → To | Meaning |
|---|---|---|
| `CONTAINS` | `Notebook → Document` | Document belongs to notebook |
| `HAS_CHUNK` | `Document → Chunk` | Chunk belongs to document |
| `MENTIONS` | `Chunk → Entity` | Chunk mentions this entity |
| `RELATES_TO` | `Entity → Entity` | Extracted semantic relationship (has `type`, `weight`) |
| `BELONGS_TO` | `Entity → Community` | Entity is a member of this community |

---

## 10. Retrieval Modes

GraphNotebook supports three retrieval strategies, selected automatically by the agentic router or manually via the UI:

### `local` — Specific Facts

Best for: *"What did the paper say about transformer attention?"*, *"Who is Dr. Smith?"*

Performs vector similarity search over `Chunk` nodes, then traverses 2 hops through the graph to surface related entities and relationships that appear in nearby chunks.

### `global` — Themes and Overviews

Best for: *"What are the main topics across my documents?"*, *"Summarize the key findings."*

Embeds the query, finds the most relevant `Community` nodes by vector similarity, generates (or retrieves cached) summaries for each community, then performs a **map-reduce synthesis** over those summaries.

### `hybrid` — Both

Best for: *"How does concept X relate to the broader themes in my collection?"*

Runs both local and global retrieval in parallel, combines the results, reranks, and synthesizes a unified answer.

---

## 11. LLM Gateway & Models

The `LLMGateway` class in `llm/gateway.py` is the *single* interface for all LLM calls. It uses **LiteLLM** internally, which means:

- One API to call regardless of the underlying model provider
- Automatic 3-attempt retry with fallback through an ordered model chain
- **Disk cache** at `./data/litellm_cache` — identical prompts never hit the API twice, even across restarts

Each task type (routing, extraction, summarization, synthesis) has its own primary model and fallback chain defined in `llm/models.py`. To change which model handles summarization, edit that registry entry — no other code changes required.

**To use a fully local LLM with no internet dependency:**

```bash
docker compose --profile local-llm up -d
```

This starts an Ollama container. The fallback chain will automatically route to it when cloud APIs are unavailable.

---

## 12. Development Guide

### Running Tests

```bash
python -m pytest tests/ -x -q
```

- `-x` stops on first failure (faster feedback loop)
- `-q` quiet output (just failures)

### Linting & Formatting

```bash
# Check for errors and import order violations
ruff check ./graphnotebook/ --select E,F,I

# Auto-format all files
ruff format ./graphnotebook/
```

### Re-initializing the Graph Schema

If you reset Neo4j or need to recreate constraints and indexes:

```bash
python -m graphnotebook.graph.schema_init
```

### Critical Development Constraints

These rules prevent subtle bugs and must be followed:

| Constraint | Why |
|---|---|
| Use **absolute imports only**: `from graphnotebook.module import X` | Prevents ambiguous relative import resolution |
| Instantiate **one `neo4j.Driver`** in `main.py`, pass via dependency injection | Connection pool is expensive; multiple drivers cause resource leaks |
| Load **one `SentenceTransformer`** (BGE-M3) at startup, pass the instance | The 1.1 GB model must not be re-loaded per request |
| **Drop GDS projections immediately** after every algorithm call | GDS projections consume significant memory and block re-projection |
| Community summaries are **generated lazily at query time only** | Eager generation during ingestion would block the pipeline unnecessarily |
| Always pass `SimpleKGPipeline(from_pdf=False)` | Documents are pre-parsed by `parsers.py`; the pipeline must not re-parse |
| All Cypher queries live in **`graph/queries.py`** and graph-layer modules only | Centralizes query management, prevents scattered/unmaintainable Cypher |

---

## 13. Roadmap

GraphNotebook is being built in four phases, each delivering a working milestone:

### ✅ Phase 1 — Foundation: *"Upload & Chat"*
Basic document upload, vector RAG, and chat UI. Deliverable: working personal search over your documents.

### 🔧 Phase 2 — Knowledge Graph: *"Extract & Connect"*
Schema-enforced entity extraction, entity deduplication, hybrid vector+graph retrieval, cross-encoder reranking. Deliverable: GraphRAG with schema-aware extraction.

### 🔧 Phase 3 — Communities + Agentic: *"Understand & Reason"*
GDS Leiden community detection, lazy summarization, global map-reduce search, LangGraph agentic router with Text-to-Cypher fallback. Deliverable: full GraphRAG with agentic reasoning.

### 🔮 Phase 4 — Polish: *"Production Personal Tool"*
Graph visualization (pyvis), notebook management, schema editor UI, streaming chat, conversation history, incremental re-indexing, GraphML/JSON-LD export. Deliverable: feature-complete personal knowledge base.

---

## 14. Architecture Decisions

| Decision | Rationale |
|---|---|
| **LiteLLM** over custom LLM client | Unified interface across 100+ providers, built-in disk cache, automatic fallback chains — replaces bespoke rate limiter + cache modules |
| **BGE-M3** for embeddings | State-of-the-art multilingual embeddings, runs fully locally, 1024-dim supports rich semantic similarity |
| **Neo4j GDS Leiden** for community detection | Native graph algorithm avoids external clustering libraries; hierarchical levels enable both fine-grained and coarse community views |
| **Lazy community summarization** | Generating summaries at query time (not ingest time) keeps ingestion fast and avoids summarizing communities that are never queried |
| **LangGraph** for all orchestration | Explicit state machines with typed state dictionaries make complex multi-step pipelines debuggable, resumable, and testable |
| **`SimpleKGPipeline(from_pdf=False)`** | Pre-parsing in `parsers.py` gives unified text normalization across all file types before KG extraction |
| **RapidFuzz + embedding cosine** for entity dedup | Two-stage deduplication catches both surface-form variations (fuzzy string match) and semantic synonyms (embedding similarity) |
| **Disk cache for LLM responses** | Eliminates redundant API calls for identical prompts across restarts — critical for iterative development with zero-cost models |

---

## 15. Glossary

| Term | Plain-English Definition |
|---|---|
| **RAG** | *Retrieval-Augmented Generation* — answering questions by first finding relevant documents, then having the LLM summarize them |
| **GraphRAG** | RAG that uses a knowledge graph to find related facts beyond simple vector similarity |
| **Knowledge Graph** | A database of *entities* (things) and *relationships* (how those things connect) |
| **Entity** | A named concept extracted from text: a person, organization, technology, idea, etc. |
| **Embedding** | A list of numbers that represents the *meaning* of a piece of text — similar meanings produce similar numbers |
| **Vector Search** | Finding the most semantically similar documents by comparing their embeddings |
| **Community** | A cluster of closely related entities detected by the Leiden algorithm |
| **Leiden Algorithm** | A graph-partitioning algorithm that finds densely connected groups (communities) in a network |
| **Lazy Summarization** | Generating a summary only when first needed, then caching the result for future use |
| **LiteLLM** | A Python library providing a single unified interface to many different LLM APIs |
| **LangGraph** | A library for building stateful, multi-step AI pipelines as explicit graph-shaped workflows |
| **Cross-Encoder** | A model that scores the relevance of a (query, document) pair — more precise than embedding similarity alone |
| **Map-Reduce** | A pattern where work is split across many items (map), then results are combined (reduce) — used here to synthesize across many community summaries |
| **Cypher** | The query language used to read and write data in Neo4j (analogous to SQL for relational databases) |
| **GDS** | *Graph Data Science* — a Neo4j plugin providing graph algorithms including Leiden community detection |
| **APOC** | *Awesome Procedures on Cypher* — a Neo4j plugin with utility procedures for data manipulation |
| **BGE-M3** | A high-quality open-source embedding model from BAAI; runs locally, no API required |
| **Bolt** | The binary network protocol used to connect to Neo4j (default port: 7687) |
| **Pydantic Settings** | A Python library for loading and validating configuration from environment variables and `.env` files |

---

*Built with Neo4j · LangGraph · LiteLLM · Gradio · sentence-transformers*
