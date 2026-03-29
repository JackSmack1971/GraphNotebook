# Current Architecture: GraphNotebook

## Core Tech Stack
- **Languages:** Python 3.11+
- **Frontend / UI:** Gradio 5.x for rapid interactions, streaming chat, and file uploads.
- **Graph Database:** Neo4j Community 5.20+ (with APOC and GDS for algorithms like Leiden community detection).
- **Orchestration Layer:** LangGraph (used for both the ingestion pipeline state machine and the agentic query router).
- **Knowledge Graph Framework:** `neo4j-graphrag` (with `SimpleKGPipeline` and Schema-builder functionality).
- **LLM Abstraction Layer:** LiteLLM (unifies OpenRouter, Ollama, and manages disk caching/fallbacks).
- **Embeddings & Reranking (Local):** `sentence-transformers` for local vectors (`BAAI/bge-m3`) and cross-encoder precision models (`ms-marco-MiniLM-L-6-v2`).
- **Data Parsing/Validation:** PyMuPDF (`fitz`), `python-docx` for document parsers; Pydantic for settings and state models.

## Design Patterns Detected
1. **Agentic Orchestration Pattern:** Both ingestion and retrieval utilize a State Machine approach via `LangGraph`, enforcing linear, recoverable steps (e.g. `parse -> chunk -> embed_store -> extract_kg`). The agentic query router dynamically determines the correct retrieval strategy (e.g., text2cypher vs local vs global).
2. **Schema-Enforced Ontology Constraint:** Entity extraction strictly limits the LLM to pre-defined entity and relationship types detailed via `SchemaBuilder`. This prevents uncontrolled generation ("ontology drift").
3. **Lazy Execution / Evaluation Pipeline:** Graph community summaries are generated lazily (on-demand at query time layer) rather than actively computed after each insertion, saving model inference costs.
4. **Dependency Injection:** Centralized initialization is handled within `graphnotebook.main`. Heavy instances like the `Neo4jClient`, `LLMGateway`, and `EmbeddingEngine` are initiated once and passed into specialized sub-modules to minimize memory load.
5. **Gateway / Adapter Pattern:** `LLMGateway` (wrapping `LiteLLM`) isolates API usage paths from the underlying generation requests, establishing straightforward fallback chains across tiers (DeepSeek -> Meta -> Local Ollama).

## Key Directories
- `graphnotebook/ui/`: Contains the Gradio blocks application layout (`app.py`), event callbacks (`callbacks.py`), and UI components. 
- `graphnotebook/ingestion/`: Handles document import pipelines, robust parsing (`parsers.py` supports PDF/docx/md), and advanced paragraph-aware token semantic chunking (`chunker.py`).
- `graphnotebook/extraction/`: Enforces domain ontology schema definitions (`schema.py`), manages KG pipeline, and operates entity resolution / fuzzy matching (`resolver.py`).
- `graphnotebook/graph/`: Controls the Neo4j client connection pooling, initializes graph DDL / vector constraints (`schema_init.py`), acts as a centralized Cypher query registry (`queries.py`), and runs advanced graph ML communities modeling (`communities.py`).
- `graphnotebook/retrieval/`: Manages agentic routing decision logic (`router.py`) and specific retrieval strategies like global summary searches, vector+graph local traversal (`local_search.py`), LLM cross-encoder reranking, and `text2cypher` logic.
- `graphnotebook/llm/`: Replaces custom API clients with unified `gateway.py` (LiteLLM registry), and manages local open-source `sentence-transformers` interactions.

## "Golden Path" Flow (Document Ingestion)
1. **Entry (UI):** User submits a file (e.g., PDF) via Gradio UI upload panel (`graphnotebook.ui.callbacks`).
2. **Parse:** Request enters `graphnotebook.ingestion.pipeline` (LangGraph state machine). Parses raw context into metadata via `parsers.py`.
3. **Chunk:** `chunker.py` deterministically divides parsed text boundaries by valid token overlap (using tiktoken semantic boundaries).
4. **Embed:** Internal `EmbeddingEngine` encodes paragraph chunks locally via `BGE-M3`.
5. **Graph Construction (Database):** State flow hands off text chunks to `SimpleKGPipeline` logic in `extraction/kg_pipeline.py`. 
   - Entities and specific relationships are identified and constrained by the rules dictated in `extraction/schema.py`.
6. **Persistence:** `Neo4jClient` executes parameterized Cypher writes to graph database. Indexes ensure entities update seamlessly.
