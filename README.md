# GraphNotebook

Personal GraphRAG knowledge base. NotebookLM meets knowledge graphs.
Single-user. Zero cost. Full graph intelligence.

## Features
- Document Ingestion: PDF, DOCX, TXT, MD
- Local Embeddings using `sentence-transformers` (BGE-M3)
- LLM Gateway: LiteLLM with OpenRouter and Ollama fallbacks
- Schema-Enforced Knowledge Graph Construction
- Native Neo4j Vector and Hybrid Retrieval

## Quick Start
1. Start Neo4j: `docker compose up -d`
2. Install dependencies: `pip install -e ".[dev]"`
3. Copy `.env.example` to `.env` and fill in API keys
4. Initialize the Graph Schema: `python -m graphnotebook.graph.schema_init`
5. Start the application: `python -m graphnotebook.main`
