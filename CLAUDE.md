# GraphNotebook

## Build & Test
- Install: `uv sync --all-extras`
- Test: `uv run pytest tests/ -x -q`
- Lint: `uv run ruff check ./graphnotebook/ --select E,F,I`
- Format: `uv run ruff format ./graphnotebook/`
- Schema init: `uv run python -m graphnotebook.graph.schema_init`
- Run: `uv run python -m graphnotebook.main`
- Docker services: `docker compose up -d`

## Conventions
- Use absolute imports exclusively: `from graphnotebook.module import X`
- All config uses Pydantic Settings with `GN_` env prefix

## Critical Constraints
- Store all Cypher queries exclusively in dedicated query modules (`graph/queries.py`, `graph/schema_init.py`, `retrieval/local_search.py`)
- Use `SimpleKGPipeline(from_pdf=False)` exclusively — all documents are pre-parsed by `parsers.py`
- Instantiate one `neo4j.Driver` in `main.py` and pass via dependency injection to all consumers
- Drop GDS graph projections immediately after every algorithm call: `gds.graph.drop('entity_graph')`
- Generate community summaries exclusively at query time (lazy, cached on Community nodes)
- Load `SentenceTransformer` (BGE-M3, 1.1GB) once at startup and pass the `EmbeddingEngine` instance
