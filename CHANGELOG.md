# Changelog: GraphNotebook

All notable changes to the GraphNotebook project are documented below. This project adheres to **v1.1 Architecture Standards** (March 2026).

---

## [1.2.0] - 2026-03-29 (Stabilization & Hardening)

### Added
- **Global Search Tie-Breaking**: Community `rank` (from GDS Leiden) is now preserved during the Map phase and used to resolve ties in LLM relevance scores.
- **Enhanced Source Attribution**: `ContextBuilder` now provides formatted `Source (Page N)` metadata for every retrieved chunk.
- **Mutation-Killing Tests**: Added 58+ new tests in `test_pipeline.py` targeting LangGraph routing logic and error-passthrough boundaries.
- **Coverage Baseline**: Initialized comprehensive HTML coverage reports (74% baseline reached).

### Changed
- **Retrieval Signatures**: Refactored `LocalSearcher` and `GlobalSearcher` to fully support dependency injection of `EmbeddingEngine` and `LLMGateway`.
- **Hybrid Search Optimization**: `LocalSearcher` now explicitly sorts hybrid results by descending score for consistent retrieval across mock and production environments.
- **Router Logic**: Updated `router.py` to correctly instantiate `EmbeddingEngine` and use the unified `hybrid_search` endpoint.
- **Semantic Chunking Boundary**: Increased stress-testing of the 512-token boundary to ensure absolute overlap and size correctness.

### Fixed
- **GlobalSearcher Syntax**: Resolved a critical prompt-string corruption issue in the Map-Reduce loop.
- **Import Errors**: Fixed missing `get_schema_hash` exports in `extraction` and `ingestion` modules.
- **ContextBuilder Deduplication**: Corrected logic for deduplicating sources when chunks from the same file/page are retrieved with different indices.
- **Settings Alignment**: Realigned Pydantic `Settings` with updated environment prefix `GN_`.

---

## [1.1.0] - 2026-03-28 (Phase 4: Multi-Tenancy & Performance)

### Added
- **Notebook Scoping**: Strict multi-tenant isolation implemented across all retrieval queries and GDS graph projections via `notebook_id` filtering.
- **Incremental Ingestion**: Hash-based duplicate detection ensures that only modified files or files with changed schemas are re-extracted.
- **Gradio Dashboard**: Refined UI to include a notebook-specific dashboard and "Load Full" graph visualization.
- **Architecture Harvesting**: Automated harvesting of LangGraph async patterns into the local knowledge base.

### Changed
- **LangGraph State Transition**: Migrated orchestration states from `TypedDict` to Pydantic `BaseModel` for improved validation.
- **uv Integration**: Standardized all build and execution commands to use `uv` for 10x faster dependency management.

---

## [1.0.0] - 2026-03-22 (Core Refactor)

### Added
- **Vectorized Aggregation**: Extracted heavy logic into `aggregation.py` using PyArrow for high-performance node processing.
- **E2E Test Suite**: Initialized Playwright-based end-to-end tests for the Gradio UI.
- **Self-Healing Locators**: Implemented robust UI testing architecture to withstand layout shifts.

### Changed
- **Decoupled Architecture**: Refactored `app.py` "God Class" into separate `ui_components.py` and `state_managers.py`.
- **Module Partitioning**: Partitioned `visualization.py` and `insights.py` to meet the <300 lines/file threshold.

---

## [0.9.0] - 2026-03-17 (Export Features)

### Added
- **1-Click Export**: Implemented GitHub Gist and Notion DB export functionality with direct UI triggers.
- **Identity Synthesis**: Initial implementation of the `LLMGateway` for structured extraction.

---
> [!NOTE]
> Detailed commit logs are available at: [github.com/JackSmack1971/GraphNotebook/commits/main/](https://github.com/JackSmack1971/GraphNotebook/commits/main/)
