# Project Roadmap: GraphNotebook Stabilization

This task list is derived from the **tasks.xml** mutation audit and represents the final hardening phase for the GraphNotebook project.

## [x] Hardening & Mutation Killers (High Priority)

- [x] **Priority 1**: Add global/hybrid mode tests to `tests/test_router.py` (Kills M9).
- [x] **Priority 2**: Assert `top_k=8` keyword argument in router → reranker integration (Kills M10).
- [x] **Priority 3**: Expand `tests/test_text2cypher.py` to cover all query construction and result mapping paths (Kills M11).
- [x] **Priority 4**: Implement fallback chain tests in `tests/test_llm_gateway.py` (Kills M12).
- [x] **Priority 5**: Verify CRUD coverage in `tests/test_notebook_manager.py` (Completed).
- [x] **Priority 6**: Add iteration ceiling assertion (max 3 cycles) in `tests/test_router.py` to prevent infinite loops.

## [x] Coverage Expansion (Target ≥ 80%)

- [x] **retrieval/router.py**: Exercise global/hybrid branches (97% coverage achieved).
- [x] **retrieval/text2cypher.py**: Unit test query construction and results (95% coverage achieved).
- [x] **llm/gateway.py**: Test fallback chain parameters (93% coverage achieved).
- [ ] **extraction/kg_pipeline.py**: Unit test `_build_pipeline` and schema enforcement (currently 52%).
- [ ] **UI Layer**: Implement unit tests for `ui/callbacks.py` and `ui/components.py` (currently 0%).

## [/] Recent Accomplishments (March 29)

- [x] Implemented **Global Search Tie-Breaking** using community `rank`.
- [x] Added formatted **Source Attribution** to `ContextBuilder`.
- [x] Added 58+ tests to `test_pipeline.py` for routing and error passthrough.
- [x] Stabilized full test suite (128 passing).
- [x] Finalized architectural alignment for searcher signatures.

---
> [!TIP]
> Use `uv run pytest tests/ --cov=graphnotebook --cov-report=term-missing` to verify progress.
