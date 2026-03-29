# GraphNotebook Lessons Learned

### Lesson 1: LiteLLM MagicMock Recursion Trap (Task: 5d8666cc, Date: 2026-03-29)
- **Root Cause**: `unittest.mock.MagicMock` attributes themselves resolve to `MagicMock` instances. When attempting to emulate a complex nested API response object (e.g., `litellm.completion.choices[0].message.content`), setting `m.choices[0].message.content = "..."` without correctly allocating specific strings resulted in dynamic dictionary accesses that still yielded MagicMocks.
- **Why It Failed**: Standard library components (e.g., `json.loads(response)`) expect string or byte objects. Because the MagicMock recursively proxied down to the `content` assignment, `json.loads` received a `MagicMock` instance, triggering `TypeError`.
- **Prevention Rule**: **NEVER** use `MagicMock` to stub out deterministic dataclass-like nested attributes (e.g., API schemas). Always create simple structured `dataclasses` or object stubs (e.g., `MockResponse`, `MockMessage`) to simulate JSON/API response trees deterministically. 
- **Regression Test**: Test suite `tests/test_llm_gateway.py` passes under `uv run pytest tests/ -x` verifying the fallback LLM invocation without throwing payload type errors.
