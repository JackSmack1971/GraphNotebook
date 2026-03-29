"""Tests for graphnotebook.llm.models MODEL_REGISTRY."""

from graphnotebook.llm.models import MODEL_REGISTRY


def test_all_task_keys_present():
    for task in ("extraction", "synthesis", "summarization", "routing"):
        assert task in MODEL_REGISTRY, f"Missing task key: {task}"


def test_each_task_has_primary_and_fallbacks():
    for task, cfg in MODEL_REGISTRY.items():
        assert "primary" in cfg, f"{task} missing 'primary'"
        assert "fallbacks" in cfg, f"{task} missing 'fallbacks'"
        assert isinstance(cfg["fallbacks"], list), f"{task}.fallbacks must be list"
        assert len(cfg["fallbacks"]) >= 1, f"{task} needs ≥1 fallback"


def test_ollama_fallback_exists():
    """Every task must have at least one local Ollama fallback for offline use."""
    for task, cfg in MODEL_REGISTRY.items():
        has_ollama = any("ollama" in fb for fb in cfg["fallbacks"])
        assert has_ollama, f"{task} has no Ollama fallback"
