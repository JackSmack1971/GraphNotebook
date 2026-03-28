"""
LiteLLM-based LLM gateway.
Provides:
  - Disk caching (survives restarts, zero infra)
  - Automatic fallbacks across free models + Ollama
  - Cost tracking + observability
  - Unified interface for all LLM tasks
"""

import json

import litellm
from litellm import completion

from .models import MODEL_REGISTRY

# Global litellm configuration
litellm.set_verbose = False

# Enable disk caching (lazy initialized on class creation, but it's a singleton pattern)
litellm.cache = litellm.Cache(type="disk", disk_cache_dir="./data/litellm_cache")


class LLMGateway:
    """Unified LLM interface via LiteLLM."""

    def __init__(self, task: str = "synthesis"):
        config = MODEL_REGISTRY.get(task, MODEL_REGISTRY["synthesis"])
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
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback if json load still fails
            raise ValueError(f"Failed to parse JSON from LLM response: {raw}")
    def invoke_stream(self, prompt: str, system: str = "", **kwargs):
        """Streaming version of invoke."""
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
                stream=True,
                **kwargs,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            raise RuntimeError(f"Streaming failed: {e}")
