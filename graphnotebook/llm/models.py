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
