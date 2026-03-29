"""
Application configuration managed by Pydantic Settings.
All environment variables use the GN_ prefix.

Python 3.14 note: pydantic-settings v2 with pydantic v2 works correctly.
Fields are declared as standard class attributes — no __getattr__ path required.
"""

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration managed by Pydantic.
    Loads from environment variables with GN_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Neo4j ──────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://127.0.0.1:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir: str = Field(default="./data")

    # ── LLM ───────────────────────────────────────────────────────────────────
    openrouter_api_key: str = Field(default="")
    litellm_cache_dir: str = Field(default="./data/litellm_cache")

    # ── Embedding / Chunking ───────────────────────────────────────────────────
    embedding_model: str = Field(default="BAAI/bge-m3")
    # Dimensions must match the loaded model.
    # BGE-M3 → 1024  |  bge-base-en-v1.5 → 768
    embedding_dimensions: int = Field(default=1024)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    encoding_name: str = Field(default="cl100k_base")

    # ── Retrieval ──────────────────────────────────────────────────────────────
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    local_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=8)
    global_top_communities: int = Field(default=5)
    max_context_tokens: int = Field(default=4000)

    # ── Derived paths (properties, not fields — never read from env) ───────────
    @property
    def uploads_dir(self) -> str:
        d = os.path.join(self.data_dir, "uploads")
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def cache_dir(self) -> str:
        d = self.litellm_cache_dir
        os.makedirs(d, exist_ok=True)
        return d
