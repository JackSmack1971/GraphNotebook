import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration managed by Pydantic.
    Loads from environment variables with GN_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GN_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Neo4j Settings
    neo4j_uri: str = Field(default="bolt://127.0.0.1:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # Paths
    data_dir: str = Field(default="./data")

    # LLM Settings
    openrouter_api_key: str = Field(default="")
    litellm_cache_dir: str = Field(default="./data/litellm_cache")

    # Embedding/Chunking Settings
    embedding_model: str = Field(default="BAAI/bge-m3")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    encoding_name: str = Field(default="cl100k_base")

    # Retrieval Settings
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

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
