"""All runtime configuration, read from env vars (and an optional .env file).

One flat settings object keeps the "what can I tune?" question answerable in one file.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["vector", "fts", "hybrid"]
RerankerName = Literal["none", "cohere"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    # Storage. DB_SCHEMA lets evals run in an isolated namespace of the same database.
    database_url: str = "postgresql://docpilot:docpilot@localhost:5432/docpilot"
    db_schema: str = "public"

    # Chat model.
    llm_provider: Literal["anthropic", "openai", "fake"] = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_output_tokens: int = 16000
    max_agent_steps: int = 6

    # OpenAI-compatible endpoint: OpenAI itself, or the LLM Gateway via OPENAI_BASE_URL.
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    chat_model: str = "gpt-4.1-mini"

    # Embeddings. Dimension is fixed by the schema: vector(1024).
    embedding_provider: Literal["openai", "fake"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1024
    embedding_batch_size: int = 64

    # Chunking (token counts are approximate: ~0.75 words per token).
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 60

    # Retrieval.
    retrieval_mode: RetrievalMode = "hybrid"
    reranker: RerankerName = "none"
    rerank_api_key: str | None = None
    rerank_model: str = "rerank-v3.5"
    candidate_k: int = 30  # how many candidates each retriever contributes before fusion

    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
