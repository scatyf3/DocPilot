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

    # production refuses to start without API keys; see security.py for what a key does and doesn't do.
    environment: Literal["development", "production"] = "development"
    api_keys: str = ""  # comma-separated; several keys allow rotation without downtime

    # Rate limits, per client, per instance.
    rate_limit_enabled: bool = True
    rate_limit_chat_per_minute: int = 30
    rate_limit_search_per_minute: int = 120
    rate_limit_upload_per_minute: int = 10
    client_ip_header: str | None = None  # e.g. fly-client-ip; only set it behind a trusted proxy

    # Bounds on one chat turn.
    max_question_chars: int = 8000
    max_history_messages: int = 40
    chat_timeout_seconds: float = 300.0

    # Upstream calls.
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    embedding_timeout_seconds: float = 30.0

    log_json: bool = False
    log_level: str = "INFO"

    # Storage. DB_SCHEMA lets evals run in an isolated namespace of the same database.
    database_url: str = "postgresql://docpilot:docpilot@localhost:5432/docpilot"
    db_schema: str = "public"
    db_pool_max_size: int = 10
    db_pool_timeout_seconds: float = 10.0  # waiting for a free connection
    db_statement_timeout_ms: int = 30_000  # one runaway query must not hold a connection forever

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
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
