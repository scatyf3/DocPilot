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

    # Production refuses to start without API keys or JWT secrets; see security.py.
    environment: Literal["development", "production"] = "development"
    # Comma-separated service keys; several allow rotation without downtime. A key acts on the
    # default tenant (id 1), which is also where everything from before multi-tenancy lives.
    api_keys: str = ""

    # User accounts: access tokens are HS256 JWTs signed with the first secret and verified
    # against all of them, so secrets rotate like API keys. Empty disables user login.
    jwt_secrets: str = ""
    access_token_ttl_seconds: int = 900
    refresh_token_ttl_days: int = 30
    allow_signup: bool = True
    # argon2id cost. Each hash holds argon2_memory_kib of RAM, so hashing is also capped in
    # concurrency: a burst of logins must not take the instance's memory with it.
    argon2_time_cost: int = 3
    argon2_memory_kib: int = 65536
    argon2_parallelism: int = 4
    password_hash_concurrency: int = 4

    # Redis: shared rate limits, caches and the ingest queue. Without it, rate limits are per
    # instance (N instances allow N times the limit) and nothing is cached.
    redis_url: str | None = None
    redis_timeout_seconds: float = 0.5  # an outage should degrade requests, not stall them

    # Rate limits, per client (user, or IP for key holders and anonymous callers).
    rate_limit_enabled: bool = True
    rate_limit_chat_per_minute: int = 30
    rate_limit_search_per_minute: int = 120
    rate_limit_upload_per_minute: int = 10
    rate_limit_auth_per_minute: int = 10
    client_ip_header: str | None = None  # e.g. fly-client-ip; only set it behind a trusted proxy

    # Caches (need Redis; 0 disables). Search results are invalidated by any write to the tenant's
    # library, so the TTL only bounds memory; query embeddings never go stale for a fixed model.
    search_cache_ttl_seconds: int = 300
    query_embedding_cache_ttl_seconds: int = 86_400

    # Ingest: `inline` parses and embeds inside the upload request; `queue` stores the file,
    # returns 202 with a job id and leaves the work to `docpilot worker` (needs Redis).
    ingest_mode: Literal["inline", "queue"] = "inline"
    worker_concurrency: int = 4
    ingest_job_timeout_seconds: int = 900
    ingest_job_max_tries: int = 3

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
    # The app's connections switch to this role, which row-level security applies to (table
    # owners and superusers bypass it). `docpilot db init` creates it. Empty: rely on the
    # tenant filters in the queries alone, e.g. on a host where you cannot create roles.
    db_app_role: str = "docpilot_app"

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
    def jwt_secret_list(self) -> list[str]:
        return [k.strip() for k in self.jwt_secrets.split(",") if k.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
