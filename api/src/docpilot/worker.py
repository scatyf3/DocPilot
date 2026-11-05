"""The background worker: `docpilot worker` (arq, on the same Redis as the API).

Runs ingest jobs, and on a schedule the job sweeper and refresh-token cleanup. Scale it
separately from the API: ingest is CPU (PDF parsing) and embedding-API bound, chat is not.
"""

import logging
import uuid

from arq import cron
from arq.connections import RedisSettings
from arq.worker import Retry, Worker, func

from docpilot import auth, jobs
from docpilot.cache import RetrievalCache, make_redis
from docpilot.config import Settings, get_settings
from docpilot.db import open_pool, schema_identity
from docpilot.embeddings import make_embedder

log = logging.getLogger(__name__)
RETRY_BACKOFF_SECONDS = (5, 30, 120)


async def ingest_document(ctx: dict, job_id: str, tenant_id: int) -> dict:
    try:
        return await jobs.run_ingest_job(
            ctx["pool"], ctx["embedder"], ctx["settings"], ctx["cache"],
            tenant_id=tenant_id, job_id=uuid.UUID(job_id), attempt=ctx["job_try"])
    except jobs.TransientIngestError as e:
        delay = RETRY_BACKOFF_SECONDS[min(ctx["job_try"], len(RETRY_BACKOFF_SECONDS)) - 1]
        raise Retry(defer=delay) from e


async def sweep_jobs(ctx: dict) -> dict:
    return await jobs.sweep(ctx["owner_pool"], ctx["redis"], ctx["settings"])


async def clean_refresh_tokens(ctx: dict) -> int:
    return await auth.delete_expired_refresh_tokens(ctx["owner_pool"])


def worker_settings(settings: Settings) -> dict:
    """Keyword arguments for arq's Worker, built from our settings rather than a module class."""
    if not settings.redis_url:
        raise RuntimeError("the worker needs REDIS_URL")

    async def startup(ctx: dict) -> None:
        ctx["settings"] = settings
        ctx["pool"] = await open_pool(settings)                    # tenant-scoped, under RLS
        ctx["owner_pool"] = await open_pool(settings, as_owner=True)  # sweeper and cleanup only
        ctx["embedder"] = make_embedder(settings)
        ctx["cache_redis"] = make_redis(settings)
        ctx["cache"] = RetrievalCache(ctx["cache_redis"],
                                      namespace=await schema_identity(ctx["pool"], settings),
                                      search_ttl=settings.search_cache_ttl_seconds,
                                      embedding_ttl=settings.query_embedding_cache_ttl_seconds)

    async def shutdown(ctx: dict) -> None:
        for name in ("pool", "owner_pool"):
            if name in ctx:
                await ctx[name].close()
        if ctx.get("cache_redis") is not None:
            await ctx["cache_redis"].aclose()

    return {
        "functions": [func(ingest_document, name=jobs.TASK, max_tries=settings.ingest_job_max_tries,
                           timeout=settings.ingest_job_timeout_seconds)],
        "cron_jobs": [cron(sweep_jobs, second=0, run_at_startup=True),            # every minute
                      cron(clean_refresh_tokens, hour=3, minute=17, second=0)],  # daily
        "redis_settings": RedisSettings.from_dsn(settings.redis_url),
        "on_startup": startup,
        "on_shutdown": shutdown,
        "max_jobs": settings.worker_concurrency,
        "keep_result": 3600,
    }


def run_worker() -> None:
    settings = get_settings()
    Worker(**worker_settings(settings)).run()
