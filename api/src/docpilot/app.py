"""FastAPI application factory. Run with: uvicorn docpilot.app:app"""

import asyncio
import logging
from contextlib import asynccontextmanager

import openai
import psycopg
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from psycopg_pool import PoolTimeout

from docpilot.auth import Passwords
from docpilot.cache import RetrievalCache, make_redis
from docpilot.config import Settings, get_settings
from docpilot.db import open_pool, schema_identity
from docpilot.embeddings import make_embedder
from docpilot.llm import make_provider
from docpilot.observability import RequestContextMiddleware, configure_logging
from docpilot.retrieval.search import Retriever
from docpilot.routes import auth, chat, documents, jobs
from docpilot.security import RateLimiter, RedisRateLimiter, get_principal

log = logging.getLogger(__name__)
READY_TIMEOUT_SECONDS = 3


def check_settings(settings: Settings) -> None:
    if settings.environment == "production":
        if not settings.api_key_list and not settings.jwt_secret_list:
            raise RuntimeError("ENVIRONMENT=production requires API_KEYS or JWT_SECRETS")
        if any(len(s) < 32 for s in settings.jwt_secret_list):
            raise RuntimeError("JWT_SECRETS must each be at least 32 characters in production")
        if not settings.redis_url:
            log.warning("no REDIS_URL: rate limits are per instance, and nothing is cached")
    if settings.ingest_mode == "queue" and not settings.redis_url:
        raise RuntimeError("INGEST_MODE=queue needs REDIS_URL")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    check_settings(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Long-lived clients are built once and shared by every request via app.state.
        pool = await open_pool(settings)
        embedder = make_embedder(settings)
        redis = make_redis(settings)
        cache = None
        if redis is not None:
            app.state.rate_limiter = RedisRateLimiter(redis)
            cache = RetrievalCache(redis, namespace=await schema_identity(pool, settings),
                                   search_ttl=settings.search_cache_ttl_seconds,
                                   embedding_ttl=settings.query_embedding_cache_ttl_seconds)
        app.state.pool = pool
        app.state.redis = redis
        app.state.cache = cache
        app.state.embedder = embedder
        app.state.retriever = Retriever(pool, embedder, settings, cache)
        app.state.llm = make_provider(settings)
        app.state.passwords = Passwords(settings)
        app.state.arq = None
        if settings.ingest_mode == "queue":
            app.state.arq = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        try:
            yield
        finally:
            await pool.close()
            for client in (redis, app.state.arq):
                if client is not None:
                    await client.aclose()

    app = FastAPI(title="DocPilot API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.rate_limiter = RateLimiter()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )
    app.add_middleware(RequestContextMiddleware)  # outermost, so it sees every response
    @app.exception_handler(openai.OpenAIError)
    async def embeddings_down(request: Request, exc: openai.OpenAIError) -> JSONResponse:
        log.warning("embedding service failed: %s", type(exc).__name__)
        return JSONResponse({"detail": "embedding service unavailable"}, status_code=503)

    @app.exception_handler(PoolTimeout)
    @app.exception_handler(psycopg.OperationalError)
    async def database_down(request: Request, exc: Exception) -> JSONResponse:
        log.warning("database failed: %s", type(exc).__name__)
        return JSONResponse({"detail": "database unavailable"}, status_code=503)

    authenticated = [Depends(get_principal)]
    app.include_router(auth.router)
    app.include_router(documents.router, dependencies=authenticated)
    app.include_router(chat.router, dependencies=authenticated)
    app.include_router(jobs.router, dependencies=authenticated)

    @app.get("/healthz")
    async def healthz() -> dict:
        """Liveness: the process is up. Must not depend on the database, or a DB blip restarts us."""
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> dict:
        """Readiness: can we serve traffic right now?"""
        try:
            async with asyncio.timeout(READY_TIMEOUT_SECONDS), app.state.pool.connection() as conn:
                await conn.execute("SELECT 1")
        except Exception as e:
            log.warning("readiness check failed: %s", type(e).__name__)
            raise HTTPException(503, "database unavailable") from e
        return {"ok": True}

    return app


configure_logging(get_settings())
app = create_app()
