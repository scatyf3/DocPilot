"""FastAPI application factory. Run with: uvicorn docpilot.app:app"""

import asyncio
import logging
from contextlib import asynccontextmanager

import openai
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from psycopg_pool import PoolTimeout

from docpilot.config import Settings, get_settings
from docpilot.db import open_pool
from docpilot.embeddings import make_embedder
from docpilot.llm import make_provider
from docpilot.observability import RequestContextMiddleware, configure_logging
from docpilot.retrieval.search import Retriever
from docpilot.routes import chat, documents
from docpilot.security import RateLimiter, require_api_key

log = logging.getLogger(__name__)
READY_TIMEOUT_SECONDS = 3


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    if settings.environment == "production" and not settings.api_key_list:
        raise RuntimeError("ENVIRONMENT=production requires API_KEYS")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Long-lived clients are built once and shared by every request via app.state.
        pool = await open_pool(settings)
        embedder = make_embedder(settings)
        app.state.pool = pool
        app.state.embedder = embedder
        app.state.retriever = Retriever(pool, embedder, settings)
        app.state.llm = make_provider(settings)
        try:
            yield
        finally:
            await pool.close()

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

    authenticated = [Depends(require_api_key)]
    app.include_router(documents.router, dependencies=authenticated)
    app.include_router(chat.router, dependencies=authenticated)

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
