"""FastAPI application factory. Run with: uvicorn docpilot.app:app"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docpilot.config import Settings, get_settings
from docpilot.db import open_pool
from docpilot.embeddings import make_embedder
from docpilot.llm import make_provider
from docpilot.retrieval.search import Retriever
from docpilot.routes import chat, documents

logging.basicConfig(level=logging.INFO)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Long-lived clients are built once and shared by every request via app.state.
        pool = await open_pool(settings)
        embedder = make_embedder(settings)
        app.state.settings = settings
        app.state.pool = pool
        app.state.embedder = embedder
        app.state.retriever = Retriever(pool, embedder, settings)
        app.state.llm = make_provider(settings)
        try:
            yield
        finally:
            await pool.close()

    app = FastAPI(title="DocPilot API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.get("/healthz")
    async def healthz() -> dict:
        async with app.state.pool.connection() as conn:
            await conn.execute("SELECT 1")
        return {"ok": True, "llm": app.state.llm.model, "embedder": app.state.embedder.name}

    return app


app = create_app()
