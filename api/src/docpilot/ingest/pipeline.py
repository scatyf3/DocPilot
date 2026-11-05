"""parse -> chunk -> embed -> store. Used by uploads, the CLI, evals, the worker and create_page."""

import asyncio
from collections.abc import Awaitable, Callable

import numpy as np
from psycopg_pool import AsyncConnectionPool

from docpilot import store
from docpilot.cache import RetrievalCache
from docpilot.config import Settings
from docpilot.embeddings import Embedder
from docpilot.ingest.chunk import chunk_text
from docpilot.ingest.parse import parse_document

Progress = Callable[[int, int], Awaitable[None]]  # (chunks embedded so far, total)


def make_chunks(settings: Settings, text: str) -> list[str]:
    chunks = chunk_text(text, settings.chunk_tokens, settings.chunk_overlap_tokens)
    if not chunks:
        raise ValueError("document has no extractable text")
    return chunks


async def embed_chunks(embedder: Embedder, settings: Settings, chunks: list[str],
                       progress: Progress | None = None) -> list[np.ndarray]:
    """Embed in batches, reporting progress after each one (a long PDF takes a while)."""
    out: list[np.ndarray] = []
    step = max(settings.embedding_batch_size, 1)
    for i in range(0, len(chunks), step):
        out.extend(await embedder.embed(chunks[i : i + step]))
        if progress is not None:
            await progress(len(out), len(chunks))
    return out


async def ingest_text(
    pool: AsyncConnectionPool, embedder: Embedder, settings: Settings,
    *, tenant_id: int, title: str, text: str, source: str, kind: str = "upload",
    created_by: int | None = None, cache: RetrievalCache | None = None,
) -> tuple[int, int]:
    """Returns (document_id, number_of_chunks)."""
    chunks = make_chunks(settings, text)
    embeddings = await embed_chunks(embedder, settings, chunks)
    doc_id = await store.insert_document(
        pool, tenant_id=tenant_id, title=title, source=source, kind=kind, chunks=chunks,
        embeddings=embeddings, created_by=created_by,
    )
    if cache is not None:
        await cache.invalidate(tenant_id)
    return doc_id, len(chunks)


async def ingest_file(
    pool: AsyncConnectionPool, embedder: Embedder, settings: Settings,
    *, tenant_id: int, filename: str, data: bytes, created_by: int | None = None,
    cache: RetrievalCache | None = None,
) -> tuple[int, int]:
    # Parsing a large PDF takes seconds of CPU; off the event loop, other requests keep flowing.
    parsed = await asyncio.to_thread(parse_document, filename, data)
    return await ingest_text(
        pool, embedder, settings, tenant_id=tenant_id, title=parsed.title, text=parsed.text,
        source=filename, created_by=created_by, cache=cache,
    )
