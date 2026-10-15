"""parse -> chunk -> embed -> store. Used by uploads, the CLI, evals and the create_page tool."""

from psycopg_pool import AsyncConnectionPool

from docpilot import store
from docpilot.config import Settings
from docpilot.embeddings import Embedder
from docpilot.ingest.chunk import chunk_text
from docpilot.ingest.parse import parse_document


async def ingest_text(
    pool: AsyncConnectionPool, embedder: Embedder, settings: Settings,
    *, title: str, text: str, source: str, kind: str = "upload",
) -> tuple[int, int]:
    """Returns (document_id, number_of_chunks)."""
    chunks = chunk_text(text, settings.chunk_tokens, settings.chunk_overlap_tokens)
    if not chunks:
        raise ValueError("document has no extractable text")
    embeddings = await embedder.embed(chunks)
    doc_id = await store.insert_document(
        pool, title=title, source=source, kind=kind, chunks=chunks, embeddings=embeddings
    )
    return doc_id, len(chunks)


async def ingest_file(
    pool: AsyncConnectionPool, embedder: Embedder, settings: Settings,
    *, filename: str, data: bytes,
) -> tuple[int, int]:
    parsed = parse_document(filename, data)
    return await ingest_text(
        pool, embedder, settings, title=parsed.title, text=parsed.text, source=filename
    )
