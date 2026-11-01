"""Raw-SQL data access for documents, conversations, messages and traces.

Retrieval queries live in `retrieval/search.py`; everything else that touches tables is here.
"""

import json
import uuid
from typing import Any

import numpy as np
from psycopg_pool import AsyncConnectionPool


async def insert_document(
    pool: AsyncConnectionPool, *, title: str, source: str, kind: str,
    chunks: list[str], embeddings: list[np.ndarray],
) -> int:
    async with pool.connection() as conn, conn.transaction():
        row = await (await conn.execute(
            "INSERT INTO documents (title, source, kind) VALUES (%s, %s, %s) RETURNING id",
            (title, source, kind),
        )).fetchone()
        doc_id = row["id"]
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO chunks (document_id, ord, text, embedding) VALUES (%s, %s, %s, %s)",
                [(doc_id, i, text, emb)
                 for i, (text, emb) in enumerate(zip(chunks, embeddings, strict=True))],
            )
    return doc_id


async def list_documents(pool: AsyncConnectionPool) -> list[dict]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """SELECT d.id, d.title, d.source, d.kind, d.created_at, count(c.id) AS chunk_count
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               GROUP BY d.id ORDER BY d.created_at DESC"""
        )
        return await cur.fetchall()


async def get_document(pool: AsyncConnectionPool, doc_id: int) -> dict | None:
    async with pool.connection() as conn:
        doc = await (await conn.execute(
            "SELECT id, title, source, kind, created_at FROM documents WHERE id = %s", (doc_id,)
        )).fetchone()
        if doc is None:
            return None
        cur = await conn.execute(
            "SELECT id, ord, text FROM chunks WHERE document_id = %s ORDER BY ord", (doc_id,)
        )
        doc["chunks"] = await cur.fetchall()
        return doc


async def delete_document(pool: AsyncConnectionPool, doc_id: int) -> bool:
    async with pool.connection() as conn:
        cur = await conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        return cur.rowcount > 0


async def get_chunks(pool: AsyncConnectionPool, chunk_ids: list[int]) -> list[dict]:
    """Chunks with their document title, in the order of `chunk_ids`."""
    if not chunk_ids:
        return []
    async with pool.connection() as conn:
        cur = await conn.execute(
            """SELECT c.id, c.document_id, c.ord, c.text, d.title
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.id = ANY(%s)""",
            (chunk_ids,),
        )
        by_id = {r["id"]: r for r in await cur.fetchall()}
    return [by_id[i] for i in chunk_ids if i in by_id]


async def conversation_history(pool: AsyncConnectionPool, conversation_id: str) -> list[dict]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """SELECT id, role, content, created_at FROM messages
               WHERE conversation_id = %s ORDER BY created_at""",
            (conversation_id,),
        )
        return await cur.fetchall()


async def retrieved_chunk_ids(pool: AsyncConnectionPool, conversation_id: str) -> list[int]:
    """Every chunk a tool has returned so far in this conversation."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            """SELECT DISTINCT unnest(t.retrieved_chunk_ids) AS id
               FROM traces t JOIN messages m ON m.id = t.message_id
               WHERE m.conversation_id = %s""",
            (conversation_id,),
        )
        return [r["id"] for r in await cur.fetchall()]


async def save_turn(
    pool: AsyncConnectionPool, *, conversation_id: str, user_text: str,
    assistant_id: uuid.UUID, assistant_text: str, trace: dict[str, Any],
) -> None:
    """Persist one user/assistant exchange and its trace atomically."""
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            "INSERT INTO conversations (id) VALUES (%s) ON CONFLICT DO NOTHING", (conversation_id,)
        )
        await conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content) VALUES (%s, %s, 'user', %s)",
            (uuid.uuid4(), conversation_id, user_text),
        )
        # clock_timestamp() (not now()) so the assistant row sorts after the user row.
        await conn.execute(
            """INSERT INTO messages (id, conversation_id, role, content, created_at)
               VALUES (%s, %s, 'assistant', %s, clock_timestamp())""",
            (assistant_id, conversation_id, assistant_text),
        )
        await conn.execute(
            """INSERT INTO traces (message_id, retrieved_chunk_ids, rejected_citation_ids, tool_calls,
                                   model, input_tokens, output_tokens, latency_ms, cost_usd)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (assistant_id, trace["retrieved_chunk_ids"], trace.get("rejected_citation_ids", []),
             json.dumps(trace["tool_calls"]),
             trace["model"], trace["input_tokens"], trace["output_tokens"],
             trace["latency_ms"], trace["cost_usd"]),
        )


async def get_trace(pool: AsyncConnectionPool, message_id: uuid.UUID) -> dict | None:
    async with pool.connection() as conn:
        trace = await (await conn.execute(
            """SELECT message_id, retrieved_chunk_ids, rejected_citation_ids, tool_calls, model, input_tokens,
                      output_tokens, latency_ms, cost_usd::float AS cost_usd, created_at
               FROM traces WHERE message_id = %s""",
            (message_id,),
        )).fetchone()
    if trace is None:
        return None
    trace["retrieved_chunks"] = await get_chunks(pool, list(trace["retrieved_chunk_ids"]))
    return trace
