"""Raw-SQL data access for documents, conversations, messages and traces.

Retrieval queries live in `retrieval/search.py`; everything else that touches tables is here.

Every function takes the caller's tenant and does two things with it: opens a transaction that
row-level security scopes to that tenant (db.tenant_connection), and filters by it in the SQL.
Either alone would keep tenants apart. Both together survive one of them being wrong.
"""

import json
import uuid
from typing import Any

import numpy as np
import psycopg
from psycopg_pool import AsyncConnectionPool

from docpilot.db import tenant_connection


async def insert_document(
    pool: AsyncConnectionPool, *, tenant_id: int, title: str, source: str, kind: str,
    chunks: list[str], embeddings: list[np.ndarray], created_by: int | None = None,
) -> int:
    async with tenant_connection(pool, tenant_id) as conn, conn.transaction():
        return await insert_document_in(conn, tenant_id=tenant_id, title=title, source=source,
                                        kind=kind, chunks=chunks, embeddings=embeddings,
                                        created_by=created_by)


async def insert_document_in(
    conn: psycopg.AsyncConnection, *, tenant_id: int, title: str, source: str, kind: str,
    chunks: list[str], embeddings: list[np.ndarray], created_by: int | None = None,
) -> int:
    """insert_document on a connection the caller holds, to commit it with other writes."""
    row = await (await conn.execute(
        """INSERT INTO documents (tenant_id, title, source, kind, created_by)
           VALUES (%s, %s, %s, %s, %s) RETURNING id""",
        (tenant_id, title, source, kind, created_by),
    )).fetchone()
    doc_id = row["id"]
    async with conn.cursor() as cur:
        await cur.executemany(
            """INSERT INTO chunks (tenant_id, document_id, ord, text, embedding)
               VALUES (%s, %s, %s, %s, %s)""",
            [(tenant_id, doc_id, i, text, emb)
             for i, (text, emb) in enumerate(zip(chunks, embeddings, strict=True))],
        )
    return doc_id


async def list_documents(pool: AsyncConnectionPool, tenant_id: int) -> list[dict]:
    async with tenant_connection(pool, tenant_id) as conn:
        cur = await conn.execute(
            """SELECT d.id, d.title, d.source, d.kind, d.created_at, count(c.id) AS chunk_count
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id AND c.tenant_id = d.tenant_id
               WHERE d.tenant_id = %s
               GROUP BY d.id ORDER BY d.created_at DESC""",
            (tenant_id,),
        )
        return await cur.fetchall()


async def get_document(pool: AsyncConnectionPool, tenant_id: int, doc_id: int) -> dict | None:
    async with tenant_connection(pool, tenant_id) as conn:
        doc = await (await conn.execute(
            """SELECT id, title, source, kind, created_at FROM documents
               WHERE id = %s AND tenant_id = %s""",
            (doc_id, tenant_id),
        )).fetchone()
        if doc is None:
            return None
        cur = await conn.execute(
            "SELECT id, ord, text FROM chunks WHERE document_id = %s AND tenant_id = %s ORDER BY ord",
            (doc_id, tenant_id),
        )
        doc["chunks"] = await cur.fetchall()
        return doc


async def delete_document(pool: AsyncConnectionPool, tenant_id: int, doc_id: int) -> bool:
    async with tenant_connection(pool, tenant_id) as conn:
        cur = await conn.execute("DELETE FROM documents WHERE id = %s AND tenant_id = %s",
                                 (doc_id, tenant_id))
        return cur.rowcount > 0


async def get_chunks(pool: AsyncConnectionPool, tenant_id: int, chunk_ids: list[int]) -> list[dict]:
    """Chunks with their document title, in the order of `chunk_ids`. Others' ids are skipped."""
    if not chunk_ids:
        return []
    async with tenant_connection(pool, tenant_id) as conn:
        cur = await conn.execute(
            """SELECT c.id, c.document_id, c.ord, c.text, d.title
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.id = ANY(%s) AND c.tenant_id = %s""",
            (chunk_ids, tenant_id),
        )
        by_id = {r["id"]: r for r in await cur.fetchall()}
    return [by_id[i] for i in chunk_ids if i in by_id]


async def claim_conversation(
    pool: AsyncConnectionPool, tenant_id: int, user_id: int | None, conversation_id: str,
) -> bool:
    """Create the conversation for this caller, or confirm they already own it.

    False when a colleague in the same tenant owns that id. Conversation ids are only unique
    per tenant, so another tenant's ids never come into it.
    """
    async with tenant_connection(pool, tenant_id) as conn:
        await conn.execute(
            """INSERT INTO conversations (tenant_id, id, user_id) VALUES (%s, %s, %s)
               ON CONFLICT (tenant_id, id) DO NOTHING""",
            (tenant_id, conversation_id, user_id),
        )
        row = await (await conn.execute(
            "SELECT user_id FROM conversations WHERE tenant_id = %s AND id = %s",
            (tenant_id, conversation_id),
        )).fetchone()
    return row is not None and row["user_id"] == user_id


async def conversation_owner(
    pool: AsyncConnectionPool, tenant_id: int, conversation_id: str,
) -> tuple[bool, int | None]:
    """(exists, user_id) for a conversation id within the tenant."""
    async with tenant_connection(pool, tenant_id) as conn:
        row = await (await conn.execute(
            "SELECT user_id FROM conversations WHERE tenant_id = %s AND id = %s",
            (tenant_id, conversation_id),
        )).fetchone()
    return (row is not None, row["user_id"] if row else None)


async def conversation_history(
    pool: AsyncConnectionPool, tenant_id: int, conversation_id: str,
) -> list[dict]:
    async with tenant_connection(pool, tenant_id) as conn:
        cur = await conn.execute(
            """SELECT id, role, content, created_at FROM messages
               WHERE tenant_id = %s AND conversation_id = %s ORDER BY created_at""",
            (tenant_id, conversation_id),
        )
        return await cur.fetchall()


async def retrieved_chunk_ids(
    pool: AsyncConnectionPool, tenant_id: int, conversation_id: str,
) -> list[int]:
    """Every chunk a tool has returned so far in this conversation."""
    async with tenant_connection(pool, tenant_id) as conn:
        cur = await conn.execute(
            """SELECT DISTINCT unnest(t.retrieved_chunk_ids) AS id
               FROM traces t JOIN messages m ON m.id = t.message_id
               WHERE m.tenant_id = %s AND m.conversation_id = %s""",
            (tenant_id, conversation_id),
        )
        return [r["id"] for r in await cur.fetchall()]


async def save_turn(
    pool: AsyncConnectionPool, *, tenant_id: int, conversation_id: str, user_text: str,
    assistant_id: uuid.UUID, assistant_text: str, trace: dict[str, Any],
) -> None:
    """Persist one user/assistant exchange and its trace atomically.

    The conversation row must exist already: the chat route claims it before the turn starts.
    """
    async with tenant_connection(pool, tenant_id) as conn, conn.transaction():
        await conn.execute(
            """INSERT INTO messages (id, tenant_id, conversation_id, role, content)
               VALUES (%s, %s, %s, 'user', %s)""",
            (uuid.uuid4(), tenant_id, conversation_id, user_text),
        )
        # clock_timestamp() (not now()) so the assistant row sorts after the user row.
        await conn.execute(
            """INSERT INTO messages (id, tenant_id, conversation_id, role, content, created_at)
               VALUES (%s, %s, %s, 'assistant', %s, clock_timestamp())""",
            (assistant_id, tenant_id, conversation_id, assistant_text),
        )
        await conn.execute(
            """INSERT INTO traces (tenant_id, message_id, retrieved_chunk_ids, rejected_citation_ids,
                                   tool_calls, model, input_tokens, output_tokens, latency_ms, cost_usd)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (tenant_id, assistant_id, trace["retrieved_chunk_ids"], trace.get("rejected_citation_ids", []),
             json.dumps(trace["tool_calls"]),
             trace["model"], trace["input_tokens"], trace["output_tokens"],
             trace["latency_ms"], trace["cost_usd"]),
        )


async def get_trace(
    pool: AsyncConnectionPool, tenant_id: int, message_id: uuid.UUID, user_id: int | None,
) -> dict | None:
    """A trace, if it belongs to a conversation this caller owns."""
    async with tenant_connection(pool, tenant_id) as conn:
        trace = await (await conn.execute(
            """SELECT t.message_id, t.retrieved_chunk_ids, t.rejected_citation_ids, t.tool_calls,
                      t.model, t.input_tokens, t.output_tokens, t.latency_ms,
                      t.cost_usd::float AS cost_usd, t.created_at
               FROM traces t
               JOIN messages m ON m.id = t.message_id AND m.tenant_id = t.tenant_id
               JOIN conversations c ON c.id = m.conversation_id AND c.tenant_id = m.tenant_id
               WHERE t.message_id = %s AND t.tenant_id = %s AND c.user_id IS NOT DISTINCT FROM %s""",
            (message_id, tenant_id, user_id),
        )).fetchone()
    if trace is None:
        return None
    trace["retrieved_chunks"] = await get_chunks(pool, tenant_id, list(trace["retrieved_chunk_ids"]))
    return trace
