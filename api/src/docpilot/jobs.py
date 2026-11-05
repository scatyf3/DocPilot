"""Background ingest jobs: create, enqueue, run, report progress, recover.

The flow: POST /documents stores the file in ingest_jobs and enqueues the job id in Redis (arq),
then returns 202. A worker claims the job, parses, chunks and embeds the file, reporting
progress after each embedding batch, and stores the document. The client polls GET /jobs/{id}.

Guarantees:
- At most one document per job. The document insert and the job's move to `succeeded` commit
  in one transaction, after locking the job row, so a duplicate delivery (a retry racing a
  slow first attempt) finds the job already done and stores nothing.
- No lost jobs. Postgres holds the job; the Redis message is only a wake-up. If enqueueing
  fails, or Redis loses its data, the sweeper re-enqueues jobs that have sat queued too long.
- Bad input fails fast. A file that can't be parsed fails on the first attempt with a message
  for the user; only errors that might pass (the embedding API down) are retried, with backoff.
"""

import asyncio
import logging
import uuid

from psycopg_pool import AsyncConnectionPool

from docpilot import store
from docpilot.cache import RetrievalCache
from docpilot.config import Settings
from docpilot.db import tenant_connection
from docpilot.embeddings import Embedder
from docpilot.ingest.parse import parse_document
from docpilot.ingest.pipeline import embed_chunks, make_chunks

log = logging.getLogger(__name__)

TASK = "ingest_document"
_PUBLIC = """id, filename, status, stage, chunks_done, chunks_total, attempts, error, document_id,
             created_at, updated_at, finished_at"""
STALE_QUEUED_SECONDS = 120


class TransientIngestError(Exception):
    """This attempt failed in a way a later one might not. The worker retries it."""


def arq_job_id(job_id: uuid.UUID | str) -> str:
    # arq refuses a second job with the same id while the first is queued or running,
    # so enqueueing twice (upload, then sweeper) never runs the job twice at once.
    return f"ingest:{job_id}"


async def create_job(pool: AsyncConnectionPool, *, tenant_id: int, created_by: int | None,
                     filename: str, data: bytes) -> uuid.UUID:
    job_id = uuid.uuid4()
    async with tenant_connection(pool, tenant_id) as conn:
        await conn.execute(
            """INSERT INTO ingest_jobs (id, tenant_id, created_by, filename, payload)
               VALUES (%s, %s, %s, %s, %s)""",
            (job_id, tenant_id, created_by, filename, data),
        )
    return job_id


async def enqueue(arq, job_id: uuid.UUID, tenant_id: int) -> None:
    await arq.enqueue_job(TASK, str(job_id), tenant_id, _job_id=arq_job_id(job_id))


async def get_job(pool: AsyncConnectionPool, tenant_id: int, job_id: uuid.UUID) -> dict | None:
    async with tenant_connection(pool, tenant_id) as conn:
        return await (await conn.execute(
            f"SELECT {_PUBLIC} FROM ingest_jobs WHERE id = %s AND tenant_id = %s", (job_id, tenant_id),
        )).fetchone()


async def list_jobs(pool: AsyncConnectionPool, tenant_id: int, limit: int = 50) -> list[dict]:
    async with tenant_connection(pool, tenant_id) as conn:
        return await (await conn.execute(
            f"SELECT {_PUBLIC} FROM ingest_jobs WHERE tenant_id = %s ORDER BY created_at DESC LIMIT %s",
            (tenant_id, limit),
        )).fetchall()


async def _update(pool, tenant_id: int, job_id: uuid.UUID, **fields) -> None:
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    async with tenant_connection(pool, tenant_id) as conn:
        await conn.execute(
            f"UPDATE ingest_jobs SET {sets}, updated_at = now() WHERE id = %(id)s AND status = 'running'",
            {**fields, "id": job_id},
        )


async def run_ingest_job(
    pool: AsyncConnectionPool, embedder: Embedder, settings: Settings, cache: RetrievalCache | None,
    *, tenant_id: int, job_id: uuid.UUID, attempt: int = 1,
) -> dict:
    """One attempt at a job. Raises TransientIngestError if it should be retried."""
    async with tenant_connection(pool, tenant_id) as conn:
        job = await (await conn.execute(
            """UPDATE ingest_jobs SET status = 'running', stage = 'parsing', attempts = attempts + 1,
                      error = NULL, updated_at = now()
               WHERE id = %s AND tenant_id = %s AND status IN ('queued', 'running')
               RETURNING filename, payload, created_by""",
            (job_id, tenant_id),
        )).fetchone()
    if job is None:  # finished already, or never existed: nothing to do
        return {"skipped": True}

    try:
        parsed = await asyncio.to_thread(parse_document, job["filename"], job["payload"])
        chunks = make_chunks(settings, parsed.text)
    except ValueError as e:  # the file itself is the problem: retrying can't help
        await _finish(pool, tenant_id, job_id, "failed", error=str(e))
        return {"failed": str(e)}

    try:
        await _update(pool, tenant_id, job_id, stage="embedding", chunks_done=0, chunks_total=len(chunks))

        async def progress(done: int, total: int) -> None:
            await _update(pool, tenant_id, job_id, chunks_done=done)

        embeddings = await embed_chunks(embedder, settings, chunks, progress)
        await _update(pool, tenant_id, job_id, stage="storing")
        async with tenant_connection(pool, tenant_id) as conn, conn.transaction():
            status = await (await conn.execute(
                "SELECT status FROM ingest_jobs WHERE id = %s FOR UPDATE", (job_id,))).fetchone()
            if status is None or status["status"] != "running":
                return {"skipped": True}  # a concurrent attempt got there first
            doc_id = await store.insert_document_in(
                conn, tenant_id=tenant_id, title=parsed.title, source=job["filename"], kind="upload",
                chunks=chunks, embeddings=embeddings, created_by=job["created_by"])
            await conn.execute(
                """UPDATE ingest_jobs SET status = 'succeeded', stage = NULL, document_id = %s,
                          chunks_done = chunks_total, payload = NULL, finished_at = now(), updated_at = now()
                   WHERE id = %s""",
                (doc_id, job_id),
            )
    except Exception as e:
        if attempt >= settings.ingest_job_max_tries:
            message = f"ingest failed after {attempt} attempts ({type(e).__name__})"
            await _finish(pool, tenant_id, job_id, "failed", error=message)
            log.exception("ingest job %s failed for good", job_id)
            return {"failed": message}
        log.warning("ingest job %s attempt %d failed, will retry: %s", job_id, attempt, type(e).__name__)
        await _update(pool, tenant_id, job_id,
                      error=f"attempt {attempt} failed ({type(e).__name__}); retrying")
        raise TransientIngestError(type(e).__name__) from e

    if cache is not None:
        await cache.invalidate(tenant_id)
    return {"document_id": doc_id, "chunks": len(chunks)}


async def _finish(pool, tenant_id: int, job_id: uuid.UUID, status: str, *, error: str) -> None:
    async with tenant_connection(pool, tenant_id) as conn:
        await conn.execute(
            """UPDATE ingest_jobs SET status = %s, stage = NULL, error = %s, payload = NULL,
                      finished_at = now(), updated_at = now()
               WHERE id = %s AND status IN ('queued', 'running')""",
            (status, error, job_id),
        )


async def sweep(owner_pool: AsyncConnectionPool, arq, settings: Settings) -> dict:
    """Recover jobs the queue lost, and fail jobs whose worker vanished. Spans all tenants.

    Runs on a pool that bypasses row-level security (the owner), so it reads only ids and
    timestamps and never touches payloads or documents.
    """
    stalled_after = settings.ingest_job_timeout_seconds + 60
    async with owner_pool.connection() as conn:
        queued = await (await conn.execute(
            """SELECT id, tenant_id FROM ingest_jobs
               WHERE status = 'queued' AND updated_at < now() - make_interval(secs => %s)
               ORDER BY updated_at LIMIT 100""",
            (STALE_QUEUED_SECONDS,),
        )).fetchall()
        # A job that outlived its timeout without reporting progress: the worker died or was
        # killed mid-attempt. Retrying is arq's job; this only stops it showing "running" forever.
        stalled = await (await conn.execute(
            """UPDATE ingest_jobs SET status = 'failed', stage = NULL, payload = NULL, finished_at = now(),
                      updated_at = now(), error = 'the worker stopped responding; please upload again'
               WHERE status = 'running' AND updated_at < now() - make_interval(secs => %s)
               RETURNING id""",
            (stalled_after,),
        )).fetchall()
        # Touch the re-enqueued ones so the next sweep doesn't enqueue them again at once.
        if queued:
            await conn.execute("UPDATE ingest_jobs SET updated_at = now() WHERE id = ANY(%s)",
                               ([r["id"] for r in queued],))
    for row in queued:
        await enqueue(arq, row["id"], row["tenant_id"])
    if queued or stalled:
        log.warning("ingest sweeper: re-enqueued %d, failed %d stalled", len(queued), len(stalled))
    return {"requeued": len(queued), "failed": len(stalled)}
