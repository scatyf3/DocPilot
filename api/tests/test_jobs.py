"""Background ingest: the job lifecycle against Postgres, then a real arq worker on Redis.

Needs DATABASE_URL; the worker tests also need REDIS_URL (a database number of their own).
"""

import asyncio
import os
import uuid

import httpx
import pytest

from docpilot import jobs
from docpilot.config import Settings

pytestmark = [pytest.mark.db,
              pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs DATABASE_URL")]
REDIS_URL = os.environ.get("REDIS_URL")
needs_redis = pytest.mark.skipif(not REDIS_URL, reason="needs REDIS_URL")

DOC = b"# Handbook\n\n" + b"Employees get 25 days of paid leave each year. " * 400


def settings(**overrides) -> Settings:
    return Settings(**{"db_schema": "docpilot_test_jobs", "embedding_provider": "fake",
                       "llm_provider": "fake", "embedding_batch_size": 2, "ingest_job_max_tries": 3,
                       "rate_limit_enabled": False, **overrides})


@pytest.fixture
async def pool():
    from docpilot.db import init_schema, open_pool

    s = settings()
    await init_schema(s, reset=True)
    pool = await open_pool(s)
    yield pool
    await pool.close()


async def job_row(pool, job_id) -> dict:
    return await jobs.get_job(pool, 1, job_id)


async def run(pool, embedder, job_id, attempt=1, **overrides):
    return await jobs.run_ingest_job(pool, embedder, settings(**overrides), None,
                                     tenant_id=1, job_id=job_id, attempt=attempt)


class Embedder:
    """The fake embedder, with a record of progress and an optional failure."""

    name = "test"

    def __init__(self, fail_times=0):
        from docpilot.embeddings import FakeEmbedder

        self.inner, self.fail_times, self.calls = FakeEmbedder(1024), fail_times, 0

    async def embed(self, texts):
        self.calls += 1
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("embedding API down")
        return await self.inner.embed(texts)


async def test_job_runs_once_and_reports_progress(pool):
    job_id = await jobs.create_job(pool, tenant_id=1, created_by=None, filename="h.md", data=DOC)
    assert (await job_row(pool, job_id))["status"] == "queued"

    seen = []
    real_update = jobs._update

    async def spy(pool_, tenant_id, job_id_, **fields):
        seen.append(fields)
        await real_update(pool_, tenant_id, job_id_, **fields)

    jobs._update, embedder = spy, Embedder()
    try:
        result = await run(pool, embedder, job_id)
    finally:
        jobs._update = real_update
    job = await job_row(pool, job_id)
    assert job["status"] == "succeeded" and job["document_id"] == result["document_id"]
    assert job["chunks_done"] == job["chunks_total"] == result["chunks"] > 2
    # Progress moved forward one embedding batch at a time.
    done = [f["chunks_done"] for f in seen if "chunks_done" in f and f.get("stage") != "embedding"]
    assert done == sorted(done) and done[-1] == job["chunks_total"] and len(done) == embedder.calls

    # A duplicate delivery does nothing: still one document, payload gone.
    assert await run(pool, Embedder(), job_id) == {"skipped": True}
    async with pool.connection() as conn:
        await conn.execute("SELECT set_config('app.tenant_id', '1', true)")
        n_docs = (await (await conn.execute("SELECT count(*) AS n FROM documents")).fetchone())["n"]
        payload = (await (await conn.execute("SELECT payload FROM ingest_jobs WHERE id = %s",
                                             (job_id,))).fetchone())["payload"]
    assert n_docs == 1 and payload is None


async def test_concurrent_attempts_store_one_document(pool):
    job_id = await jobs.create_job(pool, tenant_id=1, created_by=None, filename="h.md", data=DOC)
    results = await asyncio.gather(*[run(pool, Embedder(), job_id) for _ in range(3)])
    assert sum("document_id" in r for r in results) == 1
    assert len([d for d in await _documents(pool)]) == 1


async def _documents(pool):
    from docpilot import store
    return await store.list_documents(pool, 1)


async def test_unreadable_file_fails_at_once_with_a_reason(pool):
    job_id = await jobs.create_job(pool, tenant_id=1, created_by=None, filename="x.pdf", data=b"not a pdf")
    assert (await run(pool, Embedder(), job_id))["failed"].startswith("could not read PDF")
    job = await job_row(pool, job_id)
    assert job["status"] == "failed" and job["attempts"] == 1 and "could not read PDF" in job["error"]


async def test_transient_failures_retry_then_give_up(pool):
    job_id = await jobs.create_job(pool, tenant_id=1, created_by=None, filename="h.md", data=DOC)
    flaky = Embedder(fail_times=1)
    with pytest.raises(jobs.TransientIngestError):
        await run(pool, flaky, job_id, attempt=1)
    job = await job_row(pool, job_id)
    assert job["status"] == "running" and "retrying" in job["error"]
    assert "document_id" in await run(pool, flaky, job_id, attempt=2)  # recovered
    assert (await job_row(pool, job_id))["error"] is None

    doomed = await jobs.create_job(pool, tenant_id=1, created_by=None, filename="h.md", data=DOC)
    result = await run(pool, Embedder(fail_times=99), doomed, attempt=3)  # the last try
    assert "after 3 attempts" in result["failed"]
    assert (await job_row(pool, doomed))["status"] == "failed"


async def test_jobs_are_tenant_scoped(pool):
    async with pool.connection() as conn:
        cur = await conn.execute("INSERT INTO tenants (name) VALUES ('o') RETURNING id")
        other = (await cur.fetchone())["id"]
    job_id = await jobs.create_job(pool, tenant_id=other, created_by=None, filename="h.md", data=DOC)
    assert await jobs.get_job(pool, 1, job_id) is None
    assert await jobs.list_jobs(pool, 1) == []
    # A worker handed a forged tenant id can't run (or read) the job either.
    assert await jobs.run_ingest_job(pool, Embedder(), settings(), None, tenant_id=1,
                                     job_id=job_id) == {"skipped": True}
    assert (await jobs.get_job(pool, other, job_id))["status"] == "queued"


# --- the real thing: API in queue mode, arq worker in burst mode ---------------------------------


@pytest.fixture
async def queued_app():
    import redis.asyncio as aioredis

    from docpilot.app import create_app
    from docpilot.db import init_schema

    flush = aioredis.from_url(REDIS_URL)
    await flush.flushdb()
    s = settings(ingest_mode="queue", redis_url=REDIS_URL)
    await init_schema(s, reset=True)
    app = create_app(s)
    async with app.router.lifespan_context(app):
        yield app
    await flush.flushdb()
    await flush.aclose()


async def drain(s: Settings) -> None:
    """Run a worker until the queue is empty, as `docpilot worker` would, then stop."""
    from arq.worker import Worker

    from docpilot.worker import worker_settings

    kwargs = worker_settings(s)
    kwargs["cron_jobs"] = []  # the sweeper is tested on its own
    worker = Worker(**kwargs, burst=True, handle_signals=False, poll_delay=0.05)
    try:
        await worker.main()
    finally:
        await worker.close()


@needs_redis
async def test_upload_returns_a_job_the_worker_completes(queued_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=queued_app), base_url="http://t") as c:
        resp = await c.post("/documents", files={"file": ("h.md", DOC)})
        assert resp.status_code == 202 and resp.headers["location"] == f"/jobs/{resp.json()['job_id']}"
        job_id = resp.json()["job_id"]
        job = (await c.get(f"/jobs/{job_id}")).json()
        assert job["status"] == "queued" and job["progress"] is None and "payload" not in job

        await drain(queued_app.state.settings)

        job = (await c.get(f"/jobs/{job_id}")).json()
        assert job["status"] == "succeeded" and job["progress"] == 1.0
        docs = (await c.get("/documents")).json()
        assert [d["id"] for d in docs] == [job["document_id"]]
        hits = (await c.post("/search", json={"query": "how many days of paid leave"})).json()
        assert hits and hits[0]["document_id"] == job["document_id"]
        assert [j["id"] for j in (await c.get("/jobs")).json()] == [job_id]
        assert (await c.get(f"/jobs/{uuid.uuid4()}")).status_code == 404


@needs_redis
async def test_sweeper_recovers_a_job_whose_message_was_lost(queued_app):
    from docpilot.db import open_pool

    s = queued_app.state.settings
    pool = queued_app.state.pool
    # As if Redis was down at upload time: the job exists only in Postgres.
    job_id = await jobs.create_job(pool, tenant_id=1, created_by=None, filename="h.md", data=DOC)
    await drain(s)
    assert (await jobs.get_job(pool, 1, job_id))["status"] == "queued"  # nobody knew about it

    owner = await open_pool(s, as_owner=True)
    try:
        async with owner.connection() as conn:
            await conn.execute("UPDATE ingest_jobs SET updated_at = now() - interval '10 minutes'")
        assert await jobs.sweep(owner, queued_app.state.arq, s) == {"requeued": 1, "failed": 0}
        assert await jobs.sweep(owner, queued_app.state.arq, s) == {"requeued": 0, "failed": 0}
        await drain(s)
        assert (await jobs.get_job(pool, 1, job_id))["status"] == "succeeded"

        # A job left "running" by a worker that died is eventually marked failed, not stuck.
        stuck = await jobs.create_job(pool, tenant_id=1, created_by=None, filename="h.md", data=DOC)
        async with owner.connection() as conn:
            await conn.execute("""UPDATE ingest_jobs SET status = 'running',
                                         updated_at = now() - interval '2 hours'
                                  WHERE id = %s""", (stuck,))
        assert (await jobs.sweep(owner, queued_app.state.arq, s))["failed"] == 1
        assert (await jobs.get_job(pool, 1, stuck))["status"] == "failed"
    finally:
        await owner.close()
