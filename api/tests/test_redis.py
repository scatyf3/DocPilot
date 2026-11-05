"""Shared rate limits and retrieval caches in Redis. Skipped unless REDIS_URL is set.

Use a database number of its own (e.g. redis://localhost:6379/15): each test flushes it.
Cache tests also need DATABASE_URL.
"""

import asyncio
import os
import time
from types import SimpleNamespace as NS

import httpx
import pytest
import redis.asyncio as aioredis

from docpilot.cache import RetrievalCache
from docpilot.config import Settings
from docpilot.retrieval.search import RetrievedChunk, Retriever
from docpilot.security import RateLimiter, RedisRateLimiter

REDIS_URL = os.environ.get("REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="needs REDIS_URL")
needs_db = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs DATABASE_URL")


@pytest.fixture
async def redis():
    client = aioredis.from_url(REDIS_URL)
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


# --- rate limits ---------------------------------------------------------------------------


async def test_instances_share_one_bucket(redis):
    """The bug this fixes: with per-process buckets, N instances allowed N times the limit."""
    local = [RateLimiter(), RateLimiter()]
    clients = [aioredis.from_url(REDIS_URL) for _ in range(2)]
    shared = [RedisRateLimiter(c) for c in clients]

    async def admitted(limiters):
        return sum([await limiters[i % 2].acquire("search:ip:1.2.3.4", 5) == 0 for i in range(20)])

    assert await admitted(local) == 10  # 2 instances x 5
    assert await admitted(shared) == 5
    for c in clients:
        await c.aclose()


async def test_bucket_is_atomic_under_concurrency(redis):
    clients = [aioredis.from_url(REDIS_URL) for _ in range(4)]
    limiters = [RedisRateLimiter(c) for c in clients]
    waits = await asyncio.gather(*[limiters[i % 4].acquire("chat:user:1", 10) for i in range(60)])
    for c in clients:
        await c.aclose()
    assert sum(w == 0 for w in waits) == 10
    assert all(0 < w <= 6 for w in waits if w)  # 10/minute: one token every 6 s


async def test_bucket_refills_and_keys_expire(redis):
    limiter = RedisRateLimiter(redis)
    assert [await limiter.acquire("k", 60) for _ in range(60)] == [0.0] * 60
    wait = await limiter.acquire("k", 60)
    assert 0 < wait <= 1.0
    await asyncio.sleep(wait + 0.05)
    assert await limiter.acquire("k", 60) == 0.0
    assert 0 < await redis.pttl("docpilot:rl:k") <= 60_000  # idle clients cost nothing for long


async def test_redis_outage_falls_back_to_local_buckets():
    # Nothing listens on port 1. The limiter keeps limiting, locally, and stops retrying Redis
    # for a few seconds so an outage costs one timeout, not one per request.
    dead = aioredis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2, socket_timeout=0.2)
    limiter = RedisRateLimiter(dead)
    started = time.monotonic()
    waits = [await limiter.acquire("k", 3) for _ in range(5)]
    assert waits[:3] == [0.0] * 3 and all(w > 0 for w in waits[3:])
    assert time.monotonic() - started < 1.0  # one failed connect, not five
    await dead.aclose()


async def test_two_app_instances_enforce_one_limit(redis, monkeypatch):
    from docpilot.app import create_app

    class Pool:
        async def close(self):
            pass

    async def open_pool(settings):
        return Pool()

    async def search(query, **kw):
        return []

    async def schema_identity(pool, settings):
        return "stub"

    monkeypatch.setattr("docpilot.app.open_pool", open_pool)
    monkeypatch.setattr("docpilot.app.schema_identity", schema_identity)
    settings = Settings(llm_provider="fake", embedding_provider="fake", redis_url=REDIS_URL,
                        rate_limit_search_per_minute=3)
    apps = [create_app(settings), create_app(settings)]
    async with apps[0].router.lifespan_context(apps[0]), apps[1].router.lifespan_context(apps[1]):
        clients = []
        for app in apps:
            app.state.retriever = NS(search=search)
            clients.append(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t"))
        codes = [(await clients[i % 2].post("/search", json={"query": "x"})).status_code for i in range(6)]
        for c in clients:
            await c.aclose()
    assert codes == [200, 200, 200, 429, 429, 429]


# --- retrieval cache -----------------------------------------------------------------------


CHUNK = RetrievedChunk(id=1, document_id=1, ord=0, title="t", text="x", score=1.0)


class CountingRetriever(Retriever):
    """First-stage retrievers stubbed; counts how often each one actually runs."""

    def __init__(self, cache, vector=None, fts=None):
        super().__init__(None, NS(name="stub", embed=None), Settings(reranker="none"), cache)
        self.outcomes = {"vector": vector or [CHUNK], "fts": fts or [CHUNK]}
        self.calls = 0

    async def _run(self, name):
        self.calls += 1
        outcome = self.outcomes[name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def _vector(self, query, limit, tenant_id, document_ids=None):
        return await self._run("vector")

    async def _fts(self, query, limit, tenant_id, document_ids=None):
        return await self._run("fts")


def make_cache(redis, **kw):
    return RetrievalCache(redis, namespace="test", search_ttl=kw.get("search_ttl", 60),
                          embedding_ttl=kw.get("embedding_ttl", 60))


async def test_search_results_are_cached_per_tenant_until_a_write(redis):
    cache = make_cache(redis)
    r = CountingRetriever(cache)
    for _ in range(3):
        assert [c.id for c in await r.search("q", tenant_id=1, mode="fts")] == [1]
    assert r.calls == 1 and cache.hits == 2

    await r.search("q", tenant_id=2, mode="fts")  # another tenant: its own entry
    assert r.calls == 2
    await r.search("q", tenant_id=1, mode="fts", top_k=3)  # other parameters: other entry
    assert r.calls == 3

    await cache.invalidate(1)  # e.g. an upload committed
    await r.search("q", tenant_id=1, mode="fts")
    assert r.calls == 4
    await r.search("q", tenant_id=2, mode="fts")  # tenant 2 untouched
    assert r.calls == 4


async def test_degraded_results_are_not_cached(redis):
    cache = make_cache(redis)
    r = CountingRetriever(cache, vector=RuntimeError("embeddings down"))
    for _ in range(2):
        assert [c.id for c in await r.search("q", tenant_id=1, mode="hybrid")] == [1]
    assert cache.hits == 0  # served degraded both times, never stored
    r.outcomes["vector"] = [CHUNK]
    await r.search("q", tenant_id=1, mode="hybrid")
    await r.search("q", tenant_id=1, mode="hybrid")
    assert cache.hits == 1


async def test_cache_outage_is_a_miss_not_an_error():
    dead = aioredis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2, socket_timeout=0.2)
    r = CountingRetriever(make_cache(dead))
    assert [c.id for c in await r.search("q", tenant_id=1, mode="fts")] == [1]
    assert [c.id for c in await r.search("q", tenant_id=1, mode="fts")] == [1]
    await dead.aclose()


@needs_db
async def test_a_reset_schema_gets_a_fresh_cache_namespace():
    """Evals drop and recreate their schema; ids then name different rows, so old cached
    results must not be served, even before the first write bumps the generation."""
    from docpilot.db import init_schema, open_pool, schema_identity

    s = Settings(db_schema="docpilot_test_cache_ns")
    names = []
    for _ in range(2):
        await init_schema(s, reset=True)
        pool = await open_pool(s)
        names.append(await schema_identity(pool, s))
        await pool.close()
    assert names[0] != names[1] and all(n.startswith("docpilot_test_cache_ns.") for n in names)


@needs_db
async def test_cache_end_to_end_with_uploads_and_query_embeddings(redis):
    from docpilot.app import create_app
    from docpilot.db import init_schema

    settings = Settings(db_schema="docpilot_test_cache", embedding_provider="fake", llm_provider="fake",
                        redis_url=REDIS_URL)
    await init_schema(settings, reset=True)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        embedded = []
        real_embed = app.state.embedder.embed

        async def counting_embed(texts):
            embedded.extend(texts)
            return await real_embed(texts)

        app.state.embedder.embed = counting_embed
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            await c.post("/documents", files={"file": ("a.md", b"# A\n\nThe office opens at nine.")})
            first = (await c.post("/search", json={"query": "when does the office open"})).json()
            again = (await c.post("/search", json={"query": "when does the office open"})).json()
            assert first == again and app.state.cache.hits == 1

            # A new document must show up at once, not after the TTL.
            later = b"# B\n\nThe office opens at ten on Fridays."
            await c.post("/documents", files={"file": ("b.md", later)})
            after = (await c.post("/search", json={"query": "when does the office open", "top_k": 5})).json()
            assert len(after) == 2

            # The query was embedded once, though it was searched three times (vector side of
            # hybrid): the second search was a full hit, the third reused the embedding.
            assert embedded.count("when does the office open") == 1
