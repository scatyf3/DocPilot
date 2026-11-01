"""End-to-end through the real API and Postgres+pgvector. Skipped unless DATABASE_URL is set.

Uses the fake embedder and fake LLM, in an isolated schema that is reset for the run.
"""

import asyncio
import json
import os

import httpx
import pytest

from docpilot.config import Settings
from docpilot.llm.base import TextDelta, ToolCall, TurnResult

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs DATABASE_URL"),
]


@pytest.fixture
async def client():
    from docpilot.app import create_app
    from docpilot.db import init_schema

    settings = Settings(db_schema="docpilot_test", embedding_provider="fake", llm_provider="fake")
    await init_schema(settings, reset=True)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


DOC = b"""# Benefits

## PTO

Full-time employees receive 25 days of paid time off per calendar year.

## Parental leave

Parents get 16 weeks of fully paid parental leave.
"""


async def test_upload_search_chat_trace(client: httpx.AsyncClient):
    resp = await client.post("/documents", files={"file": ("benefits.md", DOC)})
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    docs = (await client.get("/documents")).json()
    assert docs[0]["title"] == "Benefits" and docs[0]["chunk_count"] >= 1
    assert (await client.get(f"/documents/{doc_id}")).json()["chunks"][0]["ord"] == 0

    for mode in ("vector", "fts", "hybrid"):
        hits = (await client.post("/search", json={"query": "paid time off days", "mode": mode})).json()
        assert hits and "25 days" in hits[0]["text"], mode

    resp = await client.post("/chat", json={
        "id": "conv-1",
        "messages": [{"role": "user",
                      "parts": [{"type": "text", "text": "How many days of paid time off?"}]}],
    })
    assert resp.headers["x-vercel-ai-ui-message-stream"] == "v1"
    chunks = [json.loads(e[6:]) for e in resp.text.split("\n\n") if e.startswith("data: {")]
    types = [c["type"] for c in chunks]
    assert types[0] == "start" and types[-1] == "finish" and resp.text.endswith("data: [DONE]\n\n")
    citations = next(c["data"] for c in chunks if c["type"] == "data-citations")
    assert citations and "25 days" in citations[0]["text"]

    message_id = chunks[0]["messageId"]
    trace = (await client.get(f"/traces/{message_id}")).json()
    assert trace["tool_calls"][0]["name"] == "search_docs"
    assert trace["retrieved_chunks"] and trace["latency_ms"] >= 0

    convo = (await client.get("/conversations/conv-1")).json()
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant"]

    assert (await client.delete(f"/documents/{doc_id}")).status_code == 204
    assert (await client.get(f"/documents/{doc_id}")).status_code == 404


class CitingProvider:
    """Searches once, then answers with whatever text the test gives it."""

    model = "scripted"

    def __init__(self, answer):
        self.answer = answer

    def start(self, history, user_text):
        return [{"role": "user", "content": user_text}]

    async def stream_turn(self, system, messages, tools, *, allow_tools=True):
        if messages[-1]["role"] == "user":
            call = ToolCall("c1", "search_docs", {"query": "parental leave", "top_k": 1})
            messages.append({"role": "assistant", "tool_calls": [call]})
            yield TurnResult("", [call], "tool_use", 1, 1)
            return
        yield TextDelta(self.answer)
        yield TurnResult(self.answer, [], "end", 1, 1)

    def add_tool_results(self, messages, results):
        messages.append({"role": "tool", "results": results})


class NoToolProvider(CitingProvider):
    async def stream_turn(self, system, messages, tools, *, allow_tools=True):
        yield TextDelta(self.answer)
        yield TurnResult(self.answer, [], "end", 1, 1)


async def test_citations_only_resolve_to_chunks_the_model_saw(client: httpx.AsyncClient):
    for name, body in [("a.md", b"# A\n\nParents get 16 weeks of parental leave."),
                       ("b.md", b"# B\n\nThe cafeteria opens at noon.")]:
        await client.post("/documents", files={"file": (name, body)})
    async def top_hit(query):
        return (await client.post("/search", json={"query": query, "top_k": 1, "mode": "fts"})).json()[0]

    seen, other = await top_hit("parental leave"), await top_hit("cafeteria")
    assert seen["id"] != other["id"]

    async def turn(conversation, answer):
        client._transport.app.state.llm = CitingProvider(answer)
        resp = await client.post("/chat", json={
            "id": conversation, "messages": [{"role": "user", "parts": [{"type": "text", "text": "q"}]}]})
        chunks = [json.loads(e[6:]) for e in resp.text.split("\n\n") if e.startswith("data: {")]
        return ([c["id"] for c in next(x["data"] for x in chunks if x["type"] == "data-citations")],
                next(x["data"] for x in chunks if x["type"] == "data-trace"))

    answer = f"16 weeks [c:{seen['id']}], noon [c:{other['id']}], x [c:99999999]."
    citations, trace = await turn("cite-1", answer)
    assert citations == [seen["id"]]
    assert trace["rejected_citation_ids"] == [other["id"], 99999999]
    stored = (await client.get(f"/traces/{trace['message_id']}")).json()
    assert stored["rejected_citation_ids"] == [other["id"], 99999999]

    # A follow-up may reuse an id it was shown earlier, without searching again.
    client._transport.app.state.llm = NoToolProvider(f"As I said [c:{seen['id']}], and [c:{other['id']}].")
    resp = await client.post("/chat", json={
        "id": "cite-1", "messages": [{"role": "user", "parts": [{"type": "text", "text": "and?"}]}]})
    chunks = [json.loads(e[6:]) for e in resp.text.split("\n\n") if e.startswith("data: {")]
    trace = next(x["data"] for x in chunks if x["type"] == "data-trace")
    # `other` was rejected in the first turn; being quoted in that stored answer doesn't launder it.
    assert [c["id"] for c in next(x["data"] for x in chunks if x["type"] == "data-citations")] == [seen["id"]]
    assert trace["rejected_citation_ids"] == [other["id"]]


async def test_migrations_are_tracked_and_idempotent():
    import psycopg

    from docpilot.db import _migrations, init_schema

    settings = Settings(db_schema="docpilot_test_migrations", embedding_provider="fake", llm_provider="fake")
    await init_schema(settings, reset=True)
    await asyncio.gather(init_schema(settings), init_schema(settings))  # e.g. two instances starting
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        cur = await conn.execute(
            "SELECT version FROM docpilot_test_migrations.schema_migrations ORDER BY version")
        assert [r[0] for r in await cur.fetchall()] == [v for v, _, _ in _migrations()]
        cur = await conn.execute(
            """SELECT 1 FROM information_schema.columns WHERE table_schema = 'docpilot_test_migrations'
               AND table_name = 'traces' AND column_name = 'rejected_citation_ids'""")
        assert await cur.fetchone()


async def test_failed_migration_rolls_back_and_stops(monkeypatch):
    import psycopg

    from docpilot import db

    settings = Settings(db_schema="docpilot_test_migrations", embedding_provider="fake", llm_provider="fake")
    await db.init_schema(settings, reset=True)
    good = db._migrations()
    bad = [*good, (9998, "9998_bad.sql", "CREATE TABLE half_done (id int); SELECT nonsense_column;"),
           (9999, "9999_never.sql", "CREATE TABLE never_run (id int);")]
    monkeypatch.setattr(db, "_migrations", lambda: bad)
    with pytest.raises(psycopg.Error):
        await db.init_schema(settings)
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        cur = await conn.execute(
            """SELECT table_name FROM information_schema.tables
               WHERE table_schema = 'docpilot_test_migrations'
               AND table_name IN ('half_done', 'never_run')""")
        assert await cur.fetchall() == []
        cur = await conn.execute("SELECT max(version) FROM docpilot_test_migrations.schema_migrations")
        assert (await cur.fetchone())[0] == good[-1][0]


async def test_runaway_queries_are_cancelled():
    import psycopg

    from docpilot.db import init_schema, open_pool

    settings = Settings(db_schema="docpilot_test", embedding_provider="fake", llm_provider="fake",
                        db_statement_timeout_ms=50)
    await init_schema(settings, reset=True)
    pool = await open_pool(settings)
    try:
        with pytest.raises(psycopg.errors.QueryCanceled):
            async with pool.connection() as conn:
                await conn.execute("SELECT pg_sleep(5)")
        async with pool.connection() as conn:  # search_path survives alongside the timeout
            assert (await (await conn.execute("SELECT count(*) AS n FROM documents")).fetchone())["n"] == 0
    finally:
        await pool.close()


async def test_full_text_ranks_by_rare_terms(client: httpx.AsyncClient):
    # Five documents share the common words; only one has the rare one.
    for i in range(5):
        body = f"# Doc {i}\n\nThe service team reviews the service policy for the service every week."
        await client.post("/documents", files={"file": (f"d{i}.md", body.encode())})
    await client.post("/documents", files={"file": ("k.md", b"# K\n\nKeystone rotates signing keys.")})
    hits = (await client.post("/search", json={
        "query": "what does the service team say about keystone", "mode": "fts", "top_k": 3})).json()
    assert "Keystone" in hits[0]["text"]


@pytest.mark.parametrize("query", ["the", "???", "C++", "what's", "你好", "a:b & c | !d", "it's o'neil",
                                   "back\\slash", "'; drop table chunks;--", "x" * 5000])
async def test_full_text_survives_awkward_queries(client: httpx.AsyncClient, query):
    await client.post("/documents", files={"file": ("a.md", b"# A\n\nback\\slash o'neil words here")})
    for mode in ("fts", "hybrid"):
        assert (await client.post("/search", json={"query": query, "mode": mode})).status_code == 200
