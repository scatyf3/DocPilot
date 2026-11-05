"""Tenant isolation, end to end: user B must never see anything of user A's.

Every path that can surface a chunk is tried: /search in every mode, document ids, /documents,
the agent's search_docs and read_doc tools, citations, conversations and traces. Then the
database alone is tested: row-level security has to hold even for a query with no tenant filter.

Skipped unless DATABASE_URL is set. Fake embedder and LLM; cheap argon2 parameters.
"""

import json
import os
import random
import re

import httpx
import psycopg
import pytest

from docpilot.config import Settings
from docpilot.llm.base import TextDelta, ToolCall, TurnResult

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs DATABASE_URL"),
]

SCHEMA = "docpilot_test_tenancy"
SECRET = "test-secret-" + "x" * 32


def settings(**overrides) -> Settings:
    return Settings(**{"db_schema": SCHEMA, "embedding_provider": "fake", "llm_provider": "fake",
                       "jwt_secrets": SECRET, "api_keys": "service-key", "argon2_time_cost": 1,
                       "argon2_memory_kib": 1024, "argon2_parallelism": 1,
                       "rate_limit_enabled": False, **overrides})


@pytest.fixture
async def app():
    from docpilot.app import create_app
    from docpilot.db import init_schema

    s = settings()
    await init_schema(s, reset=True)
    app = create_app(s)
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def signup(client, email) -> dict:
    resp = await client.post("/auth/register", json={"email": email, "password": "correct horse battery"})
    assert resp.status_code == 201, resp.text
    tokens = resp.json()
    return {"Authorization": f"Bearer {tokens['access_token']}", "_tokens": tokens}


def hdr(user: dict) -> dict:
    return {k: v for k, v in user.items() if not k.startswith("_")}


A_DOCS = {
    "orion.md": b"# Project Orion\n\nOrion launches on March 3. The codename for the vault is "
                b"bluefalcon and the budget is 4.2 million dollars.",
    "salaries.md": b"# Salaries\n\nThe principal engineer salary band tops out at 310000. "
                   b"Equity refreshes vest quarterly.",
}
B_DOCS = {
    "garden.md": b"# Garden club\n\nThe garden club meets on Tuesdays to plant tomatoes and basil.",
}


async def upload(client, user, docs) -> list[int]:
    ids = []
    for name, body in docs.items():
        resp = await client.post("/documents", files={"file": (name, body)}, headers=hdr(user))
        assert resp.status_code == 201, resp.text
        ids.append(resp.json()["id"])
    return ids


def stream_parts(resp) -> list[dict]:
    return [json.loads(e[6:]) for e in resp.text.split("\n\n") if e.startswith("data: {")]


async def a_chunk_ids(client, user, ids) -> set[int]:
    out = set()
    for doc_id in ids:
        doc = (await client.get(f"/documents/{doc_id}", headers=hdr(user))).json()
        out |= {c["id"] for c in doc["chunks"]}
    return out


async def test_users_get_separate_tenants(client):
    a, b = await signup(client, "a@example.com"), await signup(client, "b@example.com")
    me_a = (await client.get("/auth/me", headers=hdr(a))).json()
    me_b = (await client.get("/auth/me", headers=hdr(b))).json()
    assert me_a["email"] == "a@example.com" and me_a["tenant_id"] != me_b["tenant_id"]
    assert 1 not in (me_a["tenant_id"], me_b["tenant_id"])  # the API-key tenant is nobody's


async def test_library_routes_are_isolated(client):
    a, b = await signup(client, "a@example.com"), await signup(client, "b@example.com")
    a_ids = await upload(client, a, A_DOCS)
    b_ids = await upload(client, b, B_DOCS)

    assert {d["id"] for d in (await client.get("/documents", headers=hdr(b))).json()} == set(b_ids)
    for doc_id in a_ids:
        # The same 404 as a document that doesn't exist: ids can't be probed.
        assert (await client.get(f"/documents/{doc_id}", headers=hdr(b))).status_code == 404
        assert (await client.delete(f"/documents/{doc_id}", headers=hdr(b))).status_code == 404
        assert (await client.get(f"/documents/{doc_id}", headers=hdr(a))).status_code == 200

    # An API key acts on tenant 1, which holds neither user's library.
    assert (await client.get("/documents", headers={"X-API-Key": "service-key"})).json() == []


@pytest.mark.parametrize("mode", ["vector", "fts", "hybrid"])
async def test_search_never_returns_another_tenants_chunks(client, mode):
    a, b = await signup(client, "a@example.com"), await signup(client, "b@example.com")
    a_ids = await upload(client, a, A_DOCS)
    b_ids = await upload(client, b, B_DOCS)
    forbidden = await a_chunk_ids(client, a, a_ids)

    # Queries aimed squarely at A's text, then random words from it.
    words = re.findall(r"[a-z]+", b" ".join(A_DOCS.values()).decode().lower())
    rng = random.Random(0)
    queries = ["what is the vault codename bluefalcon", "principal engineer salary band",
               "Orion launch date and budget"]
    queries += [" ".join(rng.sample(words, 4)) for _ in range(25)]
    for query in queries:
        hits = (await client.post("/search", headers=hdr(b),
                                  json={"query": query, "mode": mode, "top_k": 50})).json()
        assert not {h["id"] for h in hits} & forbidden, query
        assert {h["document_id"] for h in hits} <= set(b_ids)
        # Naming A's documents explicitly doesn't help either.
        scoped = (await client.post("/search", headers=hdr(b), json={
            "query": query, "mode": mode, "document_ids": a_ids})).json()
        assert scoped == []

    # The control: A finds their own text.
    hits = (await client.post("/search", headers=hdr(a),
                              json={"query": "vault codename bluefalcon", "mode": mode})).json()
    assert hits and "bluefalcon" in hits[0]["text"]


class ScriptedProvider:
    """Calls the given tools in its first turn, then answers with the given text."""

    model = "scripted"

    def __init__(self, calls, answer):
        self.calls, self.answer = calls, answer

    def start(self, history, user_text):
        return [{"role": "user", "content": user_text}]

    async def stream_turn(self, system, messages, tools, *, allow_tools=True):
        if self.calls and messages[-1]["role"] == "user":
            messages.append({"role": "assistant", "tool_calls": self.calls})
            yield TurnResult("", self.calls, "tool_use", 1, 1)
            return
        yield TextDelta(self.answer)
        yield TurnResult(self.answer, [], "end", 1, 1)

    def add_tool_results(self, messages, results):
        messages.append({"role": "tool", "results": results})


async def test_agent_tools_and_citations_stay_in_tenant(app, client):
    a, b = await signup(client, "a@example.com"), await signup(client, "b@example.com")
    a_ids = await upload(client, a, A_DOCS)
    await upload(client, b, B_DOCS)
    forbidden = await a_chunk_ids(client, a, a_ids)
    stolen = min(forbidden)

    # A prompt-injected model goes looking for A's data by every means it has.
    app.state.llm = ScriptedProvider(
        [ToolCall("1", "search_docs", {"query": "vault codename bluefalcon", "top_k": 10}),
         ToolCall("2", "read_doc", {"document_id": a_ids[0]})],
        f"The codename is bluefalcon [c:{stolen}].")
    resp = await client.post("/chat", headers=hdr(b), json={
        "id": "shared-id", "messages": [{"role": "user", "parts": [{"type": "text", "text": "codename?"}]}]})
    parts = stream_parts(resp)
    trace = next(p["data"] for p in parts if p["type"] == "data-trace")
    citations = next(p["data"] for p in parts if p["type"] == "data-citations")
    assert not set(trace["retrieved_chunk_ids"]) & forbidden
    assert "error" in trace["tool_calls"][1]["summary"]  # read_doc: no such document
    assert citations == [] and trace["rejected_citation_ids"] == [stolen]
    # The model typed "bluefalcon" into its own query; nothing that came back contains it.
    outputs = [p for p in parts if p["type"].startswith("tool-output")]
    assert len(outputs) == 2 and "bluefalcon" not in json.dumps(outputs)

    # A uses the same client-side conversation id: a separate conversation, not B's.
    app.state.llm = ScriptedProvider([], "Hello A.")
    resp = await client.post("/chat", headers=hdr(a), json={
        "id": "shared-id", "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]})
    a_message = stream_parts(resp)[0]["messageId"]
    a_convo = (await client.get("/conversations/shared-id", headers=hdr(a))).json()
    b_convo = (await client.get("/conversations/shared-id", headers=hdr(b))).json()
    assert [m["content"] for m in a_convo["messages"]] == ["hi", "Hello A."]
    assert b_convo["messages"][0]["content"] == "codename?"
    assert (await client.get(f"/traces/{a_message}", headers=hdr(b))).status_code == 404
    assert (await client.get(f"/traces/{a_message}", headers=hdr(a))).status_code == 200


async def test_colleagues_share_the_library_but_not_conversations(app, client):
    a = await signup(client, "a@example.com")
    a_ids = await upload(client, a, A_DOCS)
    tenant = (await client.get("/auth/me", headers=hdr(a))).json()["tenant_id"]
    # No invitations yet: add a colleague to A's tenant directly.
    password_hash = await app.state.passwords.hash("colleague password")
    async with app.state.pool.connection() as conn:
        await conn.execute("INSERT INTO users (tenant_id, email, password_hash) VALUES (%s, %s, %s)",
                           (tenant, "c@example.com", password_hash))
    login = await client.post("/auth/login",
                              json={"email": "c@example.com", "password": "colleague password"})
    c = {"Authorization": f"Bearer {login.json()['access_token']}"}

    assert {d["id"] for d in (await client.get("/documents", headers=c)).json()} == set(a_ids)
    app.state.llm = ScriptedProvider([], "Private to A.")
    await client.post("/chat", headers=hdr(a), json={
        "id": "a-chat", "messages": [{"role": "user", "parts": [{"type": "text", "text": "x"}]}]})
    assert (await client.get("/conversations/a-chat", headers=c)).status_code == 404
    # Nor can the colleague append to it by reusing the id.
    resp = await client.post("/chat", headers=c, json={
        "id": "a-chat", "messages": [{"role": "user", "parts": [{"type": "text", "text": "y"}]}]})
    assert resp.status_code == 404
    assert len((await client.get("/conversations/a-chat", headers=hdr(a))).json()["messages"]) == 2


async def test_bm25_statistics_do_not_leak_between_tenants(client):
    """Shared IDF would let B watch a term's score move as A uploads, and learn about A's data."""
    a, b = await signup(client, "a@example.com"), await signup(client, "b@example.com")
    await upload(client, b, {"b.md": b"# Notes\n\nThe merger with Initech closes next quarter."})

    async def b_score():
        hits = (await client.post("/search", headers=hdr(b),
                                  json={"query": "initech merger", "mode": "fts"})).json()
        return hits[0]["score"]

    before = await b_score()
    await upload(client, a, {f"a{i}.md": f"# A {i}\n\nInitech merger rumours, part {i}.".encode()
                             for i in range(20)})
    assert await b_score() == pytest.approx(before)


async def new_tenant(conn) -> int:
    cur = await conn.execute("INSERT INTO tenants (name) VALUES ('t') RETURNING id")
    return (await cur.fetchone())["id"]


async def test_row_level_security_holds_without_query_filters(app):
    """The second line of defence: raw SQL with no tenant filter still sees one tenant."""
    from docpilot import store
    from docpilot.db import tenant_connection

    pool = app.state.pool
    emb = (await app.state.embedder.embed(["x"]))[0]
    async with pool.connection() as conn:
        t1, t2 = await new_tenant(conn), await new_tenant(conn)
    await store.insert_document(pool, tenant_id=t1, title="one", source="s", kind="upload",
                                chunks=["first tenant"], embeddings=[emb])
    await store.insert_document(pool, tenant_id=t2, title="two", source="s", kind="upload",
                                chunks=["second tenant"], embeddings=[emb])

    async with tenant_connection(pool, t1) as conn:
        rows = await (await conn.execute("SELECT tenant_id, text FROM chunks")).fetchall()
        assert [(r["tenant_id"], r["text"]) for r in rows] == [(t1, "first tenant")]
        docs = await (await conn.execute("SELECT title FROM documents")).fetchall()
        assert [d["title"] for d in docs] == ["one"]
        # Writing into another tenant is refused, not silently allowed.
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
            async with conn.transaction():
                await conn.execute("INSERT INTO documents (tenant_id, title, source) VALUES (%s, 'x', 'x')",
                                   (t2,))
        # Nor are its rows reachable to change or delete.
        cur = await conn.execute("UPDATE documents SET title = 'moved' WHERE tenant_id = %s", (t2,))
        assert cur.rowcount == 0
        cur = await conn.execute("DELETE FROM chunks WHERE tenant_id = %s", (t2,))
        assert cur.rowcount == 0

    # No tenant set at all: nothing is visible. Missing context fails closed.
    async with pool.connection() as conn:
        assert (await (await conn.execute("SELECT count(*) AS n FROM chunks")).fetchone())["n"] == 0

    # The owner (migrations, maintenance) sees both, which is why the app never runs as it.
    async with await psycopg.AsyncConnection.connect(app.state.settings.database_url) as conn:
        cur = await conn.execute(f"SELECT title FROM {SCHEMA}.documents WHERE tenant_id IN (%s, %s)",
                                 (t1, t2))
        assert sorted(r[0] for r in await cur.fetchall()) == ["one", "two"]


async def test_chunks_cannot_disagree_with_their_document_about_tenant(app):
    from docpilot.db import tenant_connection

    async with app.state.pool.connection() as conn:
        t1 = await new_tenant(conn)
    async with tenant_connection(app.state.pool, t1) as conn:
        doc = (await (await conn.execute(
            "INSERT INTO documents (tenant_id, title, source) VALUES (%s, 'd', 's') RETURNING id", (t1,),
        )).fetchone())["id"]
    # Even the owner, bypassing row-level security, can't attach the chunk to tenant 1.
    async with await psycopg.AsyncConnection.connect(app.state.settings.database_url) as conn:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            await conn.execute(
                f"""INSERT INTO {SCHEMA}.chunks (tenant_id, document_id, ord, text, embedding)
                    VALUES (1, %s, 0, 'x', array_fill(0.1, ARRAY[1024])::vector)""", (doc,))


async def test_small_tenant_gets_full_vector_results_in_a_big_index(app, client):
    """A tenant filter must not cut vector results short.

    At this size the planner may well scan exactly; docs/backend.md has the 45k-chunk
    measurement where it uses HNSW, and only the iterative scan returns all 30 candidates.
    """
    from docpilot import store

    big = await signup(client, "big@example.com")
    small = await signup(client, "small@example.com")
    big_tenant = (await client.get("/auth/me", headers=hdr(big))).json()["tenant_id"]
    texts = [f"Quarterly report section {i} on revenue, churn and pipeline." for i in range(1500)]
    embeddings = await app.state.embedder.embed(texts)
    for i in range(0, 1500, 300):
        await store.insert_document(app.state.pool, tenant_id=big_tenant, title=f"big {i}", source="s",
                                    kind="upload", chunks=texts[i:i + 300], embeddings=embeddings[i:i + 300])
    small_ids = await upload(client, small, {
        f"s{i}.md": f"# S{i}\n\nQuarterly revenue note number {i}.".encode() for i in range(3)})

    async with app.state.pool.connection() as conn:
        await conn.execute("ANALYZE chunks")
    hits = (await client.post("/search", headers=hdr(small),
                              json={"query": "quarterly revenue report", "mode": "vector",
                                    "top_k": 10})).json()
    assert {h["document_id"] for h in hits} == set(small_ids)
