"""End-to-end through the real API and Postgres+pgvector. Skipped unless DATABASE_URL is set.

Uses the fake embedder and fake LLM, in an isolated schema that is reset for the run.
"""

import json
import os

import httpx
import pytest

from docpilot.config import Settings

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
