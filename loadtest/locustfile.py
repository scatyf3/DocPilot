"""Load test for the DocPilot API: many tenants searching, chatting and uploading at once.

Each simulated user signs up (their own tenant), uploads three documents from the sample corpus
through the ingest queue, waits for them, then loops over a mix of requests: search (6),
chat (2), list documents (1). Questions come from evals/dataset.jsonl, so they repeat across
users the way real questions do; the search cache sees realistic reuse, but only within a
tenant, since caches are per tenant.

    pip install locust
    locust -f loadtest/locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000

LOAD_THINK_SECONDS="0.5,2" sets the pause between a user's requests (default). "0" makes a
closed loop, every user firing back to back, which measures capacity rather than a workload.

Run the API with a fake LLM unless you mean to load-test your model provider and pay for it;
the numbers then measure this backend (FastAPI, Postgres, Redis), not model latency. Raise the
rate limits (RATE_LIMIT_*_PER_MINUTE) or every user hits 429 within a minute; leave them
enabled so the Redis limiter's cost is part of what you measure.
"""

import json
import os
import random
import time
import uuid
from pathlib import Path

from locust import HttpUser, between, constant, task

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [json.loads(line)["question"] for line in (ROOT / "evals/dataset.jsonl").open()]
DOCS = sorted((ROOT / "evals/corpus").glob("*.md"))
_think = [float(x) for x in os.environ.get("LOAD_THINK_SECONDS", "0.5,2").split(",")]


class LibraryUser(HttpUser):
    wait_time = constant(_think[0]) if len(_think) == 1 else between(*_think)

    def on_start(self):
        resp = self.client.post("/auth/register", name="/auth/register", json={
            "email": f"load-{uuid.uuid4().hex[:12]}@example.com", "password": "load test password"})
        resp.raise_for_status()
        self.headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        for path in random.sample(DOCS, 3):
            resp = self.client.post("/documents", headers=self.headers, name="/documents [upload]",
                                    files={"file": (path.name, path.read_bytes())})
            if resp.status_code == 202:
                self._wait_for(resp.json()["job_id"])

    def _wait_for(self, job_id: str) -> None:
        # Once a second, like the web app. Polling faster mostly adds load while jobs are queued:
        # at 10 Hz, 200 users waiting on a backlog made 93% of all requests job polls.
        for _ in range(300):
            job = self.client.get(f"/jobs/{job_id}", headers=self.headers, name="/jobs/[id]").json()
            if job["status"] in ("succeeded", "failed"):
                return
            time.sleep(1.0)

    @task(6)
    def search(self):
        self.client.post("/search", headers=self.headers, name="/search",
                         json={"query": random.choice(QUESTIONS), "top_k": 5})

    @task(2)
    def chat(self):
        body = {"id": uuid.uuid4().hex, "messages": [
            {"role": "user", "parts": [{"type": "text", "text": random.choice(QUESTIONS)}]}]}
        with self.client.post("/chat", headers=self.headers, json=body, name="/chat",
                              stream=True, catch_response=True) as resp:
            text = resp.text  # the whole stream: latency is time to the last event
            if not text.endswith("data: [DONE]\n\n") or '"type":"error"' in text.replace(" ", ""):
                resp.failure("stream did not finish cleanly")

    @task(1)
    def list_documents(self):
        self.client.get("/documents", headers=self.headers, name="/documents")
