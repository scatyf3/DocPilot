# Backend: tenants, accounts, Redis and the ingest queue

This covers what the API does beyond answering questions: who can see which documents, how
users sign in, what Redis is for, and how uploads are processed in the background. Each
section ends with the tests that check it.

## Tenants and permission-scoped retrieval

A tenant is a workspace. Signing up creates one, and its documents, chunks, conversations,
messages, traces and ingest jobs all carry its `tenant_id`. Users in a tenant share its
library. Conversations belong to the user who started them. Service API keys act on tenant 1,
which also holds everything created before tenants existed.

In a RAG app every path that can surface a chunk is a possible leak, not just the obvious
routes. A chunk can reach a user through `/search`, `/documents/{id}`, the agent's
`search_docs` and `read_doc` tools, a citation id in the answer, conversation history and
traces. Each of these is scoped to the caller's tenant in two independent ways.

1. **Query filters.** Every query in `store.py` and `retrieval/search.py` filters on
   `tenant_id`. The agent's tools close over the caller's tenant: the model picks queries and
   ids but never the tenant, so a prompt-injected document can't point it at someone else's
   library.
2. **Row-level security.** Postgres enforces the same rule itself
   (`migrations/0004_row_level_security.sql`). The app's connections `SET ROLE` to
   `docpilot_app`, a role that doesn't own the tables (owners and superusers bypass RLS). Each
   transaction sets `app.tenant_id` (`db.tenant_connection`), and the policy compares rows
   against it. If no tenant is set, the policy matches nothing, so a missing tenant fails
   closed rather than open.

Either layer alone keeps tenants apart. Having both means one of them can be wrong, for
example a query written next year without a filter, and data still doesn't leak.

The schema keeps the data consistent as well as private:

- `chunks.tenant_id` duplicates the document's tenant so retrieval can filter without a join.
  A composite foreign key `(document_id, tenant_id) -> documents(id, tenant_id)` makes it
  impossible for the two to disagree, even for the table owner.
- Conversation ids come from the client, so their primary key is `(tenant_id, id)`. Another
  tenant's id can neither collide with ours nor reveal that it exists.
- Every `tenant_id` column was backfilled through a temporary `DEFAULT 1`, which was then
  dropped. A write that forgets its tenant now fails instead of landing in tenant 1.

**Search statistics are per tenant too.** BM25 needs document frequencies and an average chunk
length. If those were computed over all tenants, tenant B could watch the score of their own
document change as tenant A uploads, and so learn how often a term occurs in A's private
documents. A test uploads 20 documents to A and checks that B's score doesn't move.

**Caches are per tenant**, for the same reason. A cache hit is faster than a miss, so a shared
cache would let B time a query and learn whether A had asked it recently.

### Vector search under a tenant filter

Filtering an HNSW index needs care, because the index holds every tenant's vectors and Postgres
applies the `tenant_id` filter only after the index has found neighbours. By default the index
gathers `ef_search` (40) candidates, and a tenant keeps only its own share of them.

The measurement used 45,000 random 1024-dim chunks split 30,000 / 15,000 between two tenants,
asking for the 30 candidates retrieval uses. Reproduce it with
`DATABASE_URL=... python loadtest/hnsw_filter.py`. Postgres chose the HNSW index in every case.

| query, LIMIT 30 | rows returned, filter after HNSW (default) | rows returned, `hnsw.iterative_scan = relaxed_order` |
|---|---|---|
| no filter | 30 | n/a |
| tenant with 2/3 of the rows | 24–30 | 30 |
| tenant with 1/3 of the rows | **7–16** (three builds) | 30 |

Each query took 5–20 ms either way. Without an iterative scan, a tenant that owns a third of
the index silently gets back between a quarter and half of the candidates it asked for.
Retrieval quality drops and no error is raised, and nothing in the single-tenant evals would
show it. `_vector` therefore enables the iterative scan (pgvector ≥ 0.8) and re-sorts the
results, because `relaxed_order` may return them slightly out of order. Searches limited to
chosen documents skip the index entirely and scan those documents' chunks exactly.

Small tenants are not the problem: when a tenant's share is small, the planner uses the
`tenant_id` btree index and an exact sort instead of HNSW.

One loose end. In the first exploratory run (45,000 chunks inserted document by document
through the app), the planner ignored HNSW and scanned sequentially, at 466 ms a query. A
rebuild of the index fixed it. The suspicion was out-of-line (TOAST) storage of the ~4 KB
vectors, but a controlled comparison ruled that out: default, external and inline storage, with
and without a text column, and with the index built before or after loading, all used the
index. The cause wasn't found. If vector search is slow, `EXPLAIN` it before anything else.

### Tests

- `api/tests/test_tenancy.py` tries every path from user B against user A's data: `/search` in
  all three modes with 28 queries aimed at A's text, `document_ids` set to A's ids,
  `/documents/{id}` (read and delete), an agent that calls `search_docs` and `read_doc` and
  cites A's chunk id, a shared conversation id, and traces. It also checks colleagues in one
  tenant (shared library, private conversations) and BM25 statistics.
- The RLS test bypasses the application. Raw `SELECT * FROM chunks` with no filter returns only
  the current tenant's rows. A write into another tenant raises "violates row-level security".
  With no tenant set, nothing is visible.
- Retrieval quality is unchanged: `python -m evals.run --retrieval-only --embedder fake` gives
  exactly the baseline on all twelve metrics.

## Accounts and tokens

Endpoints: `POST /auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout` and
`GET /auth/me`. All of them are enabled when `JWT_SECRETS` is set.

- **Passwords: argon2id.** Each hash costs about 50 ms of CPU and 64 MiB of RAM by default.
  Hashing runs in a thread so it doesn't stall the event loop. A semaphore caps concurrent
  hashes per instance (`PASSWORD_HASH_CONCURRENCY`), because 50 simultaneous logins would
  otherwise allocate 3 GiB. Login with an unknown email still verifies against a dummy hash, so
  response time doesn't reveal which emails have accounts. The error message is the same in
  both cases. Hashes made with weaker parameters are upgraded on the next successful login.
- **Access tokens: HS256 JWTs, 15 minutes.** The algorithm list is fixed, so the token's own
  `alg` header (for example `none`) is never trusted. Issuer, audience, expiry and the tenant
  claim are required. `JWT_SECRETS` is a list: the first secret signs and every secret
  verifies, chosen by a `kid` header. Rotating a secret is therefore the same as rotating an
  API key: put the new one first, deploy, and drop the old one after 15 minutes.
- **Refresh tokens: opaque, single-use, in families.** Each one is 256 random bits and only its
  SHA-256 hash is stored. Refreshing marks the token used and issues the next one in the same
  family. Presenting a used token again means someone copied it, so the whole family is
  revoked, including the newest token, which the thief may hold. Whichever of thief and owner
  refreshes second is logged out, and the reuse is logged. Logout revokes the family.
- **Concurrent refreshes.** The rotation locks the token row (`FOR UPDATE`). When two refreshes
  race with the same token, one wins and the other sees a used token. That looks like a
  replay, so the family is revoked. Two browser tabs refreshing at the same moment therefore
  log the user out. This is the cost of being able to detect a replay at all; the usual fix is
  to refresh from a single place, such as the web app's server.
- **What stateless access tokens can't do:** a stolen one works until it expires, and logout
  can't recall it. The short TTL bounds that window.

Tests in `api/tests/test_auth.py` cover forged tokens (`alg: none`, wrong secret, edited tenant
claim, expired, wrong audience, missing claim), secret rotation, the concurrency bound,
rotation, reuse, a five-way refresh race (exactly one wins, then the family is dead), logout
and the cleanup of expired tokens.

## Redis

Redis holds nothing that can't be lost. Rate-limit buckets restart, caches refill, and queue
messages are rebuilt from Postgres (see below). Every Redis call has a 0.5 s timeout. After a
failure, the client skips Redis for 5 s instead of waiting on every request.

**Rate limits.** The in-memory token buckets from before allowed N times the limit across N
instances. The bucket is now one Lua script in Redis (`security._TOKEN_BUCKET`). It runs
atomically, takes the time from the Redis server so instances with skewed clocks agree, and
expires idle keys. Limits apply per user when a token is present, otherwise per IP, and
login is also limited per email. If Redis is down, each instance falls back to its own local
buckets: the limits loosen, but the API stays up.

- Two `RedisRateLimiter`s sharing a limit of 5 admit 5 requests; two local ones admit 10.
- Two app instances with a limit of 3 return `[200, 200, 200, 429, 429, 429]` for alternating
  requests.
- Against a live stack with `uvicorn --workers 2`, 14 uploads at 10/minute gave 10 accepted
  and 4 limited.
- 60 concurrent acquisitions against a bucket of 10 admit exactly 10.

**Caches.** There are two:

- Query embeddings, keyed by tenant, model and text, kept for 24 h.
- Complete search results, which the agent's `search_docs` tool also uses. These are
  invalidated by generation: each tenant has a counter that every write to its library
  increments, and the counter is part of the cache key. A write therefore orphans every cached
  result for that tenant at once, and the orphans expire on their own. A search that races a
  write can only store a stale result under the old generation, which nothing reads any more.
- Results from a degraded search (embeddings or reranker down) are not cached, so they don't
  outlive the outage.
- The cache namespace includes the schema's OID. When evals drop and recreate their schema, ids
  start meaning different rows, and results cached for the old schema are never served for the
  new one.

## Background ingest

With `INGEST_MODE=queue`, `POST /documents` stores the file in `ingest_jobs` and returns
`202 {"job_id"}`. `docpilot worker` (arq) parses, chunks and embeds the file, updating
`chunks_done / chunks_total` after every embedding batch. The client polls `GET /jobs/{id}`; the
web library shows "embedding 128/412 chunks (31%)". Without Redis, or with `INGEST_MODE=inline`,
uploads are processed inside the request as before.

- **Postgres is the source of truth; the Redis message is only a wake-up call.** If the enqueue
  fails or Redis loses its data, the job is still in Postgres. Once a minute, a sweeper
  re-enqueues jobs that have been queued for more than 2 minutes. arq's job ids
  (`ingest:<uuid>`) stop a job from being queued twice. The sweeper also marks jobs failed when
  they show "running" past their timeout, which happens when the worker died mid-job.
- **At most one document per job.** The document insert and the job's move to `succeeded`
  commit in one transaction, after locking the job row. Three concurrent attempts at one job
  store one document.
- **Retry only what might succeed next time.** An unreadable file fails on the first attempt,
  with the reason shown to the user. Errors such as the embedding API being down are retried
  after 5 s and then 30 s; after the third attempt the job is marked failed.
- Jobs are under RLS like everything else. The worker gets the tenant id with the job id, and a
  forged tenant id finds nothing to run. Only the sweeper and token cleanup use an owner
  connection (`open_pool(as_owner=True)`), and they read ids and timestamps only.

Tests in `api/tests/test_jobs.py` cover progress, duplicate delivery, concurrent attempts, bad
input, retry then give up, tenant scoping, a real arq worker processing an upload end to end,
and the sweeper recovering a job whose message was lost.

## Load test

`loadtest/locustfile.py` simulates users who each sign up (a tenant of their own), upload three
documents through the queue and wait for them, then loop over search (6), chat (2) and list
documents (1), with questions drawn from the eval set.

**Caveats first.**

- **Fake models.** The LLM and embedder are fakes, so these numbers measure this backend
  (FastAPI, Postgres, Redis, the agent loop) and not model latency. A real chat turn adds
  seconds of model time per step. Search would add one embedding call on a cache miss.
- **Shared, busy machine.** Everything ran on one 40-thread Xeon w7-3445 whose load average was
  already about 50 from other work. Absolute latencies are noisy; read the comparisons, not the
  digits.
- **Setup.** Two uvicorn processes would be the norm on a small host; these runs used
  `--workers 4`, one `docpilot worker` (concurrency 4), Postgres 16 with pgvector 0.8.7 and
  Redis 7 in containers, and 75-second runs. Rate limits stayed enabled, with the limits raised
  out of the way, so the Redis limiter's cost is included.

**Capacity: closed loop, 50 users, no think time.**

| endpoint | cache on: req/s | p50 | p95 | p99 | cache off: req/s | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| /search | 290 | 62 ms | 160 ms | 230 ms | 215 | 79 ms | 180 ms | 220 ms |
| /chat (whole stream) | 99 | 58 ms | 100 ms | 140 ms | 70 | 94 ms | 220 ms | 280 ms |
| /documents (list) | 49 | 31 ms | 76 ms | 100 ms | 36 | 37 ms | 81 ms | 120 ms |
| all requests | **452** | 56 ms | 140 ms | 220 ms | **335** | 68 ms | 190 ms | 240 ms |

The cache raises throughput by 35% at the same concurrency, with 0 errors in either run (33,827
and 24,941 requests). Chat gains the most because every turn's `search_docs` call is a search.
Search's own p95 moves less: at saturation the queue in front of the server dominates, and
with 50 users and 110 questions, many of a tenant's searches are first-time misses. An earlier
pair of runs gave 472 and 347 req/s, so the gap is stable even though the absolute numbers drift.

**A realistic mix: 200 users, 0.5–2 s think time, cache on.** 14,703 requests, 0 errors,
198 req/s.

| endpoint | p50 | p95 | p99 |
|---|---|---|---|
| /search | 15 ms | 93 ms | 170 ms |
| /chat (whole stream) | 13 ms | 69 ms | 120 ms |
| POST /documents (queued) | 24 ms | 99 ms | 140 ms |
| GET /jobs/{id} | 10 ms | 68 ms | 120 ms |
| /auth/register (argon2id, 19 MiB, t=2) | 170 ms | 300 ms | 320 ms |

What the runs showed:

- **Polling is a load amplifier.** The first version of this test polled job status ten times a
  second. With 600 uploads queued behind a worker, those polls were 93% of all requests and
  pushed search p95 to 270 ms and chat p95 to 240 ms. Polling once a second, as the web app
  does, gave the table above. The next step would be backoff, or server-sent events for
  progress.
- **The ingest queue absorbs bursts at the cost of waiting.** 200 users uploading 600 documents
  within a few seconds were all answered in about 24 ms (p50). One worker cleared the burst in
  75 s, about 8 documents a second with fake embeddings. The median upload waited 21 s before it
  was searchable, nearly all of it in the queue. Throughput scales with `--scale worker=N`. With
  inline ingest, those 600 requests would each have held an API connection for the whole
  parse-and-embed.
- **Sign-up is deliberately slow.** It is argon2id work, capped at 4 concurrent hashes per
  process, so a burst queues instead of exhausting memory.
- When locust stops a run, it drops in-flight requests. psycopg then logs that it discarded the
  cancelled requests' connections. The pool replaces them, and no request failed during any
  run.

To reproduce:

```bash
# API: fake models, high limits, Redis, queue mode
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake JWT_SECRETS=<32+ chars> REDIS_URL=redis://localhost:6379/0 \
INGEST_MODE=queue RATE_LIMIT_SEARCH_PER_MINUTE=100000 RATE_LIMIT_CHAT_PER_MINUTE=100000 \
RATE_LIMIT_UPLOAD_PER_MINUTE=100000 RATE_LIMIT_AUTH_PER_MINUTE=100000 \
  uvicorn docpilot.app:app --port 8000 --workers 4 --no-access-log &
docpilot worker &   # same environment
LOAD_THINK_SECONDS=0 locust -f loadtest/locustfile.py --headless -u 50 -r 25 -t 75s --host http://localhost:8000
```

Set `SEARCH_CACHE_TTL_SECONDS=0 QUERY_EMBEDDING_CACHE_TTL_SECONDS=0` for the cache-off run.
