-- Background ingest (INGEST_MODE=queue). The upload stores the file here and returns a job id.
-- `docpilot worker` parses, chunks and embeds it, updating progress as it goes. Postgres, not
-- Redis, is the source of truth: the Redis queue only says "look at job X now", so a lost
-- message (Redis restarted, enqueue failed) delays a job but never loses it. The worker's
-- sweeper finds stale queued jobs and enqueues them again.

CREATE TABLE ingest_jobs (
    id            uuid PRIMARY KEY,
    tenant_id     bigint NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    created_by    bigint REFERENCES users(id) ON DELETE SET NULL,
    filename      text NOT NULL,
    status        text NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    stage         text,                          -- parsing | embedding | storing, while running
    chunks_done   int NOT NULL DEFAULT 0,
    chunks_total  int,
    attempts      int NOT NULL DEFAULT 0,
    error         text,
    document_id   bigint REFERENCES documents(id) ON DELETE SET NULL,
    payload       bytea,                         -- the uploaded file, dropped once processed
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);
CREATE INDEX ingest_jobs_tenant_idx ON ingest_jobs (tenant_id, created_at DESC);
-- The sweeper's scan: only unfinished jobs, so it stays small however long history grows.
CREATE INDEX ingest_jobs_unfinished_idx ON ingest_jobs (updated_at) WHERE status IN ('queued', 'running');

ALTER TABLE ingest_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON ingest_jobs
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::bigint);
