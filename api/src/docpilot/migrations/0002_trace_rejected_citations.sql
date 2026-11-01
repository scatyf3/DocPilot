-- Chunk ids the model cited without having been shown them: a signal worth querying over time.
ALTER TABLE traces ADD COLUMN IF NOT EXISTS rejected_citation_ids bigint[] NOT NULL DEFAULT '{}';
