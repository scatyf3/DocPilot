-- Row-level security: the database itself keeps tenants apart.
--
-- Every query already filters by tenant. This is the second line, for the query someone writes
-- next year and forgets to filter. The app's connections run as DB_APP_ROLE (not the owner,
-- which bypasses these policies) and set app.tenant_id at the start of each transaction.
-- If it isn't set, NULLIF turns the empty setting into NULL, the comparison is never true,
-- and the transaction sees no rows: missing context fails closed.
--
-- users, tenants and refresh_tokens stay outside: login has to find a user before it knows
-- the tenant. They hold no library content.

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON documents
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::bigint);

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON chunks
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::bigint);

ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON conversations
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::bigint);

ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON messages
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::bigint);

ALTER TABLE traces ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON traces
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::bigint);
