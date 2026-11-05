"""Postgres access: one async connection pool plus the schema bootstrap.

We use raw SQL through psycopg 3 rather than an ORM so every query is visible and explainable.
"""

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from docpilot.config import Settings

log = logging.getLogger(__name__)
_SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")
_MIGRATION_FILE = re.compile(r"^(\d{4})_\w+\.sql$")


def _connect_kwargs(settings: Settings) -> dict:
    """Point search_path at DB_SCHEMA (public stays on the path for the pgvector type)."""
    if not _SCHEMA_NAME.match(settings.db_schema):
        raise ValueError(f"invalid DB_SCHEMA {settings.db_schema!r}")
    kwargs: dict = {"row_factory": dict_row}
    if settings.db_schema != "public":
        kwargs["options"] = f"-c search_path={settings.db_schema},public"
    return kwargs


async def _configure(conn: psycopg.AsyncConnection) -> None:
    # Teaches psycopg to send/receive numpy arrays as pgvector `vector` values.
    await register_vector_async(conn)


def _app_role(settings: Settings) -> str | None:
    role = settings.db_app_role
    if role and not _SCHEMA_NAME.match(role):
        raise ValueError(f"invalid DB_APP_ROLE {role!r}")
    return role or None


async def open_pool(settings: Settings, *, as_owner: bool = False) -> AsyncConnectionPool:
    """The app's pool. Its connections run as DB_APP_ROLE, so row-level security applies.

    `as_owner` keeps the connecting role instead, for maintenance that spans tenants (the
    worker's sweeper, benchmarks). Nothing that serves a request should use it.
    """
    role = None if as_owner else _app_role(settings)

    async def configure(conn: psycopg.AsyncConnection) -> None:
        await _configure(conn)
        # Only the app's pool: migrations may legitimately run longer than a request should.
        # SET, not a startup `options` parameter, which connection poolers like PgBouncer reject.
        await conn.set_autocommit(True)
        await conn.execute(f"SET statement_timeout = {int(settings.db_statement_timeout_ms)}")
        if role:
            await conn.execute(f"SET ROLE {role}")
        await conn.set_autocommit(False)

    pool = AsyncConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_pool_timeout_seconds,
        kwargs=_connect_kwargs(settings),
        configure=configure,
        check=AsyncConnectionPool.check_connection,  # hosted Postgres drops idle connections
        open=False,
    )
    await pool.open(wait=True)
    return pool


@asynccontextmanager
async def tenant_connection(pool: AsyncConnectionPool, tenant_id: int,
                            ) -> AsyncIterator[psycopg.AsyncConnection]:
    """A pooled connection inside a transaction that row-level security scopes to one tenant.

    set_config(..., true) is transaction-local, like SET LOCAL: it ends with the transaction,
    so a connection returned to the pool never carries a tenant into its next borrower.
    """
    async with pool.connection() as conn:
        await conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(int(tenant_id)),))
        yield conn


async def schema_identity(pool: AsyncConnectionPool, settings: Settings) -> str:
    """Names this incarnation of the schema: dropping and recreating it changes the OID.

    Caches key on this, so results cached for a schema that evals have since reset (and whose
    ids now mean different rows) are never served for the new one.
    """
    async with pool.connection() as conn:
        row = await (await conn.execute(
            "SELECT oid::bigint AS oid FROM pg_namespace WHERE nspname = %s", (settings.db_schema,),
        )).fetchone()
    return f"{settings.db_schema}.{row['oid'] if row else 0}"


def _migrations() -> list[tuple[int, str, str]]:
    """(version, name, sql) from migrations/NNNN_name.sql, in order. Add files, never edit them."""
    found = []
    for path in files("docpilot").joinpath("migrations").iterdir():
        match = _MIGRATION_FILE.match(path.name)
        if match:
            found.append((int(match[1]), path.name, path.read_text(encoding="utf-8")))
    versions = [v for v, _, _ in found]
    if len(versions) != len(set(versions)):
        raise RuntimeError(f"duplicate migration versions: {sorted(versions)}")
    return sorted(found)


async def init_schema(settings: Settings, reset: bool = False) -> None:
    """Create the extension and namespace, then apply pending migrations. `reset` drops it first.

    Each migration runs in its own transaction with its version record, so a failure leaves the
    database at the last good version. An advisory lock lets several instances start at once.
    """
    schema = settings.db_schema
    async with await psycopg.AsyncConnection.connect(
        settings.database_url, autocommit=True, **_connect_kwargs(settings)
    ) as conn:
        # Everything below races between instances starting together, the extension included.
        await conn.execute("SELECT pg_advisory_lock(hashtext('docpilot:init_schema'))")
        try:
            # The extension lives in public so dropping an eval schema never drops it.
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            if reset:
                if schema == "public":
                    raise ValueError("refusing to reset the public schema; set DB_SCHEMA")
                await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            await conn.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                       version int PRIMARY KEY, name text NOT NULL,
                       applied_at timestamptz NOT NULL DEFAULT now())"""
            )
            rows = await (await conn.execute("SELECT version FROM schema_migrations")).fetchall()
            applied = {r["version"] for r in rows}
            for version, name, sql in _migrations():
                if version not in applied:
                    async with conn.transaction():
                        await conn.execute(sql)
                        await conn.execute(
                            "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                            (version, name),
                        )
            if role := _app_role(settings):
                await _grant_app_role(conn, schema, role)
        finally:
            await conn.execute("SELECT pg_advisory_unlock(hashtext('docpilot:init_schema'))")


async def _grant_app_role(conn: psycopg.AsyncConnection, schema: str, role: str) -> None:
    """Create the role the app runs as, and let it use (not alter) this schema's tables.

    Re-run on every init, so tables added by later migrations are covered too.
    """
    try:
        await conn.execute(
            f"""DO $$ BEGIN
                    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') THEN
                        CREATE ROLE {role} NOLOGIN;
                    END IF;
                END $$""")
        # The connecting role must be a member to SET ROLE to it (superusers always may).
        await conn.execute(f"GRANT {role} TO CURRENT_USER")
    except psycopg.errors.InsufficientPrivilege as e:
        raise RuntimeError(
            f"cannot create or join role {role!r} ({e}). Create it yourself (CREATE ROLE {role} "
            f"NOLOGIN; GRANT {role} TO <your user>), or set DB_APP_ROLE= to run without "
            "row-level security, relying on the tenant filters in queries alone."
        ) from e
    await conn.execute(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
    await conn.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema} TO {role}")
    await conn.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {schema} TO {role}")
