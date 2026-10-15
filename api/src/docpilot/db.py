"""Postgres access: one async connection pool plus the schema bootstrap.

We use raw SQL through psycopg 3 rather than an ORM so every query is visible and explainable.
"""

import re
from importlib.resources import files

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from docpilot.config import Settings

_SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


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


async def open_pool(settings: Settings) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=10,
        kwargs=_connect_kwargs(settings),
        configure=_configure,
        open=False,
    )
    await pool.open(wait=True)
    return pool


async def init_schema(settings: Settings, reset: bool = False) -> None:
    """Create the extension, the namespace and all tables. `reset` drops the namespace first."""
    schema = settings.db_schema
    ddl = files("docpilot").joinpath("schema.sql").read_text(encoding="utf-8")
    async with await psycopg.AsyncConnection.connect(
        settings.database_url, autocommit=True, **_connect_kwargs(settings)
    ) as conn:
        # The extension lives in public so dropping an eval schema never drops it.
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        if reset:
            if schema == "public":
                raise ValueError("refusing to reset the public schema; set DB_SCHEMA")
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        await conn.execute(ddl)
