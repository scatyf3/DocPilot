"""Vector search under a tenant filter: does the HNSW index return enough rows?

The index holds every tenant's vectors and Postgres filters after it finds neighbours, so by
default a tenant gets back only its share of the ef_search (40) candidates. This builds
45,000 random 1024-dim chunks split 30,000 / 15,000 between two tenants in a scratch schema,
and runs the query retrieval runs, LIMIT 30, with and without pgvector's iterative scan,
printing the rows returned, the time and the plan Postgres chose.

    DATABASE_URL=postgresql://... python loadtest/hnsw_filter.py      # ~1 minute, ~600 MB

It connects as the owner and sets the tenant filter by hand: this measures the planner and
the index, not row-level security.
"""

import argparse
import asyncio
import os
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector_async

SCHEMA = "hnsw_filter_probe"
FILTERED = "SELECT id FROM chunks WHERE tenant_id = %(t)s ORDER BY embedding <=> %(q)s LIMIT 30"
QUERIES = {
    "no filter": ("SELECT id FROM chunks ORDER BY embedding <=> %(q)s LIMIT 30", None),
    "tenant with 2/3": (FILTERED, 1),
    "tenant with 1/3": (FILTERED, 2),
}


def unit_vectors(rng, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 1024)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


async def build(conn, sizes: dict[int, int], rng) -> None:
    await conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    await conn.execute(f"CREATE SCHEMA {SCHEMA}")
    await conn.execute(f"""CREATE TABLE {SCHEMA}.chunks (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                           tenant_id bigint NOT NULL, embedding vector(1024) NOT NULL)""")
    for tenant, n in sizes.items():
        async with conn.cursor().copy(
                f"COPY {SCHEMA}.chunks (tenant_id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int8", "vector"])
            for vec in unit_vectors(rng, n):
                await copy.write_row((tenant, vec))
    await conn.execute(f"CREATE INDEX ON {SCHEMA}.chunks (tenant_id)")
    # Serial build: a parallel one needs more shared memory than a default container has.
    await conn.execute("SET max_parallel_maintenance_workers = 0")
    await conn.execute("SET maintenance_work_mem = '512MB'")
    await conn.execute(f"CREATE INDEX ON {SCHEMA}.chunks USING hnsw (embedding vector_cosine_ops)")
    await conn.execute(f"ANALYZE {SCHEMA}.chunks")


async def measure(conn, label: str, rng, repeats: int = 5) -> None:
    q = unit_vectors(rng, 1)[0]
    heap = (await (await conn.execute(
        f"SELECT pg_size_pretty(pg_relation_size('{SCHEMA}.chunks'))")).fetchone())[0]
    print(f"\n{label} (heap {heap})")
    for name, (sql, tenant) in QUERIES.items():
        for scan in ("off", "relaxed_order"):
            if tenant is None and scan == "relaxed_order":
                continue
            await conn.execute(f"SET hnsw.iterative_scan = {scan}")
            params = {"q": q, "t": tenant}
            plan = [r[0] for r in await (await conn.execute("EXPLAIN " + sql, params)).fetchall()]
            node = next(p.strip(" ->") for p in plan if "Scan" in p).split("  (")[0]
            await (await conn.execute(sql, params)).fetchall()  # warm the cache
            started = time.perf_counter()
            for _ in range(repeats):
                rows = await (await conn.execute(sql, params)).fetchall()
            ms = (time.perf_counter() - started) / repeats * 1000
            print(f"  {name:16s} iterative_scan={scan:13s} {len(rows):2d}/30 rows {ms:7.1f} ms  {node}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--big", type=int, default=30_000)
    parser.add_argument("--small", type=int, default=15_000)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    async with await psycopg.AsyncConnection.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector_async(conn)
        await conn.execute(f"SET search_path = {SCHEMA}, public")
        try:
            await build(conn, {1: args.big, 2: args.small}, rng)
            await measure(conn, f"{args.big + args.small:,} chunks", rng)
        finally:
            await conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")


if __name__ == "__main__":
    asyncio.run(main())
