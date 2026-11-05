"""Command line: `docpilot db init` and `docpilot ingest <files...>`."""

import argparse
import asyncio
import sys
from pathlib import Path

from docpilot.config import get_settings
from docpilot.db import init_schema, open_pool
from docpilot.embeddings import make_embedder
from docpilot.ingest.pipeline import ingest_file
from docpilot.security import DEFAULT_TENANT_ID


def run_async(coro):
    """psycopg's async driver needs a selector event loop, which is not Windows' default."""
    if sys.platform == "win32":
        return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(coro)


async def _ingest(paths: list[Path], tenant_id: int) -> None:
    settings = get_settings()
    pool = await open_pool(settings)
    embedder = make_embedder(settings)
    try:
        for path in paths:
            doc_id, n = await ingest_file(pool, embedder, settings, tenant_id=tenant_id,
                                          filename=path.name, data=path.read_bytes())
            print(f"ingested {path.name}: document {doc_id}, {n} chunks")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="docpilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    db = sub.add_parser("db", help="database commands")
    db_sub = db.add_subparsers(dest="db_cmd", required=True)
    db_sub.add_parser("init", help="create the pgvector extension and all tables (idempotent)")
    ingest = sub.add_parser("ingest", help="ingest PDF/Markdown/HTML files")
    ingest.add_argument("paths", nargs="+", type=Path)
    ingest.add_argument("--tenant", type=int, default=DEFAULT_TENANT_ID,
                        help="tenant that owns the documents (default: 1, the API-key tenant)")
    args = parser.parse_args()

    if args.cmd == "db":
        run_async(init_schema(get_settings()))
        print("schema ready")
    elif args.cmd == "ingest":
        files = [p for path in args.paths for p in (sorted(path.iterdir()) if path.is_dir() else [path])]
        run_async(_ingest(files, args.tenant))


if __name__ == "__main__":
    main()
