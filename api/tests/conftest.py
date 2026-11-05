import asyncio
import sys
from pathlib import Path

import pytest

# Make the repo-level `evals` package importable from the API test suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

@pytest.fixture(autouse=True)
def no_ambient_redis(monkeypatch):
    """Settings read REDIS_URL from the environment. Tests that want Redis pass it explicitly
    (test_redis, test_jobs); everything else gets per-app limits and no cache, as before."""
    monkeypatch.delenv("REDIS_URL", raising=False)


if sys.platform == "win32":

    def pytest_asyncio_loop_factories(config, item):
        """psycopg's async driver can't run on Windows' default Proactor loop."""
        return {"selector": asyncio.SelectorEventLoop}
