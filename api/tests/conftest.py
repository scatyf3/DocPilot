import asyncio
import sys
from pathlib import Path

# Make the repo-level `evals` package importable from the API test suite.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if sys.platform == "win32":

    def pytest_asyncio_loop_factories(config, item):
        """psycopg's async driver can't run on Windows' default Proactor loop."""
        return {"selector": asyncio.SelectorEventLoop}
