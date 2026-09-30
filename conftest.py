"""Shared pytest configuration.

Two things happen here that the reference project lacked.

1. **The event loop policy is fixed on Windows.** The reference test suite
   could not collect a single passing test on Windows: all ten failed with
   ``psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop'``.
2. **No import of the application is required to run the tests.** Settings are
   built explicitly and injected, so the suite runs with zero hosted
   credentials - which is what makes the default local-embedding path
   testable at all.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.platform import ensure_compatible_event_loop  # noqa: E402

ensure_compatible_event_loop()

# Keep the default provider local and credential-free for the test run, unless
# the developer has explicitly configured something else.
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("VECTOR_STORE", "memory")
# Any real credentials in a developer's shell would otherwise make a test that
# asserts graceful degradation accidentally depend on a live service.
os.environ.setdefault("TAVILY_API_KEY", "")


@pytest.fixture(scope="session")
def settings():
    """Application settings for tests: no hosted credentials required."""
    from app.core.config import Settings

    return Settings.from_env(
        {
            "EMBEDDING_PROVIDER": os.environ.get("EMBEDDING_PROVIDER", "local"),
            "VECTOR_STORE": os.environ.get("VECTOR_STORE", "memory"),
            "MAX_ITERATIONS": "3",
        }
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
