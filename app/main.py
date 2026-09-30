"""Application entry point.

Run with::

    uvicorn app.main:app --reload

The application is built by an explicit factory rather than at import time.
The reference project constructed ``app = FastAPI(...)`` and called
``config.validate()`` at module scope, so ``import main`` raised unless four
hosted credentials were present (audit L6) and the test suite could not be
collected without production secrets.
"""

from __future__ import annotations

from app.api.routes import create_app
from app.core.config import Settings
from app.core.logging import configure_logging, get_logger
from app.core.platform import ensure_compatible_event_loop

logger = get_logger(__name__)

# Must happen before any async psycopg work. See app/core/platform.py.
_loop_policy = ensure_compatible_event_loop()

_settings = Settings.from_env()
configure_logging(_settings.log_level)

app = create_app(_settings)

logger.info(
    "app_started",
    extra={
        "event_loop_policy": _loop_policy,
        "embedding_provider": _settings.embedding_provider,
        "vector_store": _settings.vector_store,
    },
)


def main() -> None:
    """Console entry point: run the API with uvicorn."""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
