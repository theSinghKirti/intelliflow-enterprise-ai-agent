"""Memory layer: Postgres-backed conversation checkpointing.

The reference project used ``AsyncPostgresSaver.from_conn_string(...)`` inline
inside the request handler, so the API layer knew how memory was stored (audit
L8). This module puts that behind two small helpers: :func:`checkpointer`
yields the saver, and :func:`thread_config` builds the LangGraph config that
selects a conversation.

Persistence itself is the compiled graph's job. Passing a checkpointer to
``compile()`` is what makes LangGraph write each thread's state, so this module
does not implement its own read/write path.

Note the reference project also read history out of
``checkpoint["channel_values"]``, a LangGraph internal layout. The current
build does not read history at all: the graph replays the checkpointed state
into the answer node, so no history accessor is needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def checkpointer(settings: Settings) -> AsyncIterator[AsyncPostgresSaver]:
    """Yield a Postgres checkpointer, opening a connection per use.

    Why a connection per request
    ----------------------------
    This is deliberate and inherited from the reference project, whose comment
    recorded the hard-won operational detail: a single long-lived connection
    across requests meant that once a serverless provider (Neon) or free-tier
    host silently dropped an idle connection, *every* subsequent request failed
    until the process restarted. A fresh connection per call avoids that
    failure mode, at the cost of a small per-call overhead.

    ``setup()`` issues CREATE TABLE IF NOT EXISTS DDL. It is invoked on each
    use because the saver exposes no separate one-time migration hook; the
    statements are idempotent.
    """
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required for conversation memory.")

    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
        await saver.setup()
        yield saver


def thread_config(thread_id: str) -> dict[str, Any]:
    """Build the LangGraph config dict that selects a conversation thread."""
    return {"configurable": {"thread_id": thread_id}}
