"""Structured JSON logging with request and thread correlation.

The reference project created ``logger = logging.getLogger(...)`` and then
never called ``basicConfig``, attached no handler, and set no level (audit L5).
The result was a logger that emitted nothing useful, and no way to correlate a
request with the tool it called or how long it took.

This module provides:

- A single JSON log line per event, so logs are queryable rather than prose.
- A ``request_id`` carried in a :mod:`contextvars` variable, so any log call
  deep in the stack automatically attributes itself to the current request
  without every function having to thread an id through its signature.
- ``log_event`` for structured key/value fields (tool name, duration, route).
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
thread_id_var: ContextVar[str] = ContextVar("thread_id", default="-")

_RESERVED = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if (not record.exc_info) and record.levelno >= logging.ERROR:
            payload["exc"] = record.exc_text

        rid = request_id_var.get()
        if rid and rid != "-":
            payload["request_id"] = rid
        tid = thread_id_var.get()
        if tid and tid != "-":
            payload["thread_id"] = tid

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON handler on the root logger. Idempotent."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_intelliflow", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._intelliflow = True  # type: ignore[attr-defined]

    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def request_context(request_id: str | None = None, thread_id: str | None = None) -> Iterator[str]:
    """Bind correlation ids for the duration of the block.

    Tokens are reset on exit so a pooled request never inherits the previous
    caller's ids.
    """
    rid = request_id or new_request_id()
    rid_token = request_id_var.set(rid)
    tid_token = thread_id_var.set(thread_id) if thread_id else None
    try:
        yield rid
    finally:
        request_id_var.reset(rid_token)
        if tid_token is not None:
            thread_id_var.reset(tid_token)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured event.

    Extra keys are merged into the JSON object, so ``log_event(log, INFO,
    "tool_call", tool="RAG", ms=12)`` produces a queryable line.
    """
    logger.log(level, event, extra=fields)
