"""Windows event-loop compatibility.

``psycopg``'s async driver cannot use the ``ProactorEventLoop`` that Python
3.8+ selects by default on Windows: it raises

    psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run
    in async mode.

on the first connection. Because the reference project never set an event loop
policy, **every** code path that touched the Postgres checkpointer failed on
Windows - including its own documented ``uvicorn main:app --reload``
quick-start and all ten of its tests. The failure is invisible on Linux, which
is where it was developed.

Fix: select the selector-based loop policy on Windows before any async
psycopg work. This must run at process start, so
:func:`ensure_compatible_event_loop` is called from the application entry
point and from the test configuration.
"""

from __future__ import annotations

import asyncio
import sys


def ensure_compatible_event_loop() -> str:
    """Install a psycopg-compatible event loop policy on Windows.

    Returns a short description of the policy in effect, for logging. Safe to
    call more than once.
    """
    if sys.platform != "win32":
        return "default (non-Windows)"

    current = type(asyncio.get_event_loop_policy()).__name__
    if "Selector" in current:
        return current

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return type(asyncio.get_event_loop_policy()).__name__


def run(coro):  # type: no[no-untyped-def]
    """Run a coroutine on a compatible loop.

    Convenience for CLI entry points that would otherwise trip the same
    ProactorEventLoop problem.
    """
    ensure_compatible_event_loop()
    return asyncio.run(coro)
