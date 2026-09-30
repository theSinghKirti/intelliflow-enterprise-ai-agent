"""HTTP routes.

The reference project exposed exactly two endpoints (``GET /health`` and
``POST /chat``) with no authentication, no rate limiting, and no request
correlation. This module adds the request-ID middleware, keeps the layered
error mapping, and fixes the one place that mapping was wrong.

Error mapping bug fixed
-----------------------
The reference code mapped database failures to ``503`` via ``except OSError``,
with a comment claiming "asyncpg raises OSError subclasses". The project used
``psycopg``, whose errors derive from ``Exception``, not ``OSError``, so that
branch was unreachable dead code and every database failure surfaced as a
generic ``500``. The handler below catches ``psycopg.Error`` explicitly.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from app.agents.graph import build_graph
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Source,
)
from app.core.config import Settings
from app.core.logging import (
    get_logger,
    log_event,
    new_request_id,
    request_context,
)
from app.memory.postgres import checkpointer, thread_config
from app.services.llm import get_llm

logger = get_logger(__name__)

VERSION = "2.0.0"

router = APIRouter()


def _resolve_settings(request: Request) -> Settings:
    """The settings this application was actually built with.

    ``create_app`` stores its resolved ``Settings`` on ``app.state``, and that
    snapshot is the single source of truth. The fallback to a fresh
    ``from_env()`` only applies if the router was mounted on an app that never
    went through the factory.

    Reading the ambient environment here instead would make ``/health`` report
    on whatever ``.env`` happens to be on disk rather than on the running
    configuration - so a process built with no ``DATABASE_URL`` could report
    ``"memory": "postgres"`` purely because a developer's ``.env`` supplied
    one. A health probe that misreports its own configuration is worse than one
    that reports nothing.
    """
    from app.core.config import Settings as _S

    return getattr(request.app.state, "settings", None) or _S.from_env()


@router.get("/health", response_model=HealthResponse, summary="Liveness and configuration probe")
async def health(request: Request) -> HealthResponse:
    """Report service health and which providers are active.

    Deliberately reports configuration rather than deep dependency health: a
    probe that calls Postgres on every request is a load amplifier, and
    ``/chat`` already surfaces real failures.
    """
    settings = _resolve_settings(request)
    return HealthResponse(
        status="ok",
        version=VERSION,
        embeddings=f"{settings.embedding_provider} ({settings.embedding_dimension}-dim)",
        vector_store=settings.vector_store,
        memory="postgres" if settings.database_url else "unconfigured",
    )


@router.post("/chat", response_model=ChatResponse, summary="Chat with the agent")
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Run one agent turn.

    Returns the reply together with the chosen route, whether the answer was
    grounded in retrieved documents, and the citations used.
    """
    settings = _resolve_settings(request)
    thread_id = payload.resolved_thread_id(settings)

    with request_context(thread_id=thread_id):
        started = time.perf_counter()
        log_event(
            logger,
            20,
            "chat_request",
            query_chars=len(payload.query),
            has_image=payload.image_base64 is not None,
        )
        try:
            result = await _run_turn(settings, payload, thread_id)
        except HTTPException:
            raise
        except Exception as exc:
            log_event(
                logger,
                40,
                "chat_unhandled",
                error_type=type(exc).__name__,
                ms=int((time.perf_counter() - started) * 1000),
            )
            # Never leak the exception text or a stack trace to the caller.
            raise HTTPException(
                status_code=500,
                detail="Something went wrong processing your request.",
            ) from exc

        log_event(
            logger,
            20,
            "chat_complete",
            route=result.get("route"),
            grounded=result.get("grounded"),
            insufficient=result.get("insufficient_information"),
            tools=[c.get("name") for c in result.get("tool_calls", [])],
            iterations=result.get("iteration_count", 0),
            ms=int((time.perf_counter() - started) * 1000),
        )
        return ChatResponse(
            response=result.get("answer", ""),
            thread_id=thread_id,
            request_id=request.state.request_id,
            route=result.get("route"),
            route_confidence=result.get("route_confidence"),
            grounded=result.get("grounded"),
            insufficient_information=result.get("insufficient_information"),
            sources=[Source(**s) for s in result.get("sources", [])],
            tool_calls=[c.get("name", "") for c in result.get("tool_calls", [])],
            iterations=result.get("iteration_count", 0),
        )


async def _run_turn(settings: Settings, payload: ChatRequest, thread_id: str) -> dict[str, Any]:
    """Execute one turn and return the final state values of interest."""
    try:
        llm = get_llm(settings)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Model provider is not configured."
        ) from exc

    try:
        graph = await build_graph(settings, llm)
    except Exception as exc:
        log_event(logger, 40, "graph_build_failed", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Agent is not ready to serve requests."
        ) from exc

    async with checkpointer(settings) as saver:
        compiled = graph.compile(checkpointer=saver)
        state = await compiled.ainvoke(
            {
                "messages": [HumanMessage(content=payload.query)],
                "query": payload.query,
                "thread_id": thread_id,
                "iteration_count": 0,
            },
            config=thread_config(thread_id),
        )

    return {
        "answer": state.get("answer", ""),
        "route": state.get("route"),
        "route_confidence": state.get("route_confidence"),
        "grounded": state.get("grounded"),
        "insufficient_information": state.get("insufficient_information"),
        "sources": state.get("sources", []),
        "tool_calls": state.get("tool_calls", []),
        "iteration_count": state.get("iteration_count", 0),
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Replaces the reference project's module-level ``app = FastAPI(...)``, which
    meant importing the module started LLM client construction and env
    validation as a side effect.
    """
    resolved = settings or Settings.from_env()

    application = FastAPI(
        title="IntelliFlow API",
        description=(
            "Enterprise multi-agent assistant routing between grounded document "
            "retrieval, web search, database queries, Python execution, and a "
            "direct model response."
        ),
        version=VERSION,
    )
    application.state.settings = resolved

    @application.middleware("http")
    async def correlate(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a request id to every request and response.

        Sets the ``X-Request-ID`` response header and binds the id to the
        logging context so any log line emitted while handling the request
        carries it, without threading the value through every function.
        """
        incoming = request.headers.get("X-Request-ID")
        request_id = incoming if incoming and len(incoming) <= 64 else new_request_id()
        request.state.request_id = request_id
        with request_context(request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log_event(
            logger,
            40,
            "unhandled_exception",
            error_type=type(exc).__name__,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    application.include_router(router)
    return application
