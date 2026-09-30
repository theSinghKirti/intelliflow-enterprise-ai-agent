"""Unit tests for configuration, logging, platform, and the HTTP layer.

All credential-free. The reference project's ``config.validate()`` raised at
import time on four hosted keys, which is why its own test suite could not be
collected without production credentials (audit L6).
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi import FastAPI

from app.api.routes import create_app
from app.core.config import ConfigurationError, Settings
from app.core.logging import (
    JsonFormatter,
    get_logger,
    log_event,
    new_request_id,
    request_context,
    request_id_var,
    thread_id_var,
)
from app.core.platform import ensure_compatible_event_loop


# ── Settings ─────────────────────────────────────────────────────────────

def test_defaults_need_no_api_keys():
    """Default configuration needs no *hosted* credential.

    ``DATABASE_URL`` is the sole hard requirement, because conversation memory
    has no offline substitute. Everything else - the LLM key, the vector store
    key, the search key - is only needed if that provider is actually selected.
    """
    s = Settings.from_env(
        {"DATABASE_URL": "postgresql://x/y"}, include_dotenv=False
    )
    assert s.embedding_provider == "local"
    assert s.vector_store == "memory"
    s.validate()  # must not raise


def test_importing_settings_does_not_validate():
    """Construction must never raise; only validate() does.

    This is the specific behaviour that made the reference test suite
    uncollectable: it called ``config.validate()`` at import time.
    """
    s = Settings.from_env({}, include_dotenv=False)  # missing DATABASE_URL
    assert s.required_keys()  # reports the problem...
    # ...but constructing did not raise.


def test_missing_database_url_is_reported():
    s = Settings.from_env({}, include_dotenv=False)
    assert "DATABASE_URL" in s.required_keys()
    with pytest.raises(ConfigurationError):
        s.validate()


def test_local_embeddings_need_no_api_key():
    s = Settings.from_env({"DATABASE_URL": "postgresql://x/y"}, include_dotenv=False)
    assert s.required_keys() == []


def test_hosted_embedding_requires_its_key():
    s = Settings.from_env(
        {"DATABASE_URL": "postgresql://x/y", "EMBEDDING_PROVIDER": "nvidia"},
        include_dotenv=False,
    )
    assert "NVIDIA_API_KEY" in s.required_keys()
    s2 = Settings.from_env(
        {"DATABASE_URL": "postgresql://x/y", "EMBEDDING_PROVIDER": "nvidia", "NVIDIA_API_KEY": "k"},
        include_dotenv=False,
    )
    assert s2.required_keys() == []


def test_openai_provider_requires_openai_key_not_nvidia():
    s = Settings.from_env(
        {"DATABASE_URL": "postgresql://x/y", "EMBEDDING_PROVIDER": "openai"},
        include_dotenv=False,
    )
    assert "OPENAI_API_KEY" in s.required_keys()


def test_pinecone_store_requires_its_api_key():
    s = Settings.from_env(
        {"DATABASE_URL": "postgresql://x/y", "VECTOR_STORE": "pinecone"},
        include_dotenv=False,
    )
    # The index name has a usable default, so only the key is mandatory.
    assert s.required_keys() == ["PINECONE_API_KEY"]


def test_tavily_only_required_when_asked():
    s = Settings.from_env({"DATABASE_URL": "postgresql://x/y"}, include_dotenv=False)
    # Tavily is only needed by the web-search tool, not by configuration that
    # never selects it, so it is absent from the default requirement set.
    assert "TAVILY_API_KEY" not in s.required_keys()
    with pytest.raises(ConfigurationError, match="TAVILY_API_KEY"):
        s.validate(require_tavily=True)


@pytest.mark.parametrize("bad,match", [
    ({"EMBEDDING_PROVIDER": "nope"}, "EMBEDDING_PROVIDER"),
    ({"VECTOR_STORE": "nope"}, "VECTOR_STORE"),
    ({"RETRIEVAL_THRESHOLD": "1.5"}, "RETRIEVAL_THRESHOLD"),
    ({"RETRIEVAL_THRESHOLD": "-0.1"}, "RETRIEVAL_THRESHOLD"),
    ({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}, "CHUNK_OVERLAP"),
    ({"MAX_ITERATIONS": "0"}, "MAX_ITERATIONS"),
])
def test_invalid_values_rejected(bad, match):
    s = Settings.from_env(
        {"DATABASE_URL": "postgresql://x/y", **bad}, include_dotenv=False
    )
    with pytest.raises(ConfigurationError, match=match):
        s.validate()


@pytest.mark.parametrize("key", ["RETRIEVER_K", "CHUNK_SIZE", "MAX_ITERATIONS"])
def test_unparseable_numbers_fail_at_construction(key):
    """A malformed number is caught where it is read, not two calls later."""
    with pytest.raises(ConfigurationError, match=key):
        Settings.from_env({"DATABASE_URL": "postgresql://x/y", key: "abc"}, include_dotenv=False)


def test_settings_are_immutable():
    s = Settings.from_env({}, include_dotenv=False)
    with pytest.raises(Exception):
        s.max_iterations = 99  # type: ignore[misc]


def test_embedding_dimension_per_provider():
    assert Settings.from_env({"EMBEDDING_PROVIDER": "local"}, include_dotenv=False).embedding_dimension == 384
    assert Settings.from_env({"EMBEDDING_PROVIDER": "openai"}, include_dotenv=False).embedding_dimension == 1536
    assert Settings.from_env({"EMBEDDING_PROVIDER": "nvidia"}, include_dotenv=False).embedding_dimension == 1024


def test_values_are_read_per_instance_not_captured_at_import():
    """Two Settings objects can disagree - the old globals could not."""
    a = Settings.from_env({"MAX_ITERATIONS": "1"}, include_dotenv=False)
    b = Settings.from_env({"MAX_ITERATIONS": "9"}, include_dotenv=False)
    assert a.max_iterations == 1
    assert b.max_iterations == 9


# ── Event loop policy ────────────────────────────────────────────────────

def test_event_loop_policy_is_psycopg_compatible():
    import asyncio
    import sys

    name = ensure_compatible_event_loop()
    if sys.platform == "win32":
        assert "Selector" in name
    assert type(asyncio.get_event_loop_policy()).__name__ == name


def test_ensure_event_loop_is_idempotent():
    assert ensure_compatible_event_loop() == ensure_compatible_event_loop()


# ── Structured logging ───────────────────────────────────────────────────

def _render(**kwargs) -> dict:  # type: no[no-untyped-def]
    record = logging.LogRecord("t", logging.INFO, "p", 1, "hello", None, None)
    for k, v in kwargs.items():
        setattr(record, k, v)
    return json.loads(JsonFormatter().format(record))


def test_log_line_is_valid_json_with_core_fields():
    out = _render()
    assert out["message"] == "hello"
    assert out["level"] == "INFO"
    assert out["logger"] == "t"
    assert out["ts"].endswith("Z")


def test_extra_fields_are_included():
    out = _render(route="rag", ms=12)
    assert out["route"] == "rag"
    assert out["ms"] == 12


def test_request_id_is_injected_from_context():
    with request_context("req-123"):
        out = _render()
    assert out["request_id"] == "req-123"


def test_thread_id_is_injected_from_context():
    with request_context(thread_id="t-9"):
        out = _render()
    assert out["thread_id"] == "t-9"


def test_context_is_cleared_on_exit():
    with request_context("req-1"):
        pass
    assert request_id_var.get() == "-"
    assert thread_id_var.get() == "-"


def test_request_ids_are_unique():
    assert new_request_id() != new_request_id()


def test_request_context_yields_the_id():
    with request_context() as rid:
        assert rid == request_id_var.get()


def test_nested_contexts_restore_outer():
    with request_context("outer"):
        with request_context("inner"):
            assert request_id_var.get() == "inner"
        assert request_id_var.get() == "outer"


def test_log_event_emits_structured_fields(caplog):
    logger = get_logger("t.structured")
    with caplog.at_level(logging.INFO):
        log_event(logger, logging.INFO, "tool_call", tool="RAG", ms=7)
    record = caplog.records[-1]
    assert record.tool == "RAG"
    assert record.ms == 7


# ── HTTP layer ───────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Synchronous test client.

    Uses ``fastapi.testclient.TestClient``, which drives the ASGI app through
    a thread and therefore works with the installed httpx version. A bare
    ``httpx.Client`` over ``ASGITransport`` does not, because the transport is
    async-only.
    """
    from fastapi.testclient import TestClient

    settings = Settings.from_env(
        {"DATABASE_URL": "postgresql://unused/unused"}, include_dotenv=False
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


def test_health_reports_configuration(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "384-dim" in body["embeddings"]


def test_response_carries_request_id_header(client):
    r = client.get("/health")
    assert r.headers.get("X-Request-ID")


def test_incoming_request_id_is_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "caller-supplied-id"})
    assert r.headers.get("X-Request-ID") == "caller-supplied-id"


def test_blank_query_rejected(client):
    r = client.post("/chat", json={"query": "   "})
    assert r.status_code == 422


def test_oversized_query_rejected(client):
    """The reference project bounded only image size, not query length (L2)."""
    r = client.post("/chat", json={"query": "x" * 5000})
    assert r.status_code == 422


def test_malicious_thread_id_rejected(client):
    r = client.post("/chat", json={"query": "hi", "thread_id": "'; DROP TABLE--"})
    assert r.status_code == 422


def test_invalid_base64_image_rejected(client):
    r = client.post("/chat", json={"query": "hi", "image_base64": "!!!not base64!!!"})
    assert r.status_code == 422


def test_openapi_is_generated(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/chat" in r.json()["paths"]


def test_app_factory_does_not_validate_at_import():
    """create_app must not require credentials or reach the network."""
    application = create_app(Settings.from_env({}, include_dotenv=False))
    assert isinstance(application, FastAPI)
