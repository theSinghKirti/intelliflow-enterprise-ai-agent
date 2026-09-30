"""Typed application settings.

Replaces the reference project's module-level mutable globals in ``config.py``.

Why this exists
---------------
The reference version read every environment variable at import time into
module-level names, then called ``config.validate()`` at the *top* of
``main.py``. Two problems followed (audit L6, L7):

1. ``import main`` raised ``EnvironmentError`` unless four hosted credentials
   were present, so the test suite could not even be collected without
   production secrets.
2. Because the values were bound at import, no test could substitute them, and
   no deployment could share a process across environments.

:class:`Settings` reads lazily from a mapping (defaulting to the process
environment) and validates only what the *selected* providers actually need.
The default local-embedding path needs no API key at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

# Providers that require a hosted credential.
EMBEDDING_PROVIDERS = ("local", "nvidia", "openai")
VECTOR_STORES = ("memory", "pinecone")


class ConfigurationError(RuntimeError):
    """Raised when settings are missing or internally inconsistent."""


def _load_dotenv_if_present() -> dict[str, str]:
    """Parse a ``.env`` file into a plain dict.

    Implemented directly rather than via ``python-dotenv`` so that importing
    settings has no import-time side effect on the process environment.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    parsed: dict[str, str] = {}
    if not env_path.is_file():
        return parsed
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        parsed[key] = value
    return parsed


# Cache for the parsed .env file. ``None`` means "not loaded yet"; an empty
# dict means "explicitly disabled" (used by hermetic tests).
_DOTENV: dict[str, str] | None = None


def _dotenv_values() -> dict[str, str]:
    global _DOTENV
    if _DOTENV is None:
        _DOTENV = _load_dotenv_if_present()
    return _DOTENV


def _set_dotenv_override(values: dict[str, str] | None) -> None:
    """Swap the cached ``.env`` values; ``None`` restores lazy loading."""
    global _DOTENV
    _DOTENV = values


def _get(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    """Prefer the real environment; fall back to ``.env``."""
    if key in env and env[key] != "":
        return env[key]
    file_env = _dotenv_values()
    if key in file_env and file_env[key] != "":
        return file_env[key]
    return default


def _as_int(value: str | None, default: int, *, name: str) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}") from exc


def _as_float(value: str | None, default: float, *, name: str) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {value!r}") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable snapshot of configuration.

    Construct once and pass explicitly, rather than importing mutable globals.
    ``from_env()`` is the normal entry point.
    """

    # LLM (NVIDIA NIM)
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "openai/gpt-oss-20b"

    # Web search
    tavily_api_key: str | None = None

    # Vector store
    pinecone_api_key: str | None = None
    pinecone_index: str = "vector"

    # Memory
    database_url: str | None = None

    # MCP
    mcp_server_url: str = ""

    # Embeddings / RAG
    embedding_provider: str = "local"
    embedding_model: str | None = None
    vector_store: str = "memory"
    retrieval_threshold: float = 0.30
    retriever_k: int = 5
    retriever_fetch_k: int = 20
    chunk_size: int = 400
    chunk_overlap: int = 50

    # Agent
    max_iterations: int = 3

    # API
    max_query_chars: int = 4000
    max_image_base64_chars: int = 5_000_000
    default_thread_id: str = "1"

    # Observability
    langsmith_project: str = "intelliflow"
    langsmith_tracing: bool = False
    log_level: str = "INFO"

    _env: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        include_dotenv: bool = True,
    ) -> "Settings":
        """Build settings.

        Parameters
        ----------
        env:
            Source of configuration. Defaults to ``os.environ``. Passing an
            explicit mapping is how tests get hermetic, credential-free
            settings.
        include_dotenv:
            Whether a repository ``.env`` may supply values missing from
            ``env``. Left on for normal runs; tests pass ``False`` so a
            developer's real ``.env`` cannot leak into an assertion.
        """
        previous = _dotenv_values()
        if not include_dotenv:
            _set_dotenv_override({})
        try:
            return cls._build(os.environ if env is None else env)
        finally:
            _set_dotenv_override(previous)

    @classmethod
    def _build(cls, env: Mapping[str, str]) -> "Settings":
        return cls(
            nvidia_api_key=_get(env, "NVIDIA_API_KEY"),
            nvidia_base_url=_get(
                env, "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
            ) or "https://integrate.api.nvidia.com/v1",
            nvidia_model=_get(env, "NVIDIA_MODEL", "openai/gpt-oss-20b")
            or "openai/gpt-oss-20b",
            tavily_api_key=_get(env, "TAVILY_API_KEY"),
            pinecone_api_key=_get(env, "PINECONE_API_KEY"),
            pinecone_index=_get(env, "PINECONE_INDEX", "vector") or "vector",
            database_url=_get(env, "DATABASE_URL"),
            mcp_server_url=_get(env, "MCP_SERVER_URL", "") or "",
            embedding_provider=(_get(env, "EMBEDDING_PROVIDER", "local") or "local").lower(),
            embedding_model=_get(env, "EMBEDDING_MODEL"),
            vector_store=(_get(env, "VECTOR_STORE", "memory") or "memory").lower(),
            retrieval_threshold=_as_float(
                _get(env, "RETRIEVAL_THRESHOLD"), 0.30, name="RETRIEVAL_THRESHOLD"
            ),
            retriever_k=_as_int(_get(env, "RETRIEVER_K"), 5, name="RETRIEVER_K"),
            retriever_fetch_k=_as_int(
                _get(env, "RETRIEVER_FETCH_K"), 20, name="RETRIEVER_FETCH_K"
            ),
            chunk_size=_as_int(_get(env, "CHUNK_SIZE"), 400, name="CHUNK_SIZE"),
            chunk_overlap=_as_int(
                _get(env, "CHUNK_OVERLAP"), 50, name="CHUNK_OVERLAP"
            ),
            max_iterations=_as_int(_get(env, "MAX_ITERATIONS"), 3, name="MAX_ITERATIONS"),
            max_query_chars=_as_int(
                _get(env, "MAX_QUERY_CHARS"), 4000, name="MAX_QUERY_CHARS"
            ),
            max_image_base64_chars=_as_int(
                _get(env, "MAX_IMAGE_BASE64_CHARS"),
                5_000_000,
                name="MAX_IMAGE_BASE64_CHARS",
            ),
            default_thread_id=_get(env, "DEFAULT_THREAD_ID", "1") or "1",
            langsmith_project=_get(env, "LANGCHAIN_PROJECT", "intelliflow")
            or "intelliflow",
            langsmith_tracing=(_get(env, "LANGCHAIN_TRACING_V2", "false") or "false").lower()
            == "true",
            log_level=(_get(env, "LOG_LEVEL", "INFO") or "INFO").upper(),
            _env=env,
        )

    # ── Validation ────────────────────────────────────────────────────────

    def required_keys(self) -> list[str]:
        """Names of missing variables for the *selected* providers only."""
        missing: list[str] = []

        if self.database_url is None:
            missing.append("DATABASE_URL")

        if self.embedding_provider != "local" and not self.nvidia_api_key:
            if self.embedding_provider == "openai":
                if not _get(self._env, "OPENAI_API_KEY"):
                    missing.append("OPENAI_API_KEY")
            else:
                missing.append("NVIDIA_API_KEY")

        if self.vector_store == "pinecone":
            if not self.pinecone_api_key:
                missing.append("PINECONE_API_KEY")
            if not self.pinecone_index:
                missing.append("PINECONE_INDEX")

        return missing

    def validate(self, *, require_tavily: bool = False) -> "Settings":
        """Raise if configuration is unusable. Returns ``self`` to allow chaining."""
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise ConfigurationError(
                f"EMBEDDING_PROVIDER={self.embedding_provider!r} is not supported. "
                f"Expected one of: {', '.join(EMBEDDING_PROVIDERS)}."
            )
        if self.vector_store not in VECTOR_STORES:
            raise ConfigurationError(
                f"VECTOR_STORE={self.vector_store!r} is not supported. "
                f"Expected one of: {', '.join(VECTOR_STORES)}."
            )
        if not 0.0 <= self.retrieval_threshold <= 1.0:
            raise ConfigurationError(
                f"RETRIEVAL_THRESHOLD must be within [0.0, 1.0], "
                f"got {self.retrieval_threshold}."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})."
            )
        if self.max_iterations < 1:
            raise ConfigurationError(
                f"MAX_ITERATIONS must be at least 1, got {self.max_iterations}."
            )

        missing = self.required_keys()
        if require_tavily and not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")

        if missing:
            raise ConfigurationError(
                f"Missing required configuration: {sorted(set(missing))}. "
                "Copy .env.example to .env and fill in the values."
            )
        return self

    # ── Derived views ─────────────────────────────────────────────────────

    @property
    def requires_network_llm(self) -> bool:
        """True when the agent needs a hosted LLM credential to answer."""
        return True

    @property
    def embedding_dimension(self) -> int:
        return {"local": 384, "nvidia": 1024, "openai": 1536}[self.embedding_provider]
