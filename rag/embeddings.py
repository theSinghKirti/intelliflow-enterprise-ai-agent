"""Embedding provider factory.

This is the *only* module permitted to import a provider-specific embedding
client. Everything else asks :func:`get_embeddings` and receives a
``langchain_core.embeddings.Embeddings`` implementation.

Why a factory
-------------
The reference project had a single hard-coded ``MockEmbeddings`` in
``llms.py`` that returned all-zero vectors, wired directly into the retrieval
path (audit P1). Because it was a module-level singleton, swapping it meant
editing application code, and there was no way to run a test against real
semantics.

Providers
---------
``local`` (default)
    ``sentence-transformers/all-MiniLM-L6-v2`` via ``langchain-huggingface``.
    384 dimensions, ~90 MB, runs on CPU, **no API key required**. This is the
    default because the test suite must be runnable with zero credentials.

``nvidia``
    NVIDIA NIM embeddings over the OpenAI-compatible endpoint. Hosted, billed
    per token, needs ``NVIDIA_API_KEY``.

``openai``
    OpenAI ``text-embedding-3-small``. Hosted, needs ``OPENAI_API_KEY``.

Dimensionality warning
----------------------
The vector dimension differs per provider, and a vector store index is built
for exactly one dimension. The reference project's placeholder emitted
1536-dim vectors to match ``text-embedding-3-small``. The ``local`` default
emits **384**-dim vectors. Reusing a 1536-dim index with the local provider
raises a dimension error from the store; the reverse — a 384-dim index with a
hosted provider — is the more dangerous direction, because some stores truncate
instead of rejecting. :func:`get_embedding_dimension` exists so callers can
assert against the configured index rather than discover this at query time.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

# Canonical dimensions for the supported providers. Used for validation and
# surfaced in diagnostics; the real dimension is read back from the model.
PROVIDER_DIMENSIONS: dict[str, int] = {
    "local": 384,
    "nvidia": 1024,
    "openai": 1536,
}

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

VALID_PROVIDERS = tuple(PROVIDER_DIMENSIONS)


class EmbeddingConfigurationError(RuntimeError):
    """Raised when the embedding provider is misconfigured."""


def get_provider() -> str:
    """Return the configured provider name, defaulting to ``local``."""
    return os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()


def get_embedding_dimension() -> int:
    """Nominal vector dimension for the configured provider.

    Cheap: no model is loaded. Callers use this to validate a vector store
    index before the first query rather than after a failed one.
    """
    provider = get_provider()
    try:
        return PROVIDER_DIMENSIONS[provider]
    except KeyError:
        raise EmbeddingConfigurationError(
            f"Unknown EMBEDDING_PROVIDER {provider!r}. "
            f"Expected one of: {', '.join(VALID_PROVIDERS)}."
        ) from None


@lru_cache(maxsize=None)
def get_embeddings() -> Embeddings:
    """Build (and cache) the configured embedding implementation.

    Lazy on purpose: constructing the local model costs ~20 s on first call
    and the cached model still re-reads the hub. Doing that at import time
    would make ``import main`` take tens of seconds, which is the same class
    of import-time side effect the audit flagged at L6.
    """
    provider = get_provider()

    if provider == "local":
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_LOCAL_MODEL)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise EmbeddingConfigurationError(
                "EMBEDDING_PROVIDER=local requires 'langchain-huggingface' "
                "and 'sentence-transformers'. Install with: "
                "pip install langchain-huggingface sentence-transformers"
            ) from exc
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise EmbeddingConfigurationError(
                "EMBEDDING_PROVIDER=nvidia requires NVIDIA_API_KEY."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", DEFAULT_NVIDIA_EMBEDDING_MODEL),
            base_url=os.getenv(
                "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
            ),
            api_key=api_key,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingConfigurationError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL),
            api_key=api_key,
        )

    raise EmbeddingConfigurationError(
        f"Unknown EMBEDDING_PROVIDER {provider!r}. "
        f"Expected one of: {', '.join(VALID_PROVIDERS)}."
    )


def describe() -> dict[str, object]:
    """Diagnostics for the configured provider. Loads no model."""
    provider = get_provider()
    return {
        "provider": provider,
        "dimension": get_embedding_dimension(),
        "model": os.getenv("EMBEDDING_MODEL") or {
            "local": DEFAULT_LOCAL_MODEL,
            "nvidia": DEFAULT_NVIDIA_EMBEDDING_MODEL,
            "openai": DEFAULT_OPENAI_EMBEDDING_MODEL,
        }.get(provider, "unknown"),
        "requires_api_key": provider != "local",
    }
