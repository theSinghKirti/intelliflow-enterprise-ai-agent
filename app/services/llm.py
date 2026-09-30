"""LLM service construction.

The reference project constructed three ``ChatOpenAI`` clients at module import
time in ``llms.py``, using eager module-level config. That meant importing the
module built network clients whether or not anything would use them, and the
process could not run with different credentials per test.

This module builds the client on demand from a :class:`Settings` instance, so a
test can pass a stub model and production can pass real settings. Importing the
module has no side effect and requires no credential.

The reference project also declared a separate vision client for image turns.
This build has no vision path: ``ChatRequest.image_base64`` is validated for
size and base64 well-formedness but is not routed to a model, so no vision
client is constructed here.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import ConfigurationError, Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _cached_llm(
    api_key: str, base_url: str, model: str, temperature: float
) -> BaseChatModel:
    return ChatOpenAI(
        base_url=base_url, api_key=api_key, model=model, temperature=temperature
    )


def get_llm(settings: Settings, *, temperature: float = 0.0) -> BaseChatModel:
    """Return a cached chat model for the given settings.

    Caching keyed on the credential tuple avoids rebuilding a client (and its
    underlying HTTP connection pool) on every node visit.

    Raises
    ------
    ConfigurationError
        If no ``NVIDIA_API_KEY`` is configured. Callers translate this into a
        ``503`` rather than letting it surface as a ``500``.
    """
    if not settings.nvidia_api_key:
        raise ConfigurationError(
            "NVIDIA_API_KEY is required to build an LLM. Set it in the "
            "environment or in .env, or inject a stub model in tests."
        )
    return _cached_llm(
        settings.nvidia_api_key,
        settings.nvidia_base_url,
        settings.nvidia_model,
        temperature,
    )
