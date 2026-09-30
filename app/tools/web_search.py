"""Hardened web search tool.

The reference project's entire tool was one line::

    search_web = TavilySearch(max_results=5, tavily_api_key=config.TAVILY_API_KEY)

with no timeout, no retry, no empty-result normalisation, and no error
handling (audit L17). A Tavily 429 or a malformed payload propagated into
``ToolNode`` and became a ``ToolMessage`` containing raw library error text,
which the model then summarised to the user.

This wrapper normalises results into a predictable structure, degrades to an
explicit "search unavailable" message rather than an exception string, and
caps result length so a large payload cannot dominate the context window.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

from app.core.logging import get_logger, log_event

logger = get_logger(__name__)

MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 400
# Total characters of search context handed to the model.
MAX_TOTAL_CHARS = 4_000


def _get_api_key() -> str | None:
    return os.getenv("TAVILY_API_KEY") or None


def _format(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    total = 0
    for i, item in enumerate(results, start=1):
        title = str(item.get("title") or "Untitled").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS].rstrip() + "..."

        block = f"[{i}] {title}\nURL: {url}\n{snippet}".strip()
        if total + len(block) > MAX_TOTAL_CHARS:
            lines.append(f"[{i}] (truncated: result budget exhausted)")
            break
        lines.append(block)
        total += len(block)

    if not lines:
        return "No results were returned by the search provider."
    return "\n\n".join(lines)


@tool
def search_web(query: str, max_results: int = MAX_RESULTS) -> str:
    """Search the public web for current or external information.

    Use for news, live prices, weather, and third-party documentation.
    Returns numbered results with titles, URLs, and snippets.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return "No search performed: the query was empty."

    api_key = _get_api_key()
    if not api_key:
        log_event(logger, 30, "web_search_no_key")
        return (
            "Web search is not configured on this deployment, so current "
            "information could not be retrieved. Answer from your own "
            "knowledge and tell the user the result may be out of date."
        )

    try:
        from langchain_tavily import TavilySearch

        search = TavilySearch(max_results=max(1, min(max_results, 10)), tavily_api_key=api_key)
        raw = search.invoke({"query": cleaned})
    except Exception as exc:
        # Do not interpolate the exception: it can carry the API key, the
        # request URL, and provider internals into the model's context.
        log_event(
            logger,
            40,
            "web_search_failed",
            error_type=type(exc).__name__,
            query_len=len(cleaned),
        )
        return (
            "Web search failed and returned no results. Do not present any "
            "search-derived facts; tell the user the search could not be "
            "completed."
        )

    results = _normalise(raw)
    log_event(logger, 20, "web_search_ok", result_count=len(results), query_len=len(cleaned))
    return _format(results)


def _normalise(raw: Any) -> list[dict[str, Any]]:
    """Coerce whatever the provider returned into a list of dicts.

    The integration has historically returned both a bare list and a dict with
    a ``results`` key; normalising here means the formatter has one shape to
    handle and a provider change cannot crash the tool.
    """
    if isinstance(raw, dict):
        for key in ("results", "answer", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []
