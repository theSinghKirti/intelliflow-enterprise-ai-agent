"""Structured router.

The reference project's router produced a binary decision::

    class RouterDecision(BaseModel):
        decision: Literal["tools", "answer"]
        reasoning: str

and then swallowed every exception to fall back to ``"answer"`` (audit S4).
Two problems with that:

1. **Not inspectable.** Which tool category was chosen was left to the LLM in
   the next node, so the routing decision could never be asserted on, counted,
   or evaluated. The test suite parameterised an ``expected_type`` and then
   never asserted it (audit S3).
2. **Fail-silent.** A router failure meant the user got a hallucinated direct
   answer instead of an error, with nothing in the logs.

This router emits one of six categories with a reason and a confidence, and it
fails *loudly but safely*: an unusable classification falls back to
``direct`` with confidence 0.0 and an explicit reason, which is recorded in
state and logged, rather than being silently discarded.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.state import ROUTE_CATEGORIES, AgentState
from app.core.config import Settings
from app.core.logging import get_logger, log_event

logger = get_logger(__name__)

ROUTER_SYSTEM_PROMPT = """\
You are a request classifier. Choose exactly one route for the user's message.

Routes:
- rag          Questions about the user's own ingested documents, policies,
               handbooks, or internal reference material ("what does our
               refund policy say", "summarise the handbook").
- web_search   Needs information from the public internet that could be
               current or external: news, live prices, today's weather,
               documentation for a third-party library, "latest X".
- database     Questions answerable by querying the business database:
               customers, orders, products, employees, sales figures,
               aggregates and counts ("how many orders last quarter").
- python       Needs computation, maths, or data manipulation over a result
               ("average of x and y", "convert 500 USD to EUR", statistics).
- direct       General knowledge, definitions, explanations, reasoning, or
               casual conversation that needs no tool.
- clarify      The message is too vague or too ambiguous to route confidently.

Return JSON only. Choose `direct` when the query is answerable without any
tool. Prefer `clarify` over guessing when the intent is genuinely unclear.
"""


class RouterDecision(BaseModel):
    """Structured routing output.

    Extends the reference `RouterDecision` with an explicit category and a
    confidence score. `confidence` exists so low-confidence decisions can be
    reported and evaluated instead of being treated as authoritative.
    """

    route: str = Field(
        description=f"One of: {', '.join(ROUTE_CATEGORIES)}"
    )
    reason: str = Field(
        description="One short sentence explaining the choice.",
        max_length=300,
    )
    confidence: float = Field(
        description="Confidence in the route, between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )

    def normalized_route(self) -> str:
        """Route, coerced into the known set.

        The model occasionally returns a near-miss (``"web"``, ``"RAG"``).
        Rather than trusting the raw string, map what can be mapped and treat
        anything unrecognised as a direct answer.
        """
        value = (self.route or "").strip().lower()
        if value in ROUTE_CATEGORIES:
            return value
        aliases = {
            "web": "web_search",
            "websearch": "web_search",
            "search": "web_search",
            "documents": "rag",
            "knowledge_base": "rag",
            "sql": "database",
            "db": "database",
            "code": "python",
            "python_tool": "python",
            "chat": "direct",
            "llm": "direct",
            "none": "direct",
        }
        return aliases.get(value, "direct")


def _fallback(query: str, reason: str) -> RouterDecision:
    return RouterDecision(route="direct", reason=reason, confidence=0.0)


def classify(query: str, llm: Any, settings: Settings) -> RouterDecision:
    """Classify a query into one route category.

    Never raises. A router that cannot decide returns a low-confidence
    `direct` with the failure recorded in `reason`, so the caller always has a
    usable decision and the degradation is visible in logs and state.
    """
    if not query or not query.strip():
        return _fallback(query, "Empty query; defaulting to direct.")

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    try:
        structured = llm.with_structured_output(RouterDecision)
        decision = structured.invoke(messages)
    except Exception as exc:
        # Deliberately not swallowed: the reference version returned
        # "answer" with no record, so a classifier outage was indistinguishable
        # from a normal direct answer. Log it and mark confidence 0.0.
        log_event(
            logger,
            30,
            "router_fallback",
            error_type=type(exc).__name__,
            query_len=len(query),
        )
        return _fallback(
            query, f"Router classification failed ({type(exc).__name__}); defaulting to direct."
        )

    if not isinstance(decision, RouterDecision):
        return _fallback(query, f"Router returned unexpected type {type(decision).__name__}.")

    normalized = decision.normalized_route()
    if normalized != decision.route.strip().lower():
        log_event(
            logger,
            30,
            "router_route_normalized",
            raw=decision.route,
            normalized=normalized,
        )
    return RouterDecision(
        route=normalized, reason=decision.reason, confidence=decision.confidence
    )


async def router_node(state: AgentState, *, settings: Settings, llm: Any) -> dict[str, Any]:
    """LangGraph node wrapper around :func:`classify`."""
    query = state.get("query") or _latest_user_text(state)
    decision = classify(query, llm, settings)

    log_event(
        logger,
        20,
        "route_selected",
        route=decision.route,
        confidence=decision.confidence,
        reason=decision.reason,
    )

    return {
        "route": decision.route,
        "route_reason": decision.reason,
        "route_confidence": decision.confidence,
        "iteration_count": 0,
        "query": query,
    }


def _latest_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return ""


__all__ = [
    "ROUTER_SYSTEM_PROMPT",
    "RouterDecision",
    "classify",
    "router_node",
]
