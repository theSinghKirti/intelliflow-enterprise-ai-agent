"""Edge conditions for the agent graph.

Retains the reference project's bounded tool loop, which was the one piece of
its architecture genuinely worth keeping (audit §12). An unbounded
``tools -> llm_tool -> tools`` cycle lets a model call tools forever; the
iteration cap converts that into a graceful fall-through to the answer node.

The reference implementation also had a routing condition that inspected
``state["router_decision"]``. Routing here is six-way, driven by
``state["route"]``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.state import AgentState
from app.core.logging import get_logger, log_event

logger = get_logger(__name__)


def route_condition(state: AgentState) -> str:
    """Decide the first hop: a tool node, or straight to the answer.

    ``clarify`` and ``direct`` both go straight to ``answer``. The answer
    prompt handles ``clarify`` by asking the user for detail.
    """
    route = (state.get("route") or "direct").lower()
    if route in ("rag", "web_search", "database", "python"):
        return "tools"
    return "answer"


def after_tools(state: AgentState, settings: Any) -> str:
    """Loop back for another tool call, or produce the answer.

    Loops while under the iteration cap *and* the model is still requesting
    tools. Once the cap is reached the answer node runs with whatever evidence
    has accumulated, so the user gets a response instead of a hang.
    """
    messages = state.get("messages", [])
    last = messages[-1] if messages else None

    if isinstance(last, ToolMessage):
        # Tool results are in. Only continue if the model has already been
        # asked to reason about them; otherwise go to the answer node.
        return "answer"

    iteration = int(state.get("iteration_count", 0))
    if iteration >= settings.max_iterations:
        log_event(
            logger,
            30,
            "tool_loop_capped",
            iteration=iteration,
            max_iterations=settings.max_iterations,
        )
        return "answer"

    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"

    return "answer"
