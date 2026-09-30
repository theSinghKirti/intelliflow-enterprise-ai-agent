"""LangGraph graph assembly.

Shape
-----
::

    START
      │
      ▼
   ┌─────────┐  route in {rag, web_search, database, python}?  yes
   │ router  │──────────────────────────────────────────┐
   └────┬────┘                                          │
        │ no                                            ▼
        │                                     ┌──────────────┐
        │                                     │  invoke_tool │  picks the named tool,
        │                                     └──────┬───────┘  records call + result
        │                                            │
        │            under the iteration cap ────────┤
        │                                            ▼
        │                                     ┌──────────────┐
        │◄────────────────────────────────────│    reason    │  loops while the model
        │                                     └──────┬───────┘  still requests tools
        │  cap reached, or no tool requested      │
        ▼                                         │
   ┌──────────────────────────────────────────────┘
   │ answer  ──▶ END
   └─────────

Differences from the reference graph
------------------------------------
- The ``llm_tool`` node is gone. The reference version re-bound tools and
  issued a *second* LLM call purely to force a tool choice, then a third to
  execute it, then a fourth to answer. Here the router's category selects the
  tool directly, and the model reasons once in ``reason`` before answering.
  Fewer calls, and the tool choice is inspectable in state.
- The tool node runs the named tool inline so its structured output
  (citations, grounding verdict) lands in state.
- The iteration cap is enforced in ``after_tools`` and logs when it fires,
  rather than silently truncating.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.answer import answer_node
from app.agents.conditions import after_tools, route_condition
from app.agents.router import router_node
from app.agents.state import AgentState
from app.agents.tools import collect_tools, execute_tools
from app.core.config import Settings
from app.core.logging import get_logger, log_event

logger = get_logger(__name__)

_TOOL_ROUTES = {"rag", "web_search", "database", "python"}


def _make_router(settings: Settings, llm: Any):
    async def _router(state: AgentState) -> dict[str, Any]:
        return await router_node(state, settings=settings, llm=llm)

    return _router


def _make_answer(llm: Any):
    async def _answer(state: AgentState) -> dict[str, Any]:
        return await answer_node(state, llm=llm)

    return _answer


def _make_tool_runner(settings: Settings, tools: list[Any]):
    async def _run(state: AgentState) -> dict[str, Any]:
        if (state.get("route") or "").lower() not in _TOOL_ROUTES:
            return {}
        return await execute_tools(state, tools)

    return _run


def _make_reasoner(llm: Any, tools: list[Any]):
    """One LLM call that either requests another tool or produces the answer.

    The system prompt makes the model choose: call a tool if it needs one,
    otherwise reply directly. Whatever it produces becomes the last message,
    and ``after_tools`` decides where to go from there.
    """
    from langchain_core.messages import SystemMessage

    prompt = SystemMessage(
        content=(
            "You are an assistant with tools. If answering the user's request "
            "requires calling a tool, call exactly one now. If you can answer "
            "without a tool, reply directly and concisely. Use a tool when the "
            "information is not something you already know."
        )
    )
    bound = llm.bind_tools(tools)

    async def _reason(state: AgentState) -> dict[str, Any]:
        response = await bound.ainvoke([prompt, *state.get("messages", [])[-8:]])
        return {
            "messages": [response],
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
        }

    return _reason


async def build_graph(settings: Settings, llm: Any) -> StateGraph:
    """Assemble the agent graph with the given settings and model."""
    tools = await collect_tools(settings)
    tool_names = [getattr(t, "name", str(t)) for t in tools]
    log_event(logger, 20, "graph_build", tools=tool_names)

    graph = StateGraph(AgentState)

    graph.add_node("router", _make_router(settings, llm))
    graph.add_node("invoke_tool", _make_tool_runner(settings, tools))
    graph.add_node("reason", _make_reasoner(llm, tools))
    graph.add_node("answer", _make_answer(llm))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router", route_condition, {"tools": "invoke_tool", "answer": "answer"}
    )
    graph.add_edge("invoke_tool", "reason")
    graph.add_conditional_edges(
        "reason",
        lambda state: after_tools(state, settings),
        {"tools": "invoke_tool", "answer": "answer"},
    )
    graph.add_edge("answer", END)

    return graph


__all__ = ["build_graph", "after_tools", "route_condition"]
