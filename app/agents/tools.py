"""Tool execution and the tools exposed to the agent.

Retains the reference project's LangGraph ``ToolNode`` execution model, and
adds the structured result recording the new state expects.

Tool inventory
--------------
``rag``
    Grounded retrieval over ingested documents (citations preserved).
``search_web``
    Hardened Tavily wrapper with timeouts, result normalisation, and
    sanitised failures.
``run_python``
    Optional. Discovered from a configured MCP server. The reference project
    hard-coded a third party's free-tier Render deployment, which meant the
    project could not run standalone and the first tool call could hit a
    multi-second cold start (audit L19). Here the server is optional: with no
    ``MCP_SERVER_URL`` the tool is simply absent rather than the graph failing
    to build.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from app.agents.state import AgentState
from app.core.config import Settings
from app.core.logging import get_logger, log_event
from app.rag.retriever import retrieve
from app.tools.rag_tool import RAG
from app.tools.web_search import search_web

logger = get_logger(__name__)


async def mcp_tools(settings: Settings) -> list[Any]:
    """Discover tools from the configured MCP server, if any.

    Returns an empty list when no server is configured, and on discovery
    failure. Degrading to "no Python tool" is strictly better than raising:
    the agent can still answer with RAG, web search, and its own knowledge,
    whereas a build failure takes down the whole endpoint.
    """
    if not settings.mcp_server_url:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "python_tool": {
                    "transport": "streamable-http",
                    "url": settings.mcp_server_url,
                }
            }
        )
        discovered = await client.get_tools()
    except Exception as exc:
        log_event(logger, 30, "mcp_discovery_failed", error_type=type(exc).__name__)
        return []

    names = [getattr(t, "name", str(t)) for t in discovered]
    log_event(logger, 20, "mcp_discovery_ok", tools=names)
    return list(discovered)


async def collect_tools(settings: Settings) -> list[Any]:
    """Assemble the full tool list for this deployment."""
    tools: list[Any] = [RAG, search_web]
    tools.extend(await mcp_tools(settings))
    return tools


def build_tool_node(tools: list[Any]) -> ToolNode:
    return ToolNode(tools)


async def execute_tools(state: AgentState, tools: list[Any]) -> dict[str, Any]:
    """Run the tool calls recorded on the last AI message.

    Returns state updates: the ``ToolMessage`` results, the structured
    ``tool_calls``/``tool_results`` records, and any ``sources``/``errors``
    the tools produced. RAG runs inline rather than through ``ToolNode`` so
    its citations and grounding verdict can be captured as typed state.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    iteration = int(state.get("iteration_count", 0))
    tool_messages: list[ToolMessage] = []
    records_calls: list[dict[str, Any]] = []
    records_results: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[str] = []

    for call in last.tool_calls:
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        call_id = call.get("id", "")

        records_calls.append({"name": name, "args": args, "iteration": iteration})

        if name == "RAG":
            started = time.perf_counter()
            result = retrieve(args.get("query", ""))
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            if result.has_evidence:
                for chunk in result.chunks:
                    sources.append(
                        {
                            "filename": chunk.filename,
                            "page_number": chunk.page_number,
                            "chunk_id": chunk.chunk_id,
                            "score": chunk.score,
                        }
                    )
                content = result.context_block()
                ok = True
            else:
                content = (
                    "<retrieved_context>\nNo relevant information was found "
                    "in the knowledge base for this query.\n</retrieved_context>"
                )
                ok = False

            records_results.append(
                {
                    "name": name,
                    "ok": ok,
                    "summary": result.reason,
                    "citations": result.citations(),
                    "score": result.best_score,
                }
            )
            log_event(
                logger,
                20,
                "tool_call",
                tool=name,
                ms=elapsed_ms,
                ok=ok,
                best_score=result.best_score,
                sources=len(sources),
            )
        else:
            node = build_tool_node([t for t in tools if getattr(t, "name", None) == name])
            started = time.perf_counter()
            try:
                outputs = await node.ainvoke({"messages": [last]})
                content = _stringify(outputs["messages"][0].content)
                ok = True
            except Exception as exc:
                log_event(
                    logger,
                    40,
                    "tool_call_failed",
                    tool=name,
                    error_type=type(exc).__name__,
                )
                content = (
                    f"The {name} tool failed ({type(exc).__name__}). Do not "
                    "present results from it."
                )
                ok = False
                errors.append(f"{name}: {type(exc).__name__}")
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            records_results.append(
                {
                    "name": name,
                    "ok": ok,
                    "summary": f"{elapsed_ms}ms",
                    "citations": [],
                    "score": None,
                }
            )
            log_event(logger, 20, "tool_call", tool=name, ms=elapsed_ms, ok=ok)

        tool_messages.append(
            ToolMessage(content=content, name=name, tool_call_id=call_id)
        )

    update: dict[str, Any] = {
        "messages": tool_messages,
        "tool_calls": records_calls,
        "tool_results": records_results,
    }
    if sources:
        update["sources"] = sources
    if errors:
        update["errors"] = errors

    rag_results = [r for r in records_results if r["name"] == "RAG"]
    if rag_results:
        update["grounded"] = any(r["ok"] for r in rag_results)
        update["insufficient_information"] = not any(r["ok"] for r in rag_results)

    return update


def _stringify(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content).strip()
