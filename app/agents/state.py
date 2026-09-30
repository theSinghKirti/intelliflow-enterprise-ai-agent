"""Agent state schema.

Extends the reference project's four-field ``State`` TypedDict.

What changed and why
--------------------
The reference state (``state.py``) held four fields::

    messages, router_decision, reasoning, iteration_count

That is enough to run a tool loop, but it discards almost everything worth
inspecting afterwards:

- **Which tool ran.** ``router_decision`` is only ``"tools"`` or ``"answer"``;
  the specific tool is buried in the message history. You cannot answer "how
  often is the database tool used?" from a checkpoint.
- **What came back.** Tool output existed only as ``ToolMessage`` text. With
  the flattened string the reference RAG tool returned, the citation metadata
  (filename, page, chunk) was already gone by this point.
- **Whether the answer was grounded.** Nothing recorded that retrieval had
  failed and the model answered anyway.
- **Errors.** A tool that raised was invisible; the model just saw whatever
  string the tool chose to return.

The state below keeps the reference reducer semantics for ``messages``
(``add_messages``, which correctly appends and merges by id) and adds typed,
inspectable fields. Every list field uses a reducer so parallel fan-out does
not lose writes, which last-write-wins would.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Route categories. The reference router was binary (``tools`` / ``answer``).
# The six categories below make the decision structured and inspectable, which
# is what makes routing testable and measurable rather than a prompt-level hint.
#
# This tuple is the single source of truth for the route set. The router
# normalises the model's answer against it, and the graph branches on it.
ROUTE_CATEGORIES: tuple[str, ...] = (
    "rag",         # question about ingested documents
    "web_search",  # needs current/external information
    "database",    # question about the business database
    "python",      # needs computation or data analysis
    "direct",      # answer from the model's own knowledge
    "clarify",     # too ambiguous to route; ask the user
)


class ToolCallRecord(TypedDict):
    """One observed tool invocation."""

    name: str
    args: dict[str, Any]
    iteration: int


class ToolResultRecord(TypedDict):
    """One observed tool outcome, with structured citations when available."""

    name: str
    ok: bool
    summary: str
    citations: list[str]
    score: float | None


class SourceRecord(TypedDict):
    """A citation surfaced to the user."""

    filename: str
    page_number: int | None
    chunk_id: str
    score: float


def _append(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Reducer that concatenates two list updates.

    LangGraph applies reducers for every node that writes a channel. With a
    plain list default, two nodes writing in the same superstep would race and
    one write would be lost, so these fields need an explicit merge.
    """
    return [*(left or []), *(right or [])]


class AgentState(TypedDict, total=False):
    """State threaded through the agent graph.

    ``total=False`` because a node that does not write a channel should not be
    required to supply it; LangGraph treats an absent key as "leave unchanged".
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: str

    # Routing
    route: str
    route_reason: str
    route_confidence: float

    # Execution bookkeeping
    iteration_count: int
    tool_calls: Annotated[list[ToolCallRecord], _append]
    tool_results: Annotated[list[ToolResultRecord], _append]
    sources: Annotated[list[SourceRecord], _append]
    errors: Annotated[list[str], _append]

    # Output
    answer: str
    grounded: bool
    insufficient_information: bool

    # Observability
    request_id: str
    thread_id: str
