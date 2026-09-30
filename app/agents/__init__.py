"""Agent graph: structured state, routing, tool execution, and answer.

Modules
-------
``state``
    The :class:`AgentState` TypedDict threaded through the graph, and the
    canonical ``ROUTE_CATEGORIES`` tuple.
``router``
    Classifies a query into one of the six routes.
``tools``
    Assembles the tool list and executes the calls the model requested,
    recording citations and grounding verdicts into state.
``answer``
    Composes the grounded answer prompt and generates the reply.
``conditions``
    Edge predicates: the first hop, and the bounded tool loop.
``graph``
    Assembles the above into a ``StateGraph``.

Deliberately no eager re-exports
--------------------------------
This package does not import its submodules at import time. ``state`` and
``conditions`` are leaf modules, but ``tools`` pulls in the RAG stack and
``graph`` pulls in the tool stack, so an eager ``from app.agents import graph``
would drag the whole retrieval dependency chain into any process that only
wanted the state schema. It would also make an import cycle easy to introduce:
``graph`` imports ``router``, ``tools``, and ``answer``, so re-exporting them
from the package ``__init__`` would run those imports during
``app.agents`` initialisation.

Import the submodules directly instead::

    from app.agents.state import AgentState, ROUTE_CATEGORIES
    from app.agents.graph import build_graph
"""
