"""IntelliFlow application package.

The application is layered so that cross-cutting concerns do not collide:

``app.core``
    Configuration, structured logging, and platform compatibility. Imports
    nothing from the other application packages.
``app.services``
    Construction of external clients (currently the LLM). Depends on
    ``app.core``.
``app.rag``
    Embeddings, document ingestion, and grounded retrieval. Depends on
    ``app.core`` only.
``app.tools``
    The tools exposed to the agent. Each owns its own provider-specific import
    and its own failure handling, so one misconfigured integration degrades
    that tool alone. Depends on ``app.core`` and ``app.rag``.
``app.memory``
    Postgres-backed conversation checkpointing. Depends on ``app.core``.
``app.agents``
    Graph state, routing, tool execution, and answer generation. Depends on
    all of the above.
``app.api``
    HTTP schemas, routes, and the application factory. Depends on
    ``app.agents``, ``app.core``, ``app.memory``, and ``app.services``.

Dependencies point in one direction only - ``app.core`` never imports back
upward - so any module can be imported without triggering a cycle.
"""
