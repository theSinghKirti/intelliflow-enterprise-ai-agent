# Architecture

## Graph shape

```
START
  │
  ▼
┌────────┐
│ router │  six-route typed decision: route, reason, confidence
└───┬────┘
    │
    ├── route in {rag, web_search, database, python} ──▶ invoke_tool
    │                                                          │
    │                                                          ▼
    │                                                   ┌────────────┐
    │                                                   │  reason    │  bind_tools;
    │                                                   └─────┬──────┘  one call
    │                                                         │         decides
    │        under MAX_ITERATIONS and still requesting tools? │         tool vs answer
    │                                                   yes ──┘         │
    │                                                   │               │
    └───────────────────────────────────────────────────┴───────────────┤
                                                                    no ──┘
                                                                        │
    ┌───────────────────────────────────────────────────────────────────┘
    ▼
┌────────┐
│ answer │  grounded, language-matched reply
└────┬───┘
     ▼
    END
```

## Nodes

### `router` — [`app/agents/router.py`](../app/agents/router.py)

One structured-output LLM call classifies the query into one of six
categories: `rag`, `web_search`, `database`, `python`, `direct`, `clarify`.

The decision is a Pydantic model carrying a `reason` and a `confidence`, so the
routing choice is inspectable in state and assertable in a test rather than
being a prompt-level hint. The model's raw answer is normalised against
`ROUTE_CATEGORIES`; near-misses (`"web"`, `"RAG"`, `"sql"`) are mapped, and
anything unrecognised becomes `direct`.

A classification failure does not disappear. It falls back to `direct` with
`confidence=0.0` and a reason naming the exception, and it is logged. The
alternative — silently answering directly — is indistinguishable from a normal
answer once it reaches the user.

### `invoke_tool` — [`app/agents/tools.py`](../app/agents/tools.py)

Runs the tool calls on the last AI message and records them.

`RAG` is executed inline rather than through `ToolNode`, because its citations
and grounding verdict need to land in state as typed values. Other tools go
through `ToolNode`.

Tool failures are recorded in `errors` and returned to the model as a short
notice naming only the exception type. Exception text is never interpolated
into model context or into a user-visible answer.

Only `RAG` and `search_web` are bound, plus whatever MCP discovers. There is no
database or Python tool in this build, so the `database` and `python` routes
are classified correctly but have no dedicated tool behind them.

### `reason` — [`app/agents/graph.py`](../app/agents/graph.py)

A single LLM call with the tools bound. The model either requests one more
tool or answers directly. This replaces the upstream project's three-call
sequence (bind, force a tool call, then execute) with one decision per
iteration.

### `answer` — [`app/agents/answer.py`](../app/agents/answer.py)

Composes the reply.

- Output language is matched to the most recent user message (Roman Urdu or
  English).
- Output from **all** tool messages is gathered, not just the last one. In a
  bounded loop, reading only the last `ToolMessage` throws away the evidence
  from earlier steps.
- Retrieved text is fenced and labelled as reference data, and the model is
  told not to follow directives inside it. Retrieved content is untrusted
  input.
- When retrieval found nothing above the threshold, the prompt says so
  explicitly and state records `insufficient_information`, so the agent
  declines rather than answering from parametric memory.
- Multimodal messages are reduced to their text portion before reaching the
  text-only chat model.
- An empty completion is retried once before giving up.

### Edge conditions — [`app/agents/conditions.py`](../app/agents/conditions.py)

`route_condition` picks the first hop. `after_tools` loops back to
`invoke_tool` only while `iteration_count < MAX_ITERATIONS` **and** the model
is still requesting tools; otherwise it falls through to `answer`. The cap is
logged when it fires, so a truncated turn is visible rather than silent.

## State — [`app/agents/state.py`](../app/agents/state.py)

`AgentState` is a `TypedDict`. `messages` uses LangGraph's `add_messages`
reducer; every list channel (`tool_calls`, `tool_results`, `sources`, `errors`)
uses an explicit concatenating reducer, because a plain list default would let
two nodes writing in the same superstep overwrite one another.

Recorded per turn: `route`, `route_reason`, `route_confidence`, the tool calls
made, their outcomes, the sources cited, any errors, the answer, and whether it
was grounded or the information was insufficient.

## RAG — [`app/rag/`](../app/rag)

```
document → load → clean → chunk → embed → store → retrieve (MMR + threshold)
```

- **embeddings** ([`embeddings.py`](../app/rag/embeddings.py)) — the only module
  that imports a provider-specific embedding client. `local` (default) is
  `all-MiniLM-L6-v2`, 384-dim, CPU, no key; `nvidia` and `openai` are hosted
  and need credentials. The dimension differs per provider, and a store index
  is built for exactly one dimension.
- **ingestion** ([`ingestion.py`](../app/rag/ingestion.py)) — PDF via `pypdf`
  with one segment per page so a citation names the right page; `.txt`/`.md`
  as UTF-8. Null bytes are stripped and line endings normalised before
  chunking.
- **retrieval** ([`retriever.py`](../app/rag/retriever.py)) — MMR search. If
  the best score is below `RETRIEVAL_THRESHOLD`, the result carries no chunks
  and `has_evidence` is `False`; the caller is expected to decline.
- **context fencing** ([`models.py`](../app/rag/models.py)) — retrieved text is
  wrapped in `<retrieved_context>`, labelled as data, and literal fence tags
  inside a document are escaped so a document cannot break out of the block.

The store is cached process-wide. A fresh store per call would silently discard
everything ingested moments earlier.

## Memory — [`app/memory/postgres.py`](../app/memory/postgres.py)

`AsyncPostgresSaver` is compiled into the graph, which is what persists each
`thread_id`. A connection is opened per request: a single long-lived connection
meant that once a serverless provider dropped an idle connection, every
subsequent request failed until the process restarted.

History is not read out of the checkpoint. The graph replays checkpointed state
into the answer node, so no accessor reaching into LangGraph's internal
checkpoint layout is needed.

## Cross-cutting

- **Configuration** ([`app/core/config.py`](../app/core/config.py)) — an
  immutable `Settings` snapshot built from the environment, falling back to
  `.env`. Construction never raises; `validate()` does, and it only requires
  the keys the *selected* providers actually need. Nothing is validated at
  import time, so importing the application requires no credential.
- **Logging** ([`app/core/logging.py`](../app/core/logging.py)) — one JSON
  object per log line, with `request_id` and `thread_id` injected from
  `contextvars` so any log call deep in the stack is correlated without
  threading an id through every signature.
- **Platform** ([`app/core/platform.py`](../app/core/platform.py)) — installs
  the selector event loop policy on Windows. `psycopg`'s async driver cannot
  use the default `ProactorEventLoop`; without this every Postgres path fails
  on Windows while working fine on Linux.
