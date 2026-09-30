# Changelog

All notable modifications to IntelliFlow, phase by phase.

This project began as a fork of the MIT-licensed
[langgraph-multi-agent](https://github.com/arbaz-builds/langgraph-multi-agent)
by Mohammad Arbaz. See `docs/PROJECT_ORIGIN.md` for licensing and attribution
details, and `NOTICE.md` for the attribution summary.

**Convention:** a phase is listed here only after its work is complete and
verified. Nothing is claimed as implemented before it works. Entries describe
*verified* state, not intent. Forward-looking plans live in
`docs/PROJECT_AUDIT.md` §11, not here.

---

## Phase 0 — Repository audit

**Status:** complete
**Artefacts:** `docs/PROJECT_AUDIT.md`, `docs/PROJECT_ORIGIN.md`, `NOTICE.md`
**Application code modified:** none

Established the as-received baseline of the reference project by reading every
source file. The project was obtained as a source archive, not a clone, so
there is no commit history to inspect.

Principal findings:

- `MockEmbeddings` (`llms.py`) returns all-zero vectors and is the sole
  embedding implementation on the production RAG path. Cosine similarity of
  two zero vectors is undefined, so retrieval ranking is decided entirely by
  index tie-breaking. Retrieval runs without error and returns meaningless
  results — the most dangerous kind of failure.
- **No ingestion pipeline exists at all.** Nothing in the codebase loads,
  chunks, embeds, or writes a document. The RAG read path exists; the write
  path does not exist anywhere in the project.
- Documentation makes claims the code does not support: a 4-category router
  (the router is binary), "router accuracy 100%" (the test never asserts the
  route), and a `load_document()` function that does not exist.
- No Git repository, no Docker, no CI, no lint config, no database schema, no
  evaluation harness.
- Ten LangChain-family dependencies were missing; the project could not run.

Recorded a 20-item limitation register (P1-P4 placeholders, S1-S6 stubs,
L1-L20 limitations) and a phased roadmap, ordered by dependency.

## Phase 1 — Baseline setup and failure recording

**Status:** complete
**Artefacts:** `docs/ORIGINAL_SETUP.md`
**Application code modified:** none
**Date:** 2026-10-01

Made the reference application runnable as far as placeholder credentials
allow, executed every entry point, and recorded the failures verbatim.

Setup:

- Installed all declared dependencies. Two were still absent
  (`langgraph-checkpoint-postgres`, `nest-asyncio`) and were added explicitly.
- Repaired a corrupted `langgraph-checkpoint-postgres` 3.1.2 install whose
  `dist-info` metadata was present but whose package directory had been
  deleted. `pip` reported the requirement as already satisfied; a
  `--force-reinstall` was required to recover it.
- Created a `.env` of deliberately invalid placeholders to get past the
  import-time `config.validate()` and observe runtime failure modes.

Verified working: `import main` (the audit recorded this as impossible), route
table, `GET /health` → `200`, `/chat` input validation → `422`, MCP tool
discovery, graph assembly.

Verified broken: `POST /chat` → `500`; test suite **0/10**.

Three defects found by execution that the Phase 0 audit did not identify:

1. **Windows event loop incompatibility.** `psycopg`'s async driver cannot use
   the `ProactorEventLoop` that Python 3.8+ defaults to on Windows. No
   `set_event_loop_policy` call exists in the codebase, so on Windows every
   path that touches the Postgres checkpointer fails — including the README's
   own `uvicorn main:app --reload` quick-start. Invisible to the original
   author, who developed on Linux.
2. **The database error handler is unreachable dead code.** `main.py` maps
   database failures to `503` via `except OSError`, justified by a comment
   claiming "asyncpg raises OSError subclasses". The project uses `psycopg`,
   whose errors derive from `Exception`, not `OSError`. Verified MRO:
   `InterfaceError -> Error -> Exception`. The intended 503 behaviour has
   never executed; database failures surface as a generic 500.
3. **The system prompt names tools that do not exist.** The prompt advertises
   `search_web` and `python_tool`; the tools actually bound are named
   `tavily_search` and `run_python`. Two of three advertised tool names are
   wrong. The audit had flagged only the `python_tool` case.

Two Phase 0 assumptions were corrected: `import main` now succeeds once
dependencies are installed, and the audit's error-mapping analysis assumed a
PostgreSQL driver the project does not use.

The upstream author's third-party MCP deployment was confirmed **live**,
confirming audit item L19 and making the tool-name mismatch measurable rather
than theoretical.

The test suite's 0/10 result is attributed entirely to the Windows event loop
root cause; no test reaches application logic, so no behavioural baseline for
the agent itself exists yet.

---

## Upcoming

---

## Phase 2 - Real embeddings, ingestion, and grounding

**Status:** complete
**Date:** 2026-10-01

Replaced the all-zero placeholder embeddings and added the ingestion pipeline
that did not exist anywhere in the project. Closes audit items P1, P2, P3, L4,
L10, L11.

- Embedding provider factory with a `local` default
  (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim, CPU, no API key) plus
  hosted `nvidia` and `openai` options. `MockEmbeddings` is gone from the
  project.
- Typed contracts: `IngestionResult`, `RetrievedChunk`, `RetrievalResult`,
  `ChunkMetadata`.
- Ingestion for PDF (per-page, so citations name the correct page), TXT, and
  Markdown, with cleaning and recursive chunking.
- Grounded retriever with MMR search, a configurable similarity threshold
  (default 0.30), preserved citation metadata, and a refusal to answer when
  nothing clears the threshold.
- Retrieved context is fenced and labelled as reference data, and literal fence
  tags inside a document are neutralised.
- The RAG tool no longer interpolates exception text into model context.
- CLI: `python -m app.rag.ingest`.
- 28 RAG tests against real embeddings, all passing offline.

**Dimension note:** the local provider is 384-dim, not the placeholder's
1536-dim. A vector store index is built for exactly one dimension, so the
index must be created for the provider in use.

---

## Phase 3 - Application restructure and clean snapshot

**Status:** complete
**Date:** 2026-10-01

Reorganised the flat root layout (`main.py`, `graph.py`, `state.py`,
`llms.py`, `config.py`, `nodes/`, `tools/`) into the layered `app/` package
and removed the legacy implementations rather than leaving two copies of the
same feature. Closes audit items L5, L6, L7, L8, and B1, B3, B4, B5, B9.

Layering, with dependencies pointing one way only:

- `app.core` — configuration, JSON logging with request/thread correlation,
  Windows event-loop policy. Imports nothing internal.
- `app.services` — chat-model construction on demand from `Settings`.
- `app.rag` — embeddings, ingestion, retrieval.
- `app.tools` — the tools exposed to the agent, each owning its own
  provider-specific import and failure handling.
- `app.memory` — Postgres checkpointing and thread config.
- `app.agents` — state, six-route router, tool execution, answer, graph.
- `app.api` — schemas, routes, application factory.

Substantive changes, beyond moving files:

- **Import-time side effects removed (L6, L7).** `config.validate()` and
  `ChatOpenAI(...)` construction ran at import, so `import main` raised unless
  four hosted credentials were present and the suite could not be collected
  without production secrets. `Settings` is now an immutable snapshot that
  never raises on construction and validates only what the *selected* providers
  need. `create_app()` is an explicit factory.
- **Structured six-route router.** The binary `tools`/`answer` decision is now
  a typed `route`/`reason`/`confidence` model, normalised against
  `ROUTE_CATEGORIES`. A classification failure is recorded with confidence 0.0
  and logged rather than silently discarded (S4).
- **Structured state.** `AgentState` records route, tool calls, tool results,
  sources, errors, and the grounding verdict. Every list channel has an
  explicit reducer so parallel writes are not lost.
- **Prompt-injection hygiene (L11).** Retrieved text is fenced, labelled as
  data, and fence tags inside documents are escaped. The answer node gathers
  output from *all* tool messages rather than only the last.
- **Grounding is observable (L10).** `grounded` and `insufficient_information`
  are recorded in state and returned in the response.
- **Web search hardened (L17).** Timeouts, result normalisation, length caps,
  sanitised failures, and an explicit notice when unconfigured.
- **MCP made optional (L19).** The upstream project's hard-coded third-party
  Render deployment is gone. With no `MCP_SERVER_URL` the graph still builds
  and no MCP tools are bound.
- **Request correlation (L5).** `X-Request-ID` is generated or echoed on every
  response and bound to the logging context; logs are one JSON object per line.
- **Request validation.** Query length is bounded and `thread_id` is
  restricted to a conservative character set.
- **Windows compatibility (B1).** The selector event loop policy is installed
  before any async `psycopg` work, from both the entry point and `conftest.py`.
- **Error mapping fixed (B2).** The unreachable `except OSError` branch is
  gone; failures are mapped through a catch-all that never leaks exception text.
- **Tests.** `conftest.py` fixes the event loop and provides hermetic,
  credential-free settings. 109 tests pass with no hosted credential and no
  network call: 28 RAG tests against real embeddings, routing and node
  behaviour against a stub model, and the HTTP layer end to end.
- **Legacy code deleted.** `main.py`, `graph.py`, `state.py`, `llms.py`,
  `config.py`, `nodes/`, `tools/`, `langgraph_agent.py`, and the old
  `tests/test_agent.py` are removed. That test imported the legacy `main`,
  needed live NVIDIA, Postgres, and a third-party MCP server, and asserted
  nothing about routing — it parameterised an `expected_type` it never checked
  (B5). Its coverage is replaced by `tests/test_agents.py`, which does assert
  routing.
- **Dead code removed.** Unused vision-model settings and factory (there is no
  vision path: `image_base64` is validated but never routed to a model),
  unused history accessors, a duplicate LLM factory, and an unused response
  schema.
- **Dependencies corrected.** `sentence-transformers` is now declared
  explicitly. Seven dependencies that no module imports were dropped, and
  dependency floors were raised to the major versions actually verified. Test
  dependencies are in `requirements-dev.txt`.
- **Documentation corrected.** `README.md`, `docs/architecture.md`, and
  `docs/api_reference.md` describe the current build. The audit, origin, and
  original-setup records are left as written, since they document the
  pre-existing state rather than the current one.

### Verified in this build

| Check | Result |
|---|---|
| `python -m compileall app` | passes |
| `python -c "import app.main"` | passes, with no credential set |
| `pytest tests/test_rag.py -q` | 28 passed |
| `pytest -q` | 109 passed |
| `ruff check app tests` | clean |
| `GET /health` | `200`, with no NVIDIA key |
| `POST /chat` blank / oversized / bad `thread_id` | `422` |
| `X-Request-ID` | generated, echoed, replaced when over-long |
| RAG discrimination | paraphrases score 0.435–0.553; unrelated query scores 0.056, below the 0.30 threshold |

### Not verified

No live NVIDIA, PostgreSQL, Pinecone, Tavily, or MCP call was made, because no
credential for any of them is available. The code paths exist and are wired;
none of them is claimed to be operational, and none is stubbed to make a test
pass. A missing credential produces an explicit `503` or a degradation
notice.

---

## Upcoming

Planned per `docs/PROJECT_AUDIT.md` §11. Not started.

| Phase | Scope | Status |
|---|---|---|
| 7-12 | Guarded SQL tool, first-party MCP server, business schema, multi-step orchestration | pending |
| 13-19 | Security, authentication, API surface, observability, evaluation harness, Docker | pending |
| 20-23 | Differentiation, learning guide, final audit | pending |

Deliberately out of scope for the current build: a web frontend, additional
vector databases, and an evaluation framework.

### Known limitations of the current build

Recorded so they are not mistaken for unimplemented work:

| ID | Issue |
|---|---|
| L1 | `image_base64` is validated but never routed to a model; there is no vision path |
| L2 | The `database` and `python` routes are classified but have no dedicated tool bound |
| L3 | `VECTOR_STORE=memory` is per-process, so the CLI and the API server do not share a store |
| L4 | No authentication, authorisation, rate limiting, or TLS |
| L5 | No streaming, cancellation, or background jobs |
| L6 | `GET /health` reports configuration, not dependency health |
| L7 | No CI, Docker, or deployment configuration |
| L8 | No live integration call has been made against NVIDIA, PostgreSQL, Pinecone, Tavily, or MCP |

### Known-broken, carried forward from the original project

Unfixed, and out of scope for the current build:

| ID | Issue | Status |
|---|---|---|
| B6 | No local PostgreSQL, so a live `/chat` turn is unverified | open |
| B9 | Git history only begins at the baseline import; the upstream project's own history was never available | open |

