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

Planned per `docs/PROJECT_AUDIT.md` §11. Not yet started.

| Phase | Scope | Status |
|---|---|---|
| 2 | Real embeddings (provider factory, local default) + the missing ingestion pipeline | pending |
| 3 | Restructuring to the `app/` package layout | pending |
| 4-6 | Typed state, 6-category structured router, RAG with citations | pending |
| 7-12 | Guarded SQL tool, hardened web search, first-party MCP server, business schema, multi-step orchestration | pending |
| 13-19 | Security, API surface, observability, evaluation, test suite, Docker, frontend | pending |
| 20-23 | Differentiation, learning guide, README, final audit | pending |

### Known-broken, carried forward

Left unfixed at the end of Phase 1, owned by later phases:

| ID | Issue | Owner |
|---|---|---|
| B1 | Windows `ProactorEventLoop` breaks all async DB access | Phase 3 |
| B2 | `except OSError` never catches `psycopg.Error` | Phase 13 |
| B3 | Prompt/tool-name mismatch (`search_web`, `python_tool`) | Phase 5 |
| B4 | Test suite 0/10, no markers, no mocks, no `conftest.py` | Phase 18 |
| B5 | `expected_type` never asserted — "router accuracy" untested | Phase 5 |
| B6 | No local PostgreSQL; Docker daemon not running | Phase 16 |
| B7 | `MockEmbeddings` makes retrieval meaningless | **Phase 2** |
| B8 | No ingestion path exists | **Phase 2** |
| B9 | No Git repository — no incremental history is being captured | **immediately** |
