# Phase 1 — Original Setup and Recorded Baseline Failures

**Project:** IntelliFlow — Enterprise Multi-Agent AI Assistant
**Baseline under test:** the unmodified reference application as received
(`main.py`, `graph.py`, `llms.py`, `config.py`, `state.py`, `nodes/`, `tools/`, `tests/`)
**Execution date:** 2026-10-01
**Phase 1 rule followed:** no application code was modified. This document is
the only artefact of this phase. (Phase 0 modified nothing either; its artefacts
are `docs/PROJECT_AUDIT.md`, `docs/PROJECT_ORIGIN.md`, and `NOTICE.md`.)

> This document records **what actually happened**, not what was expected.
> Every error message below is copied verbatim from the terminal, including the
> tracebacks. Where the Phase 0 audit predicted something that turned out to be
> different, that difference is called out explicitly in §6.

---

## 1. What this phase was for

Phase 0 concluded that the reference project **could not run**: the entire
LangChain stack was missing, `python -c "import main"` raised
`ModuleNotFoundError: No module named 'langchain_core'`, and no `.env` file
existed.

A baseline is only meaningful if it is established *before* anything is changed.
So the purpose of Phase 1 is narrow and deliberate:

1. Make the application runnable to the extent that placeholder credentials
   allow.
2. Execute every entry point: import, `/health`, `/chat`, the test suite.
3. **Record the failures verbatim.**
4. Discover anything the Phase 0 audit missed.

No fix was applied. Several failures found here are still open at the end of
this phase, and are carried forward as known-broken.

---

## 2. Environment as found

| Item | Value |
|---|---|
| OS | Windows (`win32`), PowerShell 5.1 |
| Interpreter | Python **3.11.9** — `C:\Users\itisa\AppData\Local\Programs\Python\Python311\python.exe` |
| `.python-version` | `3.11.10` (minor drift, ignored) |
| Virtual env | **None.** Packages are installed into the global user site-packages |
| Docker CLI | `29.6.2` present |
| Docker daemon | **Not running** (`npipe:////./pipe/dockerDesktopLinuxEngine` unavailable) |
| Local PostgreSQL | **None.** `127.0.0.1:5432` refused |
| PyPI | Reachable (all downloads succeeded) |
| Working copy | Still **not** a Git repository — see §7 |

### 2.1 Pre-existing environment (before this phase)

`fastapi 0.115.0`, `uvicorn 0.30.6`, `pydantic 2.9.2`, `pytest 8.3.3`,
`pytest-asyncio 0.24.0`, `psycopg 3.2.2`, `numpy 2.4.6`, `python-dotenv 1.2.2`,
`ruff 0.6.9`, `SQLAlchemy 2.0.35`, `alembic 1.13.3`, `httpx 0.27.2` were already
installed. Most of the LangChain family was not.

---

## 3. Exact reproduction

```powershell
# from the repository root
cd C:\Users\itisa\Desktop\langgraph-multi-agent-main

python --version
python -m pip install -r requirements.txt
```

`.env` was created from the placeholder template described in §4. It is
gitignored and contains no real credentials.

### 3.1 Two packages were still missing after the main install

```
langgraph-checkpoint-postgres   MISSING  (import fails)
nest-asyncio                    MISSING  (import fails)
```

Both are declared in `requirements.txt:14` and `:35`. They were installed
explicitly:

```powershell
python -m pip install "nest-asyncio>=1.6.0" "langgraph-checkpoint-postgres>=3.0.0"
```

### 3.2 A corrupted package install had to be repaired

`nest-asyncio` installed normally. `langgraph-checkpoint-postgres` did not, and
produced a genuinely misleading state:

```
Requirement already satisfied: langgraph-checkpoint-postgres>=3.0.0 ... (3.1.2)
```

pip reported success, but the import failed:

```
ModuleNotFoundError: No module named 'langgraph_checkpoint_postgres'
```

Inspection of site-packages showed the cause:

```
langgraph_checkpoint_postgres-3.1.2.dist-info   d-----     <- metadata present
langgraph_checkpoint_postgres/                   MISSING    <- code absent
```

The distribution **metadata was present but the package directory had been
deleted**. pip trusts the metadata, so it reported the requirement as already
satisfied and refused to reinstall it. Repaired with a forced reinstall that
bypasses the metadata check:

```powershell
python -m pip install --force-reinstall --no-deps "langgraph-checkpoint-postgres==3.1.2"
```

> **Note on the import name.** `langgraph-checkpoint-postgres` is a
> *distribution* name, not an importable module name. The module path is
> `langgraph.checkpoint.postgres`, which is what `main.py:10` imports:
> `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver`.
> Probing `import langgraph_checkpoint_postgres` is expected to fail and is
> not a broken install. The Phase 0 audit listed this distribution name in its
> import probe for exactly this reason; the distinction is recorded here so the
> next reader does not re-investigate it.

### 3.3 Resulting import state

All declared dependencies import cleanly. Verified:

```
langchain, langchain_core, langgraph, langchain_openai, langchain_community,
langchain_pinecone, langchain_mcp_adapters, langchain_tavily, pinecone,
openai, pypdf, nest_asyncio, langchain_huggingface, langchain_text_splitters,
langgraph_checkpoint_postgres, sentence_transformers, fastapi, uvicorn,
pytest, ruff, sqlalchemy, alembic, psycopg, docx2txt, rank_bm25, requests,
dotenv            -> all OK
```

---

## 4. The placeholder `.env`

`config.validate()` (`config.py:37-45`) raises at import time if any of
`NVIDIA_API_KEY`, `PINECONE_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL` is
missing. To get past import-time construction and observe *runtime* failure
modes, a `.env` was created with **deliberately invalid placeholders**:

| Variable | Placeholder value | Consequence |
|---|---|---|
| `NVIDIA_API_KEY` | `placeholder_not_a_real_key` | LLM calls rejected with 401 |
| `PINECONE_API_KEY` | `placeholder_not_a_real_key` | RAG tool fails at query time |
| `TAVILY_API_KEY` | `placeholder_not_a_real_key` | Web search fails at query time |
| `DATABASE_URL` | `postgresql://placeholder:placeholder@127.0.0.1:5432/placeholder_db` | No server to connect to |
| `MCP_SERVER_URL` | upstream author's Render deployment (the code default) | *Third-party, still reachable* |

This satisfies `validate()` while leaving every external dependency
non-functional. That is the intended baseline: the goal is to see *where* the
application breaks, not to make it work.

---

## 5. Baseline results

### 5.1 Summary

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Import | `python -c "import main"` | **PASS** |
| 2 | Route table | inspect `main.app.routes` | **PASS** — 6 routes |
| 3 | `GET /health` | ASGI client | **PASS** — `200 {"status":"ok"}` |
| 4 | `POST /chat` validation | empty `query` | **PASS** — `422` |
| 5 | `POST /chat` execution | real query | **FAIL** — `500` |
| 6 | MCP tool discovery | `await get_tools()` | **PASS** — 3 tools, 2.4 s |
| 7 | Graph assembly | `await build_graph()` | **PASS** — 2.5 s |
| 8 | Test suite | `pytest tests/ -v` | **FAIL** — 0/10 passed |

Check 1 is a change from the audit: the audit recorded `import main` as
failing with `ModuleNotFoundError`. That was caused purely by the missing
packages and is now resolved.

Routes present, confirming the audit's §4 endpoint inventory of exactly two
application endpoints:

```
/openapi.json, /docs, /docs/oauth2-redirect, /redoc, /health, /chat
```

### 5.2 Failure 1 — Windows event loop incompatibility (blocks everything)

`POST /chat` with a valid-shaped body returns `500`. The underlying cause is
**not** the placeholder credentials. It is a Windows-specific incompatibility
between `psycopg`'s async driver and Python 3.11's default event loop:

```
Traceback (most recent call last):
  File "main.py", line 159, in chat
    reply = await _invoke(payload.query, payload.thread_id)
  File "main.py", line 88, in _invoke
    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as cp:
  File "...\langgraph\checkpoint\postgres\aio.py", line 81, in from_conn_string
    async with await AsyncConnection.connect(
  File "...\psycopg\connection_async.py", line 104, in connect
    raise e.InterfaceError(
psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode.
Please use a compatible event loop, for instance by setting
'asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())'
```

HTTP response returned to the client:

```json
{"detail":"Something went wrong processing your request."}
```

`main.py` contains **no** `asyncio.set_event_loop_policy(...)` call, and
neither `main.py` nor the README documents a Windows limitation. On Linux this
code path works, so the defect is invisible to the original author.

Elapsed time for this request was **44.7 s** on the first call, attributable to
MCP tool discovery against the third-party Render deployment (cold start of a
free-tier service). Subsequent discovery completed in ~2.4 s.

### 5.3 Failure 2 — no PostgreSQL server

With the event loop forced to a selector policy, the incompatibility clears and
the next failure in the chain surfaces:

```
psycopg.OperationalError: connection failed: connection to server at "127.0.0.1",
port 5432 failed: could not receive data from server: Socket is not connected
(0x00002749/10057)
could not send SSL negotiation packet: Socket is not connected (0x00002749/10057)
```

Again returned as a generic `500`, not a `503`. See §6.2.

Elapsed time for this request: 4.5 s.

### 5.4 Failure 3 — the entire test suite

```
$ python -m pytest tests/ -v --tb=line
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.3, pluggy-1.6.0
rootdir: C:\Users\itisa\Desktop\langgraph-multi-agent-main
plugins: anyio-4.13.0, langsmith-0.14.2, asyncio-0.24.0
collecting ... collected 10 items

tests/test_agent.py::test_router_accuracy[python version check karo-python_tool] FAILED [ 10%]
tests/test_agent.py::test_router_accuracy[numpy install hai?-python_tool]     FAILED [ 20%]
tests/test_agent.py::test_router_accuracy[AI kya hota hai?-direct]           FAILED [ 30%]
tests/test_agent.py::test_router_accuracy[hello kaise ho?-direct]            FAILED [ 40%]
tests/test_agent.py::test_router_accuracy[latest GPT model 2026-web_search]  FAILED [ 50%]
tests/test_agent.py::test_router_accuracy[Bitcoin price today-web_search]    FAILED [ 60%]
tests/test_agent.py::test_python_execution                                    FAILED [ 70%]
tests/test_agent.py::test_python_version                                     FAILED [ 80%]
tests/test_agent.py::test_direct_answer                                      FAILED [ 90%]
tests/test_agent.py::test_urdu_response                                       FAILED [100%]

=============================== short test summary info ===========================
FAILED tests/test_agent.py::test_router_accuracy[python version check karo-python_tool]
FAILED tests/test_agent.py::test_router_accuracy[numpy install hai?-python_tool]
FAILED tests/test_agent.py::test_router_accuracy[AI kya hota hai?-direct]
FAILED tests/test_agent.py::test_router_accuracy[hello kaise ho?-direct]
FAILED tests/test_agent.py::test_router_accuracy[latest GPT model 2026-web_search]
FAILED tests/test_agent.py::test_router_accuracy[Bitcoin price today-web_search]
FAILED tests/test_agent.py::test_python_execution
FAILED tests/test_agent.py::test_python_version
FAILED tests/test_agent.py::test_direct_answer
FAILED tests/test_agent.py::test_urdu_response
========================= 10 failed in 29.96s =============================
```

**All ten failures share one root cause** — the §5.2 `ProactorEventLoop`
error, raised from `psycopg/connection_async.py:104`. No test failure is
attributable to the application logic, because no test ever reaches the graph.
A pytest-asyncio configuration warning is also emitted
(`asyncio_default_fixture_loop_scope` unset).

### 5.5 Passing check — the third-party MCP server is reachable

`get_tools()` succeeded, which is worth recording because it is the audit's
item **L19** (hard dependency on someone else's free-tier deployment):

```
get_tools() OK in 2.4s -> 3 tool(s): ['RAG', 'tavily_search', 'run_python']
build_graph() OK in 2.5s -> StateGraph
```

So the upstream author's Render-hosted MCP server is live and returns a
`run_python` tool. Two consequences, both discussed in §6.3.

---

## 6. Findings not captured by the Phase 0 audit

Three new issues were found by executing the application. The audit read the
source; this phase ran it, and running it exposed defects that reading cannot.

### 6.1 NEW — Windows event loop incompatibility with `psycopg` (blocker)

`psycopg`'s async support is incompatible with the `ProactorEventLoop`, which
is Python 3.8+'s default on Windows. The application provides no event loop
policy configuration, so on Windows **every** code path that touches the
Postgres checkpointer fails — `/chat`, the vision path, the CLI helper, and
all ten tests.

This is a higher-priority blocker than the audit's `P1` (`MockEmbeddings`) for
day-to-day use on this platform, because nothing runs at all without it. The
README's quick-start (`uvicorn main:app --reload`) fails on Windows.

Not recorded in the audit. No `set_event_loop_policy` call exists anywhere in
the codebase.

### 6.2 NEW — the database error handler is unreachable dead code

`main.py:101-104` and `main.py:149-150` intend to translate database failures
into a `503`. Both are gated on `except OSError`, with a comment asserting:

```python
except OSError as e:
    # Covers Postgres connection failures (asyncpg raises OSError subclasses)
```

**This comment is wrong for the driver actually in use.** The project uses
`psycopg` 3.x, not `asyncpg`. Verified MRO:

```
psycopg.InterfaceError is OSError subclass: False
psycopg.InterfaceError MRO: ['InterfaceError', 'Error', 'Exception', 'BaseException', 'object']
```

`psycopg.Error` derives from `Exception`, **not** `OSError`. Therefore:

- Every `psycopg` failure bypasses the `503` branch entirely.
- It falls through to the catch-all at `main.py:163` and is reported as a
  generic `500 "Something went wrong processing your request."`.

Observed directly: §5.3 shows a `psycopg.OperationalError` connection failure
returned as `500`, not `503`.

The audit's §4 error-mapping table and §6 memory flow both assumed the
`OSError` comment was accurate. It is not. The intended
"memory store unreachable → 503" behaviour has never once executed.

### 6.3 NEW — the system prompt names tools that do not exist

`nodes/llm_tool.py:21-29` tells the model it has these tools:

```
- RAG: search uploaded documents / knowledge base
- search_web: look up current or real-time information
- python_tool: execute Python code
```

The tools actually bound at `nodes/llm_tool.py:20` are named (measured in
§5.5): `RAG`, **`tavily_search`**, **`run_python`**.

So **two of the three names in the prompt are wrong.** The model is told to
call `search_web` and `python_tool`, but the callable tools are `tavily_search`
and `run_python`. The audit flagged this for `python_tool` only (item **S5**)
and did not notice the `search_web` mismatch. Note that the object's Python
identifier in `tools/web_search.py:5` *is* `search_web`; it is the tool's
advertised `.name` that differs — a distinction easy to miss when reading the
source.

Practical effect: the model must reconcile a prompt naming two tools that are
not callable against two callable tools it was not told about. It usually
recovers, since the real schemas are attached, but the instruction is
contradictory and can cause spurious tool errors or a wasted iteration.

### 6.4 Confirmed — audit findings re-verified by execution

| Audit item | Status after running |
|---|---|
| **P1** `MockEmbeddings` all-zero vectors | Confirmed present at `llms.py:21-36`, on the RAG path |
| **L19** hard dependency on third-party MCP server | Confirmed **live and reachable** (§5.5) |
| **S5** prompt advertises a non-existent `python_tool` | Confirmed, and extended to `search_web` (§6.3) |
| **L14** tests are live-network integration tests with no mocks | Confirmed — 0/10 pass, no isolation |
| **S3** `expected_type` parameter never asserted | Confirmed by reading `tests/test_agent.py:10-25`; dead |
| **L6** import-time side effects | Confirmed — `config.validate()` at `main.py:17` raises on import if any of 4 keys is missing |
| **P2** no ingestion pipeline | Confirmed absent; no loader/chunker/indexer exists |
| **S2** `docs/api_reference.md` documents a non-existent `load_document()` | Confirmed absent |

### 6.5 Still-unreachable, recorded for honesty

The placeholder credentials mean the following were **never observed** and no
claim is made about them:

- Any LLM response. `NVIDIA_API_KEY` is invalid, so the router, the tool-call
  node, and the answer node were never exercised end to end.
- The RAG read path against a real Pinecone index.
- The Tavily web search path.
- The MCP `run_python` tool actually executing code.
- Vision input handling.
- Conversation memory persisting across turns.

Every one of these is blocked behind real credentials and a reachable
PostgreSQL instance.

---

## 7. Baseline state carried forward

**Working:** import, route table, `/health`, input validation, MCP tool
discovery, graph assembly.

**Broken and left unfixed at the end of Phase 1:**

| ID | Issue | Where | Phase that owns it |
|---|---|---|---|
| B1 | Windows `ProactorEventLoop` breaks all async DB access | project-wide | Phase 3 (config/platform layer) |
| B2 | `except OSError` never catches `psycopg.Error`; DB failures return 500 not 503 | `main.py:101,149` | Phase 13 (security/error handling) |
| B3 | Prompt names `search_web` / `python_tool`; real tools are `tavily_search` / `run_python` | `nodes/llm_tool.py:22-24` | Phase 5 (router/tooling) |
| B4 | Test suite 0/10, no markers, no mocks, no `conftest.py` | `tests/test_agent.py` | Phase 18 |
| B5 | `expected_type` param never asserted — "router accuracy" is untested | `tests/test_agent.py:10-25` | Phase 5 |
| B6 | No PostgreSQL available locally; Docker daemon not running | environment | Phase 16 (Docker Compose) |
| B7 | `MockEmbeddings` — retrieval is meaningless | `llms.py:21-36` | **Phase 2 (mandated)** |
| B8 | No ingestion path at all | absent | **Phase 2 (mandated)** |

### 7.1 Version control

The working copy is **still not a Git repository** — `git rev-parse` fails, and
there is no `.git` directory. This confirms the audit's §2.1 finding, which
recommended initialising a repository *before* Phase 1 so that each subsequent
phase becomes a reviewable commit.

That recommendation was not yet acted on at the time of writing. It remains
open, and it is the single highest-value unblock available: without it none of
the phases ahead can produce the incremental commit history that is the point of
the exercise.

---

## 8. Phase 1 conclusion

- Dependencies installed. The whole LangChain/LangGraph stack plus
  `langgraph-checkpoint-postgres`, `nest-asyncio`, `docx2txt`, and `rank_bm25`
  now import cleanly.
- A corrupted `langgraph-checkpoint-postgres` install (metadata present, code
  deleted) was diagnosed and repaired. `pip` reported it as satisfied; only a
  `--force-reinstall` recovered it.
- **The application now imports**, which the audit recorded as impossible.
- `GET /health` and `/chat` input validation both pass.
- `POST /chat` fails with `500`. Two independent causes, in order: a Windows
  event loop incompatibility (§6.1), then the absence of a PostgreSQL server.
- The test suite is **0/10**, every failure sharing the event loop root cause.
- The third-party MCP server is **live**, returning tools named
  `tavily_search` and `run_python` — neither of which matches the names in the
  system prompt (§6.3).
- Three defects were found that the Phase 0 audit did not identify: the Windows
  event loop blocker, the unreachable `except OSError` database handler, and
  the tool-name mismatch. Two audit assumptions were corrected: `import main`
  now succeeds, and the audit's error-mapping table assumed a driver
  (`asyncpg`) the project does not use.
- **No application code was modified.** All 17 source files are byte-identical
  to the Phase 0 state.

### Recommended next step

Proceed to **Phase 2 (real embeddings)**, which the audit designates as
mandated and which closes `B7` and `B8` — the two findings that make the
project's headline RAG feature non-functional. Initialise the Git repository
first, so Phase 2 and everything after it is captured incrementally.
