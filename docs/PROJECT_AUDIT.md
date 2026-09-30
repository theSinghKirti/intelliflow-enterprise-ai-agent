# Phase 0 — Repository Audit

**Project (target):** IntelliFlow — Enterprise Multi-Agent AI Assistant
**Reference repository:** <https://github.com/arbaz-builds/langgraph-multi-agent>
**Audit scope:** the code as it exists in this working copy, at commit-equivalent state of the downloaded archive.
**Audit date:** 2026-10-01
**Phase 0 rule followed:** no application code was modified. This document plus `docs/PROJECT_ORIGIN.md` are the only files created.

---

## 0. How to read this audit

Section 1 is a factual description of what the code *actually does*. Where the
code and the documentation disagree, **the code is treated as the source of
truth** and the disagreement is called out explicitly in Section 9.

Every claim in this document was verified by reading the source files listed.
Nothing here is inferred from the README.

---

## 1. Executive summary

The reference repository is a small (~440 lines of Python) but genuinely
well-structured LangGraph chatbot. It demonstrates four things competently:

1. A LangGraph `StateGraph` with conditional edges and a bounded tool loop.
2. Structured output for the router (`with_structured_output` → Pydantic).
3. Real tool calling (`bind_tools`) with multi-step chaining.
4. PostgreSQL-backed conversation checkpointing via `AsyncPostgresSaver`.

It does **not** demonstrate a working RAG system. The RAG tool performs a
retrieval against Pinecone using **all-zero placeholder vectors**
(`MockEmbeddings`), so the similarity score of every candidate is identical and
retrieval results are effectively arbitrary. Worse, the repository contains **no
ingestion code at all** — nothing ever writes to the Pinecone index from this
codebase. The RAG read path exists; the write path does not exist anywhere in
the project.

The gaps between the reference project and the IntelliFlow target fall into six
groups:

| Area | Reference repo state | IntelliFlow target |
|---|---|---|
| Embeddings | `MockEmbeddings`, all-zero vectors | Real provider, configurable, tested |
| RAG ingestion | **Absent** (no loaders, no chunking, no indexing) | PDF/TXT → extract → clean → chunk → embed → store |
| Routing | Binary `tools`/`answer` | 6-category structured router (Pydantic + confidence) |
| Data layer | Postgres used *only* as a checkpoint store | Postgres + realistic enterprise schema + guarded SQL agent tool |
| API | 2 endpoints, no auth | 5+ endpoints, auth, request IDs, validation, rate limiting |
| Process | No Docker, no CI, no lint, no eval suite | Docker Compose, pytest suite, ruff, eval harness, 20+ docs |

---

## 2. Repository inventory (complete file listing)

```
langgraph-multi-agent-main/
├── .env.example                 27 lines   documented env template
├── .gitignore                   43 lines
├── .python-version               1 line    "3.11.10"
├── CONTRIBUTING.md              33 lines
├── LICENSE                      21 lines   MIT, © 2026 Arbaz (Pokemon455)
├── README.md                   169 lines   overstates some capabilities (see §9)
├── config.py                    45 lines   env-var config + validate()
├── graph.py                     26 lines   StateGraph assembly
├── langgraph_agent.py           15 lines   DEPRECATED stub, comments only
├── llms.py                      36 lines   ChatOpenAI instances + MockEmbeddings
├── main.py                     175 lines   FastAPI app + CLI entry
├── requirements.txt             41 lines   flat requirements, gaps (see §9)
├── state.py                     21 lines   State TypedDict + RouterDecision
├── docs/
│   ├── api_reference.md         74 lines   documents functions that do not exist
│   └── architecture.md          89 lines   partially stale (see §9)
├── nodes/
│   ├── __init__.py               5 lines
│   ├── answer.py                59 lines
│   ├── conditions.py            21 lines   edge predicates
│   ├── llm_tool.py              32 lines
│   └── router.py                27 lines
├── tests/
│   └── test_agent.py            61 lines   live-network tests, no real assertions
└── tools/
    ├── __init__.py               5 lines
    ├── mcp_client.py             9 lines   client only, no server
    ├── rag.py                   24 lines   read-only retrieval
    └── web_search.py             8 lines
```

**Total Python:** ~440 lines across 17 files.

**Explicitly absent from the repository:**

- No `Dockerfile`, no `docker-compose.yml`
- No CI workflow (`.github/`), no `pyproject.toml`, no `Makefile`
- No linter/formatter/type-checker configuration (no `ruff.toml`, `mypy.ini`, `pyproject.toml`)
- No `conftest.py` for tests
- No database schema, seed data, or SQL
- No evaluation harness
- No frontend
- No `.git` directory — see §2.1

### 2.1 Critical environment finding: no version control

`git rev-parse` reports this is **not a Git repository**; there is no `.git`
directory. The code was obtained as a source archive, not cloned. Consequences:

- There is no commit history, so we cannot demonstrate incremental work via
  commit history in the portfolio unless we initialise a repository now.
- Phase 23's "check git status" step cannot be performed meaningfully until a
  repository is initialised.
- Recommendation: initialise a fresh repo *before* Phase 1 so that every phase
  of the IntelliFlow work is captured as a real, reviewable commit history.
  That history is itself interview evidence of incremental engineering.

### 2.2 Critical environment finding: dependencies are not installed

Interpreter present: **Python 3.11.9** at
`C:\Users\itisa\AppData\Local\Programs\Python\Python311\python.exe`
(`.python-version` pins `3.11.10` — trivial drift, ignore).

Import probe of every LangChain-family dependency:

```
langchain                      MISSING
langchain_core                 MISSING
langgraph                      MISSING
langchain_openai               MISSING
langchain_community            MISSING
langchain_pinecone             MISSING
langchain_mcp_adapters         MISSING
langchain_tavily               MISSING
pinecone                       MISSING
openai                         MISSING
pypdf                          MISSING
nest_asyncio                   MISSING
langchain_huggingface          MISSING
langchain_text_splitters       MISSING
langgraph_checkpoint_postgres  MISSING
```

Currently installed and relevant: `fastapi 0.115.0`, `uvicorn 0.30.6`,
`pydantic 2.9.2`, `pytest 8.3.3`, `pytest-asyncio 0.24.0`, `psycopg 3.2.2`,
`numpy 2.4.6`, `python-dotenv 1.2.2`, `ruff 0.6.9`, `SQLAlchemy 2.0.35`,
`httpx 0.27.2`.

Confirmed consequence — `python -c "import main"` fails:

```
File "main.py", line 9, in <module>
    from langchain_core.messages import HumanMessage
ModuleNotFoundError: No module named 'langchain_core'
```

**The reference project cannot currently run.** Phase 1 must install
dependencies first. PyPI is reachable from this machine (verified: HTTP 200), so
installation is feasible. No `.env` file exists either.

---

## 3. Application architecture as it exists

### 3.1 Entry points

| Entry point | Location | Description |
|---|---|---|
| FastAPI ASGI app `app` | `main.py:26` | Served via `uvicorn main:app` |
| CLI / test helper `build_and_run()` | `main.py:169` | Thin wrapper over `_invoke()` |
| `__main__` block | `main.py:174` | `asyncio.run(build_and_run("python version check karo"))` |
| `langgraph_agent.py` | whole file | **Dead.** Comment-only deprecation notice, no executable code |

### 3.2 LangGraph graph

Assembled in `graph.py:11-26`. Four nodes, three conditional edges.

**Nodes**

| Node | Function | File |
|---|---|---|
| `router` | `router_node` | `nodes/router.py:15` |
| `llm_tool` | `llm_tool_node` | `nodes/llm_tool.py:15` |
| `tools` | `ToolNode(tools)` (LangGraph prebuilt) | `graph.py:17` |
| `answer` | `answer_node` | `nodes/answer.py:41` |

**Edges**

| From | To | Condition |
|---|---|---|
| `START` | `router` | unconditional (`graph.py:20`) |
| `router` | `llm_tool` \| `answer` | `route_condition` → `state["router_decision"]` (`graph.py:21`) |
| `llm_tool` | `tools` \| `answer` | `tool_or_answer` (`graph.py:22`) |
| `tools` | `llm_tool` \| `answer` | `after_tools` (`graph.py:23`) |
| `answer` | `END` | unconditional (`graph.py:24`) |

**Graph shape**

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼            (loop while iteration_count    │
              ┌──────────┐        < MAX_ITERATIONS)              │
   START ───▶ │  router  │                                       │
              └────┬─────┘                                       │
       tools ──────┴────── answer ──▶ END                        │
         │                    ▲                                  │
         ▼                    │                                  │
    ┌─────────┐  no tool call │                                  │
    │llm_tool │───────────────┤                                  │
    └────┬────┘               │                                  │
         │ tool call          │                                  │
         ▼                    │                                  │
    ┌─────────┐               │                                  │
    │  tools  │───────────────┘                                  │
    └────┬────┘                                              │
         └──────────────────────────────────────────────────────┘
```

Loop control:

- `nodes/conditions.py:7` `tool_or_answer` — go to `tools` only if the last
  message is an `AIMessage` with `tool_calls` **and** `iteration_count < MAX_ITERATIONS`
  (`config.MAX_ITERATIONS`, default `3`).
- `nodes/conditions.py:16` `after_tools` — always loop back to `llm_tool` while
  `iteration_count < MAX_ITERATIONS`, else go to `answer`.

This is the one piece of the reference architecture that is genuinely good and
should be preserved in spirit: an explicit, configurable iteration cap that
prevents infinite tool loops.

### 3.3 State

`state.py:17-21`:

```python
class State(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]
    router_decision: str
    reasoning:       str
    iteration_count: int
```

| Field | Purpose | Reducer |
|---|---|---|
| `messages` | Conversation transcript | `add_messages` — appends, merges by `id` |
| `router_decision` | `"tools"` or `"answer"` | last-write-wins (no reducer) |
| `reasoning` | Router's explanation | last-write-wins |
| `iteration_count` | Tool-loop counter | last-write-wins (manual increment) |

`RouterDecision` (`state.py:7-14`) is a Pydantic model with
`decision: Literal["tools","answer"]` and a length-bounded `reasoning: str`.
Structured output is therefore **already used correctly** in the reference
project — the right pattern to extend in Phase 5, not to invent from scratch.

### 3.4 Tool inventory

Three tools are exposed to the agent.

| Tool | Backing implementation | File |
|---|---|---|
| `RAG` | `PineconeVectorStore.as_retriever(search_type="mmr", k=5, fetch_k=20, lambda_mult=0.7)` | `tools/rag.py:8` |
| `search_web` | `langchain_tavily.TavilySearch(max_results=5)` | `tools/web_search.py:5` |
| `python_tool` | Remote MCP server (tools discovered at runtime) | `tools/mcp_client.py:4` |

Tool assembly happens per call in `nodes/llm_tool.py:8-12`:
`mcp_client.get_tools()` → `tools = [RAG, search_web, *mcp_tools]`.

### 3.5 LLM provider

`llms.py` uses `langchain_openai.ChatOpenAI` pointed at the NVIDIA NIM
OpenAI-compatible endpoint — one provider, three client instances:

| Instance | Model | Purpose |
|---|---|---|
| `answer_LLM` | `config.NVIDIA_MODEL` (`openai/gpt-oss-20b`) | Tool calling + final answers |
| `router_LLM` | same, `temperature=0` | Router classification |
| `vision_LLM` | `meta/llama-3.2-90b-vision-instruct` (**hard-coded**) | Image input, bypasses the graph |

---

## 4. Request flow trace (as required)

```
USER QUERY
    │
    ▼
POST /chat   (main.py:153)
    │  QueryRequest Pydantic validation:
    │    - query non-empty           (field_validator, main.py:39)
    │    - image_base64 ≤ 5,000,000 chars
    │    - image_base64 valid base64 (decoded eagerly, main.py:62)
    ▼
if payload.image_base64?
    ├── YES → _invoke_vision (main.py:107)
    │           • aget_tuple(thread_id) → read channel_values["messages"]
    │           • vision_LLM.ainvoke(history + multimodal HumanMessage)
    │           • build_graph() → compile(cp) → aupdate_state(as_node="answer")
    │           • NOTE: bypasses router and tools entirely
    │
    └── NO  → _invoke (main.py:77)
                │  LANGGRAPH
                ▼
            build_graph() → get_tools() → StateGraph(...)   (graph.py:11)
                ▼
            g.compile(checkpointer=AsyncPostgresSaver.from_conn_string(DATABASE_URL))
                ▼
            ┌─────────┐  router_LLM.with_structured_output(RouterDecision)
            │ router  │  → {"router_decision": "tools"|"answer", "reasoning": str}
            └────┬────┘  on exception: silently falls back to "answer" (router.py:22)
                 │
        ┌────────┴─────────┐
        ▼                  ▼
   ┌─────────┐        ┌─────────┐
   │llm_tool │        │ answer  │ ◀── direct
   └────┬────┘        └────┬────┘
        │ tool_calls?       │
        ▼ yes, under cap    │
   ┌─────────┐              │
   │  tools  │  ToolNode executes; result → ToolMessage
   └────┬────┘              │
        │ loop while        │
        │ iteration < cap ──┘ (re-enter llm_tool)
        │
        ▼
   answer_node
        • _get_tool_output(msgs) → last ToolMessage only
        • _strip_images(msgs)    → removes raw image bytes for text-only LLM
        • history = last 10 non-ToolMessage messages
        • answer_LLM.ainvoke
        • retry once if completion is empty; fallback string if still empty
        ▼
   AIMessage  ──▶  checkpointed to Postgres  ──▶  result["messages"][-1].content
                                                          ▼
                                              QueryResponse(response=...)
                                                          ▼
                                                          JSON
```

**Error mapping** (`main.py:95-104`, mirrored in `_invoke_vision`):

| Exception | HTTP | Detail |
|---|---|---|
| `APIConnectionError`, `APITimeoutError` | 503 | Model provider unavailable |
| `APIError` | 502 | Model provider error |
| `OSError` (asyncpg) | 503 | Memory store unreachable |
| anything else (`main.py:163`) | 500 | Generic; stack trace logged, not returned |

**Endpoint surface today:** exactly two — `GET /health` (`main.py:72`) and
`POST /chat` (`main.py:153`). No document upload, no session introspection,
no tool listing, no auth.

---

## 5. RAG flow trace (as required) — **the critical finding**

### 5.1 The documented flow

```
DOCUMENT
    ↓
INGESTION          ← ❌ DOES NOT EXIST
    ↓
CHUNKING           ← ❌ DOES NOT EXIST
    ↓
EMBEDDING          ← ⚠️ MockEmbeddings — all-zero vectors
    ↓
VECTOR STORE       ← Pinecone (read-only from this codebase)
    ↓
RETRIEVAL          ← MMR over meaningless scores
    ↓
CONTEXT            ← string concatenation, no citations kept
    ↓
LLM                ← answer_node, ungrounded by design
    ↓
ANSWER
```

### 5.2 The placeholder embedding implementation — exact location

`llms.py:21-36`, verbatim:

```python
class MockEmbeddings(Embeddings):
    """Placeholder embeddings — no external API required.
    Returns fixed-size zero vectors so RAG runs without OpenAI credentials.
    Dimension (1536) matches OpenAI's text-embedding-3-small for Pinecone index compatibility.
    Replace with a real embedding model when ready for production RAG quality.
    """
    _DIM = 1536

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._DIM for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self._DIM


embeddings = MockEmbeddings()
```

**Where it is used:** exactly one place.

| Reference | Role |
|---|---|
| `llms.py:36` | `embeddings = MockEmbeddings()` — module-level singleton |
| `tools/rag.py:4` | `from llms import embeddings` |
| `tools/rag.py:14` | `embedding=embeddings` passed into `PineconeVectorStore(...)` |

A repository-wide search for `Mock` returns only these four hits: the class
definition, the instantiation, and the README note at `README.md:60`. The
import at `llms.py:4` pulls the object into `tools/rag.py`'s namespace, so the
placeholder is on the production RAG path and nowhere else. There is no second
embedding implementation to compare against.

### 5.3 Why this is only a placeholder — the mathematical reason

This is the point to be able to explain in an interview:

1. Every embedded text becomes the vector `[0, 0, 0, …, 0]` (1536 zeros).
2. Retrieval computes similarity against the query vector — which is *also*
   `[0, 0, 0, …, 0]`.
3. **Cosine similarity of two zero vectors is undefined** (`0/0`). Libraries
   guard against division by zero and return a constant — in practice `0.0` for
   every candidate, identical for all of them.
4. Therefore the ranking of retrieved chunks is determined **entirely by Pinecone's
   internal index behaviour and tie-breaking order**, not by any property of the
   text.

Consequence: the `RAG` tool returns *some* chunks, and the pipeline "works"
without raising, which is precisely what makes this dangerous. There is no
crash, no warning, and no test failure — the retrieval is silently meaningless.
Any claim that "the RAG feature works" would be false.

Two further consequences worth noting:

- The comment claims dimension 1536 keeps it "Pinecone-index-compatible", i.e.
  the author's real design intent was OpenAI `text-embedding-3-small`, but the
  requirement was never met.
- `requirements.txt:8` already declares `langchain-huggingface>=0.1.0`, which is
  the natural local, free, real-embedding path — **but it is never imported
  anywhere in the code.** A real embedding provider was anticipated and then
  not wired up.

### 5.4 There is no ingestion pipeline at all

This is the most important structural finding, and it is **not** mentioned in the
README or `docs/`.

A repository-wide search for `load_document`, `add_documents`, `add_texts`,
`from_documents`, `PyPDFLoader`, `TextLoader`, `RecursiveCharacterTextSplitter`,
and `chunk` returns **zero hits in any `.py` file**. The only hit is
`docs/api_reference.md:51`, which documents a function that does not exist:

```python
def load_document(file_path: str) -> str:
    """Loads and indexes a document into Pinecone. Supported: .pdf, .csv, .txt, .docx"""
```

So:

- **No text extraction** — no PDF/TXT/DOCX/CSV loader exists.
- **No cleaning** — no normalisation step exists.
- **No chunking** — despite `langchain-text-splitters` being declared in
  `requirements.txt:6`, it is never imported.
- **No indexing write path** — nothing in this codebase ever calls
  `add_documents`/`add_texts`/`upsert`. The Pinecone index must be populated
  entirely out-of-band, by hand, with some other unshared process.

The README's phrase "Searches previously indexed documents" is technically true
but misleading in context: *this project cannot index anything.*

### 5.5 Retrieval specifics as configured

`tools/rag.py:16-19`:
- `search_type="mmr"` — Maximal Marginal Relevance: balances relevance against
  diversity across the returned chunks.
- `k=5` (`RETRIEVER_K`) — chunks returned.
- `fetch_k=20` (`RETRIEVER_FETCH_K`) — candidates fetched before MMR re-ranking.
- `lambda_mult=0.7` — relevance/diversity trade-off, hard-coded, not configurable.

Two problems:
- MMR re-ranking over identical (meaningless) scores provides no benefit.
- `PineconeVectorStore(...)` is constructed **inside** the tool on **every
  invocation** — no client reuse, a new index lookup per query.
- Results are flattened to a plain string (`tools/rag.py:21`) with `Source 1:`,
  `Source 2:` labels. **Filename, page number, and chunk metadata are discarded**,
  so the model cannot cite sources and the API response cannot return citations.
- `tools/rag.py:23` returns `f"RAG error: {e}"` on failure — raw exception text
  is fed into the LLM context and can surface in the user-facing answer. This is
  both a security issue (internal detail leakage) and a correctness issue (the
  model may try to answer from an error string).

---

## 6. Memory flow trace

```
POST /chat { query, thread_id }
    │
    ▼
AsyncPostgresSaver.from_conn_string(config.DATABASE_URL)   (main.py:88)
    │  per-request connection, opened and closed inside the `async with`
    │  documented rationale (main.py:79-85, README:140-142):
    │  serverless/free-tier Postgres silently drops idle connections,
    │  and a long-lived pooled connection poisons every later request
    ▼
await cp.setup()                                             (main.py:89)
    │  issues CREATE TABLE IF NOT EXISTS ... DDL on EVERY request
    ▼
g.compile(checkpointer=cp).ainvoke(
    {"messages": [HumanMessage(query)], "iteration_count": 0},
    config={"configurable": {"thread_id": thread_id}})        (main.py:90-93)
    ▼
Postgres tables: checkpoints / writes / checkpoint_blobs / checkpoint_migrations
```

Observations:

- Memory works via LangGraph's official Postgres checkpointer. `thread_id` is a
  plain client-supplied string with **default `"1"`** (`main.py:36`), so all
  clients that don't pass one share a single conversation thread. There is no
  authentication binding a thread to a user — a real multi-tenant data-leakage
  vector (see §10).
- `cp.setup()` per request is wasteful but functionally safe.
- `main.py:117` reaches into `checkpoint_tuple.checkpoint["channel_values"]` —
  LangGraph **internal** checkpoint structure. This is a brittle coupling to a
  private detail of another library and a likely future breakage point.
- No memory is retained outside LangGraph checkpoints: no user profile, no
  long-term/semantic memory, no summarisation of old turns.
- The vision path writes back via `aupdate_state(..., as_node="answer")`
  (`main.py:136-140`); the code comments explain that `as_node` must be explicit
  or the write silently fails on a thread's first write. Good hard-won knowledge,
  worth preserving and re-verifying after restructuring.

---

## 7. Tool flows

### 7.1 RAG tool — see §5.

### 7.2 Web search

`tools/web_search.py` — the entire implementation:

```python
search_web = TavilySearch(max_results=5, tavily_api_key=config.TAVILY_API_KEY)
```

- Backed by `langchain_tavily.TavilySearch`, `max_results=5`.
- Returned to the LLM as whatever structure that integration produces.
- **No error handling** at this layer: no timeout configuration, no retry, no
  empty-result normalisation, no malformed-response handling. A Tavily 429 or
  malformed payload propagates as an unhandled exception into `ToolNode`, which
  produces a `ToolMessage` containing an error string rather than a clean
  structured result. It degrades rather than crashes, but not gracefully.
- **No gating:** the router only decides `tools` vs `answer`; there is no
  rule preventing web search being selected for a purely internal document
  question. Tool *selection* is entirely at the discretion of the LLM in
  `llm_tool_node`.

### 7.3 Python execution via MCP

`tools/mcp_client.py` — the entire implementation:

```python
mcp_client = MultiServerMCPClient({
    "python_tool": {
        "transport": "streamable-http",
        "url": config.MCP_SERVER_URL
    }
})
```

- Client only. **There is no MCP server in this repository.**
- `config.py:23` / `.env.example:17` default `MCP_SERVER_URL` to
  `https://deploy-9ugq.onrender.com/mcp` — a free-tier Render deployment
  belonging to the original author. Consequences:
  1. The project **cannot function standalone**; it silently depends on a third party's machine.
  2. Free-tier Render sleeps and cold-starts, so the first tool call in a session can be very slow or fail.
  3. For a portfolio project this is a liability: an interviewer asking "how does your Python tool work?" gets "it's someone else's server."
  4. Remote free-tier Python REPL = arbitrary code execution on a shared host. Documented nowhere in the project's security posture (there is no `docs/SECURITY.md`).
- `get_tools()` (`nodes/llm_tool.py:10`) is called on every `llm_tool` node
  entry, so MCP tool discovery is re-run per node visit rather than cached.
- No timeout, no retry, no capability filtering, no allow-list on which remote
  tools the agent may call.

---

## 8. Database flow

There is exactly one database interaction path: LangGraph checkpointing
(`main.py:88`, `main.py:112`). That is the whole picture.

- **No business data layer.** No `customers`, `products`, `orders`, `employees`,
  or `sales` tables. No schema, no migrations, no seed data.
- **No database tool.** No SQL of any kind is generated or executed by the agent.
- Postgres is used purely as a key-value checkpoint store.
- `requirements.txt` declares both `psycopg[binary]>=3.0.0` and
  `psycopg2-binary>=2.9.0` — redundant, as only the `psycopg` 3.x async driver
  is actually used (and it is not yet installed).
- `SQLAlchemy 2.0.35` and `alembic 1.13.3` happen to be present in the local
  environment but are not declared in `requirements.txt` and not used.

---

## 9. Documentation-vs-code discrepancies

Found by comparing every claim in `README.md` and `docs/` against the source.
These matter because several are stated as verified facts and are not.

| # | Claim | Location | Reality |
|---|---|---|---|
| 1 | Router classifies queries into **4 categories**: `python_tool`/`web_search`/`rag`/`direct` | `docs/architecture.md:27-29`, `docs/api_reference.md:69-74` | The router makes a **binary** decision, `tools` or `answer` (`state.py:8`). Category selection is delegated implicitly to the LLM in `llm_tool_node` via `bind_tools` — it is never a structured, inspectable decision. |
| 2 | "Router **Accuracy: 100%** (tested on 10+ queries)" | `docs/architecture.md:29` | **Unsubstantiated.** `tests/test_agent.py:10-25` parameterises `expected_type` and then **never asserts on it** — the fixture value is dead. The test only asserts the response is non-empty. No route is ever inspected. |
| 3 | `load_document(file_path)` — "Loads and indexes a document into Pinecone. Supported: .pdf, .csv, .txt, .docx" | `docs/api_reference.md:51-65` | **The function does not exist.** No loader, no chunker, no indexing write path anywhere in the codebase. |
| 4 | Tool named `web_search` | `docs/api_reference.md:34` | Actual symbol is `search_web` (`tools/web_search.py:5`). |
| 5 | Tool named `run_python(code)` "via MCP" | `docs/api_reference.md:42-47` | The name comes from whatever the remote server advertises; it is not defined or documented in this repo. |
| 6 | `NVIDIA_MODEL` covers model selection | `README.md:134` | The vision model `meta/llama-3.2-90b-vision-instruct` is **hard-coded** at `llms.py:17` and is not overridable by env var. |
| 7 | 3 nodes named `LLM_Tool`, `tool_condition` | `docs/architecture.md:11-16` | Actual names are `llm_tool` and `tool_or_answer`. |
| 8 | "Key fix: `tool_choice=forced`" | `docs/architecture.md:35` | `nodes/llm_tool.py:20` calls `.bind_tools(tools)` with **no** `tool_choice` argument. Behaviour is enforced only by prompt text ("always make a tool call"). |
| 9 | `langchain-huggingface`, `langchain-community`, `langchain-text-splitters`, `pypdf`, `docx2txt`, `rank-bm25`, `nest-asyncio`, `requests`, `psycopg2-binary` | `requirements.txt` | Declared but **imported nowhere** in the codebase. |

**Docs that are accurate:** `README.md` §Architecture node descriptions
(§lines 47-50), the README's RAG-quality warning (line 60), the Postgres
per-connection rationale (lines 140-142), and `docs/architecture.md` §"Bugs
Fixed" (the `MAX_ITERATIONS` dead-code fix and the test-import fix are genuine
and match the code).

---

## 10. Identified placeholders, stubs, and limitations

### Placeholders (must be eliminated)

| ID | Item | Location | Severity |
|---|---|---|---|
| P1 | `MockEmbeddings` — all-zero vectors on the production RAG path | `llms.py:21-36` | **Blocker** |
| P2 | No document ingestion / extraction / cleaning / chunking / indexing exists | absent | **Blocker** |
| P3 | Citation metadata discarded; results flattened to a string | `tools/rag.py:21` | High |
| P4 | `langchain-huggingface` declared but never wired to anything real | `requirements.txt:8` | High (opportunity) |

### Stubs and dead code

| ID | Item | Location |
|---|---|---|
| S1 | `langgraph_agent.py` — 15-line comment-only deprecation stub | whole file |
| S2 | `docs/api_reference.md` documents a non-existent `load_document()` | §Document Loading |
| S3 | `test_router_accuracy`'s `expected_type` parameter is never asserted against | `tests/test_agent.py:10-25` |
| S4 | Router swallows all exceptions and silently routes to `answer` — the user sees a hallucinated direct answer instead of an error | `nodes/router.py:22-23` |
| S5 | `llm_tool_node` advertises `- python_tool: execute Python code` in its prompt but the tool is MCP-discovered, so if MCP discovery fails the prompt still promises a Python tool | `nodes/llm_tool.py:22-28` |
| S6 | `vision_LLM` is excluded from the graph entirely — image queries bypass routing, RAG, and tools | `main.py:107-150` |

### Architectural and engineering limitations

| ID | Limitation | Location |
|---|---|---|
| L1 | **No authentication or authorization anywhere.** Anyone reaching the API can read any `thread_id`'s full conversation by guessing/using `"1"`. | `main.py:153` |
| L2 | **No input-length limit on `query`.** Only `image_base64` is size-bounded. | `main.py:34-42` |
| L3 | **No rate limiting.** Unbounded cost exposure on paid LLM/Tavily/embedding APIs. | — |
| L4 | **Raw exception text leaks into LLM context and user-visible answers.** | `tools/rag.py:23` |
| L5 | **No structured logging.** `logger = logging.getLogger(...)` exists but `basicConfig` is never called and no handler/formatter/level is configured; no request ID, thread ID, route, tool name, or duration is logged. | `main.py:19` |
| L6 | **Side effects at import time.** `config.validate()` (`main.py:17`) and `ChatOpenAI(...)` construction (`llms.py:6-18`) run on import, so `import main` raises if any of four env vars is missing. This is why `pytest tests/` cannot even collect without full production credentials. | `main.py:17`, `llms.py:6` |
| L7 | **Config is module-level mutable global state.** Provider URLs and models cannot be swapped per test or per request; every client is constructed eagerly. | `config.py`, `llms.py` |
| L8 | **Coupling to LangGraph internals.** `checkpoint["channel_values"]` is a private checkpoint layout. | `main.py:117` |
| L9 | **Single hard-coded vector store.** Pinecone is wired directly into `tools/rag.py`; no interface to swap in pgvector or an in-memory store for tests. | `tools/rag.py:12-15` |
| L10 | **RAG has no relevance threshold.** With `MockEmbeddings` every score is equal; even with a real model there is no similarity cutoff, so irrelevant-but-nearest chunks are always injected. Nothing implements "I don't have enough information in the provided documents." | `tools/rag.py:16-22` |
| L11 | **No prompt-injection consideration.** Retrieved document text is concatenated straight into the answer prompt with no delimiting or instruction to treat it as data, not instructions. | `tools/answer.py:44` |
| L12 | **No output validation.** The final answer is returned as an unvalidated string. | `main.py:94` |
| L13 | **No evaluation harness, no metrics, no regression protection.** | — |
| L14 | **Tests are live-network integration tests with no mocks, no markers, no fixtures.** They require real NVIDIA, Pinecone, Tavily, Postgres and the author's Render MCP deployment to pass, and use the removed-in-3.12 `asyncio.get_event_loop()` pattern. | `tests/test_agent.py` |
| L15 | **No dependency pinning, no lockfile.** All requirements use `>=` with wide floors across a fast-moving ecosystem; a fresh install may resolve incompatible versions. | `requirements.txt` |
| L16 | **Default `thread_id="1"`** is a shared global conversation for every caller that omits it. | `main.py:36` |
| L17 | **Tavily results unstructured; no timeout/retry/empty-result handling.** | `tools/web_search.py` |
| L18 | **Pinecone client constructed per query** instead of reused. | `tools/rag.py:12` |
| L19 | **Hard dependency on a third party's free-tier MCP deployment** that is not part of this repository. | `config.py:23` |
| L20 | **No Docker, no CI, no lint/format/type-check config, no frontend, no `docs/SECURITY.md`.** | — |

---

## 11. Proposed changes (roadmap, not executed in Phase 0)

Ordered by dependency. Each phase keeps the app runnable.

### Phase 1 — Baseline (`docs/ORIGINAL_SETUP.md`)
Install `requirements.txt`; create a local `.env`; run the app, `/health`,
`/chat`, and the existing test suite; **record the baseline failures verbatim**,
including the fact that the whole LangChain stack is missing.

### Phase 2 — Real embeddings (highest priority, mandated)
1. Introduce an embedding abstraction (`Embeddings` interface + factory) selected
   by `EMBEDDING_PROVIDER` env var. **No provider-specific code outside the factory.**
2. Default to a **local, free** provider: `HuggingFaceEmbeddings` with
   `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~90 MB, CPU, no API key).
   `langchain-huggingface` is *already* in `requirements.txt` — no new framework.
3. Because the index dimension changes from the placeholder's 1536 to 384, a
   **new Pinecone index must be created**; document the dimension mismatch
   explicitly, since reusing the old index is a silent-failure trap.
4. Provide an optional hosted provider (e.g. NVIDIA NIM embeddings or OpenAI
   `text-embedding-3-small`) behind the same interface, API key via env only.
5. **Build the missing ingestion path**: loaders (PDF via `pypdf`, TXT) → clean →
   `RecursiveCharacterTextSplitter` → embed → upsert, with `document_id`,
   `filename`, `page_number`, `chunk_id`, `source` metadata.
6. Add a retrieval relevance threshold + "insufficient information" behaviour
   (addresses L10, P3, L11).
7. Tests that prove, with **real vectors**: embedding produces non-degenerate
   output, upsert works, a paraphrased query retrieves the right document, an
   unrelated query ranks it low, and metadata round-trips.

### Phase 3 — Restructuring
Adopt the `app/` package layout from the brief (`core`, `agents`, `tools`, `rag`,
`memory`, `api`, `evaluation`, `services`). Port behaviour **without** changing
it; run the baseline tests after each move.

### Phases 4-6 — State, router, RAG
Typed state with route/tool_calls/tool_results/sources/errors/iteration_count;
6-category Pydantic `RouterDecision` with `route`/`reason`/`confidence` and
fail-safe defaulting; full RAG pipeline with citations.

### Phases 7-12 — Database tool, web search hardening, Python tool, MCP server, memory, multi-step orchestration
Guarded read-only SQL agent; structured Tavily wrapper with timeouts; a
**first-party, in-repo MCP server** replacing the third-party Render dependency
(L19); Postgres business schema + seed data; bounded multi-step
Router → Database → Python → Web → Synthesis flow.

### Phases 13-19 — Security, API, observability, evaluation, tests, Docker, frontend
Auth, rate limiting, request-ID middleware, structured JSON logs; 5 endpoints;
25-30 evaluation queries with a runner that records real results only; pytest
split into unit/integration with markers and mocks; `Dockerfile` +
`docker-compose.yml` with Postgres; minimal React + Vite UI last.

### Phase 20-23 — Differentiation, learning guide, README, final audit

---

## 12. What is worth keeping from the reference architecture

Honest assessment, so the portfolio can credit the reference author properly:

| Keep | Why |
|---|---|
| Four-node graph shape with explicit conditional edges | Clear separation of concerns; easy to explain |
| `MAX_ITERATIONS` loop cap via `tool_or_answer` / `after_tools` | Correct, non-obvious fix for a real infinite-loop bug |
| `with_structured_output` + Pydantic for the router | Right pattern; Phase 5 extends rather than replaces it |
| `Annotated[List[BaseMessage], add_messages]` reducer | Correct LangGraph idiom |
| Per-request Postgres connection, with the documented rationale | A real operational lesson worth preserving |
| `_strip_images` multimodal sanitisation (`nodes/answer.py:24`) | Genuinely thoughtful handling of a subtle failure mode |
| Explicit `as_node` on `aupdate_state` with the explanatory comment | Hard-won LangGraph knowledge |
| Empty-completion retry in `answer_node` | Pragmatic resilience |
| Layered HTTP error mapping that never leaks stack traces | Good instinct; needs the L4 leak in `tools/rag.py:23` closed too |

The "keep" list is the honest foundation. IntelliFlow's differentiation should
be the substantial work described in §11, not a rename.

---

## 13. Phase 0 conclusion

- Repository inspected in full: 17 Python files, 2 existing docs, 1 test file.
- **`MockEmbeddings` confirmed** at `llms.py:21-36`, instantiated at `llms.py:36`,
  consumed at `tools/rag.py:14` — a placeholder on the production RAG path.
- **A more serious finding:** the RAG pipeline has **no ingestion side at all**.
  Nothing in this codebase extracts, chunks, embeds, or writes a document. The
  RAG feature is read-only scaffolding over an empty index populated out-of-band.
- Dependencies are **not installed**; the project cannot run as-is.
- There is **no Git repository** in this working copy.
- **Documentation contains unverified claims** (4-category router, "100% router
  accuracy", a `load_document()` that doesn't exist). These must be corrected in
  the IntelliFlow docs rather than carried forward.
- No application code was modified in Phase 0.

**Recommendation before Phase 1:** initialise a Git repository so that each
subsequent phase is committed incrementally — the commit history is part of the
portfolio evidence.
