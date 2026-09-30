# IntelliFlow — Enterprise Multi-Agent AI Assistant

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-green)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-teal)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

IntelliFlow is a **LangGraph-based enterprise multi-agent AI assistant
prototype** demonstrating RAG, semantic retrieval, tool orchestration,
structured agent state, FastAPI APIs, and persistent-memory architecture.

It is a prototype, not a production system. Nothing here is deployed, and the
sections marked **environment-dependent** below need credentials that are not
in this repository. See [Known limitations](#known-limitations) before drawing
conclusions from anything stated here.

A LangGraph-based AI assistant that combines structured agent routing, real
semantic-embedding RAG, document ingestion, vector retrieval, tool
orchestration, FastAPI APIs, and persistent conversation-state architecture.

---

## Current capabilities

Implemented and exercised by the test suite:

- **PDF / TXT / Markdown document ingestion** — `pypdf` for PDFs, one segment
  per page so citations name the correct page
- **Real 384-dimensional embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
  on CPU, no API key. Not a placeholder
- **Semantic retrieval** — MMR search over a `VectorStore`, with a configurable
  similarity threshold
- **Grounded RAG responses** — retrieved text is fenced, labelled as reference
  data, and fence-breakout attempts are neutralised. When nothing clears the
  threshold the agent is told the documents do not cover the topic instead of
  answering from a near-miss
- **Citation metadata** — `document_id`, `filename`, `page_number`, `chunk_id`
  and score survive the full round trip and are returned to the caller
- **Structured six-route agent architecture** — `rag`, `web_search`,
  `database`, `python`, `direct`, `clarify`, emitted as a typed decision with a
  reason and a confidence score
- **Structured agent state** — a `TypedDict` with reducers, recording route,
  tool calls, tool results, sources, errors, and the grounding verdict
- **Bounded tool loop** — capped at `MAX_ITERATIONS`, and the cap is logged
  when it fires rather than silently truncating
- **FastAPI backend** — `GET /health`, `POST /chat`
- **Request validation and correlation** — bounded query length, validated
  `thread_id` and base64 image, `X-Request-ID` generated or echoed on every
  response
- **PostgreSQL memory architecture** — LangGraph checkpointing, one connection
  per request
- **MCP-aware tool integration** — tools are discovered from `MCP_SERVER_URL`
  when set; the graph still builds when it is not
- **Windows compatibility** — the selector event loop policy is installed
  before any async `psycopg` work
- **Automated RAG testing** — 109 tests, none of which require a hosted
  credential

## Environment-dependent

These paths are implemented but **cannot be verified without credentials that
are not in this repository**. They are not claimed to be operational.

| Integration | Needs | Status |
|---|---|---|
| Chat model (NVIDIA NIM) | `NVIDIA_API_KEY` | Code path implemented; no answer can be generated without a key. `/chat` returns `503` when it is absent |
| Conversation memory (PostgreSQL) | `DATABASE_URL` | Code path implemented; `/chat` returns `503` when it is absent. No local PostgreSQL was used to verify a live turn |
| Web search (Tavily) | `TAVILY_API_KEY` | Degrades to an explicit "web search is not configured" notice rather than failing the turn |
| Vector store (Pinecone) | `PINECONE_API_KEY` | Code path implemented; unverified against a live index. The default `memory` store is what the tests exercise |
| MCP server | `MCP_SERVER_URL` | Code path implemented; unverified against a live server. This project ships no MCP server |
| LangSmith tracing | `LANGCHAIN_API_KEY` | Disabled by default (`LANGCHAIN_TRACING_V2=false`) |

Nothing in this repository is stubbed to make a test pass. A tool whose
credential is missing returns an explicit degradation message; it does not
fabricate a result.

## Architecture

```
POST /chat
   │
   ▼
┌──────────────┐   six-route typed decision (route, reason, confidence)
│    router    │──────────────────────────────────────────────┐
└──────┬───────┘                                              │
       │                                                      │
  route in {rag, web_search, database, python}?              │
       │ yes                          │ no                    │
       ▼                              ▼                      ▼
┌──────────────┐              ┌──────────────┐        route, reason,
│  invoke_tool │              │    answer    │        confidence
│ (RAG, search,│              │  grounded,   │        recorded in state
│  MCP tools)  │              │  language-   │
└──────┬───────┘              │  matched     │
       │                      └──────┬───────┘
       ▼                             │
┌──────────────┐   under the cap and  │
│    reason    │───still requesting───┘
│ (bind_tools) │   tools? ── yes ── loop back to invoke_tool
└──────────────┘
```

Modules are layered so dependencies point one way only; `app.core` never
imports back upward.

```
app/
├── main.py            # entry point: uvicorn app.main:app
├── api/
│   ├── routes.py      # /health, /chat, request-ID middleware, app factory
│   └── schemas.py     # request/response models
├── agents/
│   ├── state.py       # AgentState TypedDict, ROUTE_CATEGORIES
│   ├── router.py      # six-route classification
│   ├── tools.py       # tool assembly and execution
│   ├── answer.py      # grounded answer prompt and generation
│   ├── conditions.py  # edge predicates, bounded tool loop
│   └── graph.py       # StateGraph assembly
├── core/
│   ├── config.py      # immutable Settings, validated per selected provider
│   ├── logging.py     # JSON logs, request/thread correlation
│   └── platform.py    # Windows event-loop policy
├── services/
│   └── llm.py         # cached chat-model construction
├── memory/
│   └── postgres.py    # checkpointer + thread config
├── rag/
│   ├── embeddings.py  # provider factory (local | nvidia | openai)
│   ├── ingestion.py   # load -> clean -> chunk -> embed -> store
│   ├── models.py      # typed contracts, context fencing
│   ├── retriever.py   # MMR search + threshold + citations
│   └── ingest.py      # CLI: python -m app.rag.ingest
└── tools/
    ├── rag_tool.py    # RAG tool
    └── web_search.py  # hardened Tavily wrapper
```

## Quick start

```bash
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env          # fill in what you need; see the table above
```

Run the server:

```bash
uvicorn app.main:app --reload
```

`GET /health` needs no credential at all:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","version":"2.0.0","embeddings":"local (384-dim)",
#  "vector_store":"memory","memory":"postgres"}
```

Ingest documents and try retrieval without any hosted service:

```bash
python -m app.rag.ingest --directory data/
python -m app.rag.ingest --search "what is the refund window?"
```

Note that `VECTOR_STORE=memory` is per-process, so documents ingested by the
CLI are not visible to a separately started server. Set `VECTOR_STORE=pinecone`
to persist across processes.

## Tests

```bash
pytest -q                      # whole suite
pytest tests/test_rag.py -q    # RAG only
ruff check app tests
```

| File | Covers |
|---|---|
| `tests/test_rag.py` | 28 tests: real embeddings are non-degenerate, deterministic, L2-normalised and discriminative; ingestion, chunking, PDF page numbers, citations, thresholding, context fencing |
| `tests/test_agents.py` | routing schema and normalisation, classification fallback, edge conditions, the iteration cap, answer-prompt construction |
| `tests/test_core.py` | settings validation, event-loop policy, JSON logging and correlation, and the HTTP layer end to end |

No test requires NVIDIA, Pinecone, Tavily, PostgreSQL, or a network call. The
tests that would need those are not present rather than skipped-and-green: the
agent graph is tested against a stub model, and the HTTP layer is tested up to
the point where a credential would be required.

## Environment variables

The real process environment wins over `.env`. `.env` is gitignored and must
not be committed. `.env.example` contains placeholders only.

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | for `/chat` | Postgres connection string (conversation memory) |
| `NVIDIA_API_KEY` | for answers | LLM inference via NVIDIA NIM |
| `NVIDIA_BASE_URL`, `NVIDIA_MODEL` | optional | Override the NIM endpoint / model |
| `EMBEDDING_PROVIDER` | optional | `local` (default, 384-dim, no key) \| `nvidia` (1024-dim) \| `openai` (1536-dim) |
| `EMBEDDING_MODEL` | optional | Override the provider's default model |
| `VECTOR_STORE` | optional | `memory` (default, per-process) \| `pinecone` |
| `PINECONE_API_KEY`, `PINECONE_INDEX` | for `pinecone` | Hosted vector store |
| `TAVILY_API_KEY` | optional | Web search; degrades to a notice when absent |
| `MCP_SERVER_URL` | optional | MCP server exposing extra tools; empty means no MCP tools |
| `RETRIEVAL_THRESHOLD` | optional | Minimum similarity to count as evidence (default `0.30`) |
| `RETRIEVER_K`, `RETRIEVER_FETCH_K` | optional | Retrieval breadth |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | optional | Ingestion chunking |
| `MAX_ITERATIONS` | optional | Tool-loop cap (default `3`) |
| `MAX_QUERY_CHARS`, `MAX_IMAGE_BASE64_CHARS` | optional | Request size limits |
| `LOG_LEVEL` | optional | Log level (default `INFO`) |
| `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `LANGCHAIN_TRACING_V2` | optional | LangSmith tracing, off by default |

The vector dimension depends on the embedding provider, and a store index is
built for exactly one dimension. Switching providers requires a new index.

## Known limitations

- **`image_base64` is validated but not used.** The schema checks size and
  base64 well-formedness and returns `422` for malformed input, but no vision
  model is invoked; the image is not routed to the graph. There is no vision
  path in this build.
- **`database` and `python` routes have no dedicated tool.** The router
  classifies into six routes, but the only bound tools are `RAG` and
  `search_web` (plus any discovered over MCP). A `database` query is answered
  from RAG, web search, or the model's own knowledge.
- **`VECTOR_STORE=memory` does not persist.** The store is per-process, so the
  API server sees only what was ingested inside that process.
- **No authentication, authorisation, rate limiting, or TLS.** The API is
  unauthenticated and must not be exposed publicly as-is.
- **One turn per request.** There is no streaming, no cancellation, and no
  background job handling.
- **No evaluation harness and no frontend.** Neither is part of this build.
- **The `/health` endpoint reports configuration, not dependency health.** It
  does not ping Postgres or the model provider.
- **Postgres connections are opened per request.** Deliberate, to survive
  idle-connection drops on serverless providers, at the cost of per-call
  overhead.
- **No CI, Docker, or deployment configuration.**

## Documentation

| File | Contents |
|---|---|
| `docs/CHANGELOG.md` | What was verified working, phase by phase |
| `docs/architecture.md` | Graph shape and node-by-node walkthrough |
| `docs/api_reference.md` | HTTP endpoints and request/response shapes |
| `docs/PROJECT_AUDIT.md` | Audit of the original upstream project |
| `docs/ORIGINAL_SETUP.md` | Verbatim setup and failure record for the original project |
| `docs/PROJECT_ORIGIN.md` | Upstream attribution and reference-vs-modification map |
| `NOTICE.md` | Attribution summary |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. This is a derivative work of an MIT-licensed project; see
[NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
