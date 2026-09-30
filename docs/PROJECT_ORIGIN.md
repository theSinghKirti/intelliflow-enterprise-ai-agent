# Project Origin, Licensing, and Attribution

This document exists so that IntelliFlow is presented honestly. It records where
the work came from, what licence applies, what we changed, and what is ours.

---

## 1. Reference project

**Name:** LangGraph Multi-Agent Chatbot
**Repository:** <https://github.com/arbaz-builds/langgraph-multi-agent>
**Author / copyright holder:** Mohammad Arbaz — <https://github.com/arbaz-builds>
(`arbaz-builds` / `Pokemon455`)
**Licence:** MIT
**Retrieved as:** source archive / working copy (no Git history present locally —
see `docs/PROJECT_AUDIT.md` §2.1)

A copy of the upstream `LICENSE` is retained at the repository root:

```
MIT License
Copyright (c) 2026 Arbaz (Pokemon455)
```

## 2. Our project

**Name:** IntelliFlow — Enterprise Multi-Agent AI Assistant

IntelliFlow is a derivative work. It began as a fork of the reference project and
has been substantially reworked. See `docs/CHANGELOG.md` for the per-change
record and `docs/PROJECT_AUDIT.md` for the as-received baseline analysis.

### Authorship statement

The IntelliFlow modifications, the new architecture, the new subsystems, and all
IntelliFlow-specific documentation were written for this portfolio project.

**The reference project is not our original work and is not presented as such.**
No IntelliFlow documentation claims authorship of the upstream code. Where an
IntelliFlow feature depends on upstream design, this document or the changelog
says so explicitly.

---

## 3. Licence obligations

MIT requires that the copyright notice and permission notice be retained in all
copies or substantial portions of the software.

Our compliance position:

1. The original `LICENSE` file remains at the repository root, unaltered, with
   its original copyright line intact.
2. `NOTICE.md` (this project's attribution summary) is shipped alongside it and
   is linked from the README's attribution section.
3. Upstream copyright is preserved in this file and in the README.
4. The MIT terms continue to apply to the upstream portions of the codebase.

Any future IntelliFlow-specific licence header we add must be **additive** and
must not remove or replace the upstream MIT notice. We do not relicense upstream
code.

---

## 4. Reference architecture vs. IntelliFlow changes

This is the section to point an interviewer at.

### 4.1 Taken from the reference project (upstream work, credit to Arbaz)

| Component | Upstream location | Status in IntelliFlow |
|---|---|---|
| LangGraph 4-node graph shape (`router`, `llm_tool`, `tools`, `answer`) | `graph.py` | Retained conceptually; rebuilt in `app/agents/graph.py` |
| `add_messages` reducer on the `messages` state channel | `state.py` | Retained |
| Bounded tool loop via `MAX_ITERATIONS` + conditional edges | `nodes/conditions.py` | Retained and extended |
| `with_structured_output` + Pydantic model for routing | `nodes/router.py`, `state.py` | Retained and extended |
| `bind_tools` tool-calling node | `nodes/llm_tool.py` | Retained and reworked |
| FastAPI `GET /health` + `POST /chat` shape | `main.py` | Retained and expanded |
| LangGraph Postgres checkpointing for conversation memory | `main.py` | Retained and improved |
| Per-request Postgres connection rationale | `main.py` comments | Retained — see the note below |
| `_strip_images` multimodal sanitisation | `nodes/answer.py` | Retained |
| Explicit `as_node` on `aupdate_state` | `main.py` | Retained |
| Layered HTTP error mapping | `main.py` | Retained and extended |
| Empty-completion retry | `nodes/answer.py` | Retained |

### 4.2 Removed or replaced (our changes)

| Upstream element | Why removed | Replacement |
|---|---|---|
| `MockEmbeddings` (`llms.py`) | All-zero vectors make semantic retrieval meaningless | Configurable real embedding provider (`app/rag/embeddings.py`) |
| Pinecone wired directly inside `tools/rag.py` | No abstraction, no testability, client built per query | Vector-store factory behind a config flag (`app/rag/retriever.py`) |
| Flattened `Source N:` retrieval string | Metadata/citations discarded | Structured retrieval results with `document_id`, `filename`, `page_number`, `chunk_id`, `source`, `score` |
| No ingestion pipeline | Feature did not exist upstream | `app/rag/ingestion.py` — PDF/TXT extract → clean → chunk → embed → store |
| Binary `tools`/`answer` router | Too coarse to be explainable or testable | 6-category structured router with confidence (`app/agents/router.py`) |
| `f"RAG error: {e}"` returned as context | Leaks internal detail into LLM context and answers | Structured error results; sanitised messages |
| Hard-coded third-party MCP URL on Render | Project could not run standalone | First-party MCP server in this repository |
| `config.validate()` and client construction at import time | Broke any import without full prod credentials | Lazy, explicit service construction (`app/services/llm.py`) |
| Module-level mutable config globals | Unsuitable for tests or multi-env | Typed settings object (`app/core/config.py`) |
| `Undocumented` `logger` with no handler | Effectively no logging | Structured JSON logging + request/thread IDs (`app/core/logging.py`) |

### 4.3 New functionality implemented for IntelliFlow

Nothing in the following list exists in the reference project.

| Area | Implementation |
|---|---|
| Real embedding pipeline | `app/rag/embeddings.py` — provider factory, env-configured, failure handling |
| Document ingestion | `app/rag/ingestion.py` — PDF + TXT loaders, cleaning, chunking, metadata |
| Retrieval layer | `app/rag/retriever.py` — semantic search, relevance threshold, citations |
| Grounding guardrail | "Insufficient information" behaviour when retrieval is weak |
| Structured router | `app/agents/router.py` + Pydantic `RouterDecision` (route / reason / confidence) |
| Extended agent state | `app/agents/state.py` — tool calls, tool results, sources, errors, iteration cap |
| Database / SQL tool | `app/tools/database_tool.py` — guarded, parameterised, read-only |
| Enterprise demo dataset | `customers`, `products`, `orders`, `employees`, `sales` + seed data |
| Python / data-analysis tool | `app/tools/python_tool.py` — controlled execution boundary |
| MCP server | In-repo `mcp_server/` exposing safe tools (replaces third-party dependency) |
| API layer | `app/api/routes.py`, `app/api/schemas.py` — 5+ endpoints, OpenAPI, auth, validation |
| Memory layer | `app/memory/postgres.py` — thread-scoped persistent conversation state |
| Security | `app/core/security.py`, guardrails, auth, rate limiting, input validation |
| Observability | Request ID / thread ID / route / tool / duration / iteration logging |
| Evaluation | `app/evaluation/evaluator.py` + 25-30 queries, measured results only |
| Testing | Unit + integration suites with markers; real-embedding RAG tests |
| Docker | `Dockerfile` + `docker-compose.yml` (app + Postgres) |
| Frontend | Minimal React + Vite chat UI |
| Documentation | Full `docs/` set, incl. learning guide, RAG/MCP/security/eval docs |

> The file paths above are the **planned** targets from the project plan. As each
> phase lands, this table is updated to reflect what is genuinely implemented and
> tested. Nothing is listed as implemented before it works.

---

## 5. Third-party components

IntelliFlow builds on these upstream open-source projects, each under its own
licence:

| Project | Role | Licence |
|---|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent graph orchestration | MIT |
| [LangChain](https://github.com/langchain-ai/langchain) | LLM/tool/document abstractions | MIT |
| [FastAPI](https://github.com/fastapi/fastapi) | HTTP API | MIT |
| [Pydantic](https://github.com/pydantic/pydantic) | Validation / structured output | MIT |
| [Pinecone](https://github.com/pinecone-io/pinecone-python-client) | Managed vector database | Apache-2.0 (client) |
| [Tavily](https://github.com/tavily-ai/tavily-python) | Web search | MIT |
| [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) | MCP ⇄ LangChain bridge | MIT |
| [fastmcp](https://github.com/jlowin/fastmcp) | MCP server implementation | Apache-2.0 |
| [psycopg](https://github.com/psycopg/psycopg) | PostgreSQL driver | LGPL-3.0 (with OpenSSL exception) |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Local embedding models | Apache-2.0 |

Hosted services used at runtime (NVIDIA NIM, Tavily, Pinecone, LangSmith) are
**optional and configured by environment variable**. IntelliFlow does not vendor
or redistribute any of them. No hosted service is required to run the unit test
suite or to execute local embeddings.

---

## 6. Data and model licensing

- **Embeddings.** The default local model
  (`sentence-transformers/all-MiniLM-L6-v2`) is downloaded from Hugging Face at
  runtime and is Apache-2.0 licensed. Weights are not committed to this
  repository (`.gitignore` already excludes `*.bin`, `*.pt`, `*.pth`, `*.gguf`).
- **Demo dataset.** All IntelliFlow demo business data is synthetic and generated
  by us. It contains no real personal data, no real customer records, and no
  scraped content.
- **Documents used in RAG tests.** Test fixtures we author ourselves. No
  copyrighted documents are redistributed.

---

## 7. How to cite this project

If IntelliFlow is presented publicly, please credit both:

> IntelliFlow — Enterprise Multi-Agent AI Assistant.
> Built on the MIT-licensed
> [langgraph-multi-agent](https://github.com/arbaz-builds/langgraph-multi-agent)
> by Mohammad Arbaz, substantially reworked for portfolio purposes.

---

## 8. Documentation map

| Document | Purpose |
|---|---|
| `docs/PROJECT_AUDIT.md` | Phase 0 as-received baseline analysis |
| `docs/ORIGINAL_SETUP.md` | Exact baseline setup and baseline failures |
| `docs/PROJECT_ORIGIN.md` | This file — origin, licence, attribution |
| `docs/CHANGELOG.md` | Every meaningful modification, phase by phase |
| `docs/ARCHITECTURE.md` | IntelliFlow system architecture |
| `docs/RAG_ARCHITECTURE.md` | Embedding / chunking / retrieval design |
| `docs/MCP_ARCHITECTURE.md` | MCP client, server, and security boundary |
| `docs/SECURITY.md` | Threat model, protections, honest limitations |
| `docs/EVALUATION.md` | Evaluation methodology and measured results |
| `docs/LEARNING_GUIDE.md` | Concept-by-concept WHAT / WHY / HOW / WHERE |
| `README.md` | Project overview and usage |
