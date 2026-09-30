# API Reference

Two endpoints. The application is built by a factory in
[`app/api/routes.py`](../app/api/routes.py); run it with
`uvicorn app.main:app`.

## `GET /health`

Liveness and configuration probe. **Requires no credential.**

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "2.0.0",
  "embeddings": "local (384-dim)",
  "vector_store": "memory",
  "memory": "postgres"
}
```

| Field | Meaning |
|---|---|
| `status` | Always `ok` if the process is serving |
| `version` | Application version |
| `embeddings` | Active provider and its vector dimension |
| `vector_store` | `memory` or `pinecone` |
| `memory` | `postgres` when `DATABASE_URL` is set, else `unconfigured` |

This reports **configuration, not dependency health**. It does not ping
PostgreSQL or the model provider; a probe that called a database on every
request would be a load amplifier. Failures surface on `POST /chat`.

## `POST /chat`

Runs one agent turn.

### Request

| Field | Type | Required | Rules |
|---|---|---|---|
| `query` | string | yes | 1–4000 characters, and not whitespace-only |
| `thread_id` | string | no | ≤128 characters; letters, digits, `-`, `_` only. Omitted means a fresh conversation (`anon-<DEFAULT_THREAD_ID>`) |
| `image_base64` | string | no | Data URI or raw base64; ≤5,000,000 characters; must decode. **Validated only — see the note below** |

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "what is the refund window?", "thread_id": "user_123"}'
```

> `image_base64` is validated for size and base64 well-formedness, and
> malformed input is rejected with `422`. **No vision model is invoked and the
> image is not routed to the graph.** There is no vision path in this build.

### Response `200`

| Field | Type | Meaning |
|---|---|---|
| `response` | string | The assistant's reply |
| `thread_id` | string | Thread the turn was stored under |
| `request_id` | string | Correlation id, matching the `X-Request-ID` header |
| `route` | string \| null | One of `rag`, `web_search`, `database`, `python`, `direct`, `clarify` |
| `route_confidence` | float \| null | Router's confidence, 0.0–1.0 |
| `grounded` | bool \| null | Whether retrieval produced evidence |
| `insufficient_information` | bool \| null | Whether retrieval found nothing above the threshold |
| `sources` | array | `{ filename, page_number, chunk_id, score }` per citation |
| `tool_calls` | array | Names of the tools invoked |
| `iterations` | int | Tool-loop iterations used |

```json
{
  "response": "Unopened items can be returned within 30 days of delivery for a full refund.",
  "thread_id": "user_123",
  "request_id": "3f9c1b7d2e4a5061",
  "route": "rag",
  "route_confidence": 0.9,
  "grounded": true,
  "insufficient_information": false,
  "sources": [
    {"filename": "policy.txt", "page_number": null, "chunk_id": "a1b2c3:0", "score": 0.494}
  ],
  "tool_calls": ["RAG"],
  "iterations": 1
}
```

### Errors

| Status | When |
|---|---|
| `422` | Request validation failed (see the rules above) |
| `503` | `DATABASE_URL` is not configured, or the model provider is unreachable |
| `500` | Unhandled failure. The response body is always `{"detail": "..."}` |

Exception text and stack traces are never returned to the caller. A failed
tool reports its failure to the model as a type name only.

## Request correlation

Every response carries an `X-Request-ID` header. A caller-supplied id (≤64
characters) is echoed; otherwise one is generated. The id is also bound to the
logging context, so every log line emitted while handling the request carries
it.

## Tools

### `RAG(query)`

Grounded retrieval over ingested documents. Returns fenced, cited excerpts, or
an explicit notice when the documents do not cover the topic. A retrieval
failure returns a short generic notice — the exception is never interpolated
into model context.

### `search_web(query, max_results=5)`

Tavily wrapper with result normalisation, a snippet and total-length cap, and
sanitised failures. With no `TAVILY_API_KEY` it returns a "web search is not
configured" notice instead of failing the turn.

### MCP tools

Discovered from `MCP_SERVER_URL` when set. With no server configured, no MCP
tools are bound and the graph still builds. This project ships no MCP server.

## Ingestion

Not an HTTP endpoint. Run the CLI:

```bash
python -m app.rag.ingest data/handbook.pdf data/policy.txt
python -m app.rag.ingest --directory data/
python -m app.rag.ingest --search "what is the refund window?"   # query the store
```

`VECTOR_STORE=memory` is per-process, so the CLI and a running server do not
share a store.
