"""RAG tool — grounded retrieval over ingested documents.

Replaces the reference implementation, which embedded queries with all-zero
vectors and returned a flattened string with discarded metadata.

Behaviour changes versus the reference version:

- Real embeddings via the provider factory (``rag.embeddings``).
- Citation metadata preserved (``filename``, ``page_number``, ``chunk_id``).
- Relevance threshold: when no chunk clears it, the tool says so explicitly
  instead of returning weak near-misses.
- Retrieval errors are returned as a short, generic notice. The reference
  version interpolated ``f"RAG error: {e}"`` into LLM context, leaking
  internal detail into both the context and the user-visible answer
  (audit L4).
"""

from langchain_core.tools import tool

from rag.retriever import retrieve


@tool
def RAG(query: str) -> str:
    """Search the ingested document knowledge base.

    Use this for questions about uploaded/internal documents. Returns cited
    excerpts, or a clear notice when the documents do not cover the topic.
    """
    try:
        result = retrieve(query)
    except Exception:
        # Deliberately opaque. Do not interpolate the exception: it can carry
        # connection strings, index names, or library internals into the
        # model's context and from there into the user-visible answer.
        return (
            "Document search is currently unavailable. Answer without "
            "referencing the knowledge base, and tell the user the search "
            "could not be completed."
        )

    if not result.has_evidence:
        return (
            "No relevant information was found in the knowledge base for "
            f"this query. {result.reason} Do not guess - tell the user the "
            "documents do not cover this."
        )

    parts = [result.context_block(), "", "Sources: " + "; ".join(result.citations())]
    return "\n".join(parts)
