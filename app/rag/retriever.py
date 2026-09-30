"""Semantic retrieval with grounding.

The reference project returned a flattened string from the tool (audit P3)
and had no relevance threshold (audit L10). Worse still, the retrieval model
itself was meaningless. This module returns structured results, preserves
citation metadata, and applies a configurable relevance threshold before
passing evidence to the LLM.

Grounding rule
--------------
If the best match falls below ``RETRIEVAL_THRESHOLD``, ``has_evidence`` is
``False``. The caller must answer: "I don't have enough information in the
provided documents to answer that." â€” not hallucinate from near-misses. The
``context_block()`` produced by :class:`rag.models.RetrievalResult` renders
this explicitly.

Prompt injection hygiene
------------------------
The context block is fenced, labelled as reference data, and the model is
instructed not to follow any directives it may contain (L11). Citation IDs
are preserved (``[filename#chunk_id]``).

Design
------
- Backed by ``VectorStore`` implementations.
- Defaults to ``InMemoryVectorStore`` (no credentials) so unit tests run
  offline.
- Supports Pinecone via the existing LangChain adapter without leaking
  provider-specific details into callers.
- Returns similarity scores (cosine) as produced by the store. The
  threshold is validated against measured behaviour of the chosen embedding
  model.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_pinecone import PineconeVectorStore

from app.rag.embeddings import get_embeddings
from app.rag.models import RetrievedChunk, RetrievalResult


def _get_retrieval_threshold() -> float:
    val = os.getenv("RETRIEVAL_THRESHOLD", "0.30")
    try:
        thr = float(val)
    except ValueError:
        thr = 0.30
    if thr < 0.0:
        thr = 0.0
    if thr > 1.0:
        thr = 1.0
    return thr


def _get_k() -> int:
    try:
        return max(1, int(os.getenv("RETRIEVER_K", "5")))
    except ValueError:
        return 5


def _get_fetch_k() -> int:
    try:
        return max(1, int(os.getenv("RETRIEVER_FETCH_K", "20")))
    except ValueError:
        return 20


def _document_to_chunk(doc: Document, score: float) -> RetrievedChunk | None:
    meta = doc.metadata or {}
    filename = meta.get("filename") or meta.get("source") or "unknown"
    chunk_id = meta.get("chunk_id") or ""
    document_id = meta.get("document_id") or ""

    page_number = meta.get("page_number")
    if isinstance(page_number, float):
        page_number = int(page_number)
    if page_number is not None and not isinstance(page_number, int):
        page_number = None

    text = (doc.page_content or "").strip()
    if not text and not chunk_id:
        return None

    return RetrievedChunk(
        text=text,
        score=score,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        chunk_id=chunk_id,
    )


def _build_store() -> VectorStore:
    store_type = os.getenv("VECTOR_STORE", "memory").strip().lower()

    embeddings = get_embeddings()

    if store_type == "memory":
        return InMemoryVectorStore(embedding=embeddings)

    if store_type == "pinecone":
        index_name = os.getenv("PINECONE_INDEX")
        api_key = os.getenv("PINECONE_API_KEY")
        if not index_name:
            raise RuntimeError("VECTOR_STORE=pinecone requires PINECONE_INDEX")
        if not api_key:
            raise RuntimeError("VECTOR_STORE=pinecone requires PINECONE_API_KEY")
        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=api_key,
        )

    raise ValueError(f"Unknown VECTOR_STORE {store_type!r}. Expected memory|pinecone")


_store_cache: VectorStore | None = None


def _get_store() -> VectorStore:
    """Return the process-wide store singleton.

    Caching matters: a fresh ``InMemoryVectorStore`` per call would discard
    everything ingested moments earlier, so an in-process ingest-then-search
    flow would silently return nothing. Hosted stores (Pinecone) are stateless
    clients, so caching them is safe and avoids re-initialising a gRPC
    channel on every query (audit L18).
    """
    global _store_cache
    if _store_cache is None:
        _store_cache = _build_store()
    return _store_cache


def reset_store() -> None:
    """Drop the cached store. Used by tests to isolate cases."""
    global _store_cache
    _store_cache = None


def retrieve(
    query: str,
    *,
    k: int | None = None,
    fetch_k: int | None = None,
    threshold: float | None = None,
    store: VectorStore | None = None,
) -> RetrievalResult:
    """Retrieve relevant chunks for ``query``.

    Uses MMR retrieval against the configured vector store. If the best
    similarity score is below ``threshold``, the result has ``has_evidence``
    ``False``. No chunks are returned in that case; the ``chunks`` tuple is
    empty.

    Parameters
    ----------
    store:
        Override the process-wide store. Lets a caller query an arbitrary
        corpus without mutating global state, which is how the tests keep
        per-case corpora isolated.
    """
    q = (query or "").strip()
    if not q:
        return RetrievalResult(
            query="",
            chunks=(),
            candidates_considered=0,
            threshold=threshold if threshold is not None else _get_retrieval_threshold(),
            reason="Empty query provided.",
        )

    k_eff = k if k is not None else _get_k()
    fetch_k_eff = fetch_k if fetch_k is not None else _get_fetch_k()
    thr = threshold if threshold is not None else _get_retrieval_threshold()

    target = store if store is not None else _get_store()

    raw = target.similarity_search_with_score(
        q,
        k=k_eff,
        fetch_k=fetch_k_eff,
        lambda_mult=0.7,
        search_type="mmr",
    )

    candidates_considered = len(raw)
    accepted: list[RetrievedChunk] = []
    best_score: float | None = None
    for doc, score in raw:
        chunk = _document_to_chunk(doc, score)
        if chunk is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
        if score >= thr:
            accepted.append(chunk)

    if best_score is None or best_score < thr:
        observed = (
            "no candidates returned"
            if best_score is None
            else f"best score {best_score:.3f}"
        )
        reason = (
            f"No evidence above threshold {thr:.2f} "
            f"({observed} over {candidates_considered} candidate(s))."
        )
        return RetrievalResult(
            query=q,
            chunks=(),
            candidates_considered=candidates_considered,
            threshold=thr,
            reason=reason,
        )

    accepted.sort(key=lambda c: c.score, reverse=True)
    top = accepted[:k_eff]

    reason = (
        f"Found {len(top)} evidence chunk(s) at/above threshold {thr:.2f} "
        f"(best score {best_score:.3f}, candidates={candidates_considered})."
    )
    return RetrievalResult(
        query=q,
        chunks=tuple(top),
        candidates_considered=candidates_considered,
        threshold=thr,
        reason=reason,
    )


def get_store() -> VectorStore:
    """Expose the configured store for advanced use cases."""
    return _get_store()
