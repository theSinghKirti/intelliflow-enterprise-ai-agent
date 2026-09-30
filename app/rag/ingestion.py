"""Document ingestion â€” load â†’ clean â†’ chunk â†’ embed â†’ store.

The upstream reference project had **no ingestion pipeline at all** (audit P2).
This is the first time this codebase can actually write to a vector store from
its own code. The design is test-first: it works with an in-memory vector store
(no credentials) so the test suite can verify semantics, and also with Pinecone
or a pluggable backend.

Supported formats (minimal, defensible):
- ``.txt``, ``.md`` â€” UTF-8 text
- ``.pdf`` â€” ``pypdf.PdfReader``

Chunking:
- ``RecursiveCharacterTextSplitter`` (from ``langchain-text-splitters``),
  with sensible defaults for retrieval.

Metadata:
- ``document_id`` â€” stable UUID per document
- ``filename``
- ``page_number`` â€” ``None`` for text files, 1-based for PDFs
- ``chunk_id`` â€” deterministic ``{document_id}:{chunk_index}``
- ``source`` â€” human-readable string, duplicated for convenience

Citations are preserved at retrieval time (audit P3).

The cleaning step removes null bytes and normalises line endings to prevent
embedding noise and metadata corruption.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import re
import uuid
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.rag.embeddings import get_embedding_dimension
from app.rag.models import ChunkMetadata, IngestionResult
from app.rag.retriever import get_store


def _loop_is_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def _await(awaitable):  # type: no[no-untyped-def]
    return await awaitable


_clean_nulls = re.compile(r"\x00")
_line_normalise = re.compile(r"\r\n|\r")


def _clean_text(text: str) -> str:
    text = _clean_nulls.sub("", text)
    text = _line_normalise.sub("\n", text)
    return text


def _read_text_file(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    # UTF-8 by default; if it's binary-like, still decode as utf-8 with
    # replacement rather than raising, so ingestion does not crash on odd files.
    text = data.decode("utf-8", errors="replace")
    text = _clean_text(text)
    pages = 1
    return text, pages


def _read_pdf_file(path: Path) -> tuple[list[str], int]:
    """Extract text per page.

    Returns one string per page (plus the page count) rather than a single
    concatenated blob. Page boundaries have to survive extraction so that each
    chunk can be attributed to the page it actually came from - a citation
    that names the wrong page is worse than no page number at all.
    """
    buffer = io.BytesIO(path.read_bytes())
    reader = PdfReader(buffer)
    pages_list = list(reader.pages)
    per_page: list[str] = []
    for page in pages_list:
        try:
            per_page.append(_clean_text(page.extract_text() or ""))
        except Exception:  # pragma: no cover - library robustness
            per_page.append("")
    return per_page, len(pages_list)


def _load_document(path: Path) -> tuple[list[str], int, str]:
    """Load a document as a list of page-sized segments.

    Returns ``(segments, page_count, filetype)``. A text file yields a single
    segment with ``page_number=None``; a PDF yields one segment per page so
    per-page citations stay accurate.
    """
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        text, pages = _read_text_file(path)
        return [text], pages, suffix.lstrip(".")
    if suffix == ".pdf":
        segments, pages = _read_pdf_file(path)
        return segments, pages, "pdf"
    raise ValueError(f"Unsupported document type: {suffix} ({path.name})")


def _chunk_text(
    text: str,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    separators: tuple[str, ...] = (
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ),
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
        length_function=len,
    )
    return splitter.split_text(text)


def ingest_document(
    path: str | Path,
    *,
    store: VectorStore | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> IngestionResult:
    """Ingest a single document into ``store``.

    Returns an :class:`IngestionResult` with counts and the embedding dimension
    that was actually used.

    Parameters
    ----------
    path:
        Path to ``.pdf``, ``.txt``, or ``.md``.
    store:
        Target vector store. If ``None``, uses an :class:`InMemoryVectorStore`
        (no credentials required, perfect for tests).
    chunk_size:
        Characters per chunk.
    chunk_overlap:
        Overlap between consecutive chunks.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")

    segments, page_count, filetype = _load_document(p)

    document_id = uuid.uuid4().hex
    filename = p.name
    dim = get_embedding_dimension()

    documents: list[Document] = []
    total_chars = 0
    index = 0

    for page_index, segment in enumerate(segments):
        # Text files have no meaningful page concept, so only PDFs get numbers.
        page_number = (page_index + 1) if filetype == "pdf" else None
        total_chars += len(segment)

        page_chunks = _chunk_text(
            segment, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not page_chunks and segment.strip():
            page_chunks = [segment]

        for chunk_text in page_chunks:
            chunk_id = f"{document_id}:{index}"
            metadata: ChunkMetadata = {
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "source": filename,
                "char_count": len(chunk_text),
            }
            documents.append(Document(page_content=chunk_text, metadata=metadata))
            index += 1

    if not documents:
        # Preserve the document as a single empty-content chunk so the
        # ingestion is still recorded and auditable.
        documents.append(
            Document(
                page_content="",
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": 1 if filetype == "pdf" else None,
                    "chunk_id": f"{document_id}:0",
                    "source": filename,
                    "char_count": 0,
                },
            )
        )

    # Default to the shared store so an ingest-then-search flow in the same
    # process is actually visible to retrieval.
    target = store if store is not None else get_store()

    outcome = target.add_documents(documents)
    if inspect.isawaitable(outcome):
        # A hosted store (Pinecone) exposes an async ``add_documents``. The
        # ingestion entry points are synchronous for CLI use, so drive the
        # coroutine here. Guard against being called from a live loop.
        if _loop_is_running():
            raise RuntimeError(
                "ingest_document() was called from a running event loop. "
                "Use astore.add_documents() directly from async code."
            )
        asyncio.run(_await(outcome))

    return IngestionResult(
        document_id=document_id,
        filename=filename,
        chunk_count=len(documents),
        page_count=page_count,
        total_chars=total_chars,
        embedding_dimension=dim,
    )


def ingest_documents(
    paths: Iterable[str | Path],
    *,
    store: VectorStore | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[IngestionResult]:
    """Ingest multiple documents in order."""
    results: list[IngestionResult] = []
    for p in paths:
        results.append(
            ingest_document(
                p,
                store=store,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return results
