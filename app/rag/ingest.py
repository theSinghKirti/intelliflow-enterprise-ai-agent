"""CLI for the RAG pipeline: ingest documents into the vector store.

Usage
-----
List the configured providers and dimensions::

    python -m app.rag.ingest

Ingest one or more files::

    python -m app.rag.ingest data/handbook.pdf data/policy.txt

Ingest every supported file in a directory::

    python -m app.rag.ingest --directory data/

Check what is stored::

    python -m app.rag.ingest --search "what is the refund window?"

Vector store
------------
Defaults to ``VECTOR_STORE=memory``, which is per-process: documents ingested
by this CLI are **not** visible to a separately started API server. For
persistence set ``VECTOR_STORE=pinecone`` in the environment before ingesting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.rag.embeddings import describe
from app.rag.ingestion import ingest_document
from app.rag.retriever import retrieve

SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md")


def _collect(paths: list[str], directory: str | None) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        found.append(Path(raw))
    if directory:
        base = Path(directory)
        if not base.is_dir():
            print(f"error: {base} is not a directory", file=sys.stderr)
            return []
        found.extend(
            sorted(
                p for p in base.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            )
        )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.ingest",
        description="Ingest documents into the RAG vector store.",
    )
    parser.add_argument("paths", nargs="*", help="Document paths (.pdf/.txt/.md)")
    parser.add_argument(
        "--directory", "-d", help="Recursively ingest all supported files in a directory"
    )
    parser.add_argument(
        "--search", "-s", help="Run a retrieval query against the store instead"
    )
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args(argv)

    info = describe()
    print(
        f"embeddings: provider={info['provider']} model={info['model']} "
        f"dim={info['dimension']} api_key_required={info['requires_api_key']}"
    )
    import os

    print(f"vector store: {os.getenv('VECTOR_STORE', 'memory')}")
    print()

    if args.search:
        result = retrieve(args.search)
        print(f"query: {args.search!r}")
        print(f"reason: {result.reason}")
        if not result.has_evidence:
            print("no evidence above threshold - the agent should decline to answer")
            return 0
        for chunk in result.chunks:
            print(f"  {chunk.score:.3f}  {chunk.source}#{chunk.chunk_id}")
            print(f"         {chunk.text[:110]}...")
        return 0

    targets = _collect(args.paths, args.directory)
    if not targets:
        print("nothing to ingest - pass file paths or --directory", file=sys.stderr)
        return 1

    failures = 0
    for target in targets:
        try:
            result = ingest_document(
                target,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            print(f"OK   {result.summary()}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {target}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if os.getenv("VECTOR_STORE", "memory").lower() == "memory" and not failures:
        print()
        print(
            "note: VECTOR_STORE=memory is per-process. These chunks are not "
            "visible to a separately started server - set VECTOR_STORE=pinecone "
            "to persist."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
