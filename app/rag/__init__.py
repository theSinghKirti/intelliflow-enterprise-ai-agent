"""RAG subsystem: real embeddings, ingestion, and grounded retrieval.

Replaces the reference project's placeholder retrieval path, which embedded
with all-zero vectors and had no ingestion code at all.
"""

from app.rag.embeddings import (
    EmbeddingConfigurationError,
    describe,
    get_embedding_dimension,
    get_embeddings,
    get_provider,
)
from app.rag.ingestion import ingest_document, ingest_documents
from app.rag.models import (
    ChunkMetadata,
    IngestionResult,
    RetrievedChunk,
    RetrievalResult,
)
from app.rag.retriever import get_store, reset_store, retrieve

__all__ = [
    "ChunkMetadata",
    "EmbeddingConfigurationError",
    "IngestionResult",
    "RetrievedChunk",
    "RetrievalResult",
    "describe",
    "get_embedding_dimension",
    "get_embeddings",
    "get_provider",
    "get_store",
    "ingest_document",
    "ingest_documents",
    "reset_store",
    "retrieve",
]
