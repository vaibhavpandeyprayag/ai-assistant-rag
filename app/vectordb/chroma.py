"""ChromaDB-backed implementation of the VectorStoreRepository port.

This is the only module aware of ChromaDB specifics; everything else works
against :class:`app.vectordb.base.VectorStoreRepository`.
"""

from pathlib import Path
from typing import Any

import chromadb

from app.errors import VectorStoreError
from app.ingestion.models import Chunk
from app.vectordb.base import RetrievedChunk

#: Single collection holding all indexed document chunks.
COLLECTION_NAME = "documents"

#: Chroma metadata cannot store nulls; unknown pages are persisted as -1.
_MISSING_PAGE_SENTINEL = -1


def _metadata_for_storage(chunk: Chunk) -> dict[str, Any]:
    """Build a flat, Chroma-safe metadata dict from a chunk."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "page_number": (
            _MISSING_PAGE_SENTINEL if chunk.page_number is None else chunk.page_number
        ),
    }


def _metadata_after_read(metadata: dict[str, Any]) -> dict[str, Any]:
    """Restore caller-friendly metadata (sentinel back to None)."""
    restored = dict(metadata)
    if restored.get("page_number") == _MISSING_PAGE_SENTINEL:
        restored["page_number"] = None
    return restored


class ChromaVectorStoreRepository:
    """Persistent local vector store using an embedded ChromaDB client."""

    def __init__(
        self,
        persist_directory: Path | str,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        # Cosine space matches our unit-normalized embedding vectors.
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks with precomputed vectors (same order as ``chunks``)."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=[_metadata_for_storage(chunk) for chunk in chunks],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to store chunks in the vector store ({type(exc).__name__})."
            ) from exc

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest-neighbor search; results ordered most-similar first."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Vector store query failed ({type(exc).__name__})."
            ) from exc

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            RetrievedChunk(
                text=text,
                score=1.0 - distance,
                metadata=_metadata_after_read(metadata),
            )
            for text, metadata, distance in zip(
                documents, metadatas, distances, strict=True
            )
        ]

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks of one document; returns how many were removed."""
        try:
            result = self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete document from vector store ({type(exc).__name__})."
            ) from exc
        return result if isinstance(result, int) else 0

    def count(self) -> int:
        """Total number of stored chunks."""
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to read vector store count ({type(exc).__name__})."
            ) from exc
