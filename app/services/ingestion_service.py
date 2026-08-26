"""End-to-end ingestion orchestration.

Coordinates the full pipeline: load document → chunk → embed → store,
with idempotent re-ingest and per-file error isolation in batch mode.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Literal

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunking import chunk_document
from app.ingestion.loaders import load_document
from app.ingestion.models import IngestionResult
from app.vectordb.base import VectorStoreRepository

logger = logging.getLogger(__name__)

ChunkingStrategy = Literal["recursive", "fixed"]


class IngestionService:
    """Orchestrates the load → chunk → embed → store pipeline.

    Dependencies (embedder, vector store) are injected so the service is
    testable without any real model or database.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStoreRepository,
        *,
        chunk_size: int,
        chunk_overlap: int,
        chunk_strategy: ChunkingStrategy = "recursive",
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunk_strategy = chunk_strategy

    def ingest_file(
        self,
        path: Path,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: ChunkingStrategy | None = None,
    ) -> IngestionResult:
        """Ingest one document file through the full pipeline.

        Returns an :class:`IngestionResult` with the generated document ID,
        chunk count, and elapsed time.  The document ID is a stable hash of
        the resolved file path, so re-ingesting the same file deletes the
        prior chunks first (idempotent upsert).
        """
        document_id = hashlib.sha256(path.resolve().as_posix().encode()).hexdigest()[:32]
        effective_size = chunk_size if chunk_size is not None else self._chunk_size
        effective_overlap = chunk_overlap if chunk_overlap is not None else self._chunk_overlap
        effective_strategy = strategy if strategy is not None else self._chunk_strategy

        t0 = time.perf_counter()
        loaded = load_document(path)

        chunks = chunk_document(
            loaded,
            document_id,
            chunk_size=effective_size,
            chunk_overlap=effective_overlap,
            chunk_strategy=effective_strategy,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not chunks:
            logger.info("Empty document after chunking: %s", loaded.filename)
            return IngestionResult(
                document_id=document_id,
                filename=loaded.filename,
                n_chunks=0,
                status="empty",
                elapsed_ms=elapsed_ms,
            )

        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])

        # Delete any previous chunks for this document before upserting,
        # so the caller can re-ingest the same file without duplicates.
        self._store.delete_document(document_id)
        self._store.add_chunks(chunks, embeddings)

        logger.info(
            "Ingested %s → %d chunks in %.1f ms",
            loaded.filename,
            len(chunks),
            elapsed_ms,
        )
        return IngestionResult(
            document_id=document_id,
            filename=loaded.filename,
            n_chunks=len(chunks),
            status="ok",
            elapsed_ms=elapsed_ms,
        )

    def ingest_batch(
        self,
        paths: list[Path],
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: ChunkingStrategy | None = None,
    ) -> list[IngestionResult]:
        """Ingest multiple files, continuing past individual failures."""
        results: list[IngestionResult] = []
        for path in paths:
            try:
                result = self.ingest_file(
                    path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    strategy=strategy,
                )
                results.append(result)
            except Exception as exc:
                logger.exception("Failed to ingest %s", path)
                results.append(
                    IngestionResult(
                        document_id="",
                        filename=path.name,
                        n_chunks=0,
                        status=f"error: {type(exc).__name__}",
                        elapsed_ms=0.0,
                    )
                )
        return results


def create_ingestion_service(
    settings: Settings,
    embedder: EmbeddingProvider,
    store: VectorStoreRepository,
) -> IngestionService:
    """Build an IngestionService from application settings and injected deps."""
    return IngestionService(
        embedder=embedder,
        store=store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunk_strategy=settings.chunk_strategy,
    )
