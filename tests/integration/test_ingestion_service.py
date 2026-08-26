"""Integration tests for the ingestion service.

Uses FakeEmbeddingProvider (no network) and real ChromaDB (tmp_path) so the
full pipeline runs end-to-end without any model or external dependency.
"""

from pathlib import Path

import pytest

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.services.ingestion_service import IngestionService, create_ingestion_service
from app.vectordb.base import VectorStoreRepository
from app.vectordb.chroma import ChromaVectorStoreRepository
from tests.utils.fakes import FakeEmbeddingProvider


def _txt(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _repo(tmp_path: Path) -> VectorStoreRepository:
    return ChromaVectorStoreRepository(tmp_path / "chroma")


def _embedder() -> EmbeddingProvider:
    return FakeEmbeddingProvider()


def _service(
    tmp_path: Path,
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
) -> tuple[IngestionService, VectorStoreRepository]:
    store = _repo(tmp_path)
    return create_ingestion_service(
        Settings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        embedder=_embedder(),
        store=store,
    ), store


# --- Tests ------------------------------------------------------------------


def test_happy_path_returns_ok(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    path = _txt(tmp_path, "doc.txt", "Hello world. This is a test document.")

    result = service.ingest_file(path)

    assert result.status == "ok"
    assert result.n_chunks >= 1
    assert result.filename == "doc.txt"
    assert result.document_id
    assert result.elapsed_ms > 0
    assert store.count() == result.n_chunks


def test_reingest_is_idempotent(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    path = _txt(tmp_path, "doc.txt", "Some content that will be ingested twice.")

    result1 = service.ingest_file(path)
    count_after_first = store.count()
    result2 = service.ingest_file(path)

    # Each call creates a new document_id, so old chunks are deleted first.
    assert store.count() == result2.n_chunks
    assert count_after_first == result1.n_chunks


def test_empty_document_returns_empty_status(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    path = _txt(tmp_path, "blank.txt", "   ")

    # load_document will raise EmptyDocumentError for whitespace-only files;
    # the batch path captures this, but ingest_file should propagate it.
    from app.errors import EmptyDocumentError
    with pytest.raises(EmptyDocumentError):
        service.ingest_file(path)


def test_batch_continues_past_failures(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    good = _txt(tmp_path, "ok.txt", "Valid content for ingestion.")
    bad = tmp_path / "missing.txt"  # does not exist

    results = service.ingest_batch([bad, good])

    assert len(results) == 2
    assert results[0].status.startswith("error:")
    assert results[0].n_chunks == 0
    assert results[1].status == "ok"
    assert results[1].n_chunks >= 1
    assert store.count() == results[1].n_chunks


def test_strategy_flows_through(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, chunk_size=200, chunk_overlap=40)
    path = _txt(tmp_path, "doc.txt", "Word " * 200)

    result_fixed = service.ingest_file(path, strategy="fixed")
    result_recursive = service.ingest_file(path, strategy="recursive")

    # Both strategies produce chunks; exact count differs.
    assert result_fixed.n_chunks > 0
    assert result_recursive.n_chunks > 0


def test_create_ingestion_service_uses_settings(tmp_path: Path) -> None:
    settings = Settings(chunk_size=50, chunk_overlap=10)
    service = create_ingestion_service(settings, _embedder(), _repo(tmp_path))
    path = _txt(tmp_path, "doc.txt", "A " * 100)

    result = service.ingest_file(path)

    # chunk_size=50 → more chunks than default 1000
    assert result.n_chunks > 1
