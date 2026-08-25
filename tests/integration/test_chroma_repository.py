"""Integration tests for the ChromaDB vector store repository.

Uses synthetic unit vectors along coordinate axes so similarity ordering is
exact and deterministic without any embedding model involvement.
"""

from pathlib import Path

import pytest

from app.errors import VectorStoreError
from app.ingestion.models import Chunk
from app.vectordb.chroma import ChromaVectorStoreRepository

DIM = 4
AXES = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _chunk(
    document_id: str, index: int, text: str, page_number: int | None = 1
) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}::{index}",
        document_id=document_id,
        chunk_index=index,
        text=text,
        page_number=page_number,
        filename=f"{document_id}.txt",
        source=f"data/uploads/{document_id}.txt",
    )


def _repo(tmp_path: Path) -> ChromaVectorStoreRepository:
    return ChromaVectorStoreRepository(tmp_path / "chroma")


def test_roundtrip_and_ranking(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    chunks = [_chunk("docA", i, f"text {i}") for i in range(3)]

    repo.add_chunks(chunks, AXES[:3])

    # Query leaning toward axis 0 must surface chunk 0 first.
    results = repo.similarity_search([0.9, 0.1, 0.0, 0.0], top_k=3)

    assert results[0].metadata["chunk_id"] == "docA::0"
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_result_count(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.add_chunks(
        [_chunk("docA", i, f"text {i}") for i in range(3)], AXES[:3]
    )

    results = repo.similarity_search(AXES[0], top_k=2)

    assert len(results) == 2


def test_identical_vector_scores_near_perfect_similarity(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.add_chunks([_chunk("docA", 0, "text")], [AXES[1]])

    result = repo.similarity_search(AXES[1], top_k=1)

    assert abs(result[0].score - 1.0) < 1e-6


def test_upsert_is_idempotent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    chunks = [_chunk("docA", 0, "same text")]
    embeddings = [AXES[0]]

    repo.add_chunks(chunks, embeddings)
    repo.add_chunks(chunks, embeddings)

    assert repo.count() == 1


def test_metadata_filters_restrict_results(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.add_chunks(
        [_chunk("docA", 0, "alpha"), _chunk("docA", 1, "beta")], AXES[:2]
    )
    repo.add_chunks([_chunk("docB", 0, "gamma")], [AXES[2]])

    by_document = repo.similarity_search(
        AXES[0], top_k=10, where={"document_id": "docB"}
    )
    by_filename = repo.similarity_search(
        AXES[0], top_k=10, where={"filename": "docA.txt"}
    )

    assert [result.metadata["document_id"] for result in by_document] == ["docB"]
    assert {
        result.metadata["chunk_id"] for result in by_filename
    } == {"docA::0", "docA::1"}


def test_delete_document_removes_only_its_chunks(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.add_chunks(
        [_chunk("docA", 0, "one"), _chunk("docA", 1, "two")], AXES[:2]
    )
    repo.add_chunks([_chunk("docB", 0, "three")], [AXES[2]])

    repo.delete_document("docA")

    # Chroma delete() may return None; verify removal via count().
    assert repo.count() == 1
    # Deleting an already-removed document is a harmless no-op.
    repo.delete_document("docA")
    assert repo.count() == 1


def test_persistence_across_repository_instances(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    first = ChromaVectorStoreRepository(persist_dir)
    first.add_chunks(
        [_chunk("docA", 0, "durable text")], [AXES[0]]
    )

    reopened = ChromaVectorStoreRepository(persist_dir)
    results = reopened.similarity_search(AXES[0], top_k=5)

    assert reopened.count() == 1
    assert len(results) == 1
    assert results[0].text == "durable text"


def test_dimension_mismatch_raises_typed_error(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.add_chunks(
        [_chunk("docA", 0, "four dims")], [AXES[0]]
    )

    with pytest.raises(VectorStoreError):
        repo.add_chunks(
            [_chunk("docA", 1, "three dims")], [[0.1, 0.2, 0.3]]
        )


def test_missing_page_number_round_trips_as_none(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.add_chunks(
        [
            _chunk("docA", 0, "paged", page_number=7),
            _chunk("docA", 1, "unpaged", page_number=None),
        ],
        AXES[:2],
    )

    paged = repo.similarity_search(AXES[0], top_k=1)[0].metadata
    unpaged = repo.similarity_search(AXES[1], top_k=1)[0].metadata

    assert paged["page_number"] == 7
    assert unpaged["page_number"] is None


def test_search_on_empty_store_returns_empty_list(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    assert repo.similarity_search(AXES[0], top_k=5) == []
    assert repo.count() == 0


def test_add_length_mismatch_raises_value_error(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    with pytest.raises(ValueError):
        repo.add_chunks([_chunk("docA", 0, "text")], [])
