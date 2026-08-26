"""Integration tests for the retrieval service.

Uses FakeEmbeddingProvider (similarity-preserving, no network) and real
ChromaDB (tmp_path) so the full retrieval pipeline runs end-to-end.
"""

from pathlib import Path

import pytest

from app.ingestion.models import Chunk
from app.retrieval.retriever import Retriever
from app.vectordb.chroma import ChromaVectorStoreRepository
from tests.utils.fakes import FakeEmbeddingProvider

SEED_CHUNKS = [
    Chunk(chunk_id="docA::0", document_id="docA", chunk_index=0,
          text="How to reset your account password step by step.",
          page_number=1, filename="guide.txt", source="guide.txt"),
    Chunk(chunk_id="docA::1", document_id="docA", chunk_index=1,
          text="Your billing invoice is available in the account dashboard.",
          page_number=2, filename="guide.txt", source="guide.txt"),
    Chunk(chunk_id="docB::0", document_id="docB", chunk_index=0,
          text="Password recovery instructions for locked accounts.",
          page_number=1, filename="help.txt", source="help.txt"),
    Chunk(chunk_id="docC::0", document_id="docC", chunk_index=0,
          text="Our restaurant serves Italian food and fresh pasta daily.",
          page_number=1, filename="menu.txt", source="menu.txt"),
]


def _setup(tmp_path: Path) -> tuple[Retriever, FakeEmbeddingProvider]:
    embedder = FakeEmbeddingProvider()
    store = ChromaVectorStoreRepository(tmp_path / "chroma")
    embeddings = embedder.embed_documents([c.text for c in SEED_CHUNKS])
    store.add_chunks(SEED_CHUNKS, embeddings)
    retriever = Retriever(embedder, store, top_k=5)
    return retriever, embedder


def test_related_text_ranks_higher(tmp_path: Path) -> None:
    retriever, _ = _setup(tmp_path)

    results = retriever.retrieve("password reset")

    assert len(results) >= 2
    top_two_ids = {results[0].metadata["chunk_id"], results[1].metadata["chunk_id"]}
    # Password-related chunks must outrank the cooking chunk.
    assert "docC::0" not in top_two_ids


def test_top_k_limits_results(tmp_path: Path) -> None:
    retriever, _ = _setup(tmp_path)

    results = retriever.retrieve("account", top_k=2)

    assert len(results) == 2


def test_filter_restricts_by_document(tmp_path: Path) -> None:
    retriever, _ = _setup(tmp_path)

    results = retriever.retrieve("account", filter={"document_id": "docB"})

    assert all(r.metadata["document_id"] == "docB" for r in results)
    assert len(results) >= 1


def test_min_score_excludes_low_relevance(tmp_path: Path) -> None:
    retriever, _ = _setup(tmp_path)

    all_results = retriever.retrieve("password reset")
    high_score = all_results[0].score
    # Filter above the highest score → nothing passes.
    results = retriever.retrieve("password reset", min_score=high_score + 0.1)

    assert results == []


def test_empty_query_raises_value_error(tmp_path: Path) -> None:
    retriever, _ = _setup(tmp_path)

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("")


def test_whitespace_query_raises_value_error(tmp_path: Path) -> None:
    retriever, _ = _setup(tmp_path)

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("   ")


def test_empty_index_returns_empty_list(tmp_path: Path) -> None:
    embedder = FakeEmbeddingProvider()
    store = ChromaVectorStoreRepository(tmp_path / "empty")
    retriever = Retriever(embedder, store)

    results = retriever.retrieve("anything")

    assert results == []
