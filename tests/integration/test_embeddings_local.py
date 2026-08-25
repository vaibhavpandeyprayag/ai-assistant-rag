"""Integration test against the real local embedding model.

Runs only on demand: ``pytest -m slow`` (excluded from default runs).
First execution downloads model weights (~90 MB) into the HF cache.
"""

import pytest

from app.embeddings.local import SentenceTransformersEmbeddingProvider

pytestmark = [pytest.mark.slow, pytest.mark.requires_model]

EXPECTED_MINILM_DIMENSION = 384


def test_document_and_query_embeddings_share_dimensions_and_scale() -> None:
    provider = SentenceTransformersEmbeddingProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = provider.embed_documents(["first document text", "second document text"])
    query = provider.embed_query("first document text")

    assert len(documents) == 2
    lengths = {len(vector) for vector in documents} | {len(query)}
    assert lengths == {EXPECTED_MINILM_DIMENSION}

    # Normalized vectors: squared magnitude ~ 1.0.
    magnitude = sum(value * value for value in query)
    assert abs(magnitude - 1.0) < 1e-3


def test_identical_text_embeds_identically() -> None:
    provider = SentenceTransformersEmbeddingProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    first = provider.embed_query("determinism check")
    second = provider.embed_query("determinism check")

    assert first == second
