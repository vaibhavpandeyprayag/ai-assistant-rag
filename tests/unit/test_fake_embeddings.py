"""Contract tests for the FakeEmbeddingProvider used by later-phase tests."""

from tests.utils.fakes import FakeEmbeddingProvider


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_deterministic_across_calls() -> None:
    provider = FakeEmbeddingProvider()
    first = provider.embed_query("identical text")
    second = provider.embed_query("identical text")
    assert first == second


def test_vectors_are_unit_normalized() -> None:
    vector = FakeEmbeddingProvider().embed_query("some words here")
    assert abs(_dot(vector, vector) - 1.0) < 1e-9


def test_related_texts_closer_than_unrelated_texts() -> None:
    provider = FakeEmbeddingProvider()
    anchor = provider.embed_query("the cat sat on the mat")
    related = provider.embed_query("a cat sat on the mat today")
    unrelated = provider.embed_query("quarterly revenue spreadsheet totals")

    assert _dot(anchor, related) > _dot(anchor, unrelated)


def test_call_logging_separates_documents_and_queries() -> None:
    provider = FakeEmbeddingProvider()
    provider.embed_documents(["doc one", "doc two"])
    provider.embed_query("question?")

    assert provider.embedded_documents == ["doc one", "doc two"]
    assert provider.embedded_queries == ["question?"]
