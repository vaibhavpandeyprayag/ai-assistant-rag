"""Integration tests for the RAG pipeline using fakes (no network)."""

import pytest

from app.rag.pipeline import INSUFFICIENT_CONTEXT_RESPONSE, RAGPipeline
from app.vectordb.base import RetrievedChunk
from tests.utils.fakes import FakeLLMProvider


class StubRetriever:
    """Returns a canned list (or augments it) for deterministic tests."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.queries: list[str] = []

    def retrieve(self, query: str, *, top_k: int | None = None, **kwargs) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.chunks[: top_k if top_k is not None else len(self.chunks)]


def _chunk(
    text: str,
    chunk_id: str,
    filename: str = "doc.txt",
    page: int | None = 1,
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        metadata={
            "chunk_id": chunk_id,
            "filename": filename,
            "page_number": page,
        },
    )


RETRIEVED = [
    _chunk("context one", "a::0", "a.txt", page=3, score=0.9),
    _chunk("context two", "b::0", "b.txt", page=None, score=0.7),
]


def test_answer_returns_llm_text_and_matching_sources() -> None:
    llm = FakeLLMProvider(response="The answer is 42.")
    pipeline = RAGPipeline(StubRetriever(RETRIEVED), llm, top_k=5)

    result = pipeline.answer("what is the answer?")

    assert result.answer == "The answer is 42."
    assert result.insufficient_context is False
    assert result.latency_ms >= 0
    # Sources map 1:1 to the chunks the builder included.
    assert [(s.chunk_id, s.filename) for s in result.sources] == [
        ("a::0", "a.txt"),
        ("b::0", "b.txt"),
    ]
    assert result.sources[0].page_number == 3
    assert result.sources[0].score == 0.9
    # The LLM received a system + user message.
    assert len(llm.calls) == 1
    assert llm.calls[0][0][1].content.startswith("Context:\n[1] (a.txt, p.3)")


def test_insufficient_context_short_circuits_without_llm() -> None:
    llm = FakeLLMProvider(response="should not be called")
    pipeline = RAGPipeline(StubRetriever([]), llm)

    result = pipeline.answer("anything")

    assert result.answer == INSUFFICIENT_CONTEXT_RESPONSE
    assert result.insufficient_context is True
    assert result.sources == []
    assert llm.calls == []  # No LLM call was made.


def test_empty_query_rejected() -> None:
    pipeline = RAGPipeline(StubRetriever([]), FakeLLMProvider())

    with pytest.raises(ValueError, match="query must not be empty"):
        pipeline.answer("   ")
