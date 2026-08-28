"""API tests for the grounded chat endpoint (network-free via fakes)."""

from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.main import create_app
from app.rag.pipeline import RAGPipeline
from app.services.chat_service import ChatService
from app.vectordb.base import RetrievedChunk
from tests.api.conftest import FakeRetriever
from tests.utils.fakes import FakeLLMProvider


def _chunk(
    text: str, chunk_id: str, filename: str = "a.txt", page: int | None = 1
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=0.9,
        metadata={
            "chunk_id": chunk_id,
            "filename": filename,
            "page_number": page,
        },
    )


def _chat_service(chunks: list[RetrievedChunk], response: str) -> ChatService:
    pipeline = RAGPipeline(FakeRetriever(chunks), FakeLLMProvider(response=response))
    return ChatService(pipeline)


def test_chat_returns_answer_and_sources(settings) -> None:
    app = create_app(settings)
    service = _chat_service(
        [_chunk("the sky is blue", "a::0", "facts.txt", 3)],
        response="The sky is blue on clear days.",
    )
    app.dependency_overrides[get_chat_service] = lambda: service

    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "why is the sky blue?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The sky is blue on clear days."
    assert body["insufficient_context"] is False
    assert body["latency_ms"] >= 0
    assert len(body["sources"]) == 1
    source = body["sources"][0]
    assert source["chunk_id"] == "a::0"
    assert source["filename"] == "facts.txt"
    assert source["page_number"] == 3
    assert source["score"] == 0.9


def test_chat_insufficient_context_shape(settings) -> None:
    llm = FakeLLMProvider(response="should not be called")
    pipeline = RAGPipeline(FakeRetriever([]), llm)
    app = create_app(settings)
    app.dependency_overrides[get_chat_service] = lambda: ChatService(pipeline)

    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_context"] is True
    assert body["sources"] == []
    assert llm.calls == []  # No LLM call for insufficient context.


def test_chat_rejects_blank_query(settings) -> None:
    app = create_app(settings)
    app.dependency_overrides[get_chat_service] = lambda: _chat_service([], "x")

    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
