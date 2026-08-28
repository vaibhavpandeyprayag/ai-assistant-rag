"""API tests for the semantic search endpoint."""

from fastapi.testclient import TestClient

from app.api.dependencies import get_retriever
from app.main import create_app
from app.vectordb.base import RetrievedChunk
from tests.api.conftest import FakeRetriever


def _chunk(text: str, filename: str = "a.txt", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=0.9,
        metadata={"chunk_id": "a::0", "filename": filename, "page_number": page},
    )


def test_search_returns_retrieved_chunks(settings) -> None:
    app = create_app(settings)
    fake = FakeRetriever([_chunk("the sky is blue"), _chunk("grass is green", "b.txt")])
    app.dependency_overrides[get_retriever] = lambda: fake

    with TestClient(app) as client:
        response = client.post("/search", json={"query": "what color is the sky?"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["text"] == "the sky is blue"
    assert body["results"][0]["score"] == 0.9
    assert body["results"][0]["metadata"]["filename"] == "a.txt"
    assert fake.calls[0]["query"] == "what color is the sky?"


def test_search_passes_top_k_and_filter(settings) -> None:
    app = create_app(settings)
    fake = FakeRetriever([_chunk("text")])
    app.dependency_overrides[get_retriever] = lambda: fake

    with TestClient(app) as client:
        response = client.post(
            "/search",
            json={"query": "q", "top_k": 3, "filter": {"filename": "a.txt"}},
        )

    assert response.status_code == 200
    assert fake.calls[0]["top_k"] == 3
    assert fake.calls[0]["filter"] == {"filename": "a.txt"}


def test_search_rejects_blank_query(settings) -> None:
    app = create_app(settings)
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever([])

    with TestClient(app) as client:
        response = client.post("/search", json={"query": "   "})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert "traceback" not in body.get("message", "").lower()
