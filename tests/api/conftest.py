"""Shared fixtures for API-level tests (network-free via provider overrides)."""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.vectordb.base import RetrievedChunk


class FakeRetriever:
    """Returns a canned (configurable) list of retrieved chunks."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.calls: list[dict] = []

    def retrieve(self, query, *, top_k=None, filter=None, min_score=None):
        self.calls.append(
            {"query": query, "top_k": top_k, "filter": filter, "min_score": min_score}
        )
        return self.chunks[: top_k if top_k is not None else len(self.chunks)]


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings with runtime directories isolated under pytest's tmp dir."""
    return Settings(
        _env_file=None,
        chroma_persist_directory=tmp_path / "chroma",
        upload_directory=tmp_path / "uploads",
    )


@pytest.fixture
def client(settings):
    """A TestClient whose settings point at isolated temp directories."""
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
