"""API tests for document upload and ingestion endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.dependencies import get_ingestion_service
from app.config import Settings
from app.ingestion.models import IngestionResult
from app.main import create_app


class FakeIngestionService:
    def __init__(self) -> None:
        self.batch_paths: list[list] = []

    def ingest_batch(self, paths):
        self.batch_paths.append(paths)
        return [
            IngestionResult(
                document_id=f"id:{path.name}",
                filename=path.name,
                n_chunks=2,
                status="ok",
                elapsed_ms=5.0,
            )
            for path in paths
        ]


def _app_with(settings: Settings):
    return create_app(settings)


def _override_ingestion(app, fake):
    app.dependency_overrides[get_ingestion_service] = lambda: fake


def test_upload_valid_txt(settings) -> None:
    app = _app_with(settings)
    fake = FakeIngestionService()
    _override_ingestion(app, fake)

    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "stored"
    assert body["filename"] == "notes.txt"
    assert body["size_bytes"] == 11
    assert body["document_id"]
    # File was persisted for a later ingest step.
    assert (settings.upload_directory / "notes.txt").read_bytes() == b"hello world"


def test_upload_rejects_unsupported_extension(settings) -> None:
    app = _app_with(settings)
    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("virus.exe", b"MZ", "application/octet-stream")},
        )

    assert response.status_code == 415
    body = response.json()
    assert body["code"] == "unsupported_format"


def test_upload_rejects_oversized_file() -> None:
    settings = Settings(_env_file=None, upload_max_size_mb=1)
    app = _app_with(settings)
    with TestClient(app) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("big.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
        )

    assert response.status_code == 413
    body = response.json()
    assert body["code"] == "file_too_large"


def test_ingest_specific_filenames(settings) -> None:
    app = _app_with(settings)
    fake = FakeIngestionService()
    _override_ingestion(app, fake)
    settings.upload_directory.mkdir(parents=True, exist_ok=True)
    (settings.upload_directory / "a.txt").write_text("content")
    (settings.upload_directory / "b.md").write_text("content")

    with TestClient(app) as client:
        response = client.post(
            "/documents/ingest", json={"filenames": ["a.txt"]}
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["filename"] == "a.txt"
    assert body["results"][0]["n_chunks"] == 2
    assert body["results"][0]["status"] == "ok"
    assert body["results"][0]["error"] is None


def test_ingest_unknown_file_returns_404(settings) -> None:
    app = _app_with(settings)
    fake = FakeIngestionService()
    _override_ingestion(app, fake)

    with TestClient(app) as client:
        response = client.post(
            "/documents/ingest", json={"filenames": ["missing.txt"]}
        )

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "document_not_found"


def test_ingest_empty_list_returns_no_results(settings) -> None:
    app = _app_with(settings)
    _override_ingestion(app, FakeIngestionService())

    with TestClient(app) as client:
        response = client.post("/documents/ingest", json={"filenames": []})

    assert response.status_code == 200
    assert response.json()["results"] == []
