"""Tests for the GET /health endpoint."""

from fastapi.testclient import TestClient

from app import __version__
from app.config import Settings
from app.main import create_app


def _isolated_app(**overrides):
    """Build an app from pristine defaults (ignoring any local .env file)."""
    return create_app(Settings(_env_file=None, **overrides))


def test_health_returns_expected_shape() -> None:
    client = TestClient(_isolated_app())
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["embedding_model"]


def test_health_does_not_expose_secrets(clean_env) -> None:
    """Secrets present in configuration never reach the API surface."""
    clean_env.setenv("HUGGINGFACE_API_KEY", "super-secret-key-value")
    settings = Settings(_env_file=None)
    client = TestClient(create_app(settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert "super-secret-key-value" not in response.text
