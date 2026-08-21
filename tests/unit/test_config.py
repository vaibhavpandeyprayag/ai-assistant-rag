"""Unit tests for application configuration."""

import json

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults(base_settings: Settings) -> None:
    """Sensible documented defaults are applied when nothing is configured."""
    assert base_settings.llm_provider == "local"
    assert base_settings.local_llm_model == "llama3.1:8b"
    assert base_settings.embedding_provider == "local"
    assert base_settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert base_settings.chunk_size == 1000
    assert base_settings.chunk_overlap == 200
    assert base_settings.chunk_strategy == "recursive"
    assert base_settings.top_k == 5
    assert base_settings.retrieval_min_score == 0.0
    assert base_settings.upload_max_size_mb == 20
    assert base_settings.huggingface_api_key is None


def test_environment_overrides(clean_env) -> None:
    """Environment variables take precedence over defaults."""
    clean_env.setenv("CHUNK_SIZE", "512")
    clean_env.setenv("TOP_K", "9")
    clean_env.setenv("LLM_PROVIDER", "hf")
    clean_env.setenv("CHROMA_PERSIST_DIRECTORY", "somewhere/else")

    settings = Settings(_env_file=None)

    assert settings.chunk_size == 512
    assert settings.top_k == 9
    assert settings.llm_provider == "hf"
    assert settings.chroma_persist_directory.as_posix() == "somewhere/else"


def test_chunk_overlap_must_be_smaller_than_chunk_size(clean_env) -> None:
    clean_env.setenv("CHUNK_SIZE", "500")
    clean_env.setenv("CHUNK_OVERLAP", "500")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_llm_provider_rejected(clean_env) -> None:
    clean_env.setenv("LLM_PROVIDER", "bogus-provider")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_api_key_is_masked_in_serialization(clean_env) -> None:
    """SecretStr ensures the API key never leaks through dumps or str()."""
    clean_env.setenv("HUGGINGFACE_API_KEY", "super-secret-key-value")

    settings = Settings(_env_file=None)

    dumped = json.dumps(settings.model_dump(mode="json"))
    assert "super-secret-key-value" not in dumped
    assert "super-secret-key-value" not in repr(settings)
    assert settings.huggingface_api_key is not None
    assert settings.huggingface_api_key.get_secret_value() == "super-secret-key-value"


def test_ensure_runtime_directories_creates_paths(base_settings: Settings, tmp_path) -> None:
    """Runtime directories are created on demand under the configured roots."""
    base_settings.chroma_persist_directory = tmp_path / "chroma"
    base_settings.upload_directory = tmp_path / "uploads"

    base_settings.ensure_runtime_directories()

    assert (tmp_path / "chroma").is_dir()
    assert (tmp_path / "uploads").is_dir()
