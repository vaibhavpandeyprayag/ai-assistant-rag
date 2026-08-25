"""Unit tests for embedding provider factory selection."""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.embeddings.factory import create_embedding_provider
from app.embeddings.hf_api import HuggingFaceInferenceEmbeddingProvider
from app.embeddings.local import SentenceTransformersEmbeddingProvider
from app.errors import ConfigurationError


def _isolated(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_local_settings_build_sentence_transformers_provider() -> None:
    settings = _isolated(embedding_model="sentence-transformers/custom-model")

    provider = create_embedding_provider(settings)

    assert isinstance(provider, SentenceTransformersEmbeddingProvider)
    assert provider.model_name == "sentence-transformers/custom-model"


def test_construction_does_not_load_model_weights() -> None:
    """Lazy loading keeps provider construction free of downloads/torch."""
    provider = create_embedding_provider(_isolated())

    assert provider._model is None  # noqa: SLF001 - deliberate white-box check


def test_hf_settings_build_hf_provider(clean_env) -> None:
    clean_env.setenv("EMBEDDING_PROVIDER", "hf")
    clean_env.setenv("HUGGINGFACE_API_KEY", "test-key-123")

    provider = create_embedding_provider(Settings(_env_file=None))

    assert isinstance(provider, HuggingFaceInferenceEmbeddingProvider)
    assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_hf_without_api_key_raises_configuration_error(clean_env) -> None:
    clean_env.setenv("EMBEDDING_PROVIDER", "hf")

    with pytest.raises(ConfigurationError, match="HUGGINGFACE_API_KEY"):
        create_embedding_provider(Settings(_env_file=None))


def test_unknown_provider_raises_configuration_error() -> None:
    # Settings' Literal blocks invalid values; exercise the defensive branch.
    bogus = SimpleNamespace(
        embedding_provider="quantum", embedding_model="m", huggingface_api_key=None
    )

    with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
        create_embedding_provider(bogus)  # type: ignore[arg-type]
