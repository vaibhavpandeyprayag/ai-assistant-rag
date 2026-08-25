"""Config-driven selection of the active embedding provider."""

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.hf_api import HuggingFaceInferenceEmbeddingProvider
from app.embeddings.local import SentenceTransformersEmbeddingProvider
from app.errors import ConfigurationError


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the embedding provider selected by ``EMBEDDING_PROVIDER``.

    Raises:
        ConfigurationError: If the selected provider lacks required settings.
    """
    if settings.embedding_provider == "local":
        return SentenceTransformersEmbeddingProvider(
            model_name=settings.embedding_model,
        )

    if settings.embedding_provider == "hf":
        api_key = settings.huggingface_api_key
        if api_key is None:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER='hf' requires HUGGINGFACE_API_KEY to be set."
            )
        return HuggingFaceInferenceEmbeddingProvider(
            api_key=api_key.get_secret_value(),
            model_name=settings.embedding_model,
        )

    # Unreachable while Settings validates the Literal, kept as a safeguard.
    raise ConfigurationError(f"Unknown embedding provider: {settings.embedding_provider!r}.")
