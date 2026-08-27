"""Config-driven selection of the active LLM provider."""

from app.config import Settings
from app.errors import ConfigurationError
from app.llm.base import LLMProvider
from app.llm.hf_llm import HuggingFaceLLM


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Build the LLM provider selected by ``LLM_PROVIDER``.

    Only the Hugging Face provider is supported (``LLM_PROVIDER="hf"``).

    Raises:
        ConfigurationError: If the provider is unknown or a required setting
            (the Hugging Face API key) is missing.
    """
    if settings.llm_provider != "hf":
        raise ConfigurationError(
            f"Unknown LLM provider: {settings.llm_provider!r}. Only 'hf' is supported."
        )

    api_key = settings.huggingface_api_key
    if api_key is None:
        raise ConfigurationError(
            "LLM_PROVIDER='hf' requires HUGGINGFACE_API_KEY to be set."
        )

    return HuggingFaceLLM(
        api_key=api_key.get_secret_value(),
        model_name=settings.huggingface_model,
    )
