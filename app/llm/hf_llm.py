"""Hugging Face Inference API LLM provider.

Generation is served by hosted models through ``InferenceClient.chat_completion``.
The provider accepts an optional pre-built client so tests can substitute one
backed by httpx ``MockTransport``.
"""

import logging

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from app.errors import ConfigurationError, LLMAuthError, LLMUnavailableError
from app.llm.base import ChatMessage, LLMOptions

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """Chat completion through the Hugging Face Inference API."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        client: InferenceClient | None = None,
        timeout: float | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "HUGGINGFACE_API_KEY is required for LLM_PROVIDER='hf'."
            )
        self._model_name = model_name
        if client is not None:
            self._client = client
        else:
            self._client = InferenceClient(
                api_key=api_key,
                model=model_name,
                timeout=timeout or 60.0,
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        messages: list[ChatMessage],
        options: LLMOptions | None = None,
    ) -> str:
        """Generate a completion and return the assistant message text.

        Raises:
            LLMAuthError: The backend rejected the API key (401/403).
            LLMUnavailableError: The backend was unreachable or errored.
        """
        opts = options or LLMOptions()
        try:
            completion = self._client.chat_completion(
                [{"role": message.role, "content": message.content} for message in messages],
                model=self._model_name,
                max_tokens=opts.max_tokens,
                temperature=opts.temperature,
                top_p=opts.top_p,
                seed=opts.seed,
                extra_body=opts.extra_body or None,
            )
        except Exception as exc:
            raise _wrap_hf_error(exc, self._model_name) from exc
        return completion.choices[0].message.content


def _wrap_hf_error(exc: Exception, model_name: str) -> Exception:
    """Convert a Hugging Face client error into a typed domain error.

    Authentication problems are surfaced distinctly so a misconfigured key is
    diagnosable; every other failure is treated as unavailability. Credentials
    and stack details are never forwarded to callers or logged.
    """
    if isinstance(exc, HfHubHTTPError) and exc.response is not None:
        if exc.response.status_code in (401, 403):
            logger.error("Hugging Face LLM authentication failed for '%s'.", model_name)
            return LLMAuthError(
                "Hugging Face LLM authentication failed. Check the API key."
            )
    logger.error(
        "Hugging Face LLM request failed for '%s' (%s).",
        model_name,
        type(exc).__name__,
    )
    return LLMUnavailableError("The language model is currently unavailable.")
