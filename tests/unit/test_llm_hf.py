"""Unit tests for the Hugging Face LLM provider and factory.

The Hugging Face InferenceClient is pointed at an httpx ``MockTransport`` so
requests are intercepted locally (no network) while still exercising the real
client decode path.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

import httpx
import huggingface_hub.inference._client as hf_client_module
import pytest
from huggingface_hub import InferenceClient

from app.config import Settings
from app.errors import ConfigurationError, LLMAuthError, LLMUnavailableError
from app.llm.base import ChatMessage, LLMOptions
from app.llm.factory import create_llm_provider
from app.llm.hf_llm import HuggingFaceLLM

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
API_KEY = "hf_mocked0000000000000000000000000000"

_captured: list[dict] = []


@contextmanager
def _patched_llm(handler) -> Iterator[HuggingFaceLLM]:
    """Provider whose HTTP calls hit the handler while the mock session is live."""
    client = InferenceClient(api_key=API_KEY, model=MODEL)
    provider = HuggingFaceLLM(api_key=API_KEY, model_name=MODEL, client=client)
    with mock.patch.object(
        hf_client_module,
        "get_session",
        return_value=httpx.Client(transport=httpx.MockTransport(handler)),
    ):
        yield provider


def _ok_handler(request: httpx.Request) -> httpx.Response:
    _captured.append(json.loads(request.content.decode()))
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
    )


def test_generate_sends_messages_and_options() -> None:
    _captured.clear()
    with _patched_llm(_ok_handler) as provider:
        messages = [
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hi"),
        ]
        options = LLMOptions(temperature=0.2, max_tokens=64, top_p=0.9)

        result = provider.generate(messages, options)

    assert result == "hello"
    body = _captured[0]
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9


def test_generate_defaults_when_options_none() -> None:
    _captured.clear()
    with _patched_llm(_ok_handler) as provider:
        provider.generate([ChatMessage(role="user", content="hi")])

    body = _captured[0]
    # None options are dropped from the request body entirely.
    assert "max_tokens" not in body
    assert "temperature" not in body


def test_generate_auth_error_maps_to_llm_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    with _patched_llm(handler) as provider:
        with pytest.raises(LLMAuthError):
            provider.generate([ChatMessage(role="user", content="hi")])


def test_generate_transport_error_maps_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with _patched_llm(handler) as provider:
        with pytest.raises(LLMUnavailableError):
            provider.generate([ChatMessage(role="user", content="hi")])


def test_generate_500_maps_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "overloaded"})

    with _patched_llm(handler) as provider:
        with pytest.raises(LLMUnavailableError):
            provider.generate([ChatMessage(role="user", content="hi")])


def test_requires_api_key_at_construction() -> None:
    with pytest.raises(ConfigurationError):
        HuggingFaceLLM(api_key="", model_name=MODEL)


# --- Factory ----------------------------------------------------------------


def test_factory_returns_hf_provider() -> None:
    settings = Settings(
        llm_provider="hf",
        huggingface_model=MODEL,
        huggingface_api_key=API_KEY,
        _env_file=None,
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, HuggingFaceLLM)
    assert provider.model_name == MODEL


def test_factory_requires_api_key() -> None:
    settings = Settings(llm_provider="hf", _env_file=None)

    with pytest.raises(ConfigurationError, match="HUGGINGFACE_API_KEY"):
        create_llm_provider(settings)


def test_factory_rejects_unknown_provider() -> None:
    settings = Settings(_env_file=None)
    settings.llm_provider = "bogus"  # type: ignore[assignment]

    with pytest.raises(ConfigurationError, match="Only 'hf' is supported"):
        create_llm_provider(settings)
