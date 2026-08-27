"""LLM provider contract.

This module is free of any specific provider library: it defines the port
that the RAG pipeline depends on, so the LLM backend can be changed without
touching RAG or API logic.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class ChatMessage:
    """A single turn in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(slots=True)
class LLMOptions:
    """Generation parameters passed to the provider."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    #: Reserved for provider-specific settings the abstraction doesn't model.
    extra_body: dict[str, object] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Text-generation port implemented by concrete backends."""

    #: A stable identifier of the underlying model for diagnostics.
    model_name: str

    def generate(
        self,
        messages: list[ChatMessage],
        options: LLMOptions | None = None,
    ) -> str:
        """Generate a completion for ``messages`` and return its text.

        Raises:
            LLMAuthError: If the backend rejects the credentials.
            LLMUnavailableError: If the backend cannot be reached or fails.
        """
        ...
