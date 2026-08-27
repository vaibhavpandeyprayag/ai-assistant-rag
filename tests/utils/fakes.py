"""Fake implementations of application ports for tests."""

import hashlib
import math
import re

from app.llm.base import ChatMessage, LLMOptions

_WHITESPACE = re.compile(r"\s+")


class FakeEmbeddingProvider:
    """Deterministic, similarity-preserving embedding provider.

    Vectors come from feature hashing: each character trigram of the text is
    hashed into one of ``dimension`` bins, counts are accumulated and the
    result is L2-normalized. Properties:

    - no model download, no network, instant
    - identical texts always produce identical vectors
    - texts that share substrings land close together (meaningful cosine /
      dot-product similarity), which ranking-dependent tests rely on
    """

    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension
        self.embedded_documents: list[str] = []
        self.embedded_queries: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_documents.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # Same math as documents so queries live in the same vector space.
        self.embedded_queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        normalized = _WHITESPACE.sub(" ", text.lower()).strip()
        padded = f"  {normalized}  "
        bins = [0.0] * self.dimension
        for position in range(len(padded) - 2):
            trigram = padded[position : position + 3]
            digest = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
            bins[digest % self.dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in bins)) or 1.0
        return [value / norm for value in bins]


class FakeLLMProvider:
    """Deterministic LLM provider that records inputs and returns canned text.

    Used so the RAG pipeline is testable without any network or model. The
    response can be fixed, or raised to simulate a failure.
    """

    def __init__(
        self,
        model_name: str = "fake-llm",
        response: str = "fake response",
        error: Exception | None = None,
    ) -> None:
        self.model_name = model_name
        self.response = response
        self.error = error
        #: Records every (messages, options) pair passed to generate().
        self.calls: list[tuple[list[ChatMessage], LLMOptions | None]] = []

    def generate(
        self,
        messages: list[ChatMessage],
        options: LLMOptions | None = None,
    ) -> str:
        self.calls.append((messages, options))
        if self.error is not None:
            raise self.error
        return self.response
