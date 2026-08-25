"""Embedding provider contract shared by ingestion and retrieval."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generates vectors for document chunks and user queries.

    Document chunks and queries must be embedded through the same provider
    configuration so they share one vector space and dimensionality.
    """

    @property
    def model_name(self) -> str:
        """Identifier of the underlying embedding model."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts; output order matches input order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...
