"""Vector store contract and retrieval result model.

This module is intentionally free of any specific database library: it defines
the port that application code depends on, so the vector store can be replaced
without touching ingestion or retrieval logic.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from app.ingestion.models import Chunk


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by similarity search.

    ``score`` is a cosine similarity (1 − cosine distance); higher means more
    similar. ``metadata`` mirrors what was stored alongside the chunk.
    """

    text: str
    score: float
    metadata: dict[str, Any]


class VectorStoreRepository(Protocol):
    """Persistence/search port for storing chunks and finding similar ones."""

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their vectors; re-adding an existing chunk_id updates it."""
        ...

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks nearest to the query vector.

        ``where`` optionally restricts matches via metadata equality filters
        (supported keys depend on the implementation).
        """
        ...

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to a document; returns removed count."""
        ...

    def count(self) -> int:
        """Return the total number of stored chunks."""
        ...
