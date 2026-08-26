"""Retrieval service: query-time top-K semantic search.

This module is independent of any LLM — it embeds the user query, searches
the vector store, and returns ranked results that downstream consumers
(e.g. the RAG pipeline) assemble into a prompt.
"""

from typing import Any

from app.embeddings.base import EmbeddingProvider
from app.vectordb.base import RetrievedChunk, VectorStoreRepository


class Retriever:
    """Nearest-neighbor retrieval with optional score thresholding.

    Dependencies (embedder, vector store) are injected so the retriever is
    testable without any real model or database.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStoreRepository,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Embed the query, search the store, and filter low-relevance results.

        Raises:
            ValueError: If ``query`` is empty or whitespace-only.
        """
        if not query.strip():
            raise ValueError("query must not be empty")

        effective_top_k = top_k if top_k is not None else self._top_k
        effective_min_score = min_score if min_score is not None else self._min_score

        query_embedding = self._embedder.embed_query(query)
        results = self._store.similarity_search(
            query_embedding,
            top_k=effective_top_k,
            where=filter,
        )

        if effective_min_score > 0.0:
            results = [r for r in results if r.score >= effective_min_score]

        return results
