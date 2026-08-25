"""Local embedding provider backed by sentence-transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbeddingProvider:
    """Embeds text with a local Hugging Face sentence-transformers model.

    The model is loaded lazily on first use so that importing this module or
    constructing the provider stays cheap (and test-friendly). Vectors are
    unit-normalized, making cosine similarity equivalent to a dot product.
    """

    def __init__(self, *, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            # Deferred import keeps module import cheap and avoids loading
            # torch unless this provider actually runs.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        # Reuses the document path so queries and chunks share dimensionality.
        return self.embed_documents([text])[0]
