"""Hugging Face Inference API embedding provider."""

from huggingface_hub import InferenceClient

from app.errors import ConfigurationError, EmbeddingError


def _to_vector(raw: object) -> list[float]:
    """Normalize Inference API feature-extraction output to a flat vector.

    Endpoints return either a sentence-level vector or a token-level matrix
    (one row per token). Token matrices are mean-pooled into one fixed-size
    vector so the output dimension is independent of text length.
    """
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise ValueError("Unexpected embedding payload shape")
    data = list(raw)
    if not data:
        raise ValueError("Empty embedding payload")
    if isinstance(data[0], (list, tuple)):
        # Mean-pool across tokens: transpose rows and average each dimension.
        # Rows are expected rectangular; ragged input raises and is wrapped.
        data = [sum(column) / len(column) for column in zip(*data, strict=True)]
        if not data:
            # e.g. a matrix of zero-length rows pools to nothing usable.
            raise ValueError("Empty embedding payload")
    return [float(value) for value in data]


class HuggingFaceInferenceEmbeddingProvider:
    """Embeds text through the Hugging Face Inference API (hosted models).

    Requests are sent one text at a time; adequate for this project's scale
    and kept simple intentionally.
    """

    def __init__(self, *, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ConfigurationError(
                "HUGGINGFACE_API_KEY is required for EMBEDDING_PROVIDER='hf'."
            )
        self._model_name = model_name
        self._client = InferenceClient(api_key=api_key, model=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _embed_one(self, text: str) -> list[float]:
        try:
            raw = self._client.feature_extraction(text)
            return _to_vector(raw)
        except Exception as exc:
            # Trust boundary: any external API failure (auth, availability,
            # payload shape) becomes one typed domain error without leaking
            # credentials or stack traces.
            raise EmbeddingError(
                f"Hugging Face embedding request failed for model '{self._model_name}'."
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # Reuses the document path so queries and chunks share dimensionality.
        return self._embed_one(text)
