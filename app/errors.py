"""Domain error types shared across application layers.

Expected errors derive from :class:`RagError`; the FastAPI layer maps them to
safe JSON responses (see ``app.main.register_exception_handlers``). Unexpected
errors propagate and are converted to a generic 500 without internal details.
"""


class RagError(Exception):
    """Base class for expected, user-presentable application errors."""

    #: Stable machine-readable identifier returned to API clients.
    code: str = "internal_error"
    #: HTTP status the API layer should respond with.
    http_status: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(RagError):
    """Raised when required configuration is missing or invalid."""

    code = "configuration_error"
    http_status = 500


class UnsupportedFormatError(RagError):
    """Raised when a document has a file extension we cannot process."""

    code = "unsupported_format"
    http_status = 415


class DocumentParseError(RagError):
    """Raised when a document exists but cannot be read or parsed."""

    code = "document_parse_error"
    http_status = 422


class EmptyDocumentError(RagError):
    """Raised when a document yields no extractable text."""

    code = "empty_document"
    http_status = 422


class EmbeddingError(RagError):
    """Raised when embedding generation fails."""

    code = "embedding_error"
    http_status = 500


class VectorStoreError(RagError):
    """Raised when vector store operations fail."""

    code = "vector_store_error"
    http_status = 500


class LLMAuthError(RagError):
    """Raised when LLM authentication fails (missing/invalid API key)."""

    code = "llm_auth_error"
    http_status = 503


class LLMUnavailableError(RagError):
    """Raised when the LLM cannot be reached or fails to generate."""

    code = "llm_unavailable"
    http_status = 503


class FileTooLargeError(RagError):
    """Raised when an uploaded file exceeds the configured size limit."""

    code = "file_too_large"
    http_status = 413


class DocumentNotFoundError(RagError):
    """Raised when a referenced uploaded document does not exist."""

    code = "document_not_found"
    http_status = 404
