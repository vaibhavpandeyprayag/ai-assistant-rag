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
