"""Health endpoint response schema."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Public health summary; intentionally free of secrets and internals."""

    status: str
    version: str
    embedding_model: str
