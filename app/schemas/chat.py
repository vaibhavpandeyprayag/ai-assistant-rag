"""Request/response schemas for the grounded chat endpoint."""

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """A question to answer against the indexed corpus."""

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class ChatSource(BaseModel):
    """Supporting source metadata for one chunk used in the answer."""

    chunk_id: str
    filename: str
    page_number: int | None = None
    score: float


class ChatResponse(BaseModel):
    """The RAG answer with its supporting sources."""

    answer: str
    sources: list[ChatSource]
    insufficient_context: bool
    latency_ms: float
