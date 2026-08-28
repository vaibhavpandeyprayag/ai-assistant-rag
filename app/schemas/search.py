"""Request/response schemas for the similarity search endpoint."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """A semantic search against the indexed corpus."""

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    #: Optional metadata equality filter, e.g. {"filename": "a.pdf"}.
    filter: dict[str, Any] | None = None

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchResultItem(BaseModel):
    """One retrieved chunk with its stored source metadata."""

    text: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Ranked list of retrieved chunks."""

    results: list[SearchResultItem]
