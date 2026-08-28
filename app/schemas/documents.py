"""Request/response schemas for the document upload and ingest endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Outcome of persisting an uploaded file to local storage."""

    filename: str
    document_id: str
    size_bytes: int
    status: Literal["stored"]


class IngestRequest(BaseModel):
    """Selects which uploaded files to index.

    ``filenames`` are relative to the configured upload directory; when
    omitted every stored file is ingested.
    """

    filenames: list[str] | None = None


class IngestResultItem(BaseModel):
    """Per-file outcome of the ingestion pipeline."""

    filename: str
    document_id: str
    n_chunks: int
    status: str
    error: str | None = None


class IngestResponse(BaseModel):
    """Aggregate result of an ingestion request."""

    results: list[IngestResultItem]
    elapsed_ms: float = Field(default=0.0)
