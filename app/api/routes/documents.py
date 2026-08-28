"""Document upload and ingestion endpoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.dependencies import (
    get_ingestion_service,
    get_settings,
)
from app.config import Settings
from app.errors import DocumentNotFoundError, FileTooLargeError, UnsupportedFormatError
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.schemas.documents import (
    IngestRequest,
    IngestResponse,
    IngestResultItem,
    UploadResponse,
)
from app.services.ingestion_service import (
    IngestionService,
    document_id_for_path,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _safe_name(filename: str) -> str:
    """Reduce an uploaded filename to a safe leaf name."""
    return Path(filename).name


def _extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: Annotated[UploadFile, File(description="A PDF, DOCX, TXT or Markdown file")],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadResponse:
    """Validate and persist an uploaded file, returning its stable ID.

    Rejects unsupported extensions (415) and files over the configured size
    limit (413). The file is written to the upload directory; indexing happens
    separately via ``POST /documents/ingest``.
    """
    name = _safe_name(file.filename or "upload")
    extension = _extension_of(name)
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Unsupported file format '{extension or 'unknown'}'. "
            f"Supported formats: {supported}."
        )

    max_bytes = settings.upload_max_size_mb * 1024 * 1024
    content = file.file.read()
    if len(content) > max_bytes:
        raise FileTooLargeError(
            f"File exceeds the configured size limit of "
            f"{settings.upload_max_size_mb} MB."
        )

    destination = settings.upload_directory / name
    destination.write_bytes(content)

    document_id = document_id_for_path(destination)
    logger.info("Uploaded %s (%d bytes) as %s", name, len(content), document_id)
    return UploadResponse(
        filename=name,
        document_id=document_id,
        size_bytes=len(content),
        status="stored",
    )


def _resolve_uploaded(upload_dir: Path, filename: str) -> Path:
    """Resolve a filename to a path inside the upload directory."""
    leaf = _safe_name(filename)
    candidate = (upload_dir / leaf).resolve()
    upload_root = upload_dir.resolve()
    if not candidate.is_relative_to(upload_root):
        raise DocumentNotFoundError(f"Document not found: {leaf}")
    if not candidate.is_file():
        raise DocumentNotFoundError(f"Document not found: {leaf}")
    return candidate


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    request: IngestRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    ingestion: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestResponse:
    """Index uploaded files into the vector store.

    A ``filenames`` list restricts ingestion to those files; when omitted,
    every stored file is ingested. Per-file failures are reported in the
    response without aborting the batch.
    """
    t0 = time.perf_counter()

    if request.filenames:
        paths = [_resolve_uploaded(settings.upload_directory, name) for name in request.filenames]
    else:
        paths = sorted(settings.upload_directory.glob("*"))
        paths = [p for p in paths if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not paths:
        return IngestResponse(results=[], elapsed_ms=0.0)

    results = ingestion.ingest_batch(paths)
    items = [_to_result_item(result) for result in results]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return IngestResponse(results=items, elapsed_ms=elapsed_ms)


def _to_result_item(result) -> IngestResultItem:
    error = None
    if result.status.startswith("error:"):
        error = result.status[len("error:"):].strip()
    return IngestResultItem(
        filename=result.filename,
        document_id=result.document_id,
        n_chunks=result.n_chunks,
        status=result.status,
        error=error,
    )
