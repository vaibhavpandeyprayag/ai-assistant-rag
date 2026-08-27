"""Health check endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app import __version__
from app.api.dependencies import get_settings
from app.config import Settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def read_health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Report basic service status without exposing secrets or internals."""
    return HealthResponse(
        status="ok",
        version=__version__,
        embedding_model=settings.embedding_model,
    )
