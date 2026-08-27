"""FastAPI application factory.

Run locally with::

    uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api.dependencies import get_settings
from app.api.routes import health
from app.config import Settings
from app.errors import RagError

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """Configure root logging once; unknown levels fall back to INFO."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        # getattr above returns INFO for any unrecognized string.
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare runtime directories on startup; cleanup hook for later phases."""
    settings: Settings = app.state.settings
    settings.ensure_runtime_directories()
    logger.info("Application started (embedding_model=%s)", settings.embedding_model)
    yield
    logger.info("Application shutdown")


def register_exception_handlers(app: FastAPI) -> None:
    """Map domain errors to a consistent, secret-free JSON error envelope."""

    @app.exception_handler(RagError)
    async def rag_error_handler(request: Request, exc: RagError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"code": exc.code, "message": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Summarize only the first error to avoid leaking request internals.
        errors = exc.errors()
        detail = "Invalid request"
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first.get("loc", []))
            detail = f"Invalid request at '{location}': {first.get('msg', 'validation failed')}"
        return JSONResponse(
            status_code=422,
            content={"code": "validation_error", "message": detail},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Full details stay in server logs; clients receive a generic message.
        logger.exception("Unhandled application error")
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "An unexpected error occurred."},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings if settings is not None else Settings()
    _configure_logging(resolved_settings.log_level)

    app = FastAPI(
        title="AI Assistant with RAG",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    # Inject the provided settings instance so routes consistently depend on
    # get_settings() while tests can pass isolated Settings objects.
    app.dependency_overrides[get_settings] = lambda: resolved_settings

    app.include_router(health.router)
    register_exception_handlers(app)
    return app


# ASGI entry point for `uvicorn app.main:app`.
app = create_app()
