"""Shared FastAPI dependency providers.

Providers are cached so expensive objects (settings, and later embedders,
vector stores, LLM clients) are constructed once per process. Tests replace
them via ``app.dependency_overrides``.
"""

from functools import lru_cache

from app.config import Settings


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide application settings."""
    return Settings()
