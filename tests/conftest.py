"""Shared pytest fixtures."""

import pytest

from app.config import Settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every application env var so tests see pristine defaults.

    Field names map 1:1 to environment variable names (case-insensitive
    matching is disabled via ``case_sensitive=False`` in Settings, so the
    upper-cased field name is the canonical env var key).
    """
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    return monkeypatch


@pytest.fixture
def base_settings(clean_env) -> Settings:
    """Settings built purely from defaults, ignoring any local .env file."""
    return Settings(_env_file=None)
