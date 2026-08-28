"""Centralized application configuration.

All runtime settings are read from environment variables; an optional ``.env``
file in the project root is supported for local development. Secrets are held
in ``SecretStr`` so they are never leaked through logs or serialization.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    log_level: str = Field(default="INFO")

    # --- CORS ----------------------------------------------------------------
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Cross-origin request origins allowed by the API",
    )

    # --- LLM ----------------------------------------------------------------
    llm_provider: Literal["hf"] = "hf"
    huggingface_model: str = "microsoft/Phi-3-mini-4k-instruct"
    huggingface_api_key: SecretStr | None = None

    # --- Embeddings -----------------------------------------------------------
    embedding_provider: Literal["local", "hf"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage --------------------------------------------------------------
    chroma_persist_directory: Path = Path("data/chroma")
    upload_directory: Path = Path("data/uploads")

    # --- Chunking ---------------------------------------------------------------
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    chunk_strategy: Literal["recursive", "fixed"] = "recursive"

    # --- Retrieval -----------------------------------------------------------------
    top_k: int = Field(default=5, ge=1)
    retrieval_min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Upload limits ------------------------------------------------------------
    upload_max_size_mb: int = Field(default=20, ge=1)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        """Parse a comma-separated origin list from the environment."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
        """Ensure chunk overlap stays strictly smaller than the chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    def ensure_runtime_directories(self) -> None:
        """Create runtime storage directories if they do not exist."""
        self.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
