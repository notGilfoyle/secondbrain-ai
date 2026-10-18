"""Validated application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, with environment variables prefixed by ``SB_``."""

    app_name: str = "SecondBrain AI"
    environment: str = "local"
    debug: bool = False
    database_path: Path = Path("data/secondbrain.db")
    document_storage_path: Path = Path("data/documents")
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, gt=0)

    model_config = SettingsConfigDict(
        env_prefix="SB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings instance for dependency injection."""
    return Settings()
