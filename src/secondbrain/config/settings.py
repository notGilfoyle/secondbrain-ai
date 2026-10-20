"""Validated application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, with environment variables prefixed by ``SB_``."""

    app_name: str = "SecondBrain AI"
    environment: str = "local"
    debug: bool = False
    database_path: Path = Path("data/secondbrain.db")
    document_storage_path: Path = Path("data/documents")
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    cohere_api_key: SecretStr | None = None
    cohere_model: str | None = None
    cohere_timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("cohere_api_key")
    @classmethod
    def validate_cohere_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        """Reject a configured Cohere key that contains only whitespace."""
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("cohere_api_key must not be blank")
        return value

    @field_validator("cohere_model")
    @classmethod
    def validate_cohere_model(cls, value: str | None) -> str | None:
        """Reject a configured Cohere model name that contains only whitespace."""
        if value is not None and not value.strip():
            raise ValueError("cohere_model must not be blank")
        return value

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
