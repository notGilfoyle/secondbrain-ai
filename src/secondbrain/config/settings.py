"""Validated application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, with environment variables prefixed by ``SB_``."""

    app_name: str = "SecondBrain AI"
    environment: str = "local"
    debug: bool = False

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
