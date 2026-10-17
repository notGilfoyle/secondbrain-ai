"""FastAPI application entry point."""

from fastapi import FastAPI

from secondbrain.api.routes.health import router as health_router
from secondbrain.config.settings import get_settings


def create_app() -> FastAPI:
    """Create the application with configured metadata and API routes."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug)
    app.include_router(health_router)
    return app


app = create_app()
