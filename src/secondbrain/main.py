"""FastAPI application entry point."""

from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException

from secondbrain.api.middleware import (
    MULTIPART_OVERHEAD_BYTES,
    RequestBodySizeLimitMiddleware,
    request_body_limit_exception_handler,
)
from secondbrain.api.routes.documents import router as documents_router
from secondbrain.api.routes.health import router as health_router
from secondbrain.config.settings import get_settings


def create_app() -> FastAPI:
    """Create the application with configured metadata and API routes."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug)
    app.add_middleware(
        RequestBodySizeLimitMiddleware,
        max_body_size_bytes=(settings.max_upload_size_bytes + MULTIPART_OVERHEAD_BYTES),
    )
    app.add_exception_handler(
        StarletteHTTPException, request_body_limit_exception_handler
    )
    app.include_router(health_router)
    app.include_router(documents_router)
    return app


app = create_app()
