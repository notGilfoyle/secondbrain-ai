"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import PlainTextResponse

from secondbrain.api.middleware import (
    MULTIPART_OVERHEAD_BYTES,
    RequestBodySizeLimitMiddleware,
    request_body_limit_exception_handler,
)
from secondbrain.api.observability import RequestCorrelationMiddleware
from secondbrain.api.routes.documents import router as documents_router
from secondbrain.api.routes.health import router as health_router
from secondbrain.config.settings import get_settings


def create_app() -> FastAPI:
    """Create the application with configured metadata and API routes."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug)

    async def server_error_response(
        request: Request, _exception: Exception
    ) -> PlainTextResponse:
        """Keep Starlette's generic 500 body and expose the canonical request ID."""
        request_id = request.scope.get("state", {}).get("request_id")
        headers = {"X-Request-ID": request_id} if isinstance(request_id, str) else {}
        return PlainTextResponse(
            "Internal Server Error", status_code=500, headers=headers
        )

    # An integer 500 handler is used by ServerErrorMiddleware, which wraps
    # user middleware and therefore creates unhandled responses outside the
    # correlation middleware's send wrapper. Starlette still re-raises after
    # sending the response, so this does not swallow application exceptions.
    app.add_exception_handler(500, server_error_response)
    app.add_middleware(
        RequestBodySizeLimitMiddleware,
        max_body_size_bytes=(settings.max_upload_size_bytes + MULTIPART_OVERHEAD_BYTES),
    )
    # Starlette places the most recently added user middleware outermost, so the
    # correlation layer also observes direct 413 responses from the body limiter.
    app.add_middleware(RequestCorrelationMiddleware)
    app.add_exception_handler(
        StarletteHTTPException, request_body_limit_exception_handler
    )
    app.include_router(health_router)
    app.include_router(documents_router)
    return app


app = create_app()
