"""ASGI request-body limits for document multipart uploads."""

import json
from collections.abc import Sequence

from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

DOCUMENT_UPLOAD_PATH = "/api/v1/documents"
MULTIPART_OVERHEAD_BYTES = 64 * 1024
BODY_LIMIT_EXCEPTION_DETAIL = "secondbrain_request_body_limit_exceeded"
BODY_LIMIT_RESPONSE_DETAIL = "Request body exceeds the configured limit"


class RequestBodyLimitExceeded(MultiPartException):
    """Multipart parser exception used to close partial uploads on overflow."""

    def __init__(self) -> None:
        super().__init__(BODY_LIMIT_EXCEPTION_DETAIL)


class RequestBodySizeLimitMiddleware:
    """Bound document request bodies without buffering or replaying their stream."""

    def __init__(self, app: ASGIApp, max_body_size_bytes: int) -> None:
        if max_body_size_bytes <= 0:
            raise ValueError("max_body_size_bytes must be positive")
        self.app = app
        self.max_body_size_bytes = max_body_size_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._is_document_upload(scope):
            await self.app(scope, receive, send)
            return

        if self._declared_length_exceeds_limit(scope):
            await self._send_too_large(send)
            return

        bytes_received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal bytes_received
            message = await receive()
            if message["type"] == "http.request":
                bytes_received += len(message.get("body", b""))
                if bytes_received > self.max_body_size_bytes:
                    # Starlette's multipart parser catches this exception and closes
                    # any UploadFile objects it has already spooled.
                    raise RequestBodyLimitExceeded()
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyLimitExceeded:
            # Non-multipart consumers do not translate MultiPartException themselves.
            # Multipart parsing translates it to HTTPException and uses the handler
            # below, after closing its temporary upload files.
            if response_started:
                raise
            await self._send_too_large(send)

    @staticmethod
    def _is_document_upload(scope: Scope) -> bool:
        return (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path", "").rstrip("/") == DOCUMENT_UPLOAD_PATH
        )

    def _declared_length_exceeds_limit(self, scope: Scope) -> bool:
        headers: Sequence[tuple[bytes, bytes]] = scope.get("headers", [])
        content_lengths = [
            value for name, value in headers if name.lower() == b"content-length"
        ]
        has_transfer_encoding = any(
            name.lower() == b"transfer-encoding" for name, _ in headers
        )
        if len(content_lengths) != 1 or has_transfer_encoding:
            return False
        try:
            raw_length = content_lengths[0].decode("ascii")
        except UnicodeDecodeError:
            return False
        return raw_length.isdecimal() and int(raw_length) > self.max_body_size_bytes

    @staticmethod
    async def _send_too_large(send: Send) -> None:
        body = json.dumps({"detail": BODY_LIMIT_RESPONSE_DETAIL}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


async def request_body_limit_exception_handler(
    request: Request, exc: Exception
) -> Response:
    """Translate Starlette's multipart overflow response to HTTP 413."""
    if (
        isinstance(exc, StarletteHTTPException)
        and exc.status_code == 400
        and exc.detail == BODY_LIMIT_EXCEPTION_DETAIL
    ):
        return JSONResponse(
            status_code=413,
            content={"detail": BODY_LIMIT_RESPONSE_DETAIL},
        )
    if isinstance(exc, StarletteHTTPException):
        return await http_exception_handler(request, exc)
    raise exc
