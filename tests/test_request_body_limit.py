"""ASGI-level tests for request-body limits before multipart parsing finishes."""

import asyncio
import json
from collections.abc import Callable
from typing import Annotated, Any

import pytest
from fastapi import FastAPI, File, UploadFile
from starlette import formparsers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Message, Receive, Scope, Send

from secondbrain.api.middleware import (
    BODY_LIMIT_RESPONSE_DETAIL,
    RequestBodySizeLimitMiddleware,
    request_body_limit_exception_handler,
)


class ReceiveCountingApp:
    """Count messages actually forwarded from middleware to the parser app."""

    def __init__(self, app: Callable[..., Any]) -> None:
        self.app = app
        self.forwarded_bytes = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def counted_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                self.forwarded_bytes += len(message.get("body", b""))
            return message

        await self.app(scope, counted_receive, send)


def make_multipart_body() -> tuple[bytes, bytes]:
    boundary = b"asgi-test-boundary"
    body = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="notes.txt"'
        + b"\r\nContent-Type: text/plain\r\n\r\n"
        + b"x" * 800
        + b"\r\n--"
        + boundary
        + b"--\r\n"
    )
    content_type = b"multipart/form-data; boundary=" + boundary
    return body, content_type


def run_asgi_request(
    *, content_length: bytes | None, max_body_size: int = 256
) -> tuple[int, dict[str, Any], int, int, int, bool]:
    body, content_type = make_multipart_body()
    chunks = [body[offset : offset + 64] for offset in range(0, len(body), 64)]
    receive_calls = 0
    sent_messages: list[Message] = []
    route_called = False
    fastapi_app = FastAPI()
    fastapi_app.add_exception_handler(
        StarletteHTTPException, request_body_limit_exception_handler
    )

    @fastapi_app.post("/api/v1/documents")
    async def parse_upload(file: Annotated[UploadFile, File()]) -> dict[str, bool]:
        nonlocal route_called
        route_called = True
        await file.close()
        return {"accepted": True}

    counting_app = ReceiveCountingApp(fastapi_app)
    limited_app = RequestBodySizeLimitMiddleware(
        counting_app, max_body_size_bytes=max_body_size
    )

    async def receive() -> Message:
        nonlocal receive_calls
        if receive_calls < len(chunks):
            chunk = chunks[receive_calls]
            receive_calls += 1
            return {
                "type": "http.request",
                "body": chunk,
                "more_body": receive_calls < len(chunks),
            }
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent_messages.append(message)

    headers = [(b"content-type", content_type), (b"host", b"testserver")]
    if content_length is not None:
        headers.append((b"content-length", content_length))
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/documents",
        "raw_path": b"/api/v1/documents",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "app": fastapi_app,
    }

    asyncio.run(limited_app(scope, receive, send))
    status_code = next(
        message["status"]
        for message in sent_messages
        if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in sent_messages
        if message["type"] == "http.response.body"
    )
    return (
        status_code,
        json.loads(response_body),
        receive_calls,
        len(chunks),
        counting_app.forwarded_bytes,
        route_called,
    )


@pytest.mark.parametrize("content_length", [None, b"1"])
def test_stream_limit_stops_before_multipart_parser_consumes_full_body(
    content_length: bytes | None,
) -> None:
    status_code, response, receive_calls, chunk_count, forwarded, route_called = (
        run_asgi_request(content_length=content_length)
    )

    assert status_code == 413
    assert response == {"detail": BODY_LIMIT_RESPONSE_DETAIL}
    assert receive_calls < chunk_count
    assert forwarded <= 256
    assert route_called is False


def test_declared_oversize_request_is_rejected_without_reading_body() -> None:
    status_code, response, receive_calls, _, forwarded, route_called = run_asgi_request(
        content_length=b"257"
    )

    assert status_code == 413
    assert response == {"detail": BODY_LIMIT_RESPONSE_DETAIL}
    assert receive_calls == 0
    assert forwarded == 0
    assert route_called is False


def test_stream_overflow_closes_partial_multipart_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_factory = formparsers.SpooledTemporaryFile
    created_files: list[Any] = []

    def track_spooled_file(*args: Any, **kwargs: Any) -> Any:
        spooled_file = real_factory(*args, **kwargs)
        created_files.append(spooled_file)
        return spooled_file

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", track_spooled_file)
    status_code, _, _, _, _, route_called = run_asgi_request(content_length=None)

    assert status_code == 413
    assert route_called is False
    assert created_files
    assert all(spooled_file.closed for spooled_file in created_files)
