"""Request-correlation and safe completion-event tests."""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Annotated, Any

import pytest
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from secondbrain.api.middleware import RequestBodySizeLimitMiddleware
from secondbrain.api.observability import (
    RequestCorrelationMiddleware,
    emit_structured_event,
)
from secondbrain.main import create_app


def completion_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "secondbrain.http"
    ]


def test_structured_event_rejects_unapproved_fields() -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        emit_structured_event(
            logging.getLogger("secondbrain.http"),
            "http.request.completed",
            prompt="PRIVATE_PROMPT",
        )


async def invoke_asgi_with_request_id(
    header_values: list[tuple[bytes, bytes]],
    request_id_factory: Callable[[], str] | None = None,
) -> tuple[list[Message], Scope]:
    sent: list[Message] = []
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/not-logged",
        "raw_path": b"/not-logged",
        "query_string": b"secret-query=value",
        "root_path": "",
        "headers": header_values,
        "server": ("testserver", 80),
        "client": ("testclient", 1234),
    }

    async def app(inner_scope: Scope, receive: Receive, send: Send) -> None:
        assert inner_scope["state"]["request_id"]
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestCorrelationMiddleware(
        app, request_id_factory=request_id_factory
    )
    await middleware(scope, receive, send)
    return sent, scope


@pytest.mark.parametrize(
    ("headers", "accepted_id"),
    [
        ([], None),
        ([(b"x-request-id", b"client.correlation-42")], "client.correlation-42"),
        ([(b"x-request-id", b"bad id")], None),
        ([(b"x-request-id", b"x" * 65)], None),
        ([(b"x-request-id", b"caf\xe9")], None),
        (
            [
                (b"x-request-id", b"valid-one"),
                (b"x-request-id", b"valid-two"),
            ],
            None,
        ),
    ],
)
def test_request_id_policy(
    headers: list[tuple[bytes, bytes]],
    accepted_id: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="secondbrain.http")
    sent, scope = asyncio.run(invoke_asgi_with_request_id(headers))
    request_id = scope["state"]["request_id"]
    response_start = next(
        message for message in sent if message["type"] == "http.response.start"
    )
    response_headers = dict(response_start["headers"])
    event = completion_events(caplog)[0]

    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    assert response_headers[b"x-request-id"].decode("ascii") == request_id
    assert request_id not in [
        value.decode("ascii", errors="ignore") for _, value in headers
    ]
    assert event["request_id"] == request_id
    assert event.get("client_request_id") == accepted_id
    assert len(completion_events(caplog)) == 1


def test_server_id_is_regenerated_if_uuid_matches_client_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="secondbrain.http")
    candidate_ids = iter(["client-id-123", "a" * 32])
    sent, scope = asyncio.run(
        invoke_asgi_with_request_id(
            [(b"x-request-id", b"client-id-123")],
            request_id_factory=lambda: next(candidate_ids),
        )
    )
    assert scope["state"]["request_id"] == "a" * 32
    event = completion_events(caplog)[0]
    assert event["client_request_id"] == "client-id-123"
    assert event["request_id"] != event["client_request_id"]
    assert next(
        message for message in sent if message["type"] == "http.response.start"
    )["headers"]


def test_route_template_error_event_and_log_redaction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="secondbrain.http")
    app = FastAPI()

    @app.get("/api/v1/documents/{document_id}")
    def get_document(document_id: str, request: Request, q: str = "") -> dict[str, str]:
        if q == "trigger-error":
            raise HTTPException(status_code=409, detail="RAW_EXCEPTION_SENTINEL")
        return {"request_id": request.state.request_id, "document_id": document_id}

    app.add_middleware(RequestCorrelationMiddleware)
    document_id = "DOCUMENT_ID_SENTINEL_abc123"
    client = TestClient(app)
    response = client.get(
        f"/api/v1/documents/{document_id}?q=QUERY_SENTINEL",
        headers={"Authorization": "AUTHORIZATION_SENTINEL"},
    )
    error = client.get(
        f"/api/v1/documents/{document_id}?q=trigger-error",
        headers={"Authorization": "AUTHORIZATION_SENTINEL"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == response.json()["request_id"]
    assert error.status_code == 409
    events = completion_events(caplog)
    assert len(events) == 2
    assert [event["status_code"] for event in events] == [200, 409]
    assert all(event["route"] == "/api/v1/documents/{document_id}" for event in events)
    assert all(event["http_method"] == "GET" for event in events)
    assert all(isinstance(event["duration_ms"], float | int) for event in events)
    assert all("client_request_id" not in event for event in events)
    event_text = " ".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "secondbrain.http"
    )
    for secret in (
        document_id,
        "QUERY_SENTINEL",
        "AUTHORIZATION_SENTINEL",
        "RAW_EXCEPTION_SENTINEL",
    ):
        assert secret not in event_text


def test_unexpected_exception_is_correlated_without_logging_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="secondbrain.http")

    async def fail(scope: Scope, receive: Receive, send: Send) -> None:
        message = await receive()
        assert message["body"] == b"DOCUMENT_TEXT_SENTINEL PROMPT_SENTINEL"
        raise RuntimeError("RAW_EXCEPTION_SENTINEL API_KEY_SENTINEL")

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/answer/DOCUMENT_ID_SENTINEL",
        "raw_path": b"/answer/DOCUMENT_ID_SENTINEL",
        "query_string": b"query=QUERY_SENTINEL",
        "root_path": "",
        "headers": [
            (b"authorization", b"AUTHORIZATION_SENTINEL"),
            (b"x-request-id", b"client-correlation-1"),
        ],
    }

    async def receive() -> Message:
        return {
            "type": "http.request",
            "body": b"DOCUMENT_TEXT_SENTINEL PROMPT_SENTINEL",
            "more_body": False,
        }

    async def send(message: Message) -> None:
        pytest.fail(f"unexpected response message: {message!r}")

    with pytest.raises(RuntimeError, match="RAW_EXCEPTION_SENTINEL"):
        asyncio.run(RequestCorrelationMiddleware(fail)(scope, receive, send))

    events = completion_events(caplog)
    assert len(events) == 1
    assert events[0]["status_code"] == 500
    event_text = caplog.records[0].getMessage()
    for secret in (
        "RAW_EXCEPTION_SENTINEL",
        "PROMPT_SENTINEL",
        "DOCUMENT_TEXT_SENTINEL",
        "DOCUMENT_ID_SENTINEL",
        "API_KEY_SENTINEL",
        "QUERY_SENTINEL",
        "AUTHORIZATION_SENTINEL",
    ):
        assert secret not in event_text


def test_unhandled_500_response_has_request_id_and_one_completion_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="secondbrain.http")
    app = create_app()

    @app.get("/test/unhandled")
    def fail() -> None:
        raise RuntimeError("PRIVATE_EXCEPTION_SENTINEL")

    response = TestClient(app, raise_server_exceptions=False).get(
        "/test/unhandled?secret=QUERY_SENTINEL"
    )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    request_id = response.headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    events = completion_events(caplog)
    assert len(events) == 1
    assert events[0]["request_id"] == request_id
    assert events[0]["status_code"] == 500
    assert events[0]["route"] == "/test/unhandled"
    assert "PRIVATE_EXCEPTION_SENTINEL" not in caplog.text
    assert "QUERY_SENTINEL" not in caplog.text


def test_early_body_limit_response_has_request_id_and_one_completion_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="secondbrain.http")
    app = FastAPI()

    @app.post("/api/v1/documents")
    async def upload(file: Annotated[UploadFile, File()]) -> dict[str, str]:
        await file.close()
        return {"filename": file.filename or ""}

    app.add_middleware(RequestBodySizeLimitMiddleware, max_body_size_bytes=256)
    app.add_middleware(RequestCorrelationMiddleware)
    response = TestClient(app).post(
        "/api/v1/documents?query=DO_NOT_LOG",
        files={"file": ("private.txt", b"x" * 2048, "text/plain")},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the configured limit"}
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])
    events = completion_events(caplog)
    assert len(events) == 1
    assert events[0]["status_code"] == 413
    # Body-limit rejection occurs before routing, so use the safe fallback.
    assert events[0]["route"] == "unmatched"
    assert "DO_NOT_LOG" not in events[0].__str__()
