"""Safe JSON event logging and server-side HTTP request correlation."""

import json
import logging
import math
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_CLIENT_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}\Z", re.ASCII)
_EVENT_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,79}\Z", re.ASCII)
_METHOD_PATTERN = re.compile(r"[A-Z]{1,16}\Z", re.ASCII)
_ROUTE_SEGMENT_PATTERN = re.compile(
    r"(?:[A-Za-z0-9._~-]+|\{[A-Za-z_][A-Za-z0-9_]*\})\Z", re.ASCII
)
_SAFE_EVENT_FIELDS = {
    "request_id",
    "client_request_id",
    "http_method",
    "route",
    "status_code",
    "duration_ms",
    "operation",
    "failure_category",
}
_ALLOWED_EVENTS = {
    "http.request.completed",
    "document.ingestion.failed",
    "document.list.failed",
    "document.get.failed",
}


def emit_structured_event(
    logger: logging.Logger, event: str, **fields: str | int | float
) -> None:
    """Emit one JSON event with a fixed field allowlist and validated values."""
    if (
        not isinstance(event, str)
        or not _EVENT_PATTERN.fullmatch(event)
        or event not in _ALLOWED_EVENTS
    ):
        raise ValueError("event name is invalid")
    unknown_fields = fields.keys() - _SAFE_EVENT_FIELDS
    if unknown_fields:
        raise ValueError("structured event contains unsupported fields")

    payload: dict[str, str | int | float] = {"event": event}
    for field, value in fields.items():
        if field in {"request_id"}:
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
                raise ValueError(f"{field} is invalid")
        elif field == "client_request_id":
            if not isinstance(value, str) or not _CLIENT_ID_PATTERN.fullmatch(value):
                raise ValueError(f"{field} is invalid")
        elif field == "http_method":
            if not isinstance(value, str) or not _METHOD_PATTERN.fullmatch(value):
                raise ValueError(f"{field} is invalid")
        elif field == "route":
            if not isinstance(value, str) or not _is_safe_route_template(value):
                raise ValueError(f"{field} is invalid")
        elif field == "status_code":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 100 <= value <= 599
            ):
                raise ValueError(f"{field} is invalid")
        elif field == "duration_ms":
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field} is invalid")
        elif field == "operation":
            if value not in {"ingest", "list", "get"}:
                raise ValueError(f"{field} is invalid")
        elif field == "failure_category":
            if value != "unexpected":
                raise ValueError(f"{field} is invalid")
        payload[field] = value
    logger.info(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))


def _is_safe_route_template(route: str) -> bool:
    if route in {"/", "unmatched"}:
        return True
    return route.startswith("/") and all(
        _ROUTE_SEGMENT_PATTERN.fullmatch(segment) for segment in route[1:].split("/")
    )


class RequestCorrelationMiddleware:
    """Add a server request ID and one safe completion event around HTTP work."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        event_logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.perf_counter,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.app = app
        self.event_logger = event_logger or logging.getLogger("secondbrain.http")
        self.clock = clock
        self.request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        client_request_id = _validated_client_request_id(scope)
        request_id = self.request_id_factory()
        while request_id == client_request_id:
            request_id = self.request_id_factory()
        scope.setdefault("state", {})["request_id"] = request_id
        if client_request_id is not None:
            scope["state"]["client_request_id"] = client_request_id

        started = self.clock()
        response_status = 500

        async def correlated_send(message: Message) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, correlated_send)
        finally:
            fields: dict[str, str | int | float] = {
                "request_id": request_id,
                "http_method": _safe_method(scope.get("method")),
                "route": _route_template(scope),
                "status_code": response_status,
                "duration_ms": round(max(0.0, self.clock() - started) * 1000, 3),
            }
            if client_request_id is not None:
                fields["client_request_id"] = client_request_id
            emit_structured_event(self.event_logger, "http.request.completed", **fields)


def _validated_client_request_id(scope: Scope) -> str | None:
    values = [
        value
        for name, value in scope.get("headers", [])
        if name.lower() == b"x-request-id"
    ]
    if len(values) != 1:
        return None
    try:
        decoded = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    return decoded if _CLIENT_ID_PATTERN.fullmatch(decoded) else None


def _safe_method(method: Any) -> str:
    if isinstance(method, str) and _METHOD_PATTERN.fullmatch(method):
        return method
    return "OTHER"


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    template = getattr(route, "path", None)
    return (
        template
        if isinstance(template, str) and _is_safe_route_template(template)
        else "unmatched"
    )
