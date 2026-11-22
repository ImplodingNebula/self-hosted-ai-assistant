"""Request-id middleware: correlation ids plus one access log line per request.

Written against the raw ASGI protocol (not BaseHTTPMiddleware) so streaming
responses - the SSE chat endpoint in particular - pass through one message at
a time with no buffering or background-task pitfalls.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_var

_access_logger = logging.getLogger("app.access")

_HEADER_NAME = b"x-request-id"
_MAX_REQUEST_ID_LENGTH = 128

# Probe endpoints, polled every few seconds by compose healthchecks and by
# whatever orchestrates the deployment. Their access lines are logged at DEBUG
# so they do not bury the real traffic in the INFO stream.
_PROBE_PATHS = frozenset({"/healthz", "/readyz"})


def _sanitize_request_id(raw: bytes | None) -> str | None:
    """Return a safe client-supplied request id, or None if unusable.

    The id is reflected into response headers and log lines, so anything
    non-ASCII, non-printable, empty, or over 128 chars is discarded (the
    middleware then generates a fresh id) rather than repaired.
    """
    if raw is None:
        return None
    try:
        value = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if not value or len(value) > _MAX_REQUEST_ID_LENGTH or not value.isprintable():
        return None
    return value


def _content_length(scope: Scope) -> int | None:
    """Read the request's declared Content-Length, or None if absent/malformed."""
    for name, value in scope.get("headers") or []:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class MaxBodySizeMiddleware:
    """Answer 413 for an over-large request body before it is read or spooled.

    FastAPI parses (and spools to disk) a multipart body BEFORE any route
    dependency or handler line runs, so a size check inside the upload endpoint
    fires only after the bytes have already landed - too late to stop a body
    that could fill the container's disk and take the whole stack down. This ASGI
    middleware runs first: a request whose declared Content-Length exceeds the
    ceiling is refused outright, its body never consumed.

    The ceiling is the app's largest legitimate body (an attachment upload) plus
    room for the multipart envelope; the upload handler still applies the exact
    per-file limit after the (now-bounded) body is parsed. A body sent without a
    Content-Length (chunked) is not measurable here and is bounded by the ingress
    proxy in production - defence the app does not have to duplicate.

    The problem+json body is written directly rather than raised: an exception
    from user middleware escapes the app's exception handlers (they sit inside
    it) and would render as a 500. RequestIdMiddleware wraps this one, so the
    response still gets its X-Request-ID header and access-log line.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject an oversized http body; every other scope passes through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._reject(send)
            return
        await self.app(scope, receive, send)

    async def _reject(self, send: Send) -> None:
        """Send a 413 problem+json without reading the request body."""
        body = json.dumps(
            {
                "type": "about:blank",
                "title": "Payload Too Large",
                "status": 413,
                "detail": f"The request body exceeds the {self.max_bytes} byte limit.",
                "code": "payload_too_large",
                "request_id": request_id_var.get(),
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/problem+json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class RequestIdMiddleware:
    """Assign a per-request id, expose it via contextvar, and log an access line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection; non-http scopes pass through untouched."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        request_id = _sanitize_request_id(headers.get(_HEADER_NAME)) or uuid.uuid4().hex
        request_id_var.set(request_id)

        status_code = 500  # assume the worst; overwritten on http.response.start
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                raw_headers = [
                    (name, value)
                    for name, value in message.get("headers") or []
                    if name.lower() != _HEADER_NAME
                ]
                raw_headers.append((_HEADER_NAME, request_id.encode("ascii")))
                message["headers"] = raw_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            # finally: an escaping exception still gets its access line (as 500)
            # before the outermost error middleware renders the response.
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            method = scope.get("method", "")
            path = scope.get("path", "")
            level = logging.DEBUG if path in _PROBE_PATHS else logging.INFO
            _access_logger.log(
                level,
                "%s %s -> %d",
                method,
                path,
                status_code,
                extra={
                    "method": method,
                    "path": path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                },
            )
