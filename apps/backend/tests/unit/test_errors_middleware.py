"""Tests for the RFC 7807 error surface and the request-id middleware."""

from __future__ import annotations

import json
import logging
import sys

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.core.logging import JsonLineFormatter, configure_logging
from app.core.middleware import MaxBodySizeMiddleware
from app.main import create_app

PROBLEM_KEYS = {"type", "title", "status", "detail", "code", "request_id"}


def assert_problem_shape(resp, status: int, code: str) -> dict:
    """Assert the response is a well-formed problem+json and return its body."""
    assert resp.status_code == status
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert set(body) >= PROBLEM_KEYS
    assert body["type"] == "about:blank"
    assert body["status"] == status
    assert body["code"] == code
    assert isinstance(body["title"], str) and body["title"]
    assert isinstance(body["detail"], str) and body["detail"]
    return body


async def test_404_problem_json_full_shape(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/definitely-not-a-route")
    body = assert_problem_shape(resp, 404, "not_found")
    assert body["title"] == "Not Found"
    assert body["request_id"]
    assert resp.headers["x-request-id"] == body["request_id"]


async def test_supplied_request_id_echoed_in_header_and_body(client: AsyncClient) -> None:
    resp = await client.get(
        "/api/v1/definitely-not-a-route", headers={"X-Request-ID": "trace-abc-123"}
    )
    assert resp.headers["x-request-id"] == "trace-abc-123"
    assert resp.json()["request_id"] == "trace-abc-123"


async def test_request_id_generated_when_absent(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/definitely-not-a-route")
    generated = resp.headers["x-request-id"]
    # uuid4().hex: 32 lowercase hex chars.
    assert len(generated) == 32
    int(generated, 16)
    assert resp.json()["request_id"] == generated


async def test_unusable_request_id_is_replaced(client: AsyncClient) -> None:
    oversized = "x" * 200
    resp = await client.get("/healthz", headers={"X-Request-ID": oversized})
    assert resp.headers["x-request-id"] != oversized
    assert len(resp.headers["x-request-id"]) == 32


async def test_422_validation_problem_from_invalid_chat_body(client: AsyncClient) -> None:
    # An empty body is neither words, files, nor a confirmation - the model
    # validator rejects it. Whether an empty message is valid depends on the other
    # fields, so the error is reported against the body as a whole, and its message
    # names what was missing.
    resp = await client.post("/api/v1/chat/stream", json={})
    body = assert_problem_shape(resp, 422, "validation_error")
    assert body["request_id"]
    errors = body["errors"]
    assert isinstance(errors, list) and errors
    for err in errors:
        assert set(err) == {"loc", "msg", "type"}
    assert any("message" in err["msg"] for err in errors)
    # No internal reprs (pydantic ctx/input) may leak.
    assert "ctx" not in json.dumps(body)


async def test_catch_all_500_problem_with_request_id() -> None:
    app = create_app()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("secret internal detail")

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/boom", headers={"X-Request-ID": "boom-1"})

    body = assert_problem_shape(resp, 500, "internal_error")
    assert body["request_id"] == "boom-1"
    assert resp.headers["x-request-id"] == "boom-1"
    # The generic detail must not leak the real exception text.
    assert "secret internal detail" not in json.dumps(body)


async def test_healthz_unaffected(client: AsyncClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"status": "ok"}
    assert resp.headers["x-request-id"]


async def test_sse_stream_survives_middleware(client: AsyncClient) -> None:
    lines: list[str] = []
    async with client.stream(
        "POST", "/api/v1/chat/stream", json={"message": "hi"}, headers={"X-Request-ID": "sse-1"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["x-request-id"] == "sse-1"
        async for line in resp.aiter_lines():
            if line:
                lines.append(line)

    body = "\n".join(lines)
    assert "event: token" in body
    assert "event: done" in body


async def test_access_log_levels(client: AsyncClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="app.access"):
        await client.get("/healthz")
        await client.get("/api/v1/models")

    by_path = {record.path: record for record in caplog.records if record.name == "app.access"}
    assert by_path["/healthz"].levelno == logging.DEBUG
    assert by_path["/api/v1/models"].levelno == logging.INFO
    for record in by_path.values():
        assert record.method == "GET"
        assert isinstance(record.status, int)
        assert record.duration_ms >= 0


def _http_scope(headers: list[tuple[bytes, bytes]]) -> dict:
    return {"type": "http", "method": "POST", "path": "/api/v1/attachments", "headers": headers}


async def _drive(middleware: MaxBodySizeMiddleware, scope: dict) -> tuple[list[dict], bool]:
    """Run one request through the middleware; return (sent messages, inner-called)."""
    called = False

    async def inner(scope, receive, send):  # noqa: ARG001 - ASGI signature
        nonlocal called
        called = True
        await receive()  # reading the body is exactly what a rejection must prevent
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware.app = inner
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def receive() -> dict:
        return {"type": "http.request", "body": b"x", "more_body": False}

    await middleware(scope, receive, send)
    return sent, called


async def test_oversized_body_is_refused_before_it_is_read() -> None:
    mw = MaxBodySizeMiddleware(None, max_bytes=100)  # app replaced in _drive
    sent, called = await _drive(mw, _http_scope([(b"content-length", b"101")]))

    # The inner app never ran, so the body was never read or spooled.
    assert called is False
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert (b"content-type", b"application/problem+json") in sent[0]["headers"]
    body = json.loads(sent[1]["body"])
    assert body["code"] == "payload_too_large"
    assert body["status"] == 413


async def test_body_within_limit_passes_through() -> None:
    mw = MaxBodySizeMiddleware(None, max_bytes=100)
    sent, called = await _drive(mw, _http_scope([(b"content-length", b"100")]))

    assert called is True
    assert sent[0]["status"] == 201


async def test_body_without_content_length_passes_through() -> None:
    # A chunked upload cannot be measured here; the ingress proxy bounds it.
    mw = MaxBodySizeMiddleware(None, max_bytes=100)
    sent, called = await _drive(mw, _http_scope([]))

    assert called is True
    assert sent[0]["status"] == 201


async def test_malformed_content_length_passes_through() -> None:
    mw = MaxBodySizeMiddleware(None, max_bytes=100)
    sent, called = await _drive(mw, _http_scope([(b"content-length", b"not-a-number")]))

    assert called is True


def test_body_size_guard_is_installed_with_an_upload_sized_ceiling() -> None:
    # Wiring check: the guard is on the stack (httpx recomputes Content-Length
    # from the body, so a real over-limit request cannot be forged cheaply), and
    # its ceiling clears the largest legitimate upload plus envelope headroom.
    app = create_app()
    guards = [m for m in app.user_middleware if m.cls is MaxBodySizeMiddleware]
    assert len(guards) == 1
    assert guards[0].kwargs["max_bytes"] > get_settings().attachment_max_bytes


def test_configure_logging_is_idempotent() -> None:
    settings = get_settings()
    configure_logging(settings)
    configure_logging(settings)
    root = logging.getLogger()
    json_handlers = [h for h in root.handlers if isinstance(h.formatter, JsonLineFormatter)]
    assert len(json_handlers) == 1


def test_json_formatter_emits_parseable_lines() -> None:
    formatter = JsonLineFormatter()
    record = logging.LogRecord("app.test", logging.ERROR, __file__, 1, "kaboom: %s", ("why",), None)
    try:
        raise ValueError("inner cause")
    except ValueError:
        record.exc_info = sys.exc_info()
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "ERROR"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "kaboom: why"
    assert "inner cause" in payload["exc_info"]
    assert "timestamp" in payload
