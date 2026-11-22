"""Per-client rate limiting: budgets, keying, exemptions, and failing open.

Most of these drive the raw ASGI middleware directly. That is not a shortcut -
the middleware IS raw ASGI precisely so a streaming response passes through
message by message, and driving it at that level is the only way to see that
happening.
"""

from __future__ import annotations

import json
import logging
import re

import pytest
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.main as main_module
from app.api.deps import get_model_provider
from app.core.concurrency import StreamLimiter
from app.core.config import get_settings
from app.core.middleware import RequestIdMiddleware
from app.core.ratelimit import (
    _HEAVY_EXACT,
    _HEAVY_SUFFIXES,
    _LIGHT_ROUTES,
    WINDOW_SECONDS,
    RateCounter,
    RateLimitMiddleware,
    is_trusted,
    limit_key,
    route_class,
    strip_port,
)
from app.main import create_app
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.storage.local import FakeStorageBackend
from app.workers.queue import ArqJobQueue, FakeJobQueue
from tests.conftest import make_memory_store


class Counter:
    """An in-memory RateCounter that can be told to fail."""

    def __init__(self, *, fail: bool = False) -> None:
        self.counts: dict[str, int] = {}
        self.fail = fail
        self.calls: list[str] = []

    async def incr_window(self, key: str, ttl_seconds: int) -> int:
        self.calls.append(key)
        if self.fail:
            raise ConnectionError("redis is gone")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]


class _App:
    """The inner app: records what reached it and answers 200."""

    def __init__(self) -> None:
        self.scopes: list[dict] = []

    async def __call__(self, scope, receive, send) -> None:
        self.scopes.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


class _State:
    def __init__(self, counter) -> None:
        self.rate_counter = counter


class _Host:
    """Stands in for the FastAPI app that ASGI puts in ``scope["app"]``."""

    def __init__(self, counter) -> None:
        self.state = _State(counter)


def _scope(
    method: str = "GET",
    path: str = "/api/v1/models",
    *,
    counter=None,
    client: tuple[str, int] | None = ("203.0.113.9", 5555),
    headers: list[tuple[bytes, bytes]] | None = None,
) -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
        "client": client,
        "app": _Host(counter),
    }


async def _drive(middleware: RateLimitMiddleware, scope: dict) -> list[dict]:
    """Run one request through the middleware and return the messages it sent."""
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, receive, send)
    return sent


def _limiter(inner, *, per_minute: int = 3, heavy: int = 2, light: int = 50, trusted=frozenset()):
    return RateLimitMiddleware(
        inner,
        enabled=True,
        per_minute=per_minute,
        heavy_per_minute=heavy,
        light_per_minute=light,
        trusted_proxies=trusted,
    )


# --- route classification -------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("GET", "/healthz", None),
        ("GET", "/readyz", None),
        ("GET", "/docs", None),
        ("GET", "/openapi.json", None),
        ("GET", "/", None),
        ("GET", "/api/v1/models", "general"),
        ("POST", "/api/v1/chat/stream", "general"),
        # The status poll: its own class. Deleting the same resource is not.
        ("GET", "/api/v1/attachments/abc", "light"),
        ("DELETE", "/api/v1/attachments/abc", "general"),
        # Nothing deeper than one segment falls into the light class.
        ("GET", "/api/v1/attachments/abc/extra", "general"),
        ("GET", "/api/v1/integrations/oauth/redirect-uri", "general"),
        ("POST", "/api/v1/models/pull", "heavy"),
        ("POST", "/api/v1/rag/ingest", "heavy"),
        ("POST", "/api/v1/attachments", "heavy"),
        ("POST", "/api/v1/attachments/", "heavy"),
        ("GET", "/api/v1/integrations/oauth/callback", "heavy"),
        ("POST", "/api/v1/integrations/servers/test-connection", "heavy"),
        ("POST", "/api/v1/integrations/servers/an-id/test-connection", "heavy"),
        ("POST", "/api/v1/integrations/servers/an-id/oauth/connect", "heavy"),
        ("POST", "/api/v1/integrations/servers/an-id/oauth/disconnect", "heavy"),
        ("POST", "/api/v1/models/verify", "heavy"),
        ("POST", "/api/v1/models/embedder-check", "heavy"),
        # Adding to the knowledge base parses a file and embeds every passage of
        # it before answering - the same cost as the ingest line above, reached
        # with a file instead of JSON.
        ("POST", "/api/v1/knowledge/documents", "heavy"),
        # Reading and curating that list are ordinary requests.
        ("GET", "/api/v1/knowledge/documents", "general"),
        ("DELETE", "/api/v1/knowledge/documents", "general"),
        # The download buffers a whole object (up to attachment_max_bytes)
        # before answering, so it cannot sit in the general class.
        ("GET", "/api/v1/attachments/abc/content", "heavy"),
        # ... but asking ABOUT the same file reads no object at all, so the
        # cheap method stays ordinary traffic. This is what keeps a real
        # download to one heavy request: a client checks, then transfers.
        ("HEAD", "/api/v1/attachments/abc/content", "general"),
        # The tail is what makes it heavy, not the prefix: reading, listing and
        # deleting the same attachment are ordinary requests.
        ("GET", "/api/v1/attachments/abc/contents", "general"),
        ("POST", "/api/v1/attachments/abc/content", "general"),
        # Neighbours that stay general, so "heavy" is not creeping.
        ("POST", "/api/v1/models/preflight", "general"),
        ("POST", "/api/v1/models/warm", "general"),
        ("POST", "/api/v1/rag/query", "general"),
        # The method matters: a GET of the pull endpoint is not the download.
        ("GET", "/api/v1/models/pull", "general"),
    ],
)
def test_route_class_map(method: str, path: str, expected) -> None:
    assert route_class(method, path) == expected


# --- budgets --------------------------------------------------------------


async def test_requests_under_the_budget_pass_through() -> None:
    inner, counter = _App(), Counter()
    middleware = _limiter(inner, per_minute=3)

    for _ in range(3):
        sent = await _drive(middleware, _scope(counter=counter))
        assert sent[0]["status"] == 200

    assert len(inner.scopes) == 3


async def test_the_request_past_the_budget_is_refused() -> None:
    inner, counter = _App(), Counter()
    middleware = _limiter(inner, per_minute=2)

    await _drive(middleware, _scope(counter=counter))
    await _drive(middleware, _scope(counter=counter))
    sent = await _drive(middleware, _scope(counter=counter))

    # The inner app never ran for the refused request.
    assert len(inner.scopes) == 2
    assert sent[0]["status"] == 429
    headers = dict(sent[0]["headers"])
    assert headers[b"content-type"] == b"application/problem+json"
    retry_after = int(headers[b"retry-after"])
    assert 1 <= retry_after <= WINDOW_SECONDS
    body = json.loads(sent[1]["body"])
    assert body["code"] == "too_many_requests"
    assert body["status"] == 429
    assert body["title"] == "Too Many Requests"
    assert body["type"] == "about:blank"
    assert "2 per minute" in body["detail"]


async def test_heavy_routes_have_their_own_smaller_budget() -> None:
    inner, counter = _App(), Counter()
    middleware = _limiter(inner, per_minute=10, heavy=1)

    first = await _drive(middleware, _scope("POST", "/api/v1/rag/ingest", counter=counter))
    second = await _drive(middleware, _scope("POST", "/api/v1/rag/ingest", counter=counter))
    # The general budget is untouched by the heavy one: separate keys.
    general = await _drive(middleware, _scope(counter=counter))

    assert first[0]["status"] == 200
    assert second[0]["status"] == 429
    assert general[0]["status"] == 200


async def test_checking_a_file_does_not_spend_the_download_budget() -> None:
    # The arithmetic the split exists for. With the check in the heavy class a
    # client gets HALF the downloads its budget claims, because every saved file
    # costs two heavy requests; here the checks run out of the general budget
    # and every heavy request is a file actually transferred.
    inner, counter = _App(), Counter()
    middleware = _limiter(inner, per_minute=10, heavy=2)
    path = "/api/v1/attachments/abc/content"

    for _ in range(4):
        check = await _drive(middleware, _scope("HEAD", path, counter=counter))
        assert check[0]["status"] == 200

    assert (await _drive(middleware, _scope("GET", path, counter=counter)))[0]["status"] == 200
    assert (await _drive(middleware, _scope("GET", path, counter=counter)))[0]["status"] == 200
    # ... and the cheap method is still a budget, not an exemption: it just
    # spends the general one.
    assert (await _drive(middleware, _scope("GET", path, counter=counter)))[0]["status"] == 429


async def test_the_polled_route_has_its_own_larger_budget() -> None:
    # The case this class exists for: a client polling several parsing files
    # must not be refused while a heavier route stays strictly bounded.
    inner, counter = _App(), Counter()
    middleware = _limiter(inner, per_minute=2, light=5)
    poll = _scope("GET", "/api/v1/attachments/abc", counter=counter)

    for _ in range(5):
        assert (await _drive(middleware, dict(poll)))[0]["status"] == 200
    # Its own budget, so the general one is untouched by all that polling.
    assert (await _drive(middleware, _scope(counter=counter)))[0]["status"] == 200
    # ... and it is a budget, not an exemption.
    assert (await _drive(middleware, dict(poll)))[0]["status"] == 429


async def test_the_window_rolls(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.core.ratelimit as ratelimit_module

    now = 1_000_000.0
    monkeypatch.setattr(ratelimit_module.time, "time", lambda: now)

    inner, counter = _App(), Counter()
    middleware = _limiter(inner, per_minute=1)

    assert (await _drive(middleware, _scope(counter=counter)))[0]["status"] == 200
    assert (await _drive(middleware, _scope(counter=counter)))[0]["status"] == 429

    now += WINDOW_SECONDS
    assert (await _drive(middleware, _scope(counter=counter)))[0]["status"] == 200


# --- keying ---------------------------------------------------------------


async def test_each_client_address_gets_its_own_budget() -> None:
    inner, counter = _App(), Counter()
    middleware = _limiter(inner, per_minute=1)

    await _drive(middleware, _scope(counter=counter, client=("10.0.0.1", 1)))
    other = await _drive(middleware, _scope(counter=counter, client=("10.0.0.2", 1)))
    same = await _drive(middleware, _scope(counter=counter, client=("10.0.0.1", 2)))

    assert other[0]["status"] == 200
    assert same[0]["status"] == 429


def test_a_forwarded_header_from_an_untrusted_peer_is_ignored() -> None:
    # The spoofing case: without this, one caller sends a different
    # X-Forwarded-For per request and never spends the same budget twice.
    scope = _scope(headers=[(b"x-forwarded-for", b"198.51.100.7")], client=("203.0.113.9", 1))

    assert limit_key(scope, frozenset()) == "203.0.113.9"


def test_a_trusted_proxy_supplies_the_client_address() -> None:
    # The canonical proxied deployment: the peer IS the proxy, and the one
    # header entry is the address it appended for the real client.
    scope = _scope(
        headers=[(b"x-forwarded-for", b"198.51.100.7")],
        client=("203.0.113.9", 1),
    )

    assert limit_key(scope, frozenset({"203.0.113.9"})) == "198.51.100.7"


def test_only_the_rightmost_untrusted_hop_is_believed() -> None:
    # The header reads client, proxy1, proxy2. Only the two proxies are ours,
    # so the leftmost entry is client-supplied text and the first hop nobody
    # vouches for is what gets counted.
    scope = _scope(
        headers=[(b"x-forwarded-for", b"1.1.1.1, 198.51.100.7, 10.0.0.1")],
        client=("10.0.0.2", 1),
    )
    trusted = frozenset({"10.0.0.2", "10.0.0.1"})

    assert limit_key(scope, trusted) == "198.51.100.7"


def test_a_trusted_peer_with_no_forwarded_header_falls_back_to_itself() -> None:
    scope = _scope(client=("10.0.0.2", 1))

    assert limit_key(scope, frozenset({"10.0.0.2"})) == "10.0.0.2"


def test_a_connection_with_no_peer_address_still_gets_a_key() -> None:
    scope = _scope(client=None)

    assert limit_key(scope, frozenset()) == "unknown"


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("203.0.113.5:44321", "203.0.113.5"),
        ("203.0.113.5", "203.0.113.5"),
        # Bracketed IPv6 is the only form that carries a port.
        ("[2001:db8::1]:8080", "2001:db8::1"),
        ("[::1]", "::1"),
        # Bare IPv6 is all colons and has NO port - splitting would destroy it.
        ("::1", "::1"),
        ("2001:db8::1", "2001:db8::1"),
        ("unknown", "unknown"),
    ],
)
def test_a_forwarded_entry_is_reduced_to_a_bare_address(entry: str, expected: str) -> None:
    assert strip_port(entry) == expected


def test_a_port_in_the_forwarded_chain_does_not_split_one_client_into_many() -> None:
    # Left alone, the port lands in the key and every connection gets a fresh
    # budget: the limiter counts forever and never refuses anything.
    trusted = frozenset({"10.0.0.2"})
    first = _scope(headers=[(b"x-forwarded-for", b"203.0.113.5:1111")], client=("10.0.0.2", 1))
    second = _scope(headers=[(b"x-forwarded-for", b"203.0.113.5:2222")], client=("10.0.0.2", 1))

    assert limit_key(first, trusted) == limit_key(second, trusted) == "203.0.113.5"


def test_a_bracketed_ipv6_client_behind_a_trusted_proxy_keys_on_its_address() -> None:
    scope = _scope(headers=[(b"x-forwarded-for", b"[2001:db8::1]:9000")], client=("10.0.0.2", 1))

    assert limit_key(scope, frozenset({"10.0.0.2"})) == "2001:db8::1"


def test_a_trusted_proxy_may_be_named_by_cidr() -> None:
    # The setting has to be fillable: a container's address is handed out by
    # the bridge at boot, so an exact address is not knowable in advance.
    scope = _scope(headers=[(b"x-forwarded-for", b"198.51.100.7")], client=("172.18.0.4", 1))

    assert limit_key(scope, frozenset({"172.16.0.0/12"})) == "198.51.100.7"


@pytest.mark.parametrize(
    ("address", "entries", "expected"),
    [
        ("10.0.0.5", {"10.0.0.5"}, True),
        ("10.0.0.5", {"10.0.0.0/24"}, True),
        ("10.0.1.5", {"10.0.0.0/24"}, False),
        # Host bits set: read as the network, not rejected.
        ("10.0.0.9", {"10.0.0.5/24"}, True),
        ("::1", {"::/0"}, True),
        # A typo must fail CLOSED - "trust nobody", never "trust everybody".
        ("10.0.0.5", {"not-an-address"}, False),
        ("not-an-address", {"10.0.0.0/24"}, False),
        ("10.0.0.5", set(), False),
    ],
)
def test_trusted_proxy_matching(address: str, entries: set[str], expected: bool) -> None:
    assert is_trusted(address, frozenset(entries)) is expected


# --- exemptions and lapses ------------------------------------------------


# NOTE on what actually guards the exemption: this next test and the /healthz
# assertion in the end-to-end test below BOTH still pass with _EXEMPT_PATHS
# emptied, because the probes already sit outside the limited prefix. They
# describe the behaviour; they do not pin the constant. The only test that
# holds _EXEMPT_PATHS to account is the prefix-widening one right after it.
@pytest.mark.parametrize("path", ["/healthz", "/readyz"])
async def test_probe_paths_are_never_counted(path: str) -> None:
    inner, counter = _App(), Counter()
    middleware = _limiter(inner, per_minute=1)

    for _ in range(5):
        sent = await _drive(middleware, _scope("GET", path, counter=counter))
        assert sent[0]["status"] == 200

    # Not merely allowed - not counted at all, so the counter was never dialled.
    assert counter.calls == []
    assert len(inner.scopes) == 5


def test_probes_stay_exempt_even_if_the_limited_prefix_widens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Today the probes are also outside /api/v1, so the prefix rule alone would
    # spare them and the explicit exemption looks like decoration. It is not:
    # the day anything widens what gets limited, a limiter that could answer
    # 429 to a readiness probe would take the deployment down BY ITSELF. This
    # widens the prefix on purpose to hold the exemption to account.
    import app.core.ratelimit as ratelimit_module

    monkeypatch.setattr(ratelimit_module, "_LIMITED_PREFIX", "/")

    assert route_class("GET", "/healthz") is None
    assert route_class("GET", "/readyz") is None
    assert route_class("GET", "/something-else") == "general"


async def test_an_unreachable_counter_fails_open(caplog: pytest.LogCaptureFixture) -> None:
    inner, counter = _App(), Counter(fail=True)
    middleware = _limiter(inner, per_minute=1)

    with caplog.at_level(logging.WARNING, logger="app.core.ratelimit"):
        for _ in range(4):
            sent = await _drive(middleware, _scope(counter=counter))
            assert sent[0]["status"] == 200

    assert len(inner.scopes) == 4
    warnings = [r for r in caplog.records if r.name == "app.core.ratelimit"]
    # Reported, but throttled: an outage affects every request and must not
    # bury its own explanation.
    assert len(warnings) == 1
    assert "not being enforced" in warnings[0].message


async def test_the_queue_on_app_state_is_used_when_no_counter_was_named() -> None:
    # Only the lifespan sets `rate_counter`, but an app assembled without one
    # still has the Redis-backed queue - and limiting nothing while looking
    # installed is the quietest possible failure. So the queue is the fallback.
    inner = _App()
    middleware = _limiter(inner, per_minute=1)
    host = _Host(None)
    host.state.rate_counter = None
    host.state.job_queue = FakeJobQueue()
    scope = {**_scope(), "app": host}

    assert (await _drive(middleware, dict(scope)))[0]["status"] == 200
    assert (await _drive(middleware, dict(scope)))[0]["status"] == 429


async def test_a_queue_that_cannot_count_reads_as_no_counter() -> None:
    # A JobQueue is not obliged to be a RateCounter. One that is not must fail
    # open like any missing counter, not raise on every single request.
    class _Enqueuer:
        async def enqueue(self, *_a, **_k):
            return "x"

    inner = _App()
    middleware = _limiter(inner, per_minute=1)
    host = _Host(None)
    host.state.rate_counter = None
    host.state.job_queue = _Enqueuer()
    scope = {**_scope(), "app": host}

    for _ in range(3):
        assert (await _drive(middleware, dict(scope)))[0]["status"] == 200


async def test_a_missing_counter_fails_open(caplog: pytest.LogCaptureFixture) -> None:
    inner = _App()
    middleware = _limiter(inner, per_minute=1)

    with caplog.at_level(logging.WARNING, logger="app.core.ratelimit"):
        for _ in range(3):
            sent = await _drive(middleware, _scope(counter=None))
            assert sent[0]["status"] == 200

    assert len(inner.scopes) == 3
    assert any("no counter is wired" in r.message for r in caplog.records)


async def test_disabled_does_no_counting_at_all() -> None:
    inner, counter = _App(), Counter()
    middleware = RateLimitMiddleware(
        inner,
        enabled=False,
        per_minute=1,
        heavy_per_minute=1,
        light_per_minute=1,
        trusted_proxies=frozenset(),
    )

    for _ in range(5):
        sent = await _drive(middleware, _scope(counter=counter))
        assert sent[0]["status"] == 200

    assert counter.calls == []


async def test_non_http_scopes_pass_straight_through() -> None:
    inner, counter = _App(), Counter()
    middleware = _limiter(inner)

    await _drive(middleware, {"type": "lifespan", "app": _Host(counter)})

    assert counter.calls == []
    assert len(inner.scopes) == 1


# --- streaming ------------------------------------------------------------


async def test_the_limiter_never_wraps_send() -> None:
    # The strongest form of "streaming is untouched", and the one that survives
    # a refactor: assert the inner app is handed the VERY SAME send callable,
    # so there is no wrapper at all to buffer, delay, or rewrite a frame. A
    # merely pass-through wrapper (or a BaseHTTPMiddleware rewrite) still
    # yields the right number of chunks, so counting chunks cannot see it.
    seen: list = []

    async def inner(scope, receive, send) -> None:
        seen.append(send)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = _limiter(inner)
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(_scope("POST", "/api/v1/chat/stream", counter=Counter()), receive, send)

    assert seen[0] is send


async def test_a_streaming_response_is_not_buffered() -> None:
    class _Streamer:
        async def __call__(self, scope, receive, send) -> None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            for index in range(3):
                await send(
                    {
                        "type": "http.response.body",
                        "body": f"event: token\ndata: {index}\n\n".encode(),
                        "more_body": True,
                    }
                )
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    counter = Counter()
    middleware = _limiter(_Streamer())
    sent = await _drive(middleware, _scope("POST", "/api/v1/chat/stream", counter=counter))

    # Each chunk arrived as its OWN message: the limiter never wrapped send, so
    # nothing coalesced the stream into one body.
    bodies = [m for m in sent if m["type"] == "http.response.body"]
    assert len(bodies) == 4
    assert bodies[0]["body"] == b"event: token\ndata: 0\n\n"
    assert bodies[0]["more_body"] is True
    assert bodies[-1]["more_body"] is False


# --- wiring ---------------------------------------------------------------


def test_the_limiter_is_installed_inside_cors_and_the_request_id() -> None:
    # Order matters twice over: a 429 must carry CORS headers (or a browser
    # sees an opaque failure) and must get an id and an access line. BOTH
    # relations are asserted - checking only the request-id one leaves the CORS
    # half of the name unguarded, and swapping CORS below the limiter is a
    # mutation that keeps a one-sided assertion green.
    app = create_app()
    classes = [m.cls for m in app.user_middleware]

    assert RateLimitMiddleware in classes
    # user_middleware lists outermost first.
    assert classes.index(RequestIdMiddleware) < classes.index(RateLimitMiddleware)
    assert classes.index(CORSMiddleware) < classes.index(RateLimitMiddleware)
    limiter = next(m for m in app.user_middleware if m.cls is RateLimitMiddleware)
    settings = get_settings()
    assert limiter.kwargs["per_minute"] == settings.rate_limit_per_minute
    assert limiter.kwargs["heavy_per_minute"] == settings.rate_limit_heavy_per_minute


def _served_routes() -> set[tuple[str, str]]:
    """Every (METHOD, path) the app actually serves, read from its own schema.

    The OpenAPI document rather than ``app.routes``: this FastAPI version keeps
    included routers nested rather than flattened onto the app, and the schema
    is the same authoritative surface the generated API types are built from.
    """
    paths = create_app().openapi()["paths"]
    return {(method.upper(), path) for path, ops in paths.items() for method in ops}


def test_every_heavy_route_in_the_map_is_a_route_the_app_actually_serves() -> None:
    # The map is a table of literal strings, and test_route_class_map checks it
    # against more literal strings - i.e. against itself. Without this, renaming
    # or moving a route silently demotes it to the general budget with every
    # other test still green.
    missing = sorted(pair for pair in _HEAVY_EXACT if pair not in _served_routes())

    assert not missing, (
        f"these entries name no registered route: {missing}. A moved or renamed "
        "route silently falls back to the general budget."
    )


def test_every_light_route_in_the_map_is_a_route_the_app_actually_serves() -> None:
    # Same guard as the heavy map: the light class is a pattern plus the
    # OpenAPI path it is meant to cover, so a moved poll endpoint is caught
    # here rather than silently reverting to the general budget.
    served = _served_routes()
    missing = [(verb, doc) for verb, doc, _p in _LIGHT_ROUTES if (verb, doc) not in served]

    assert not missing, f"these entries name no registered route: {missing}"


def test_every_light_pattern_matches_its_own_documented_path() -> None:
    # The pattern and the documented path must describe the same route; a
    # drifting pair would pass the existence check above while matching nothing.
    for verb, doc, pattern in _LIGHT_ROUTES:
        concrete = re.sub(r"\{[^}]+\}", "a-real-id", doc)
        assert pattern.match(concrete), f"{verb} {doc} does not match its own pattern"


def test_every_heavy_suffix_matches_at_least_one_served_route() -> None:
    served = _served_routes()
    unmatched = [
        (verb, tail)
        for verb, tail in _HEAVY_SUFFIXES
        if not any(method == verb and path.endswith(tail) for method, path in served)
    ]

    assert not unmatched, f"these suffix rules match no route at all: {unmatched}"


async def test_the_lifespan_wires_a_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without this the limiter would fail open forever and say so only in a
    # throttled log line - the quietest possible way to ship no limiting.
    async def _no_seed(_sessionmaker) -> None:
        return None

    monkeypatch.setattr(main_module, "seed_defaults", _no_seed)
    app = create_app()
    async with main_module._lifespan(app):
        counter = app.state.rate_counter

    assert counter is not None
    assert isinstance(counter, RateCounter)


@pytest.mark.parametrize("queue_cls", [ArqJobQueue, FakeJobQueue])
def test_the_queue_implementations_satisfy_the_counter_protocol(queue_cls) -> None:
    # The lifespan hands the queue to the limiter as a counter; if an
    # implementation ever stops offering incr_window, that hand-off degrades
    # into a permanent silent fail-open.
    assert issubclass(queue_cls, RateCounter)


async def test_the_arq_queue_counts_in_one_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.workers.queue as queue_module

    class _Pipeline:
        def __init__(self) -> None:
            self.commands: list[tuple] = []

        def incr(self, key):
            self.commands.append(("incr", key))

        def expire(self, key, ttl):
            self.commands.append(("expire", key, ttl))

        async def execute(self):
            return [7, True]

    pipeline = _Pipeline()

    class _Pool:
        def pipeline(self, transaction=True):
            assert transaction is False  # a counter needs no MULTI/EXEC
            return pipeline

    async def _create(*_args, **_kwargs):
        return _Pool()

    monkeypatch.setattr(queue_module, "create_pool", _create)
    count = await ArqJobQueue("redis://localhost:6379/0").incr_window("k", 60)

    assert count == 7
    assert pipeline.commands == [("incr", "k"), ("expire", "k", 60)]


def test_the_pipeline_api_the_counter_relies_on_is_still_synchronous() -> None:
    # incr_window queues two commands WITHOUT awaiting them, which is correct
    # only while a buffered pipeline's execute_command returns the pipeline
    # rather than a coroutine. The hand-rolled fake pool above cannot see that
    # change: if redis-py ever made this async, `pipeline.incr(key)` would
    # return an un-awaited coroutine, queue nothing, and every caller would
    # count as zero forever - limiting silently switched off, with the test
    # above still green. So the dependency's contract is pinned here directly.
    import inspect

    from redis.asyncio.client import Pipeline

    assert not inspect.iscoroutinefunction(Pipeline.execute_command)


# --- end to end -----------------------------------------------------------


@pytest.fixture
def tight_limits(monkeypatch: pytest.MonkeyPatch):
    """Rebuild the settings with a budget a test can actually reach."""
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def relayed(monkeypatch: pytest.MonkeyPatch):
    """Tight budget plus a trusted relay, configured the way a deployment does.

    Set through the environment rather than poked onto the middleware, so the
    CSV parsing and the create_app wiring are part of what is under test.
    """
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "10.9.9.9")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _wired_app(db_sessionmaker) -> object:
    """A real app with every collaborator in-process and the limiter counting."""
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider()
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.storage = FakeStorageBackend()
    queue = FakeJobQueue()
    app.state.job_queue = queue
    app.state.rate_counter = queue
    return app


async def test_two_relayed_callers_do_not_share_one_budget(
    relayed,
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """The whole point of the forwarded chain, asserted end to end.

    Every browser request reaches the backend from the frontend server, so
    without the relayed address two different people would spend the same
    budget and the second would be refused for the first one's traffic.
    """
    app = _wired_app(db_sessionmaker)
    # Both requests arrive FROM THE RELAY; only the forwarded address differs.
    relay = ASGITransport(app=app, client=("10.9.9.9", 1))
    async with AsyncClient(transport=relay, base_url="http://test") as client:
        alice = [
            (
                await client.get("/api/v1/models", headers={"X-Forwarded-For": "198.51.100.1"})
            ).status_code
            for _ in range(3)
        ]
        bob = await client.get("/api/v1/models", headers={"X-Forwarded-For": "198.51.100.2"})

    # Alice spends her own budget (2) and is refused on her third.
    assert alice == [200, 200, 429]
    # Bob is a different caller and is unaffected by Alice hitting her limit.
    assert bob.status_code == 200


async def test_relayed_callers_share_one_budget_without_the_forwarded_address(
    tight_limits,
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The bug the forwarding fixes, pinned so it cannot come back quietly: with
    # no header relayed, two different people are one client to the limiter.
    app = _wired_app(db_sessionmaker)
    relay = ASGITransport(app=app, client=("10.9.9.9", 1))
    async with AsyncClient(transport=relay, base_url="http://test") as client:
        statuses = [(await client.get("/api/v1/models")).status_code for _ in range(3)]

    assert statuses == [200, 200, 429]


async def test_an_attacker_supplied_forwarded_entry_is_inert(
    relayed,
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """A caller cannot mint a fresh identity by seeding the chain themselves.

    The relay APPENDS the address it saw, so anything the caller sent sits to
    its left and the rightmost-untrusted rule never selects it.
    """
    app = _wired_app(db_sessionmaker)
    relay = ASGITransport(app=app, client=("10.9.9.9", 1))
    async with AsyncClient(transport=relay, base_url="http://test") as client:
        # One caller (198.51.100.9) forging a different left-hand entry each
        # time. All four requests must land on the SAME budget.
        statuses = [
            (
                await client.get(
                    "/api/v1/models",
                    headers={"X-Forwarded-For": f"10.0.0.{n}, 198.51.100.9"},
                )
            ).status_code
            for n in range(4)
        ]

    assert statuses == [200, 200, 429, 429]


async def test_the_real_app_refuses_over_the_limit_with_a_correlated_problem(
    tight_limits,
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider()
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.storage = FakeStorageBackend()
    queue = FakeJobQueue()
    app.state.job_queue = queue
    app.state.rate_counter = queue

    origin = {"Origin": "http://localhost:3000"}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        statuses = [
            (await client.get("/api/v1/models", headers=origin)).status_code for _ in range(2)
        ]
        refused = await client.get(
            "/api/v1/models", headers={**origin, "X-Request-ID": "limited-1"}
        )
        # Exempt even while the caller is over budget.
        probe = await client.get("/healthz")

    assert statuses == [200, 200]
    assert refused.status_code == 429
    assert refused.headers["content-type"].startswith("application/problem+json")
    assert int(refused.headers["retry-after"]) >= 1
    # The refusal went out through the request-id middleware, so it is
    # traceable like any other response.
    assert refused.headers["x-request-id"] == "limited-1"
    assert refused.json()["request_id"] == "limited-1"
    # ... and out through CORS, so a browser sees a real 429 rather than an
    # opaque network error it cannot report.
    assert refused.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert probe.status_code == 200
