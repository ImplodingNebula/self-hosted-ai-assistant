"""Per-client request-rate limiting, as ASGI middleware.

Abuse protection, not a quota. The defaults sit far above anything a person's
session produces - a chat turn is one request that then streams for many
seconds - so the limiter is invisible in normal use and only bites a script
hammering the port. That framing matters: a limit tight enough to be felt by a
real user would be a bug, not a stricter policy.

It lives in the APP rather than in the proxy in front of it on purpose. The app
is the only layer that knows which routes are expensive (a model download costs
gigabytes, an ingest costs an embedding pass, an OAuth callback costs an
outbound call to a third party) and it is the layer that survives a deployment
without a proxy - a local `docker compose up` has no Caddy in front of it.

Written against the raw ASGI protocol, like the request-id and body-size
middleware beside it, and for the same reason: streaming responses must pass
through one message at a time. The limiter decides once, at request start, and
then gets out of the way - it never wraps ``send``, so an SSE chat turn streams
exactly as it did before.

Two deliberate weaknesses, both chosen:

Window edges
    Counters are FIXED windows, not sliding ones. A caller can therefore spend
    a full budget at the end of one minute and another at the start of the
    next - a 2x burst across the boundary. Sliding windows cost either a sorted
    set per client or a second round trip, and the point here is to stop
    sustained hammering, which a fixed window does at one INCR.

Fail open
    A counter that cannot be reached lets the request through. The store is
    shared infrastructure that can blink; refusing traffic because the counter
    is down would turn a bookkeeping outage into a site outage. Every such
    lapse is logged (throttled, so an outage cannot flood the log).
"""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import re
import time
from typing import Literal, Protocol, runtime_checkable

from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.errors import TooManyRequestsError
from app.core.logging import request_id_var

logger = logging.getLogger(__name__)

RouteClass = Literal["light", "general", "heavy"]

WINDOW_SECONDS = 60

# How often the "not enforcing" warning may repeat. A counter outage affects
# every request, so an unthrottled warning would bury the reason for it.
_WARNING_INTERVAL_SECONDS = 60.0

# Only the versioned API is limited. Everything else the app serves is either a
# probe (below) or documentation, and neither is worth a Redis round trip.
_LIMITED_PREFIX = "/api/v1/"

# Probes, exempted explicitly rather than by falling outside the prefix above.
# They are polled every few seconds by compose healthchecks and by whatever
# orchestrates the deployment, and a limiter that could answer 429 to a
# readiness probe would take the deployment down by ITSELF.
_EXEMPT_PATHS = frozenset({"/healthz", "/readyz"})

# THE map of expensive routes. One call to any of these costs real resources -
# a model download, an embedding pass over a document, an upload's bytes, or an
# outbound request to a third party made on the caller's behalf - so they get
# their own much smaller budget. Everything else under /api/v1 is "general".
#
# POST /chat/stream is deliberately NOT here. It is limited as general traffic,
# which caps how often a stream may be STARTED; how many may run AT ONCE is a
# different axis, governed by StreamLimiter (app.core.concurrency).
_HEAVY_EXACT: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/v1/models/pull"),
        ("POST", "/api/v1/rag/ingest"),
        # The same embedding pass as the line above, reached with a file instead
        # of JSON: it parses the upload and embeds every passage of it before it
        # answers. Listing and deleting knowledge are ordinary requests.
        ("POST", "/api/v1/knowledge/documents"),
        # The upload itself. Reading, listing and deleting an attachment are
        # ordinary requests and stay general.
        ("POST", "/api/v1/attachments"),
        # Unauthenticated by necessity - the provider's browser redirect lands
        # here - which makes it the most exposed route in the app.
        ("GET", "/api/v1/integrations/oauth/callback"),
        # The two probe endpoints. Each unloads the resident model, loads the
        # one under test, and re-warms afterwards, so a handful of calls can
        # spend minutes of GPU time and evict whatever was serving chat.
        ("POST", "/api/v1/models/verify"),
        ("POST", "/api/v1/models/embedder-check"),
    }
)

# Routes a healthy client polls, where the general budget is the wrong shape.
# The composer polls EVERY parsing attachment concurrently every 1.2 seconds,
# so N files in flight is N*50 requests a minute from one entirely legitimate
# user - five files would trip a 240 budget on their own. Raising the general
# limit to cover that would raise it for everything; a separate, generous
# budget for the one polled route keeps the general ceiling meaningful.
#
# The real fix is a batched poll (one list call instead of N), which is a
# frontend change and not part of this slice. Revisit this class when it lands.
# Each entry pairs the OpenAPI path (so a test can prove the route still
# exists) with the pattern the concrete request path is matched against.
_LIGHT_ROUTES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "GET",
        "/api/v1/attachments/{attachment_id}",
        re.compile(r"^/api/v1/attachments/[^/]+$"),
    ),
)

# Routes that carry a resource id mid-path, so they match on their tail rather
# than exactly. The integration actions are here because each reaches out to a
# third party - somebody else's rate limit being spent - and the attachment
# download because of memory.
#
# The download reads a whole object into the process before answering (uploads
# are capped at attachment_max_bytes, 25 MB), so the general budget of 240/min
# would authorise ~6 GB of concurrent buffering a minute on a box with 8 GB to
# its name. That is not a cost model the general class was written for. It is
# NOT here because the bytes are expensive to produce - they are already sitting
# in storage - which is why it is worth revisiting the moment the read stops
# being buffered.
#
# Every entry names its METHOD, and on /content that is deliberate: HEAD on the
# same path answers from the row and one existence check WITHOUT reading the
# object, so it costs what a general request costs and is classified as one.
# Matching the suffix regardless of method would ration a request that spends
# none of the memory this class exists to bound.
#
# The web client does not currently use that method - it fetches the file and
# reads the outcome of that single request, so a refusal cannot be saved to disk
# as the file - which makes RATE_LIMIT_HEAVY_PER_MINUTE the ceiling on files
# saved per minute per client, one request to one file, with nothing else on
# that path spending it.
_HEAVY_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("POST", "/test-connection"),
    ("POST", "/oauth/connect"),
    ("POST", "/oauth/disconnect"),
    ("GET", "/content"),
)


@runtime_checkable
class RateCounter(Protocol):
    """A shared, expiring counter - the only thing the limiter needs to persist.

    Deliberately not a whole seam of its own. The process already holds one
    Redis connection pool for the job queue, and the queue implementations
    satisfy this protocol, so the limiter borrows that pool rather than opening
    a second one. Nothing here is Redis-shaped: "increment a named counter that
    expires" is a capability any backing store can offer.
    """

    async def incr_window(self, key: str, ttl_seconds: int) -> int:
        """Increment ``key``, arm its expiry, and return the new count."""
        ...


def route_class(method: str, path: str) -> RouteClass | None:
    """Classify one request for limiting purposes.

    Args:
        method: The HTTP method, uppercase.
        path: The request path.

    Returns:
        The route's class, or None when the route is exempt and must not be
        counted at all.
    """
    normalized = path.rstrip("/") or "/"
    if normalized in _EXEMPT_PATHS or not normalized.startswith(_LIMITED_PREFIX):
        return None
    if (method, normalized) in _HEAVY_EXACT:
        return "heavy"
    if any(method == verb and normalized.endswith(tail) for verb, tail in _HEAVY_SUFFIXES):
        return "heavy"
    if any(method == verb and pattern.match(normalized) for verb, _doc, pattern in _LIGHT_ROUTES):
        return "light"
    return "general"


def _counter_from(scope: Scope) -> RateCounter | None:
    """Find the counter this app was wired with, or None if it has none.

    Two names, because two things populate them and only one is guaranteed:
    the lifespan sets ``rate_counter`` explicitly, while ``job_queue`` is the
    Redis-backed seam that a request-scoped fallback may have built instead.
    Reading only the first would leave any app assembled without a lifespan
    silently unlimited - the quietest possible way to ship no limiting, and the
    bug class this repo has been bitten by more than once.
    """
    state = getattr(scope.get("app"), "state", None)
    if state is None:
        return None
    counter = getattr(state, "rate_counter", None)
    if counter is not None:
        return counter
    fallback = getattr(state, "job_queue", None)
    # A queue implementation is not obliged to count; one that cannot must read
    # as "no counter" rather than raise on every request.
    return fallback if isinstance(fallback, RateCounter) else None


def _peer_address(scope: Scope) -> str:
    """Return the address the connection actually came from."""
    client = scope.get("client")
    if not client:
        return "unknown"
    return str(client[0])


def strip_port(entry: str) -> str:
    """Return ``entry`` without a trailing port, leaving the bare address.

    Some proxies append ``ip:port`` rather than a bare address. Left alone the
    port becomes part of the key, so one client would get a fresh budget per
    connection - the limiter would still count, just never reach a limit.

    IPv6 is why this cannot split on the last colon: a bare ``::1`` is all
    colons and has no port at all. Only the bracketed form carries one, and an
    unbracketed address with several colons is left exactly as it is.

    Args:
        entry: One forwarded-chain entry, already stripped of whitespace.

    Returns:
        The address without its port, or the entry unchanged when it has none.
    """
    if entry.startswith("["):
        # [::1]:8080 or [::1] - the bracketed IPv6 form.
        closing = entry.find("]")
        return entry[1:closing] if closing != -1 else entry
    if entry.count(":") == 1:
        # Exactly one colon means IPv4 with a port; IPv6 always has more.
        return entry.split(":", 1)[0]
    return entry


def _forwarded_for(scope: Scope) -> list[str]:
    """Return every X-Forwarded-For entry, in the order the header lists them."""
    entries: list[str] = []
    for name, value in scope.get("headers") or []:
        if name.lower() != b"x-forwarded-for":
            continue
        entries.extend(part.strip() for part in value.decode("latin-1").split(","))
    return [strip_port(entry) for entry in entries if entry]


def is_trusted(address: str, trusted_proxies: frozenset[str]) -> bool:
    """Report whether ``address`` is one of the configured trusted proxies.

    Entries may be a plain address or a CIDR range, because the address a
    proxy actually dials from is often not knowable in advance: on a compose
    network the frontend container gets whatever the bridge hands it that
    boot, so an exact-match-only setting would be advice nobody could follow.

    A malformed entry or a non-address candidate simply does not match. That
    is the safe direction - the failure is "trust nobody", never "trust
    everybody" - and it keeps a typo in the setting from being a bypass.

    Args:
        address: The candidate address, as text.
        trusted_proxies: The configured entries (addresses and/or CIDRs).

    Returns:
        True when the address is covered by one of the entries.
    """
    if not trusted_proxies:
        return False
    if address in trusted_proxies:
        return True
    try:
        candidate = ipaddress.ip_address(address)
    except ValueError:
        return False
    for entry in trusted_proxies:
        try:
            # strict=False so "10.0.0.5/24" is read as its network rather than
            # rejected for having host bits set.
            network = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            continue
        if candidate in network:
            return True
    return False


def limit_key(scope: Scope, trusted_proxies: frozenset[str]) -> str:
    """Identify the caller whose budget this request spends.

    The ONE place that decision is made, and the seam the identity model will
    move through: today it is the client's address, and once real login exists
    it becomes the authenticated principal - a change to this function alone,
    with the middleware around it untouched.

    X-Forwarded-For is believed only when the peer is a configured trusted
    proxy, and then only from the right: the entries a proxy did not add
    itself are client-supplied text, so reading the LEFTMOST one (the usual
    mistake) lets any caller mint a fresh identity per request and walk
    straight through the limiter. Walking from the right past the addresses
    that are themselves trusted proxies yields the first hop nobody vouches
    for, which is the closest thing to a real client address available.

    Args:
        scope: The ASGI connection scope.
        trusted_proxies: Peer addresses or CIDR ranges whose forwarded headers
            may be believed. Empty (the default) means no header is ever
            believed.

    Returns:
        An opaque, stable key for this caller.
    """
    peer = _peer_address(scope)
    if not is_trusted(peer, trusted_proxies):
        return peer
    for candidate in reversed(_forwarded_for(scope)):
        if not is_trusted(candidate, trusted_proxies):
            return candidate
    # Every hop is a trusted proxy of ours, or the header was absent; the peer
    # is the most specific thing left.
    return peer


class RateLimitMiddleware:
    """Refuse a client that exceeds its per-minute budget for a route class."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: bool,
        per_minute: int,
        heavy_per_minute: int,
        light_per_minute: int,
        trusted_proxies: frozenset[str],
    ) -> None:
        self.app = app
        self.enabled = enabled
        self._limits: dict[RouteClass, int] = {
            "light": light_per_minute,
            "general": per_minute,
            "heavy": heavy_per_minute,
        }
        self.trusted_proxies = trusted_proxies
        self._last_warning = -math.inf

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Count this request, or refuse it; either way the response is untouched."""
        # Disabled is the first branch and allocates nothing: the limiter must
        # cost a deployment that turned it off exactly one attribute read.
        if not self.enabled or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limited = route_class(scope.get("method", ""), scope.get("path", ""))
        if limited is None:
            await self.app(scope, receive, send)
            return

        counter = _counter_from(scope)
        if counter is None:
            self._warn("no counter is wired on the application")
            await self.app(scope, receive, send)
            return

        now = time.time()
        window = int(now // WINDOW_SECONDS)
        key = f"ratelimit:{limited}:{window}:{limit_key(scope, self.trusted_proxies)}"
        try:
            count = await counter.incr_window(key, WINDOW_SECONDS)
        except Exception as exc:  # noqa: BLE001 - a blinking counter must not blink the app
            self._warn(f"the counter store is unavailable ({type(exc).__name__})")
            await self.app(scope, receive, send)
            return

        if count > self._limits[limited]:
            await self._refuse(send, limit=self._limits[limited], now=now)
            return
        await self.app(scope, receive, send)

    def _warn(self, reason: str) -> None:
        """Report that limiting lapsed, at most once per interval."""
        elapsed = time.monotonic() - self._last_warning
        if elapsed < _WARNING_INTERVAL_SECONDS:
            return
        self._last_warning = time.monotonic()
        logger.warning("Rate limiting is not being enforced: %s.", reason)

    async def _refuse(self, send: Send, *, limit: int, now: float) -> None:
        """Answer 429 as problem+json, with the seconds until the window rolls.

        The document is written here rather than raised: an exception from user
        middleware escapes the app's exception handlers (they sit inside it)
        and would render as a 500. The wording and codes come from the typed
        error the rest of the app raises for this, so one 429 is
        indistinguishable from another on the wire.
        """
        retry_after = max(1, WINDOW_SECONDS - int(now % WINDOW_SECONDS))
        body = json.dumps(
            {
                "type": "about:blank",
                "title": TooManyRequestsError.title,
                "status": TooManyRequestsError.status,
                "detail": (
                    f"Too many requests: at most {limit} per minute from one client. "
                    f"Retry in {retry_after}s."
                ),
                "code": TooManyRequestsError.code,
                "request_id": request_id_var.get(),
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": TooManyRequestsError.status,
                "headers": [
                    (b"content-type", b"application/problem+json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", str(retry_after).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
