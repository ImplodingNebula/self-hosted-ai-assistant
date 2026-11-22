"""Turning a request into the principal it acts as.

Three answers, checked in this order:

1. **Login is off** - every request is the seeded default principal, exactly as
   it was before login existed. Byte-identical, and no database read.
2. **The request came from inside the backend container itself** - the
   deployment's own operator path (see :func:`is_local_operator`).
3. **A session cookie** - read the server-side session, load the users row, and
   act as that person with the role the row says RIGHT NOW.

Anything else is refused with a typed 401 the frontend recognises.

The role is deliberately re-read per request rather than cached in the session.
It is one indexed primary-key lookup, and it is what makes a demotion or a
removal take effect on the next request instead of whenever a session happens
to expire.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import TYPE_CHECKING

from app.core.errors import AuthRequiredError, ServiceUnavailableError
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import User
from app.services.auth.store import read_session
from app.workers.queue import JobQueueUnavailableError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from starlette.requests import Request

    from app.core.config import Settings
    from app.services.auth.store import KeyValueStore

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "session_id"

# Headers a relay adds. Their PRESENCE disqualifies the operator grant below -
# see is_local_operator for why that matters more than it looks.
_RELAY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip")


def seeded_principal() -> TenantContext:
    """The pre-login principal: the seeded tenant's seeded admin user."""
    return TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


def is_local_operator(request: Request) -> bool:
    """Whether this request came from inside the backend container itself.

    **What this exists for.** A deployment publishes nothing but the reverse
    proxy, so the operator targets (`make warm`, `make status`, `make perf`)
    reach the API by running a request from INSIDE the backend
    container - ``docker compose exec -T backend uv run python -`` with
    ``scripts/backend_http.py`` on stdin, dialling ``http://localhost:8000``.
    That request has no browser, no cookie, and nobody to sign in; without this,
    turning login on breaks every one of those targets.

    **Why the peer address is a sound signal - IN THE CONTAINERIZED LAYOUTS,
    which is the scope of this claim.** Under compose, reaching the backend's
    own loopback interface requires already being INSIDE that container, which
    requires the ability to run ``docker exec`` - i.e. control of the host's
    Docker socket, which is root-equivalent on that host. Somebody holding that
    can read the database directly; a request granted the operator context is
    strictly less than they already have. Every other path in gets a different
    address: the reverse proxy and the frontend relay dial over the compose
    network from their own container addresses, and a published host port
    arrives via the bridge gateway. The production layout is stronger still -
    it publishes NO backend port at all (``ports: !reset []``), so the only
    thing that can reach this interface is a process already in the container.

    **Where the claim does NOT hold: the backend run bare on a host.** A plain
    ``uvicorn`` on a developer's machine has a loopback interface anything local
    can dial, and a relay on that same host (the Next dev server, say) would
    launder every one of its callers into a loopback peer - so with login
    configured, anyone who can reach that relay would be handed the operator
    context. That shape is not what the repository ships or documents, but the
    grant is only as narrow as the deployment is, and this is the sentence that
    says so out loud. A bare-metal deployment wanting login should put the
    backend behind a proxy that sets a forwarding header (which disqualifies the
    grant, see below) or run the containerized layout.

    **The two conditions, and why the second one is not belt-and-braces.** The
    peer must be a loopback address AND the request must carry no forwarding
    header. The peer address this reads is ``scope["client"]``, which uvicorn's
    proxy-header middleware REWRITES from ``X-Forwarded-For`` when the immediate
    peer is trusted; its default trusts loopback, and a deployment can widen it
    to anything through ``FORWARDED_ALLOW_IPS``. On a deployment that widened it,
    the address alone would let a request relayed through Caddy claim
    ``X-Forwarded-For: 127.0.0.1`` and be handed the operator context. Refusing
    any request that carries a forwarding header at all closes that: the
    operator probe sends none, and every relayed request has one (Caddy's
    ``reverse_proxy`` appends ``X-Forwarded-For`` unconditionally).

    **SSRF.** This grant is only worth having if nothing can be made to issue a
    request to the app's own loopback on an outsider's behalf. Nothing in the
    app fetches a caller-supplied URL: the web fetcher only follows search
    results and applies a public-host guard, OAuth discovery applies the same
    guard to every remote-named target, and the model registry client talks to a
    fixed configured host. The one operator-supplied fetch target - a registered
    MCP server's base URL - is admin-gated and audited, so reaching it already
    requires the role this grant would hand out.

    Args:
        request: The incoming request.

    Returns:
        True when the request is the deployment's own in-container operator
        path.
    """
    for header in _RELAY_HEADERS:
        if header in request.headers:
            return False
    client = request.client
    if client is None:
        return False
    try:
        return ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        return False


async def resolve_principal(
    request: Request,
    *,
    settings: Settings,
    store: KeyValueStore,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> TenantContext:
    """Resolve the acting principal, or refuse the request.

    Args:
        request: The incoming request (cookie and peer address).
        settings: The deployment's configuration.
        store: The session store.
        sessionmaker: Session factory for the one users-row read.

    Returns:
        The principal this request acts as.

    Raises:
        AuthRequiredError: Login is on and there is no valid session.
        ServiceUnavailableError: The session store cannot be reached. This
            FAILS CLOSED, unlike the rate limiter's counter beside it: a
            limiter that cannot count must not take the site down, but an
            authenticator that cannot check must not wave people through.
    """
    if not settings.oidc_enabled:
        return seeded_principal()

    if is_local_operator(request):
        return seeded_principal()

    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        raise AuthRequiredError("Sign in to continue.")

    try:
        record = await read_session(store, session_id, ttl_seconds=settings.session_ttl_seconds)
    except JobQueueUnavailableError as exc:
        raise ServiceUnavailableError(
            "Sessions cannot be checked right now; try again shortly."
        ) from exc
    if record is None:
        raise AuthRequiredError("This session has expired. Sign in again.")

    # A session of its own rather than the request-scoped one: the request
    # session is held open for the whole handler, and the streaming endpoints
    # deliberately never take it - making the auth seam depend on it would open
    # a connection for the life of every SSE stream.
    async with sessionmaker() as db:
        user = await db.get(User, record.user_id)
    if user is None:
        # The session outlived the row it names - a removed account, or a
        # database restored from before the sign-in. Refuse rather than fall
        # back to the seeded principal, which would silently promote a stranger.
        raise AuthRequiredError("This session is no longer valid. Sign in again.")

    return TenantContext(tenant_id=user.tenant_id, user_id=user.id, role=user.role)
