"""The login routes, the dependency swap, and the auth coverage of the API.

Two things are proved here that nothing else can prove:

- **With no issuer configured, nothing changed.** That is the no-regression
  clamp the whole phase rests on, so it is asserted against the real app rather
  than reasoned about.
- **With an issuer configured, EVERY ``/api/v1`` route resolves a principal.**
  A route that takes no principal dependency is a route that answers an
  unauthenticated caller, and the four ``/auth`` endpoints are the only ones
  allowed to be that. The test walks the live route surface, so a new route
  joins the exempt list only by being added to it on purpose.

**A trap worth naming, because it would make every test here hollow.**
``httpx.ASGITransport`` defaults its peer address to ``127.0.0.1``, which is
exactly the address the in-container operator grant honours - so a test app
driven through the default transport is ALWAYS an operator and would sail past
every session check while proving nothing. Every client below therefore names a
peer explicitly, and :func:`operator_client` is the only one that says loopback.
"""

from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.services.auth.login as login_mod
from app.api.deps import get_current_tenant, get_model_provider, get_settings
from app.api.v1.auth import LOGIN_STATE_COOKIE_NAME
from app.core.concurrency import StreamLimiter
from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID
from app.main import create_app
from app.models import User
from app.providers.base import ModelCapabilities, ModelInfo
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.services.auth.discovery import clear_discovery_cache
from app.services.auth.jwks import clear_jwks_cache
from app.services.auth.principal import SESSION_COOKIE_NAME
from app.services.auth.store import create_session
from app.storage.local import FakeStorageBackend
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store
from tests.idp import DEFAULT_CLIENT_ID, FakeIdp

# Anything that is plainly not this machine. The value matters: see the module
# docstring on the ASGITransport default.
REMOTE_PEER = ("203.0.113.7", 51234)
LOOPBACK_PEER = ("127.0.0.1", 51234)


@pytest.fixture(autouse=True)
def _fresh_auth_caches() -> Iterator[None]:
    """Isolate the process-wide discovery and key caches between tests."""
    clear_discovery_cache()
    clear_jwks_cache()
    yield
    clear_discovery_cache()
    clear_jwks_cache()


def login_settings(**overrides: object) -> Settings:
    """Settings with login configured against the fake provider."""
    base: dict[str, object] = {
        "oidc_issuer": FakeIdp().issuer,
        "oidc_client_id": DEFAULT_CLIENT_ID,
        "oidc_client_secret": "a-client-secret",
        "oidc_admin_emails": ["boss@example.com"],
        "oidc_allowed_domains": ["example.com"],
        "oauth_redirect_base_url": "http://localhost:3000",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def build_app(
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    queue: FakeJobQueue,
) -> FastAPI:
    """A real app wired to fakes, with the given settings in force."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider()
    app.state.db_sessionmaker = sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.storage = FakeStorageBackend()
    app.state.job_queue = queue
    app.state.rate_counter = queue
    return app


def client_for(app: FastAPI, *, peer: tuple[str, int] = REMOTE_PEER) -> AsyncClient:
    """An httpx client whose requests arrive from ``peer``."""
    return AsyncClient(transport=ASGITransport(app=app, client=peer), base_url="http://test")


def operator_client(app: FastAPI) -> AsyncClient:
    """A client standing in for a request made INSIDE the backend container."""
    return client_for(app, peer=LOOPBACK_PEER)


async def add_user(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    email: str,
    role: str,
    subject: str = "sub-1",
) -> UUID:
    """Insert a signed-in-looking user row and return its id."""
    async with sessionmaker() as session:
        user = User(
            tenant_id=DEFAULT_TENANT_ID,
            email=email,
            display_name=email,
            role=role,
            oidc_issuer=FakeIdp().issuer,
            oidc_sub=subject,
        )
        session.add(user)
        await session.commit()
        return user.id


# --------------------------------------------------------- no issuer at all


async def test_with_no_issuer_the_api_behaves_exactly_as_before(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The no-regression clamp: no cookie, no session store, no login page - and
    # every route answers as the seeded admin, which is what it always did.
    app = build_app(Settings(oidc_issuer=None), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        listing = await client.get("/api/v1/models")
        manager_gate = await client.get("/api/v1/models/residency")
        who = await client.get("/api/v1/auth/me")

    assert listing.status_code == 200
    assert manager_gate.status_code == 200
    assert who.json() == {
        "auth": "none",
        "user_id": str(DEFAULT_USER_ID),
        "email": None,
        "display_name": "Default User",
        "role": "admin",
    }


async def test_with_no_issuer_nothing_reaches_the_session_store_at_all(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # Sharper than the behavioural check above, and it is the claim that matters
    # for a deployment with no Redis worth speaking of: the store is wired to
    # REFUSE every call, and the API answers anyway - so the no-login path
    # genuinely reads nothing rather than reading something that happens to work.
    app = build_app(Settings(oidc_issuer=None), db_sessionmaker, FakeJobQueue(unavailable=True))

    async with client_for(app) as client:
        listing = await client.get("/api/v1/models")
        who = await client.get("/api/v1/auth/me")

    assert listing.status_code == 200
    assert who.status_code == 200


async def test_with_no_issuer_the_sign_in_route_names_nothing(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    app = build_app(Settings(oidc_issuer=None), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        response = await client.get("/api/v1/auth/login")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_signing_out_with_no_login_configured_is_still_a_no_op_204(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    app = build_app(Settings(oidc_issuer=None), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        response = await client.post("/api/v1/auth/logout")

    assert response.status_code == 204


# ------------------------------------------------------------ boot refusal


@pytest.mark.parametrize(
    "missing",
    ["oidc_client_id", "oidc_client_secret", "oidc_admin_emails"],
)
def test_a_half_configured_login_refuses_to_boot(missing: str) -> None:
    # Each of these fails late and confusingly otherwise: no secret means the
    # provider rejects the first exchange, and no admin means a workspace that
    # nobody can administer - discovered after the first person signs in.
    values: dict[str, object] = {
        "oidc_issuer": "https://idp.example",
        "oidc_client_id": "an-id",
        "oidc_client_secret": "a-secret",
        "oidc_admin_emails": "boss@example.com",
    }
    values[missing] = "" if missing != "oidc_admin_emails" else ""

    with pytest.raises(ValidationError, match=missing.upper()):
        Settings(**values)  # type: ignore[arg-type]


def test_a_fully_configured_login_boots() -> None:
    assert login_settings().oidc_enabled is True


def test_a_blank_issuer_reads_as_no_login_not_as_a_blank_issuer() -> None:
    # Compose forwards an unset key as an empty string, so this is the ORDINARY
    # shape for "no login" - read literally it would turn the switch on and fail
    # every request with a discovery error.
    assert Settings(oidc_issuer="   ").oidc_enabled is False


# ----------------------------------------------------------- session gating


async def test_a_request_with_no_session_is_refused_with_the_typed_code(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        response = await client.get("/api/v1/models")

    assert response.status_code == 401
    # The exact contract the frontend keys on to send somebody to sign in.
    assert response.json()["code"] == "auth_required"


async def test_a_minted_session_is_accepted_and_carries_the_stored_role(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    queue = FakeJobQueue()
    user_id = await add_user(db_sessionmaker, email="person@example.com", role="member")
    session_id = await create_session(
        queue, user_id=user_id, tenant_id=DEFAULT_TENANT_ID, ttl_seconds=99
    )
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, session_id)
        response = await client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json() == {
        "auth": "oidc",
        "user_id": str(user_id),
        "email": "person@example.com",
        "display_name": "person@example.com",
        "role": "member",
    }


async def test_a_session_naming_a_row_that_is_gone_is_refused(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # Falling back to the seeded principal here would silently promote a
    # stranger holding a stale cookie to the pre-login admin.
    queue = FakeJobQueue()
    session_id = await create_session(
        queue,
        user_id=UUID("00000000-0000-0000-0000-0000000000ff"),
        tenant_id=DEFAULT_TENANT_ID,
        ttl_seconds=99,
    )
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, session_id)
        response = await client.get("/api/v1/models")

    assert response.status_code == 401


async def test_an_expired_session_is_refused(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    queue = FakeJobQueue()
    user_id = await add_user(db_sessionmaker, email="person@example.com", role="member")
    session_id = await create_session(
        queue, user_id=user_id, tenant_id=DEFAULT_TENANT_ID, ttl_seconds=99
    )
    # Expiry, as the reader sees it: the key is simply no longer there.
    queue.values.clear()
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, session_id)
        response = await client.get("/api/v1/models")

    assert response.status_code == 401


async def test_a_demotion_applies_on_the_very_next_request(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The point of a server-side session: the role is re-read per request rather
    # than frozen into the cookie when it was minted.
    queue = FakeJobQueue()
    user_id = await add_user(db_sessionmaker, email="person@example.com", role="admin")
    session_id = await create_session(
        queue, user_id=user_id, tenant_id=DEFAULT_TENANT_ID, ttl_seconds=99
    )
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, session_id)
        before = await client.get("/api/v1/models/residency")
        async with db_sessionmaker() as session:
            user = await session.get(User, user_id)
            assert user is not None
            user.role = "member"
            await session.commit()
        after = await client.get("/api/v1/models/residency")

    assert before.status_code == 200
    assert after.status_code == 403


async def test_signing_out_takes_effect_on_the_next_request(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    queue = FakeJobQueue()
    user_id = await add_user(db_sessionmaker, email="person@example.com", role="member")
    session_id = await create_session(
        queue, user_id=user_id, tenant_id=DEFAULT_TENANT_ID, ttl_seconds=99
    )
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, session_id)
        signed_out = await client.post("/api/v1/auth/logout")
        # The cookie is deliberately put BACK, because a real holder of a stolen
        # copy would still have it. The record is what must be gone.
        client.cookies.set(SESSION_COOKIE_NAME, session_id)
        after = await client.get("/api/v1/models")

    assert signed_out.status_code == 204
    assert after.status_code == 401


async def test_a_session_store_that_is_down_fails_closed(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The asymmetry with the rate limiter next door, which fails OPEN: a counter
    # that cannot count must not take the site down, but an authenticator that
    # cannot check must not wave people through.
    queue = FakeJobQueue(unavailable=True)
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, "any-session")
        response = await client.get("/api/v1/models")

    assert response.status_code == 503
    assert response.json()["code"] == "service_unavailable"


# -------------------------------------------------------------- role gating


@pytest.mark.parametrize("path", ["/api/v1/models/manager", "/api/v1/models/residency"])
async def test_a_member_is_refused_the_operator_views(
    db_sessionmaker: async_sessionmaker[AsyncSession], path: str
) -> None:
    # Each of these discloses deployment-wide operator information: a
    # cross-tenant space inventory, and what the shared model server is holding.
    queue = FakeJobQueue()
    user_id = await add_user(db_sessionmaker, email="person@example.com", role="member")
    session_id = await create_session(
        queue, user_id=user_id, tenant_id=DEFAULT_TENANT_ID, ttl_seconds=99
    )
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, session_id)
        response = await client.get(path)

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


@pytest.mark.parametrize("path", ["/api/v1/models/manager", "/api/v1/models/residency"])
async def test_an_admin_reaches_the_operator_views(
    db_sessionmaker: async_sessionmaker[AsyncSession], path: str
) -> None:
    queue = FakeJobQueue()
    user_id = await add_user(db_sessionmaker, email="boss@example.com", role="admin")
    session_id = await create_session(
        queue, user_id=user_id, tenant_id=DEFAULT_TENANT_ID, ttl_seconds=99
    )
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, session_id)
        response = await client.get(path)

    assert response.status_code == 200


# --------------------------------------------------- the ops (loopback) path


async def test_the_in_container_operator_grant_is_honoured(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # `make warm` / `make status` on a deployment run their request from INSIDE
    # the backend container, with no browser and nobody to sign in. Reaching
    # that loopback interface requires docker exec on the host, which is already
    # root-equivalent there.
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())
    # build_app's bare FakeProvider reports completion=True for every tag (it
    # stands in for the chat model everywhere else), so the embedding-inventory
    # rung would find nothing to resolve. Give it one tag that is actually
    # embedding-only, so a 200 here proves the operator grant rather than
    # tripping over unrelated resolution machinery this test does not cover.
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider(
        models=[ModelInfo(name="embed:0.6b")],
        capabilities_by_model={"embed:0.6b": ModelCapabilities(embedding=True, completion=False)},
    )

    async with operator_client(app) as client:
        response = await client.post("/api/v1/models/warm-embed", json={})

    # 200, not merely "not 401" - a weaker assertion here would be satisfied by
    # a 500, which is not the operator path working.
    assert response.status_code == 200


async def test_a_remote_peer_gets_no_operator_grant(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        response = await client.post("/api/v1/models/warm-embed", json={})

    assert response.status_code == 401


@pytest.mark.parametrize("header", ["x-forwarded-for", "forwarded", "x-real-ip"])
async def test_a_forwarded_header_disqualifies_the_operator_grant(
    db_sessionmaker: async_sessionmaker[AsyncSession], header: str
) -> None:
    # The peer address alone is not enough. uvicorn's proxy-header middleware
    # REWRITES the peer from X-Forwarded-For when the immediate peer is trusted,
    # and a deployment can widen who is trusted through FORWARDED_ALLOW_IPS - at
    # which point a request relayed through the proxy could claim to be
    # loopback. The operator probe sends no forwarding header; every relayed
    # request has one.
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())

    async with operator_client(app) as client:
        response = await client.post(
            "/api/v1/models/warm-embed", json={}, headers={header: "127.0.0.1"}
        )

    assert response.status_code == 401


# ------------------------------------------------------ full round trip


def _fake_provider_client(idp: FakeIdp, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the login flow's own httpx client at the fake provider.

    The flow builds its client itself (that is the shipped behaviour, and the
    route is what is under test here), so the class has to be substituted. The
    substitute fills in a transport ONLY where the caller named none: the test's
    own ASGI client passes one explicitly and must be left alone, and the real
    class has to be captured first or the replacement calls itself forever.
    """
    real_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.setdefault("transport", idp.transport())
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(login_mod.httpx, "AsyncClient", factory)


async def test_the_whole_sign_in_round_trip_works_end_to_end(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    idp = FakeIdp()
    _fake_provider_client(idp, monkeypatch)
    queue = FakeJobQueue()
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as client:
        refused_before = await client.get("/api/v1/models")

        started = await client.get("/api/v1/auth/login", params={"next": "/console/models"})
        consent = urlsplit(started.headers["location"])
        query = parse_qs(consent.query)
        state = query["state"][0]
        idp.next_id_token = idp.id_token(nonce=query["nonce"][0], email="person@example.com")

        landed = await client.get(
            "/api/v1/auth/callback", params={"code": "the-code", "state": state}
        )
        allowed_after = await client.get("/api/v1/models")
        who = await client.get("/api/v1/auth/me")

    assert refused_before.status_code == 401
    assert started.status_code == 302
    # The 302 that starts a sign-in CARRIES A COOKIE, and any relay in front of
    # this route has to pass it through - without it the callback has nothing to
    # match and every sign-in fails at the last hop, which is the quietest way
    # for login to be broken.
    assert LOGIN_STATE_COOKIE_NAME in started.headers.get("set-cookie", "")
    assert consent.path == "/authorize"
    assert landed.status_code == 302
    assert landed.headers["location"] == "/console/models"
    # The cookie the browser was handed is what makes the next request work.
    assert SESSION_COOKIE_NAME in landed.headers["set-cookie"]
    assert "httponly" in landed.headers["set-cookie"].lower()
    assert "samesite=lax" in landed.headers["set-cookie"].lower()
    assert allowed_after.status_code == 200
    assert who.json()["auth"] == "oidc"
    assert who.json()["email"] == "person@example.com"


async def test_an_unadmitted_person_gets_a_readable_refusal_not_a_cookie(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    idp = FakeIdp()
    _fake_provider_client(idp, monkeypatch)
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        started = await client.get("/api/v1/auth/login")
        query = parse_qs(urlsplit(started.headers["location"]).query)
        idp.next_id_token = idp.id_token(nonce=query["nonce"][0], email="stranger@evil.example")
        refused = await client.get(
            "/api/v1/auth/callback",
            params={"code": "the-code", "state": query["state"][0]},
        )

    assert refused.status_code == 403
    # The attempt's own cookie is cleared, but no SESSION is ever handed out.
    assert SESSION_COOKIE_NAME not in refused.headers.get("set-cookie", "")
    assert refused.headers["content-type"].startswith("text/html")
    assert "not permitted" in refused.text.lower()


async def test_a_replayed_callback_is_refused_without_leaking_which_check_failed(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    idp = FakeIdp()
    _fake_provider_client(idp, monkeypatch)
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        started = await client.get("/api/v1/auth/login")
        query = parse_qs(urlsplit(started.headers["location"]).query)
        state = query["state"][0]
        idp.next_id_token = idp.id_token(nonce=query["nonce"][0])
        params = {"code": "the-code", "state": state}
        first = await client.get("/api/v1/auth/callback", params=params)
        # The binding cookie is spent by the first callback, so a genuine replay
        # is already refused. Putting it back is what an attacker who somehow
        # HELD it would have, and it drives the replay onto the state check
        # underneath - which must refuse it too. Two independent guards, and this
        # is how the second one gets exercised rather than shadowed by the first.
        client.cookies.set(LOGIN_STATE_COOKIE_NAME, state)
        replayed = await client.get("/api/v1/auth/callback", params=params)

    assert first.status_code == 302
    assert replayed.status_code == 400
    assert "already used" in replayed.text.lower()
    assert SESSION_COOKIE_NAME not in replayed.headers.get("set-cookie", "")


async def test_a_callback_that_did_not_start_in_this_browser_is_refused(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Login CSRF: an attacker runs a real sign-in as themselves, keeps the code
    # and state, and gets a victim to open the callback - which would leave the
    # victim's browser holding a session for the ATTACKER'S account, with
    # everything they do next landing somewhere the attacker can read. The state
    # value cannot detect this (the attacker has the genuine one); the cookie
    # can, because the attacker cannot set one on somebody else's browser.
    idp = FakeIdp()
    _fake_provider_client(idp, monkeypatch)
    queue = FakeJobQueue()
    app = build_app(login_settings(), db_sessionmaker, queue)

    async with client_for(app) as attacker:
        started = await attacker.get("/api/v1/auth/login")
        query = parse_qs(urlsplit(started.headers["location"]).query)
        idp.next_id_token = idp.id_token(nonce=query["nonce"][0])
        params = {"code": "the-code", "state": query["state"][0]}

    # A different browser entirely - it never visited /auth/login.
    async with client_for(app) as victim:
        planted = await victim.get("/api/v1/auth/callback", params=params)

    assert planted.status_code == 400
    assert SESSION_COOKIE_NAME not in planted.headers.get("set-cookie", "")
    # And the attempt is still spendable by the browser that owns it, because a
    # refusal here must not consume somebody else's pending sign-in either.
    assert any(key.startswith("authpending:") for key in queue.values)


async def test_the_provider_cancelling_is_reported_as_a_refusal(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        response = await client.get("/api/v1/auth/callback", params={"error": "access_denied"})

    assert response.status_code == 400
    # The provider's error code is attacker-influencable text; it is never
    # rendered back onto the page.
    assert "access_denied" not in response.text


# -------------------------------------------------------- coverage sweep


def _api_routes(app: FastAPI) -> list[tuple[str, APIRoute]]:
    """Every APIRoute the app serves, with its full path.

    This FastAPI version keeps an included router nested behind a wrapper
    object rather than flattening its routes onto the app, so the walk has to
    follow ``original_router`` - the same reason the rate-limit guard reads the
    OpenAPI document instead.
    """
    found: list[tuple[str, APIRoute]] = []

    def walk(routes: list[object], prefix: str) -> None:
        for route in routes:
            if isinstance(route, APIRoute):
                found.append((prefix + route.path, route))
            elif hasattr(route, "original_router"):
                context = route.include_context  # type: ignore[attr-defined]
                walk(route.original_router.routes, prefix + getattr(context, "prefix", ""))  # type: ignore[attr-defined]
            elif hasattr(route, "routes"):
                walk(route.routes, prefix + getattr(route, "prefix", ""))  # type: ignore[attr-defined]

    walk(list(app.routes), "")
    return found


def _resolves_a_principal(dependant: object, seen: set[int] | None = None) -> bool:
    """Whether ``get_current_tenant`` appears anywhere in a route's dependencies."""
    seen = set() if seen is None else seen
    for sub in dependant.dependencies:  # type: ignore[attr-defined]
        if sub.call is get_current_tenant:
            return True
        if id(sub) in seen:
            continue
        seen.add(id(sub))
        if _resolves_a_principal(sub, seen):
            return True
    return False


# The only routes allowed to answer without a principal, and why each one must.
# Adding to this list is a deliberate act; the test below is what makes it one.
SESSIONLESS_PATHS = frozenset(
    {
        # Liveness and readiness, polled by compose healthchecks and by whatever
        # orchestrates the deployment. A probe that could answer 401 would take
        # the deployment down by itself.
        "/healthz",
        "/readyz",
        # Three of these run BEFORE there is a principal, and the fourth exists
        # to report that there is not one.
        "/api/v1/auth/login",
        "/api/v1/auth/callback",
        "/api/v1/auth/logout",
        "/api/v1/auth/me",
    }
)


def test_every_api_route_resolves_a_principal() -> None:
    # The sweep. With login configured, a route with no principal dependency
    # answers an unauthenticated caller - and the failure is silent, because the
    # route works perfectly. Three model routes were exactly that before this
    # test existed.
    ungated = sorted(
        path
        for path, route in _api_routes(create_app())
        if path not in SESSIONLESS_PATHS and not _resolves_a_principal(route.dependant)
    )

    assert not ungated, (
        "These routes resolve no principal, so with login configured they answer "
        f"anybody who can reach the port: {ungated}. Give each a TenantDep or an "
        "AdminDep, or - only if it genuinely must be reachable signed out - add it "
        "to SESSIONLESS_PATHS with the reason."
    )


def test_the_sessionless_list_names_only_routes_that_exist() -> None:
    # The other direction: a stale entry here would silently exempt nothing
    # while looking like it exempts something, and would keep exempting a path
    # that later comes back meaning something else.
    served = {path for path, _route in _api_routes(create_app())}
    missing = sorted(path for path in SESSIONLESS_PATHS if path not in served)

    assert not missing, f"these exemptions name no registered route: {missing}"


def test_the_login_routes_are_limited_as_general_traffic() -> None:
    # Deliberate, and worth pinning: the callback is unauthenticated by
    # necessity, so it must be counted rather than exempted.
    from app.core.ratelimit import route_class

    assert route_class("GET", "/api/v1/auth/login") == "general"
    assert route_class("GET", "/api/v1/auth/callback") == "general"
    assert route_class("POST", "/api/v1/auth/logout") == "general"
    assert route_class("GET", "/api/v1/auth/me") == "general"


async def test_a_non_ascii_state_is_refused_rather_than_crashing(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # `hmac.compare_digest` raises TypeError on a str holding any non-ASCII
    # character, and `state` here is raw query text from an unauthenticated
    # request. Comparing the strings therefore hands anybody a 500 with a
    # traceback on the authentication boundary - a log-flood primitive, and a
    # hole in this module's rule that every refusal reads the same. Comparing
    # bytes makes it an ordinary refusal.
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        client.cookies.set(LOGIN_STATE_COOKIE_NAME, "a-real-looking-state")
        response = await client.get(
            "/api/v1/auth/callback", params={"code": "x", "state": "\u00e9\u00e9\u00e9"}
        )

    assert response.status_code == 400
    assert "did not start in this browser" in response.text


async def test_the_landing_path_comes_from_the_stored_attempt_not_the_callback(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The redirect after a successful sign-in must be the value validated when
    # the attempt STARTED, never anything read back off the callback request. A
    # callback that could name its own landing page would be an open redirector
    # reached with a valid session already attached.
    idp = FakeIdp()
    _fake_provider_client(idp, monkeypatch)
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as client:
        started = await client.get("/api/v1/auth/login", params={"next": "/console/models"})
        query = parse_qs(urlsplit(started.headers["location"]).query)
        idp.next_id_token = idp.id_token(nonce=query["nonce"][0])
        landed = await client.get(
            "/api/v1/auth/callback",
            params={
                "code": "the-code",
                "state": query["state"][0],
                # Ignored entirely - the callback takes no landing instruction.
                "next": "https://evil.example/steal",
            },
        )

    assert landed.status_code == 302
    assert landed.headers["location"] == "/console/models"


async def test_a_planted_callback_does_not_cancel_the_victims_own_sign_in(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Refusing a planted callback must not clear the browser's cookie. Somebody
    # midway through their OWN sign-in holds one, and clearing it would let an
    # attacker who cannot steal a session still cancel every attempt to start
    # one - a denial of service handed out as a consolation prize. So the
    # mismatch branch is the one exit that leaves the cookie alone; it expires
    # by itself within ten minutes.
    idp = FakeIdp()
    _fake_provider_client(idp, monkeypatch)
    app = build_app(login_settings(), db_sessionmaker, FakeJobQueue())

    async with client_for(app) as attacker:
        attacker_start = await attacker.get("/api/v1/auth/login")
        attacker_query = parse_qs(urlsplit(attacker_start.headers["location"]).query)

    async with client_for(app) as victim:
        victim_start = await victim.get("/api/v1/auth/login")
        victim_query = parse_qs(urlsplit(victim_start.headers["location"]).query)

        planted = await victim.get(
            "/api/v1/auth/callback",
            params={"code": "the-attackers-code", "state": attacker_query["state"][0]},
        )

        # The victim now finishes their own sign-in, which only works if their
        # cookie survived the planted callback.
        idp.next_id_token = idp.id_token(nonce=victim_query["nonce"][0])
        finished = await victim.get(
            "/api/v1/auth/callback",
            params={"code": "the-victims-code", "state": victim_query["state"][0]},
        )

    assert planted.status_code == 400
    assert finished.status_code == 302
    assert SESSION_COOKIE_NAME in finished.headers["set-cookie"]
