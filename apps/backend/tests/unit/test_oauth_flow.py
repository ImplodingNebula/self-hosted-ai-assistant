"""The connect flow: start, callback, disconnect - and the guards each rests on."""

from __future__ import annotations

import asyncio
import base64
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

import app.services.oauth.discovery as discovery_mod
from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.models import McpAuthType, McpServer, utcnow
from app.providers.secrets import SecretNotFoundError, SecretScope, build_cipher
from app.repositories.mcp_servers import McpServerRepository
from app.repositories.oauth_authorizations import OAuthAuthorizationRepository
from app.services.oauth import flow, locks
from app.services.oauth.authorize import OAuthEndpoints
from app.services.oauth.bundle import (
    OAuthTokenBundle,
    oauth_client_secret_name,
    oauth_token_secret_name,
)
from app.services.oauth.flow import OAuthFlowError
from app.services.secret_store import SecretStore

pytestmark = pytest.mark.anyio

_KEY = base64.b64encode(b"o" * 32).decode()
CIPHER = build_cipher(_KEY)
TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_TENANT_ID, role="admin")

BYPASS = OAuthEndpoints(
    issuer="https://as.example",
    authorization_endpoint="https://as.example/authorize",
    token_endpoint="https://as.example/token",
    revocation_endpoint="https://as.example/revoke",
    send_resource=False,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, model_provider="fake", secret_master_key=_KEY, **overrides)


async def _add_server(
    db_sessionmaker,
    *,
    base_url: str = "https://mcp.example/mcp",
    credential_scope: str = "workspace",
) -> McpServer:
    async with db_sessionmaker() as session:
        server = await McpServerRepository(session, TENANT).create(
            name="crm", base_url=base_url, credential_scope=credential_scope
        )
        await session.commit()
    return server


async def _start(
    db_sessionmaker,
    server,
    *,
    secret: str | None = None,
    endpoints: OAuthEndpoints | None = BYPASS,
    **kwargs,
) -> str:
    """Run start_connect on the bypass path inside one session; return the URL."""
    async with db_sessionmaker() as session:
        url = await flow.start_connect(
            server,
            _settings(),
            TENANT,
            repo=OAuthAuthorizationRepository(session, TENANT),
            store=SecretStore(session, TENANT, CIPHER),
            cipher=CIPHER,
            client_id="cid",
            client_secret=secret,
            scopes=["repo"],
            endpoints=endpoints,
            **kwargs,
        )
    return url


def _token_client(captured: dict, payload: dict | None = None) -> httpx.AsyncClient:
    """A MockTransport token endpoint recording the exchange's form fields."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["form"] = parse_qs(request.content.decode())
        return httpx.Response(
            200,
            json=payload
            if payload is not None
            else {
                "access_token": "at-123",
                "refresh_token": "rt-456",
                "expires_in": 3600,
                "scope": "repo",
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _callback(db_sessionmaker, state: str, *, client: httpx.AsyncClient, code="c0de"):
    async with db_sessionmaker() as session:
        result = await flow.complete_callback(
            state=state,
            code=code,
            settings=_settings(),
            tenant=TENANT,
            repo=OAuthAuthorizationRepository(session, TENANT),
            servers=McpServerRepository(session, TENANT),
            store=SecretStore(session, TENANT, CIPHER),
            cipher=CIPHER,
            http_client=client,
        )
    return result


def _state_from(url: str) -> str:
    return parse_qs(urlsplit(url).query)["state"][0]


# --- start ------------------------------------------------------------------


async def test_start_stores_a_sealed_verifier_never_plaintext(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    params = parse_qs(urlsplit(url).query)
    async with db_sessionmaker() as session:
        repo = OAuthAuthorizationRepository(session, TENANT)
        row = await repo.consume(params["state"][0])
    assert row is not None
    # The row holds AES-GCM bytes; the verifier reappears only through the
    # cipher with the row's own AAD - a database read alone completes nothing.
    verifier = CIPHER.decrypt(
        row.code_verifier_sealed, aad=f"oauth-authz:{TENANT.tenant_id}:{row.state}".encode()
    )
    assert verifier not in row.code_verifier_sealed.decode("latin1")
    assert len(verifier) == 128


async def test_start_seals_the_client_secret_write_only(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)
    await _start(db_sessionmaker, server, secret="s3cret")
    async with db_sessionmaker() as session:
        revealed = await SecretStore(session, TENANT, CIPHER).reveal(
            oauth_client_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
        )
    assert revealed == "s3cret"


# --- callback ---------------------------------------------------------------


async def test_callback_exchanges_and_seals_the_grant(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    captured: dict = {}
    async with _token_client(captured) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)

    # The exchange hit the snapshotted token endpoint with the exact
    # redirect_uri the consent used, the unsealed verifier, and the client id.
    assert captured["url"] == "https://as.example/token"
    form = captured["form"]
    assert form["redirect_uri"] == ["http://localhost:3000/api/v1/integrations/oauth/callback"]
    assert form["grant_type"] == ["authorization_code"]
    assert form["client_id"] == ["cid"]
    assert len(form["code_verifier"][0]) == 128
    # Bypass path: generic providers take no resource parameter.
    assert "resource" not in form

    async with db_sessionmaker() as session:
        sealed = await SecretStore(session, TENANT, CIPHER).reveal(
            oauth_token_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
        )
        bundle = OAuthTokenBundle.unseal(sealed)
        row = await McpServerRepository(session, TENANT).get(server.id)
    assert bundle.access_token == "at-123"
    assert bundle.refresh_token == "rt-456"
    assert bundle.issuer == "https://as.example"
    assert row is not None and row.auth_type == McpAuthType.OAUTH.value


async def test_a_state_is_single_use(db_sessionmaker) -> None:
    # The replay guard: the second callback with the SAME state finds nothing,
    # whatever code it carries. Delete the consume-deletes behaviour and this
    # second exchange would run.
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    state = _state_from(url)
    async with _token_client({}) as client:
        await _callback(db_sessionmaker, state, client=client)
    async with _token_client({}) as client:
        with pytest.raises(OAuthFlowError, match="state"):
            await _callback(db_sessionmaker, state, client=client)


async def test_an_expired_state_is_refused(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = OAuthAuthorizationRepository(session, TENANT)
        await repo.create(
            mcp_server_id=server.id,
            state="expired-state",
            code_verifier_sealed=b"x",
            client_id="cid",
            redirect_uri="http://localhost:3000/cb",
            resource=None,
            endpoints=BYPASS.model_dump(),
            scopes="repo",
            created_by=TENANT.user_id,
            expires_at=utcnow() - timedelta(seconds=1),
        )
        await repo.commit()
    async with _token_client({}) as client:
        with pytest.raises(OAuthFlowError, match="state"):
            await _callback(db_sessionmaker, "expired-state", client=client)


async def test_a_response_without_an_access_token_stores_nothing(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    async with _token_client({}, payload={"error": "invalid_grant"}) as client:
        with pytest.raises(OAuthFlowError, match="HTTP|access token"):
            await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        with pytest.raises(SecretNotFoundError):
            await SecretStore(session, TENANT, CIPHER).reveal(
                oauth_token_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
            )
    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT).get(server.id)
    assert row is not None and row.auth_type == McpAuthType.HEADER.value


async def test_no_expiry_declared_gets_the_documented_horizon(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    async with _token_client({}, payload={"access_token": "at"}) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        sealed = await SecretStore(session, TENANT, CIPHER).reveal(
            oauth_token_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
        )
    bundle = OAuthTokenBundle.unseal(sealed)
    # No expires_in and no scope in the response: the horizon stands in, and
    # the REQUESTED scopes are recorded (RFC 6749: absent means as-requested).
    assert bundle.expires_at > utcnow() + timedelta(days=300)
    assert bundle.granted_scope == "repo"


async def test_the_sweep_clears_expired_rows_on_insert(db_sessionmaker) -> None:
    # Abandoned consents must not accumulate: the next start clears the
    # tenant's dead states. Delete the sweep in create() and this fails.
    server = await _add_server(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = OAuthAuthorizationRepository(session, TENANT)
        await repo.create(
            mcp_server_id=server.id,
            state="dead-state",
            code_verifier_sealed=b"x",
            client_id="cid",
            redirect_uri="http://localhost:3000/cb",
            resource=None,
            endpoints=BYPASS.model_dump(),
            scopes="repo",
            created_by=TENANT.user_id,
            expires_at=utcnow() - timedelta(seconds=1),
        )
        await repo.commit()
    await _start(db_sessionmaker, server)
    async with db_sessionmaker() as session:
        from sqlalchemy import select

        from app.models import OAuthAuthorization

        states = (await session.execute(select(OAuthAuthorization.state))).scalars().all()
    assert "dead-state" not in states
    assert len(states) == 1  # only the fresh consent remains


async def test_a_denial_burns_the_real_pending_state(db_sessionmaker) -> None:
    # The user clicking "deny" on the provider's page must end the flow for
    # good: the state is consumed on the error leg, so a later callback that
    # DOES carry a code (an injected one) finds nothing to complete.
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    state = _state_from(url)
    async with db_sessionmaker() as session:
        repo = OAuthAuthorizationRepository(session, TENANT)
        burned = await repo.consume(state)  # what the router's error leg runs
        await repo.commit()
    assert burned is not None
    async with _token_client({}) as client:
        with pytest.raises(OAuthFlowError, match="state"):
            await _callback(db_sessionmaker, state, client=client)


async def test_the_sealed_verifier_is_welded_to_its_state(db_sessionmaker) -> None:
    # AAD negative: the same ciphertext under a DIFFERENT state's AAD must
    # refuse to open - a mixed-up or copied row cannot complete an exchange.
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    async with db_sessionmaker() as session:
        row = await OAuthAuthorizationRepository(session, TENANT).consume(_state_from(url))
    assert row is not None
    with pytest.raises(Exception):  # noqa: B017 - the cipher lib raises its own type
        CIPHER.decrypt(
            row.code_verifier_sealed,
            aad=f"oauth-authz:{TENANT.tenant_id}:some-other-state".encode(),
        )


async def test_the_native_path_echoes_the_resource_in_the_exchange(db_sessionmaker) -> None:
    # Discovery path: the RFC 8707 binding rides BOTH legs - the consent URL
    # and the token exchange - naming the canonical MCP server URI.
    server = await _add_server(db_sessionmaker, base_url="https://mcp.example/tools/mcp")

    def discovery_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://mcp.example/.well-known/oauth-protected-resource/tools/mcp":
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example/tools/mcp",
                    "authorization_servers": ["https://as.example"],
                },
            )
        if url == "https://as.example/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "issuer": "https://as.example",
                    "authorization_endpoint": "https://as.example/authorize",
                    "token_endpoint": "https://as.example/token",
                },
            )
        return httpx.Response(404)

    async with (
        db_sessionmaker() as session,
        httpx.AsyncClient(transport=httpx.MockTransport(discovery_handler)) as client,
    ):
        url = await flow.start_connect(
            server,
            _settings(),
            TENANT,
            repo=OAuthAuthorizationRepository(session, TENANT),
            store=SecretStore(session, TENANT, CIPHER),
            cipher=CIPHER,
            client_id="cid",
            client_secret=None,
            scopes=["mcp"],
            endpoints=None,
            http_client=client,
        )
    assert "resource=https%3A%2F%2Fmcp.example%2Ftools%2Fmcp" in url
    captured: dict = {}
    async with _token_client(captured) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    assert captured["form"]["resource"] == ["https://mcp.example/tools/mcp"]


async def test_the_native_binding_uses_the_declared_resource_identifier(db_sessionmaker) -> None:
    # Asana's shape: the metadata declares .../v2 while the connection URL is
    # .../v2/mcp. The DECLARED identifier is what the authorization server
    # issues tokens for, so it - not a recomputation from the base URL - must
    # ride the consent URL, the exchange, and the sealed grant.
    server = await _add_server(db_sessionmaker, base_url="https://mcp.example/v2/mcp")

    def discovery_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://mcp.example/.well-known/oauth-protected-resource/v2/mcp":
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example/v2",
                    "authorization_servers": ["https://as.example"],
                },
            )
        if url == "https://as.example/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "issuer": "https://as.example",
                    "authorization_endpoint": "https://as.example/authorize",
                    "token_endpoint": "https://as.example/token",
                },
            )
        return httpx.Response(404)

    async with (
        db_sessionmaker() as session,
        httpx.AsyncClient(transport=httpx.MockTransport(discovery_handler)) as client,
    ):
        url = await flow.start_connect(
            server,
            _settings(),
            TENANT,
            repo=OAuthAuthorizationRepository(session, TENANT),
            store=SecretStore(session, TENANT, CIPHER),
            cipher=CIPHER,
            client_id="cid",
            client_secret=None,
            scopes=["default"],
            endpoints=None,
            http_client=client,
        )
    assert parse_qs(urlsplit(url).query)["resource"] == ["https://mcp.example/v2"]
    captured: dict = {}
    async with _token_client(captured) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    assert captured["form"]["resource"] == ["https://mcp.example/v2"]
    assert (await _stored_bundle(db_sessionmaker, server)).resource == "https://mcp.example/v2"


# --- disconnect -------------------------------------------------------------


async def test_disconnect_revokes_and_deletes(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    async with _token_client({}) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)

    revoked: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        revoked["url"] = str(request.url)
        revoked["form"] = parse_qs(request.content.decode())
        return httpx.Response(200)

    async with db_sessionmaker() as session:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            had = await flow.disconnect(
                server,
                _settings(),
                TENANT,
                store=SecretStore(session, TENANT, CIPHER),
                http_client=client,
            )
        await session.commit()
    assert had is True
    # RFC 7009: the refresh token (the wider grant) is what gets revoked.
    assert revoked["url"] == "https://as.example/revoke"
    assert revoked["form"]["token"] == ["rt-456"]
    async with db_sessionmaker() as session:
        with pytest.raises(SecretNotFoundError):
            await SecretStore(session, TENANT, CIPHER).reveal(
                oauth_token_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
            )


async def test_disconnect_with_nothing_connected_is_a_quiet_no_op(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)
    async with db_sessionmaker() as session:
        had = await flow.disconnect(
            server, _settings(), TENANT, store=SecretStore(session, TENANT, CIPHER)
        )
    assert had is False


async def test_disconnect_keeps_the_client_registration(db_sessionmaker) -> None:
    # Disconnect means "not connected", not "forget the app": the client
    # id/secret must survive so the same registration can grant again without
    # the operator re-entering it. forget_grant is the variant that erases it.
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server, secret="s3cret")
    async with _token_client({}) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        ) as revoker:
            await flow.disconnect(server, _settings(), TENANT, store=store, http_client=revoker)
        await session.commit()
    async with db_sessionmaker() as session:
        assert (
            await SecretStore(session, TENANT, CIPHER).reveal(
                oauth_client_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
            )
            == "s3cret"
        )


async def test_forget_grant_revokes_and_erases_both_secrets(db_sessionmaker) -> None:
    # The delete/re-point path. Both secret names derive from (id, base_url),
    # so once the row is gone or re-pointed NOTHING can name them again: a
    # surviving :oauth-client (which disconnect alone leaves behind) points at
    # a live registration no API or UI could ever list. Revoke, then erase both.
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server, secret="s3cret")
    async with _token_client({}) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)

    revoked: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        revoked["form"] = parse_qs(request.content.decode())
        return httpx.Response(200)

    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as revoker:
            await flow.forget_grant(server, _settings(), TENANT, store=store, http_client=revoker)
        await session.commit()

    # The revocation went out with the client secret the registration held...
    assert revoked["form"]["token"] == ["rt-456"]
    assert revoked["form"]["client_secret"] == ["s3cret"]
    # ...and NEITHER secret survives it.
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        for name in (
            oauth_token_secret_name(server.id, server.base_url),
            oauth_client_secret_name(server.id, server.base_url),
        ):
            with pytest.raises(SecretNotFoundError):
                await store.reveal(name, scope=SecretScope.WORKSPACE)


async def test_forget_grant_erases_locally_when_the_provider_is_unreachable(
    db_sessionmaker,
) -> None:
    # A provider that is down (or gone) must never make a grant undeletable -
    # the caller is usually mid-delete of the row that owns it. Revocation is
    # best effort; the local erase is not.
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server, secret="s3cret")
    async with _token_client({}) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(transport=httpx.MockTransport(dead)) as revoker:
            await flow.forget_grant(server, _settings(), TENANT, store=store, http_client=revoker)
        await session.commit()
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        for name in (
            oauth_token_secret_name(server.id, server.base_url),
            oauth_client_secret_name(server.id, server.base_url),
        ):
            with pytest.raises(SecretNotFoundError):
                await store.reveal(name, scope=SecretScope.WORKSPACE)


async def test_forget_grant_erases_with_no_master_key(db_sessionmaker) -> None:
    # A keyless deployment cannot reveal (so cannot revoke), but must still be
    # able to delete: the rows carry no cipher requirement.
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server, secret="s3cret")
    async with _token_client({}) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        # cipher=None is exactly what an unset SECRET_MASTER_KEY produces.
        await flow.forget_grant(
            server, _settings(), TENANT, store=SecretStore(session, TENANT, None)
        )
        await session.commit()
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        for name in (
            oauth_token_secret_name(server.id, server.base_url),
            oauth_client_secret_name(server.id, server.base_url),
        ):
            with pytest.raises(SecretNotFoundError):
                await store.reveal(name, scope=SecretScope.WORKSPACE)


# --- refresh ----------------------------------------------------------------


def _bundle(**overrides) -> OAuthTokenBundle:
    """A connected grant nearing expiry, overridable per test."""
    base = {
        "access_token": "at-old",
        "refresh_token": "rt-old",
        "expires_at": utcnow() + timedelta(seconds=30),
        "granted_scope": "repo",
        "token_endpoint": "https://as.example/token",
        "revocation_endpoint": "https://as.example/revoke",
        "client_id": "cid",
        "issuer": "https://as.example",
    }
    base.update(overrides)
    return OAuthTokenBundle(**base)


async def _seal_grant(db_sessionmaker, server, bundle: OAuthTokenBundle | None = None) -> str:
    """Seal a grant for ``server`` and return the secret name it lives under.

    Every renewal path is UPDATE-ONLY (a disconnect mid-renewal must not be
    undone), so a refresh test has to start from a grant that actually exists -
    which is also what the tool path always does: it loads the bundle from this
    very row before it renews it.
    """
    name = oauth_token_secret_name(server.id, server.base_url)
    async with db_sessionmaker() as session:
        await SecretStore(session, TENANT, CIPHER).store(
            name, (bundle or _bundle()).seal(), scope=SecretScope.WORKSPACE
        )
        await session.commit()
    return name


async def _stored_bundle(db_sessionmaker, server) -> OAuthTokenBundle:
    """Read back the sealed grant a refresh should (or should not) have replaced."""
    async with db_sessionmaker() as session:
        sealed = await SecretStore(session, TENANT, CIPHER).reveal(
            oauth_token_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
        )
    return OAuthTokenBundle.unseal(sealed)


async def test_refresh_renews_reseals_and_logs(db_sessionmaker, caplog) -> None:
    server = await _add_server(db_sessionmaker)
    await _seal_grant(db_sessionmaker, server)
    captured: dict = {}
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with _token_client(
            captured, {"access_token": "at-new", "expires_in": 3600}
        ) as client:
            with caplog.at_level("INFO"):
                renewed = await flow.refresh_grant(
                    server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=client
                )
        await session.commit()

    assert renewed.access_token == "at-new"
    # RFC 6749 section 6: a response with no refresh_token keeps the old one,
    # and an omitted scope means unchanged.
    assert renewed.refresh_token == "rt-old"
    assert renewed.granted_scope == "repo"
    assert renewed.expires_at > utcnow() + timedelta(seconds=3000)
    assert captured["form"]["grant_type"] == ["refresh_token"]
    assert captured["form"]["refresh_token"] == ["rt-old"]
    # The renewal is visible at rest and in the audit log.
    assert (await _stored_bundle(db_sessionmaker, server)).access_token == "at-new"
    assert any("oauth token renewed" in r.message for r in caplog.records)


async def test_refresh_repeats_the_resource_binding(db_sessionmaker) -> None:
    # RFC 8707: the audience binding rides EVERY token request, or a renewed
    # token could come back scoped to a different audience - and the renewed
    # bundle keeps it for the renewal after that.
    server = await _add_server(db_sessionmaker)
    await _seal_grant(db_sessionmaker, server, _bundle(resource="https://mcp.example/v2"))
    captured: dict = {}
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with _token_client(
            captured, {"access_token": "at-new", "expires_in": 3600}
        ) as client:
            await flow.refresh_grant(
                server,
                _settings(),
                TENANT,
                store=store,
                bundle=_bundle(resource="https://mcp.example/v2"),
                http_client=client,
            )
        await session.commit()
    assert captured["form"]["resource"] == ["https://mcp.example/v2"]
    assert (await _stored_bundle(db_sessionmaker, server)).resource == "https://mcp.example/v2"


async def test_refresh_without_a_binding_sends_no_resource(db_sessionmaker) -> None:
    # A generic provider's bundle has no resource; the refresh form must not
    # grow a spurious parameter the provider never saw at consent.
    server = await _add_server(db_sessionmaker)
    captured: dict = {}
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with _token_client(
            captured, {"access_token": "at-new", "expires_in": 3600}
        ) as client:
            await flow.refresh_grant(
                server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=client
            )
        await session.commit()
    assert "resource" not in captured["form"]


async def test_refresh_rotates_the_refresh_token_when_the_provider_sends_one(
    db_sessionmaker,
) -> None:
    server = await _add_server(db_sessionmaker)
    await _seal_grant(db_sessionmaker, server)
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with _token_client({}, {"access_token": "at-new", "refresh_token": "rt-new"}) as c:
            renewed = await flow.refresh_grant(
                server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=c
            )
        await session.commit()
    assert renewed.refresh_token == "rt-new"
    assert (await _stored_bundle(db_sessionmaker, server)).refresh_token == "rt-new"


async def test_refresh_sends_the_stored_client_secret(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)
    captured: dict = {}
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        await store.store(
            oauth_client_secret_name(server.id, server.base_url),
            "s3cret",
            scope=SecretScope.WORKSPACE,
        )
        await session.commit()
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with _token_client(captured) as client:
            await flow.refresh_grant(
                server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=client
            )
        await session.commit()
    assert captured["form"]["client_secret"] == ["s3cret"]


async def test_a_denied_refresh_marks_needs_reconnect_and_raises(db_sessionmaker) -> None:
    # The invalid_grant ERROR CODE (RFC 6749 section 5.2) means the refresh
    # token is dead, so the grant is re-sealed with needs_reconnect BEFORE
    # raising - no later turn should hammer the endpoint with a dead token.
    server = await _add_server(db_sessionmaker)
    await _seal_grant(db_sessionmaker, server)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(flow.OAuthRefreshDenied):
                await flow.refresh_grant(
                    server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=client
                )
        await session.commit()
    stored = await _stored_bundle(db_sessionmaker, server)
    assert stored.needs_reconnect is True
    assert stored.access_token == "at-old"


async def test_a_transient_refresh_failure_stores_nothing(db_sessionmaker) -> None:
    # A 5xx is the provider's outage, not a dead grant: no needs_reconnect, no
    # stored change - the next turn simply retries.
    server = await _add_server(db_sessionmaker)
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        await store.store(
            oauth_token_secret_name(server.id, server.base_url),
            _bundle().seal(),
            scope=SecretScope.WORKSPACE,
        )
        await session.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(OAuthFlowError) as excinfo:
                await flow.refresh_grant(
                    server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=client
                )
        await session.commit()
    assert not isinstance(excinfo.value, flow.OAuthRefreshDenied)
    stored = await _stored_bundle(db_sessionmaker, server)
    assert stored.needs_reconnect is False
    assert stored.access_token == "at-old"


async def test_refresh_without_a_refresh_token_raises_before_any_http(db_sessionmaker) -> None:
    server = await _add_server(db_sessionmaker)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call belongs here")

    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(OAuthFlowError):
                await flow.refresh_grant(
                    server,
                    _settings(),
                    TENANT,
                    store=store,
                    bundle=_bundle(refresh_token=None),
                    http_client=client,
                )


async def test_a_401_without_a_terminal_error_code_is_transient(db_sessionmaker) -> None:
    # invalid_client (bad client auth, a misconfig or a blip) is NOT a dead
    # grant: condemning it would cost a human re-consent, so it must stay
    # transient - nothing stored, retried next turn.
    server = await _add_server(db_sessionmaker)
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        await store.store(
            oauth_token_secret_name(server.id, server.base_url),
            _bundle().seal(),
            scope=SecretScope.WORKSPACE,
        )
        await session.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_client"})

    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(OAuthFlowError) as excinfo:
                await flow.refresh_grant(
                    server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=client
                )
        await session.commit()
    assert not isinstance(excinfo.value, flow.OAuthRefreshDenied)
    assert (await _stored_bundle(db_sessionmaker, server)).needs_reconnect is False


async def test_a_github_style_200_with_an_error_field_is_terminal(db_sessionmaker) -> None:
    # GitHub answers a dead refresh token with HTTP 200 and an error body -
    # judging by status alone would retry it forever. The error CODE decides.
    server = await _add_server(db_sessionmaker)
    await _seal_grant(db_sessionmaker, server)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "bad_refresh_token"})

    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(flow.OAuthRefreshDenied):
                await flow.refresh_grant(
                    server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=client
                )
        await session.commit()
    assert (await _stored_bundle(db_sessionmaker, server)).needs_reconnect is True


async def test_malformed_refresh_token_and_scope_keep_the_prior_values(db_sessionmaker) -> None:
    # model_copy skips validation, so a non-string refresh_token/scope from a
    # misbehaving provider would seal a bundle the NEXT unseal rejects -
    # bricking the grant a turn later. The known-good prior values win.
    server = await _add_server(db_sessionmaker)
    await _seal_grant(db_sessionmaker, server)
    payload = {"access_token": "at-new", "refresh_token": 123, "scope": ["repo", "read:org"]}
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with _token_client({}, payload) as client:
            renewed = await flow.refresh_grant(
                server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=client
            )
        await session.commit()
    assert renewed.refresh_token == "rt-old"
    assert renewed.granted_scope == "repo"
    # And the stored form round-trips: unseal (full validation) succeeds.
    assert (await _stored_bundle(db_sessionmaker, server)).access_token == "at-new"


async def test_start_forwards_extra_authorize_params_to_the_consent_url(
    db_sessionmaker,
) -> None:
    # The Google-preset path: non-RFC extras ride the consent URL verbatim,
    # and only that URL - the pending row and token exchange never see them.
    server = await _add_server(db_sessionmaker)
    url = await _start(
        db_sessionmaker,
        server,
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
    )
    params = parse_qs(urlsplit(url).query)
    assert params["access_type"] == ["offline"]
    assert params["prompt"] == ["consent"]


# --- the connected-account label --------------------------------------------


def _routed_client(userinfo: httpx.Response | None) -> httpx.AsyncClient:
    """A MockTransport serving the token exchange AND the identity lookup."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://as.example/token":
            return httpx.Response(200, json={"access_token": "at-123", "refresh_token": "rt-456"})
        if str(request.url) == "https://api.example/user":
            assert request.headers["Authorization"] == "Bearer at-123"
            return userinfo if userinfo is not None else httpx.Response(404)
        raise AssertionError(f"unexpected URL {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_callback_records_the_connected_account(db_sessionmaker) -> None:
    # The wrong-account guardrail: the callback asks the provider WHO was
    # connected and stores the label in the server's non-secret config, so
    # the card can say "Connected as octocat" without decrypting anything.
    server = await _add_server(db_sessionmaker)
    url = await _start(
        db_sessionmaker,
        server,
        userinfo_endpoint="https://api.example/user",
        userinfo_label_field="login",
    )
    async with _routed_client(httpx.Response(200, json={"login": "octocat"})) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT).get(server.id)
    assert row is not None
    assert row.config["connected_account"] == "octocat"


async def test_a_failed_identity_lookup_still_connects(db_sessionmaker) -> None:
    # The label is a nicety: a 500 from the identity endpoint must not fail
    # the callback or block the grant from sealing.
    server = await _add_server(db_sessionmaker)
    url = await _start(
        db_sessionmaker,
        server,
        userinfo_endpoint="https://api.example/user",
        userinfo_label_field="login",
    )
    async with _routed_client(httpx.Response(500)) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT).get(server.id)
        sealed = await SecretStore(session, TENANT, CIPHER).reveal(
            oauth_token_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
        )
    assert row is not None
    assert "connected_account" not in row.config
    assert OAuthTokenBundle.unseal(sealed).access_token == "at-123"


async def test_a_plain_http_identity_endpoint_is_refused_at_start(db_sessionmaker) -> None:
    # The callback sends the fresh ACCESS TOKEN to this URL, so it gets the
    # same hygiene bar as the token endpoint - and the refusal happens at
    # start, before any consent runs.
    server = await _add_server(db_sessionmaker)
    with pytest.raises(OAuthFlowError, match="identity endpoint"):
        await _start(
            db_sessionmaker,
            server,
            userinfo_endpoint="http://collector.example/user",
            userinfo_label_field="login",
        )


async def test_a_dotted_label_field_walks_a_nested_envelope(db_sessionmaker) -> None:
    # Asana wraps /users/me in a data envelope, so its label field is
    # data.name - the dotted path walks nested objects.
    server = await _add_server(db_sessionmaker)
    url = await _start(
        db_sessionmaker,
        server,
        userinfo_endpoint="https://api.example/user",
        userinfo_label_field="data.name",
    )
    async with _routed_client(httpx.Response(200, json={"data": {"name": "Priya K"}})) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT).get(server.id)
    assert row is not None
    assert row.config["connected_account"] == "Priya K"


async def test_a_dotted_label_field_over_a_flat_payload_stores_nothing(db_sessionmaker) -> None:
    # A non-object where the path expects one resolves to no label - never an
    # error, and never a stringified blob stored as the account line.
    server = await _add_server(db_sessionmaker)
    url = await _start(
        db_sessionmaker,
        server,
        userinfo_endpoint="https://api.example/user",
        userinfo_label_field="data.name",
    )
    async with _routed_client(httpx.Response(200, json={"data": "flat"})) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT).get(server.id)
    assert row is not None
    assert "connected_account" not in row.config


async def test_the_account_label_is_sanitized(db_sessionmaker) -> None:
    # A hostile provider's label could carry RTL overrides, newlines, or
    # zero-width characters to spoof the card line; only printable text
    # survives. (Delete the printable filter and the raw string stores.)
    server = await _add_server(db_sessionmaker)
    url = await _start(
        db_sessionmaker,
        server,
        userinfo_endpoint="https://api.example/user",
        userinfo_label_field="login",
    )
    hostile = "octo‮cat\n​"
    async with _routed_client(httpx.Response(200, json={"login": hostile})) as client:
        await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT).get(server.id)
    assert row is not None
    assert row.config["connected_account"] == "octocat"


# --- the SSRF anchor on the endpoints the flow POSTs to ---------------------


def _public_only(*hosts: str):
    """A host_is_public stand-in: only ``hosts`` resolve public."""

    async def fake_public(host: str) -> bool:
        return host in hosts

    return fake_public


async def _pending_row(db_sessionmaker, state: str):
    """The pending authorization row for ``state`` (for asserting or tampering)."""
    from sqlalchemy import select

    from app.models import OAuthAuthorization

    async with db_sessionmaker() as session:
        stmt = select(OAuthAuthorization).where(OAuthAuthorization.state == state)
        return (await session.execute(stmt)).scalar_one()


async def test_start_refuses_a_private_token_endpoint_for_a_public_server(
    db_sessionmaker, monkeypatch
) -> None:
    # The registered server is public, so the operator-supplied token endpoint
    # may not be an internal address: the exchange POSTs the client secret, the
    # code and the verifier there. Delete the require_fetchable call in
    # start_connect and the consent starts happily.
    monkeypatch.setattr(discovery_mod, "host_is_public", _public_only("mcp.example"))
    server = await _add_server(db_sessionmaker)
    internal = OAuthEndpoints(
        issuer="https://as.example",
        authorization_endpoint="https://as.example/authorize",
        token_endpoint="https://10.0.0.5/token",
        send_resource=False,
    )
    async with db_sessionmaker() as session:
        with pytest.raises(OAuthFlowError, match="token endpoint"):
            await flow.start_connect(
                server,
                _settings(),
                TENANT,
                repo=OAuthAuthorizationRepository(session, TENANT),
                store=SecretStore(session, TENANT, CIPHER),
                cipher=CIPHER,
                client_id="cid",
                client_secret=None,
                scopes=["repo"],
                endpoints=internal,
            )


async def test_start_allows_a_private_endpoint_for_a_private_server(db_sessionmaker) -> None:
    # The dev/compose case that must keep working: nothing resolves public
    # here, so the registered server is private and the admin may point the
    # consent at a loopback authorization server.
    server = await _add_server(db_sessionmaker, base_url="http://chassis.internal/mcp")
    local = OAuthEndpoints(
        issuer="http://localhost:9000",
        authorization_endpoint="http://localhost:9000/authorize",
        token_endpoint="http://localhost:9000/token",
        send_resource=False,
    )
    async with db_sessionmaker() as session:
        url = await flow.start_connect(
            server,
            _settings(),
            TENANT,
            repo=OAuthAuthorizationRepository(session, TENANT),
            store=SecretStore(session, TENANT, CIPHER),
            cipher=CIPHER,
            client_id="cid",
            client_secret=None,
            scopes=["repo"],
            endpoints=local,
        )
    assert url.startswith("http://localhost:9000/authorize?")
    # And the snapshot records the verdict the callback will re-apply.
    row = await _pending_row(db_sessionmaker, _state_from(url))
    assert row.endpoints["require_public_endpoints"] is False


async def test_the_callback_refuses_a_snapshot_that_names_an_internal_target(
    db_sessionmaker, monkeypatch
) -> None:
    # Defense in depth: the endpoints snapshot is a plain JSON blob on a row an
    # attacker with DB write could edit, and the exchange below carries the
    # client secret, the code and the verifier. The recorded public-host
    # verdict is re-applied before anything is POSTed. Remove that re-check and
    # the exchange fires at 10.0.0.5 - the handler asserts if it ever does.
    monkeypatch.setattr(discovery_mod, "host_is_public", _public_only("mcp.example", "as.example"))
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    state = _state_from(url)
    async with db_sessionmaker() as session:
        from sqlalchemy import select

        from app.models import OAuthAuthorization

        stmt = select(OAuthAuthorization).where(OAuthAuthorization.state == state)
        row = (await session.execute(stmt)).scalar_one()
        assert row.endpoints["require_public_endpoints"] is True
        row.endpoints = {**row.endpoints, "token_endpoint": "https://10.0.0.5/collect"}
        session.add(row)
        await session.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"the exchange must never reach {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthFlowError, match="validation"):
            await _callback(db_sessionmaker, state, client=client)


async def test_a_private_identity_endpoint_is_refused_for_a_public_server(
    db_sessionmaker, monkeypatch
) -> None:
    # The identity lookup sends the FRESH ACCESS TOKEN, so it is held to the
    # same anchor as the token endpoint - and refused at start, before consent.
    monkeypatch.setattr(discovery_mod, "host_is_public", _public_only("mcp.example", "as.example"))
    server = await _add_server(db_sessionmaker)
    with pytest.raises(OAuthFlowError, match="identity endpoint"):
        await _start(
            db_sessionmaker,
            server,
            userinfo_endpoint="https://192.168.0.10/whoami",
            userinfo_label_field="login",
        )


# --- the per-user OAuth invariant on the connect paths -----------------------


async def test_start_refuses_a_per_user_row(db_sessionmaker) -> None:
    # oauth + credential_scope=user is refused at create and at patch; connect
    # is the third write path into it, and the one that mints a real grant.
    server = await _add_server(db_sessionmaker, credential_scope="user")
    with pytest.raises(OAuthFlowError, match="per-user"):
        await _start(db_sessionmaker, server)


async def test_the_callback_refuses_a_row_rescoped_mid_consent(db_sessionmaker) -> None:
    # The row is editable while the consent is open, and the callback is what
    # flips auth_type to oauth - so the forbidden combination would be minted
    # right there. Nothing is stored and the row keeps its prior auth_type.
    server = await _add_server(db_sessionmaker)
    url = await _start(db_sessionmaker, server)
    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT).get(server.id)
        row.credential_scope = "user"
        session.add(row)
        await session.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no exchange belongs here")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthFlowError, match="per-user"):
            await _callback(db_sessionmaker, _state_from(url), client=client)
    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT).get(server.id)
        with pytest.raises(SecretNotFoundError):
            await SecretStore(session, TENANT, CIPHER).reveal(
                oauth_token_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
            )
    assert row is not None and row.auth_type == McpAuthType.HEADER.value


# --- disconnect versus renewal ----------------------------------------------


async def test_a_renewal_into_a_disconnected_grant_stores_nothing(db_sessionmaker) -> None:
    # The interleaving: a renewal reads the bundle, the admin's Disconnect
    # revokes and deletes it, and only THEN does the renewal's write land. An
    # upsert would re-insert a grant holding the NEW, still-live tokens - a
    # workspace credential to an account the admin believes is severed. The
    # write is update-only, so the deleted row stays deleted.
    server = await _add_server(db_sessionmaker)
    name = await _seal_grant(db_sessionmaker, server)
    in_flight = _bundle()  # what the renewal read before the disconnect landed

    async def handler(request: httpx.Request) -> httpx.Response:
        # The disconnect lands while the token endpoint is answering.
        async with db_sessionmaker() as session:
            await SecretStore(session, TENANT, CIPHER).delete(name, scope=SecretScope.WORKSPACE)
            await session.commit()
        return httpx.Response(200, json={"access_token": "at-new", "expires_in": 3600})

    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            renewed = await flow.refresh_grant(
                server, _settings(), TENANT, store=store, bundle=in_flight, http_client=client
            )
        await session.commit()
    # The in-flight turn still got its token (it was revoked upstream anyway)...
    assert renewed.access_token == "at-new"
    # ...but nothing durable came back from the dead.
    async with db_sessionmaker() as session:
        with pytest.raises(SecretNotFoundError):
            await SecretStore(session, TENANT, CIPHER).reveal(name, scope=SecretScope.WORKSPACE)


async def test_a_denied_refresh_marker_does_not_resurrect_a_deleted_grant(
    db_sessionmaker,
) -> None:
    # The same rule on the terminal leg: marking a grant needs_reconnect must
    # not RE-CREATE one the admin already disconnected.
    server = await _add_server(db_sessionmaker)
    name = oauth_token_secret_name(server.id, server.base_url)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(flow.OAuthRefreshDenied):
                await flow.refresh_grant(
                    server, _settings(), TENANT, store=store, bundle=_bundle(), http_client=client
                )
        await session.commit()
    async with db_sessionmaker() as session:
        with pytest.raises(SecretNotFoundError):
            await SecretStore(session, TENANT, CIPHER).reveal(name, scope=SecretScope.WORKSPACE)


async def test_disconnect_waits_for_an_in_flight_renewal(db_sessionmaker) -> None:
    # Both paths must serialize on the SAME lock, or a disconnect can slip
    # between a renewal's read and its write. The tool path takes it from
    # app.services.oauth.locks; so must disconnect. Remove the lock from
    # disconnect and the task completes while the lock is still held.
    server = await _add_server(db_sessionmaker)
    name = oauth_token_secret_name(server.id, server.base_url)
    async with db_sessionmaker() as session:
        store = SecretStore(session, TENANT, CIPHER)
        async with locks.grant_lock(name):  # stands in for the renewal block
            task = asyncio.create_task(flow.disconnect(server, _settings(), TENANT, store=store))
            await asyncio.sleep(0.05)
            assert not task.done(), "disconnect ran while a renewal held the grant lock"
        assert await task is False


async def test_a_public_server_may_not_point_a_credential_bearing_endpoint_inward(
    db_sessionmaker, monkeypatch
) -> None:
    # The SSRF the anchor exists for. A registered server that resolves
    # publicly may not name a PRIVATE token endpoint - the exchange POSTs the
    # client secret, the code and the PKCE verifier there, so an accepted
    # internal target is blind SSRF with credential exfiltration. Only the
    # scheme rule ran before, and it permits any https host.
    # Everything resolves publicly EXCEPT the endpoint under test, so the only
    # guard that can refuse this is the token endpoint's own. An earlier draft
    # left the authorization host private too, and passed with the token check
    # deleted - a different guard was doing the work.
    async def fake_host_is_public(host: str) -> bool:
        return host != "10.0.0.5"

    monkeypatch.setattr(discovery_mod, "host_is_public", fake_host_is_public)
    server = await _add_server(db_sessionmaker)
    inward = OAuthEndpoints(
        issuer="https://as.example",
        authorization_endpoint="https://as.example/authorize",
        token_endpoint="https://10.0.0.5/collect",
        send_resource=False,
    )
    with pytest.raises(OAuthFlowError, match="refused"):
        await _start(db_sessionmaker, server, endpoints=inward)


async def test_a_private_server_may_still_name_private_endpoints(
    db_sessionmaker, monkeypatch
) -> None:
    # The other half, and the reason the anchor is relative rather than
    # absolute: a sidecar on the compose network or a loopback dev stack is
    # itself private, so its authorization server is too. Anchoring on
    # "public targets only" would make local development unconnectable.
    async def fake_host_is_public(host: str) -> bool:
        return False

    monkeypatch.setattr(discovery_mod, "host_is_public", fake_host_is_public)
    server = await _add_server(db_sessionmaker, base_url="https://chassis.internal/mcp")
    inward = OAuthEndpoints(
        issuer="https://as.internal",
        authorization_endpoint="https://as.internal/authorize",
        token_endpoint="https://as.internal/token",
        send_resource=False,
    )
    url = await _start(db_sessionmaker, server, endpoints=inward)
    assert url.startswith("https://as.internal/authorize?")
