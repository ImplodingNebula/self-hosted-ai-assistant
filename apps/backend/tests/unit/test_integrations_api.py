"""The Integrations write API (slice 6b PR3).

Drives the real endpoints over the in-process app (ASGITransport + the sqlite
sessionmaker), overriding only the auth seam (``get_current_tenant``) and
``get_settings`` to exercise the admin gate, tenant isolation, the credential
scope rules, and the 503-without-a-master-key path. Asserts what actually crosses
the wire - in particular that a credential VALUE never appears in any response.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_tenant, get_settings
from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.main import create_app

pytestmark = pytest.mark.anyio

SERVERS = "/api/v1/integrations/servers"
MASTER_KEY = base64.b64encode(b"k" * 32).decode()

# A member of the same tenant, and an admin of a DIFFERENT tenant.
MEMBER = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
OTHER_TENANT = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="admin")

CREATE = {"name": "github", "base_url": "https://mcp.example/github/mcp"}


def _settings(*, master_key: str | None = None, **overrides) -> Settings:
    return Settings(
        _env_file=None, model_provider="fake", secret_master_key=master_key, **overrides
    )


class _SpyProvider:
    """Records invalidate() calls; the endpoints only ever call invalidate on it."""

    def __init__(self) -> None:
        self.invalidated: list[tuple] = []

    def invalidate(self, tenant_id, server_id) -> None:
        self.invalidated.append((tenant_id, server_id))


@asynccontextmanager
async def integrations_client(db_sessionmaker, *, tenant=None, settings=None, provider=None):
    """An app-wired client; override the tenant/settings/source-provider as needed."""
    app = create_app()
    app.state.db_sessionmaker = db_sessionmaker
    if provider is not None:
        app.state.mcp_source_provider = provider
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings
    if tenant is not None:
        app.dependency_overrides[get_current_tenant] = lambda: tenant
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def _create(client, **overrides) -> dict:
    body = {**CREATE, **overrides}
    response = await client.post(SERVERS, json=body)
    assert response.status_code == 201, response.text
    return response.json()


# --- server CRUD -----------------------------------------------------------


async def test_create_and_get_returns_metadata_only(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        created = await _create(client)
        assert created["name"] == "github"
        assert created["credential_scope"] == "workspace"
        # No credential material ever appears in a response.
        for leaked in ("value", "ciphertext", "credential", "tenant_id"):
            assert leaked not in created

        got = await client.get(f"{SERVERS}/{created['id']}")
        assert got.status_code == 200
        assert got.json()["base_url"] == CREATE["base_url"]


async def test_create_location_header(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        response = await client.post(SERVERS, json=CREATE)
        assert response.headers["Location"] == f"{SERVERS}/{response.json()['id']}"


async def test_create_duplicate_name_conflicts(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        await _create(client)
        again = await client.post(SERVERS, json=CREATE)
        assert again.status_code == 409


async def test_auth_type_defaults_to_header_and_round_trips(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        created = await _create(client)
        # The default is the pre-column behaviour - an unmodified client's
        # servers keep sending their stored or env bearer exactly as before.
        assert created["auth_type"] == "header"

        explicit = await _create(client, name="open-server", auth_type="none")
        assert explicit["auth_type"] == "none"
        got = await client.get(f"{SERVERS}/{explicit['id']}")
        assert got.json()["auth_type"] == "none"


async def test_patch_auth_type_persists_and_invalidates(db_sessionmaker) -> None:
    spy = _SpyProvider()
    async with integrations_client(db_sessionmaker, provider=spy) as client:
        created = await _create(client)
        patched = await client.patch(f"{SERVERS}/{created['id']}", json={"auth_type": "none"})
        assert patched.status_code == 200, patched.text
        assert patched.json()["auth_type"] == "none"
        # The cached source rebuilds from the committed row on the next turn.
        assert spy.invalidated


async def test_create_rejects_an_unknown_auth_type(db_sessionmaker) -> None:
    # oauth became a real value with the consent flow; a value no build knows
    # is still a clean 422, never a row that fails open later.
    async with integrations_client(db_sessionmaker) as client:
        bad = await client.post(SERVERS, json={**CREATE, "auth_type": "saml"})
        assert bad.status_code == 422


async def test_oauth_connect_returns_a_consent_url(db_sessionmaker) -> None:
    key = base64.b64encode(b"k" * 32).decode()
    async with integrations_client(db_sessionmaker, settings=_settings(master_key=key)) as client:
        created = await _create(client)
        resp = await client.post(
            f"{SERVERS}/{created['id']}/oauth/connect",
            json={
                "client_id": "cid",
                "scopes": ["repo"],
                "endpoints": {
                    "issuer": "https://as.example",
                    "authorization_endpoint": "https://as.example/authorize",
                    "token_endpoint": "https://as.example/token",
                },
            },
        )
        assert resp.status_code == 200, resp.text
        url = resp.json()["authorization_url"]
        assert url.startswith("https://as.example/authorize?")
        assert "code_challenge=" in url and "state=" in url
        # The bypass path binds no resource - generic providers do not take it.
        assert "resource=" not in url


async def test_oauth_connect_without_a_master_key_is_503(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        created = await _create(client)
        resp = await client.post(
            f"{SERVERS}/{created['id']}/oauth/connect",
            json={"client_id": "cid", "scopes": ["repo"]},
        )
        assert resp.status_code == 503


async def test_oauth_connect_is_admin_gated(db_sessionmaker) -> None:
    key = base64.b64encode(b"k" * 32).decode()
    async with integrations_client(db_sessionmaker, settings=_settings(master_key=key)) as admin:
        created = await _create(admin)
    async with integrations_client(
        db_sessionmaker, tenant=MEMBER, settings=_settings(master_key=key)
    ) as member:
        resp = await member.post(
            f"{SERVERS}/{created['id']}/oauth/connect",
            json={"client_id": "cid", "scopes": ["repo"]},
        )
        assert resp.status_code == 403


async def test_oauth_callback_redirects_clean_with_no_query_string(db_sessionmaker) -> None:
    # Whatever happens - denial, junk state, no code - the answer is a 302 to
    # the Integrations page with NOTHING appended: the authorization code and
    # state must not ride into another access log.
    async with integrations_client(db_sessionmaker) as client:
        for query in (
            "",
            "?state=junk",
            "?state=junk&error=access_denied",
            # A code with no master key configured: the exchange cannot run,
            # but the browser still gets the clean redirect - never a 503
            # describing the deployment's configuration. (This case is the
            # mutation guard for the keyless-callback fix.)
            "?state=junk&code=abc123",
        ):
            resp = await client.get(f"/api/v1/integrations/oauth/callback{query}")
            assert resp.status_code == 302
            location = resp.headers["location"]
            assert location == "http://localhost:3000/console/integrations"


async def test_an_oauth_server_reports_its_grant_status(db_sessionmaker) -> None:
    # The card's "Connected" line: metadata only - the listing never decrypts
    # the grant, and a keyless deployment can still answer it.
    from datetime import UTC, datetime
    from uuid import UUID
    from uuid import uuid4 as _uuid4

    from app.providers.secrets import SecretScope, build_cipher
    from app.services.oauth.bundle import OAuthTokenBundle, oauth_token_secret_name
    from app.services.secret_store import SecretStore

    key = base64.b64encode(b"k" * 32).decode()
    async with integrations_client(db_sessionmaker, settings=_settings(master_key=key)) as client:
        created = await _create(client, name="gh", auth_type="oauth")
        before = await client.get(f"{SERVERS}/{created['id']}")
        assert before.json()["oauth_connected"] is False
        assert before.json()["oauth_connected_at"] is None

        bundle = OAuthTokenBundle(
            access_token="at",
            expires_at=datetime(2030, 1, 1, tzinfo=UTC),
            granted_scope="repo",
            token_endpoint="https://as.example/token",
            client_id="cid",
            issuer="https://as.example",
        )
        async with db_sessionmaker() as session:
            actor = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=_uuid4(), role="admin")
            await SecretStore(session, actor, build_cipher(key)).store(
                oauth_token_secret_name(UUID(created["id"]), created["base_url"]),
                bundle.seal(),
                scope=SecretScope.WORKSPACE,
            )
            await session.commit()

        after = await client.get(f"{SERVERS}/{created['id']}")
        assert after.json()["oauth_connected"] is True
        assert after.json()["oauth_connected_at"] is not None
        # And the list endpoint agrees with the detail one.
        listed = await client.get(SERVERS)
        row = next(r for r in listed.json()["items"] if r["id"] == created["id"])
        assert row["oauth_connected"] is True


async def test_grant_status_enriches_patch_responses_and_keyless_listings(db_sessionmaker) -> None:
    # The two serializer paths the detail/list test skips: a PATCH response is
    # enriched too, and a deployment with NO master key still lists connected
    # status (the read is metadata-only, it never touches the cipher).
    from datetime import UTC, datetime
    from uuid import UUID
    from uuid import uuid4 as _uuid4

    from app.providers.secrets import SecretScope, build_cipher
    from app.services.oauth.bundle import OAuthTokenBundle, oauth_token_secret_name
    from app.services.secret_store import SecretStore

    key = base64.b64encode(b"k" * 32).decode()
    async with integrations_client(db_sessionmaker, settings=_settings(master_key=key)) as client:
        created = await _create(client, name="gh", auth_type="oauth")
        bundle = OAuthTokenBundle(
            access_token="at",
            expires_at=datetime(2030, 1, 1, tzinfo=UTC),
            granted_scope="repo",
            token_endpoint="https://as.example/token",
            client_id="cid",
            issuer="https://as.example",
        )
        async with db_sessionmaker() as session:
            actor = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=_uuid4(), role="admin")
            await SecretStore(session, actor, build_cipher(key)).store(
                oauth_token_secret_name(UUID(created["id"]), created["base_url"]),
                bundle.seal(),
                scope=SecretScope.WORKSPACE,
            )
            await session.commit()

        patched = await client.patch(f"{SERVERS}/{created['id']}", json={"enabled": False})
        assert patched.status_code == 200
        assert patched.json()["oauth_connected"] is True
        assert patched.json()["oauth_connected_at"] is not None

    # Same rows, no master key: the listing must still answer, enriched.
    async with integrations_client(db_sessionmaker, settings=_settings(master_key=None)) as keyless:
        listed = await keyless.get(SERVERS)
        row = next(r for r in listed.json()["items"] if r["id"] == created["id"])
        assert row["oauth_connected"] is True
        assert row["oauth_connected_at"] is not None


async def test_oauth_disconnect_is_a_204(db_sessionmaker) -> None:
    key = base64.b64encode(b"k" * 32).decode()
    async with integrations_client(db_sessionmaker, settings=_settings(master_key=key)) as client:
        created = await _create(client)
        resp = await client.post(f"{SERVERS}/{created['id']}/oauth/disconnect")
        assert resp.status_code == 204


async def test_create_rejects_bad_url_and_missing_name(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        bad_url = await client.post(SERVERS, json={"name": "x", "base_url": "ftp://nope"})
        assert bad_url.status_code == 422
        no_name = await client.post(SERVERS, json={"base_url": "https://x/mcp"})
        assert no_name.status_code == 422


async def test_list_includes_disabled_name_ordered(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        a = await _create(client, name="alpha", base_url="https://a/mcp")
        await client.patch(f"{SERVERS}/{a['id']}", json={"enabled": False})
        await _create(client, name="beta", base_url="https://b/mcp")

        items = (await client.get(SERVERS)).json()["items"]
        assert [s["name"] for s in items] == ["alpha", "beta"]
        assert next(s for s in items if s["name"] == "alpha")["enabled"] is False


async def test_update_disable_repoint_rescope(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        server = await _create(client)
        patched = await client.patch(
            f"{SERVERS}/{server['id']}",
            json={"enabled": False, "base_url": "https://new/mcp", "credential_scope": "user"},
        )
        assert patched.status_code == 200
        body = patched.json()
        assert body["enabled"] is False
        assert body["base_url"] == "https://new/mcp"
        assert body["credential_scope"] == "user"


async def test_delete_then_gone(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        server = await _create(client)
        assert (await client.delete(f"{SERVERS}/{server['id']}")).status_code == 204
        assert (await client.get(f"{SERVERS}/{server['id']}")).status_code == 404


async def test_unknown_and_malformed_ids_are_404(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        unknown = str(uuid4())
        assert (await client.get(f"{SERVERS}/{unknown}")).status_code == 404
        assert (await client.get(f"{SERVERS}/not-a-uuid")).status_code == 404
        assert (
            await client.patch(f"{SERVERS}/{unknown}", json={"enabled": True})
        ).status_code == 404
        assert (await client.delete(f"{SERVERS}/{unknown}")).status_code == 404


# --- tenant isolation + admin gate ----------------------------------------


async def test_a_server_is_invisible_to_another_tenant(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as owner:
        server = await _create(owner)
    async with integrations_client(db_sessionmaker, tenant=OTHER_TENANT) as other:
        assert (await other.get(f"{SERVERS}/{server['id']}")).status_code == 404
        assert (
            await other.patch(f"{SERVERS}/{server['id']}", json={"enabled": False})
        ).status_code == 404
        assert (await other.delete(f"{SERVERS}/{server['id']}")).status_code == 404
        assert (await other.get(SERVERS)).json()["items"] == []


async def test_a_member_cannot_mutate_but_can_list(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as admin:
        server = await _create(admin)
    async with integrations_client(db_sessionmaker, tenant=MEMBER) as member:
        create = await member.post(SERVERS, json={"name": "z", "base_url": "https://z/mcp"})
        assert create.status_code == 403
        assert (
            await member.patch(f"{SERVERS}/{server['id']}", json={"enabled": False})
        ).status_code == 403
        assert (await member.delete(f"{SERVERS}/{server['id']}")).status_code == 403
        test = await member.post(f"{SERVERS}/test-connection", json={"base_url": "https://z/mcp"})
        assert test.status_code == 403
        # Reads are allowed.
        assert (await member.get(SERVERS)).status_code == 200


async def test_saved_test_connection_is_admin_gated_for_a_workspace_server(db_sessionmaker) -> None:
    # Re-testing a workspace server uses its shared credential, so it is admin-only
    # (same gate as the credential ops), never a dial for a member.
    async with integrations_client(db_sessionmaker) as admin:
        server = await _create(admin)
    async with integrations_client(db_sessionmaker, tenant=MEMBER) as member:
        resp = await member.post(f"{SERVERS}/{server['id']}/test-connection", json={})
        assert resp.status_code == 403


async def test_saved_test_connection_missing_server_is_404(db_sessionmaker) -> None:
    async with integrations_client(db_sessionmaker) as client:
        resp = await client.post(f"{SERVERS}/{uuid4()}/test-connection", json={})
        assert resp.status_code == 404


async def test_saved_test_connection_member_can_test_a_user_scoped_server(db_sessionmaker) -> None:
    # A user-scoped server is the member's own to test. The probe is always 200 - a
    # down target is a diagnostic - here unreachable at a closed local port (no
    # network, no credential connected, so it degrades to an unauthenticated dial).
    async with integrations_client(db_sessionmaker) as admin:
        server = await _create(
            admin, name="user-srv", base_url="http://127.0.0.1:9/mcp", credential_scope="user"
        )
    async with integrations_client(db_sessionmaker, tenant=MEMBER) as member:
        resp = await member.post(f"{SERVERS}/{server['id']}/test-connection", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] in ("unreachable", "timeout")


# --- credentials -----------------------------------------------------------


async def test_credential_write_and_read_never_leak_plaintext(db_sessionmaker) -> None:
    settings = _settings(master_key=MASTER_KEY)
    async with integrations_client(db_sessionmaker, settings=settings) as client:
        server = await _create(client)
        written = await client.put(
            f"{SERVERS}/{server['id']}/credential", json={"value": "ghp_secrettoken1234"}
        )
        assert written.status_code == 200
        body = written.json()
        assert body["connected"] is True
        assert body["last4"] == "1234"
        assert body["scope"] == "workspace"
        assert "value" not in body

        read = await client.get(f"{SERVERS}/{server['id']}/credential")
        assert read.status_code == 200
        assert read.json()["connected"] is True
        assert read.json()["last4"] == "1234"
        assert "value" not in read.json()


async def test_credential_read_reports_not_connected(db_sessionmaker) -> None:
    settings = _settings(master_key=MASTER_KEY)
    async with integrations_client(db_sessionmaker, settings=settings) as client:
        server = await _create(client)
        read = await client.get(f"{SERVERS}/{server['id']}/credential")
        assert read.status_code == 200
        assert read.json()["connected"] is False


async def test_credential_write_is_503_without_a_master_key(db_sessionmaker) -> None:
    settings = _settings(master_key=None)
    async with integrations_client(db_sessionmaker, settings=settings) as client:
        server = await _create(client)
        written = await client.put(
            f"{SERVERS}/{server['id']}/credential", json={"value": "whatever12345"}
        )
        assert written.status_code == 503
        # Server CRUD does not need the key and still works.
        assert (await client.get(f"{SERVERS}/{server['id']}")).status_code == 200


async def test_the_503_carries_the_operator_remedy(db_sessionmaker) -> None:
    """Naming the variable is not enough: an unset key is fixed in the deployment.

    Nobody can act on "SECRET_MASTER_KEY is not configured" from the page that
    reports it, so the detail has to say where the fix lives.
    """
    settings = _settings(master_key=None)
    async with integrations_client(db_sessionmaker, settings=settings) as client:
        server = await _create(client)
        written = await client.put(f"{SERVERS}/{server['id']}/credential", json={"value": "v12345"})
        assert written.status_code == 503
        detail = written.json()["detail"]
        assert "SECRET_MASTER_KEY" in detail
        assert "deployment environment" in detail
        assert "restart the backend" in detail

        # The OAuth path answers the same way, for the same reason.
        started = await client.post(
            f"{SERVERS}/{server['id']}/oauth/connect",
            json={"client_id": "cid", "scopes": ["repo"]},
        )
        assert started.status_code == 503
        assert "deployment environment" in started.json()["detail"]
        assert "restart the backend" in started.json()["detail"]


async def test_the_list_reports_whether_credentials_can_be_stored(db_sessionmaker) -> None:
    """The deployment fact rides on the list, so a client says so BEFORE a write.

    Without it the only way to learn that storage is off is to register a server
    and fail the credential write - which leaves the server registered and
    authenticating against nothing.
    """
    async with integrations_client(db_sessionmaker, settings=_settings(master_key=None)) as off:
        await _create(off)
        assert (await off.get(SERVERS)).json()["credential_storage_available"] is False

    # The SAME rows, read by a deployment that CAN store: only the flag moves.
    async with integrations_client(
        db_sessionmaker, settings=_settings(master_key=MASTER_KEY)
    ) as on:
        listing = (await on.get(SERVERS)).json()
        assert listing["credential_storage_available"] is True
        # The flag describes the DEPLOYMENT, not any server: it reads true while
        # the one registered server still has no credential of its own.
        connected = await on.get(f"{SERVERS}/{listing['items'][0]['id']}/credential")
        assert connected.json()["connected"] is False


async def test_credential_state_answers_for_every_auth_type(db_sessionmaker) -> None:
    """One field covers header, oauth and none, so a listing needs no per-row fetch.

    Before this, the list reported grant status for oauth rows and nothing for
    header rows, so "does this server have a credential" was invisible until a
    card was expanded - and every row looked alike.
    """
    settings = _settings(master_key=MASTER_KEY)
    async with integrations_client(db_sessionmaker, settings=settings) as client:
        header = await _create(client, name="h", base_url="https://h.example/mcp")
        oauth = await _create(client, name="o", base_url="https://o.example/mcp", auth_type="oauth")
        bare = await _create(client, name="n", base_url="https://n.example/mcp", auth_type="none")

        by_id = {
            row["id"]: row["credential_state"]
            for row in (await client.get(SERVERS)).json()["items"]
        }
        assert by_id[header["id"]] == "missing"
        assert by_id[oauth["id"]] == "missing", "no grant yet"
        assert by_id[bare["id"]] == "not_required", "auth_type none is never a warning"

        # Storing one flips exactly that row, on the list and the single read.
        await client.put(f"{SERVERS}/{header['id']}/credential", json={"value": "tok_12345"})
        after = {
            row["id"]: row["credential_state"]
            for row in (await client.get(SERVERS)).json()["items"]
        }
        assert after[header["id"]] == "connected"
        assert after[oauth["id"]] == "missing"
        assert (await client.get(f"{SERVERS}/{header['id']}")).json()["credential_state"] == (
            "connected"
        )


async def test_the_env_backed_chassis_is_named_not_reported_as_connected(
    db_sessionmaker,
) -> None:
    """The bootstrap key is a real credential and a fragile one; it says which it is.

    ``_workspace_bearer`` backs ONLY a server pointed at the chassis URL with the
    env key. Reporting that row as plain "connected" would hide that it breaks
    when the deployment drops the variable; reporting it as "missing" would be a
    lie about a server whose calls work.
    """
    settings = _settings(
        master_key=MASTER_KEY,
        mcp_chassis_url="http://chassis.internal:8765/mcp",
        mcp_chassis_api_key="bootstrap-key",
    )
    async with integrations_client(db_sessionmaker, settings=settings) as client:
        seeded = await _create(client, name="c", base_url="http://chassis.internal:8765/mcp")
        other = await _create(client, name="x", base_url="http://other.internal:8765/mcp")
        states = {
            r["id"]: r["credential_state"] for r in (await client.get(SERVERS)).json()["items"]
        }
        assert states[seeded["id"]] == "deployment_key"
        # The env key belongs to ONE url; an unrelated server never inherits it.
        assert states[other["id"]] == "missing"

        # A stored credential takes over, exactly as the chat path resolves it.
        await client.put(f"{SERVERS}/{seeded['id']}/credential", json={"value": "stored_9999"})
        assert (await client.get(f"{SERVERS}/{seeded['id']}")).json()["credential_state"] == (
            "connected"
        )


async def test_a_malformed_master_key_reads_as_unavailable_not_available(db_sessionmaker) -> None:
    """A key that is SET but unusable must not advertise storage as working.

    The cipher builder degrades a bad key to None; the flag has to follow that
    same judgement, or the page invites an admin into a write that 503s.
    """
    async with integrations_client(
        db_sessionmaker, settings=_settings(master_key="not-base64-at-all!!")
    ) as client:
        assert (await client.get(SERVERS)).json()["credential_storage_available"] is False


async def test_a_member_manages_a_user_scoped_credential(db_sessionmaker) -> None:
    settings = _settings(master_key=MASTER_KEY)
    async with integrations_client(db_sessionmaker, settings=settings) as admin:
        server = await _create(admin, name="u", base_url="https://u/mcp", credential_scope="user")
    async with integrations_client(db_sessionmaker, settings=settings, tenant=MEMBER) as member:
        written = await member.put(
            f"{SERVERS}/{server['id']}/credential", json={"value": "member_token_123"}
        )
        assert written.status_code == 200
        assert written.json()["scope"] == "user"


async def test_a_member_cannot_set_a_workspace_credential(db_sessionmaker) -> None:
    settings = _settings(master_key=MASTER_KEY)
    async with integrations_client(db_sessionmaker, settings=settings) as admin:
        server = await _create(admin)  # workspace scope by default
    async with integrations_client(db_sessionmaker, settings=settings, tenant=MEMBER) as member:
        written = await member.put(
            f"{SERVERS}/{server['id']}/credential", json={"value": "member_token_123"}
        )
        assert written.status_code == 403


async def test_a_member_cannot_read_or_clear_a_workspace_credential(db_sessionmaker) -> None:
    # The scope gate applies to GET and DELETE too, not just the write - a member
    # must not read a workspace credential's hint or wipe the tenant's shared one.
    settings = _settings(master_key=MASTER_KEY)
    async with integrations_client(db_sessionmaker, settings=settings) as admin:
        server = await _create(admin)  # workspace scope
    async with integrations_client(db_sessionmaker, settings=settings, tenant=MEMBER) as member:
        assert (await member.get(f"{SERVERS}/{server['id']}/credential")).status_code == 403
        assert (await member.delete(f"{SERVERS}/{server['id']}/credential")).status_code == 403


async def test_a_member_clears_their_own_user_credential(db_sessionmaker) -> None:
    settings = _settings(master_key=MASTER_KEY)
    async with integrations_client(db_sessionmaker, settings=settings) as admin:
        server = await _create(admin, name="u", base_url="https://u/mcp", credential_scope="user")
    async with integrations_client(db_sessionmaker, settings=settings, tenant=MEMBER) as member:
        await member.put(f"{SERVERS}/{server['id']}/credential", json={"value": "member_token_123"})
        assert (await member.delete(f"{SERVERS}/{server['id']}/credential")).status_code == 204
        assert (await member.get(f"{SERVERS}/{server['id']}/credential")).json()[
            "connected"
        ] is False


async def test_credential_delete_invalidates_only_for_workspace_scope(db_sessionmaker) -> None:
    settings = _settings(master_key=MASTER_KEY)
    # A workspace credential delete invalidates the cached source (it fed the
    # shared-header path); a user credential delete does not (revealed per call).
    workspace_spy = _SpyProvider()
    async with integrations_client(db_sessionmaker, settings=settings, provider=workspace_spy) as c:
        server = await _create(c)
        await c.put(f"{SERVERS}/{server['id']}/credential", json={"value": "workspace_tok"})
        workspace_spy.invalidated.clear()  # ignore the write's invalidation; test the DELETE
        await c.delete(f"{SERVERS}/{server['id']}/credential")
    assert len(workspace_spy.invalidated) == 1

    user_spy = _SpyProvider()
    async with integrations_client(db_sessionmaker, settings=settings) as admin:
        server2 = await _create(admin, name="u", base_url="https://u/mcp", credential_scope="user")
    async with integrations_client(
        db_sessionmaker, settings=settings, tenant=MEMBER, provider=user_spy
    ) as member:
        await member.put(f"{SERVERS}/{server2['id']}/credential", json={"value": "member_token_1"})
        await member.delete(f"{SERVERS}/{server2['id']}/credential")
    assert user_spy.invalidated == []


async def test_create_rejects_a_base_url_with_userinfo(db_sessionmaker) -> None:
    # A url embedding a credential would land verbatim in the audit log; reject it.
    async with integrations_client(db_sessionmaker) as client:
        response = await client.post(
            SERVERS, json={"name": "x", "base_url": "https://user:token@host/mcp"}
        )
        assert response.status_code == 422


async def test_recreating_a_deleted_server_starts_with_no_credential(db_sessionmaker) -> None:
    # The delete-recreate leak, closed end to end: connect a user credential,
    # delete the server, recreate it under the SAME name - the new server (new id,
    # so a new credential key) must start disconnected, never inheriting the old
    # token that a re-pointed same-name server would otherwise forward.
    settings = _settings(master_key=MASTER_KEY)
    async with integrations_client(db_sessionmaker, settings=settings) as client:
        first = await _create(
            client, name="github", base_url="https://a/mcp", credential_scope="user"
        )
        await client.put(f"{SERVERS}/{first['id']}/credential", json={"value": "secret_token_1"})
        assert (await client.get(f"{SERVERS}/{first['id']}/credential")).json()["connected"] is True

        await client.delete(f"{SERVERS}/{first['id']}")
        second = await _create(
            client, name="github", base_url="https://a/mcp", credential_scope="user"
        )
        assert second["id"] != first["id"]
        assert (await client.get(f"{SERVERS}/{second['id']}/credential")).json()[
            "connected"
        ] is False


# --- cache invalidation wiring --------------------------------------------


async def test_update_invalidates_the_cached_source(db_sessionmaker) -> None:
    spy = _SpyProvider()
    async with integrations_client(db_sessionmaker, provider=spy) as client:
        server = await _create(client)
        await client.patch(f"{SERVERS}/{server['id']}", json={"enabled": False})
    assert len(spy.invalidated) == 1
    assert str(spy.invalidated[0][1]) == server["id"]


async def test_workspace_credential_write_invalidates(db_sessionmaker) -> None:
    settings = _settings(master_key=MASTER_KEY)
    spy = _SpyProvider()
    async with integrations_client(db_sessionmaker, settings=settings, provider=spy) as client:
        server = await _create(client)
        await client.put(f"{SERVERS}/{server['id']}/credential", json={"value": "workspace_tok"})
    assert len(spy.invalidated) == 1


async def test_user_credential_write_does_not_invalidate(db_sessionmaker) -> None:
    # A user credential is revealed per call, so a member connecting theirs must
    # NOT cold-start the tenant's whole listing cache.
    settings = _settings(master_key=MASTER_KEY)
    spy = _SpyProvider()
    async with integrations_client(db_sessionmaker, settings=settings) as admin:
        server = await _create(admin, name="u", base_url="https://u/mcp", credential_scope="user")
    async with integrations_client(
        db_sessionmaker, settings=settings, tenant=MEMBER, provider=spy
    ) as member:
        await member.put(f"{SERVERS}/{server['id']}/credential", json={"value": "member_token_123"})
    assert spy.invalidated == []


async def test_presets_list_their_prefills_and_caveats(db_sessionmaker) -> None:
    # The curated catalog: static prefill metadata any member may read. Pins
    # the two shipped presets' load-bearing facts - GitHub rides MCP-native
    # discovery (no endpoints), Gmail prefills endpoints AND the visible
    # refresh-token ask (the owner-approved reading of the no-silent-
    # standing-access guardrail).
    async with integrations_client(db_sessionmaker, tenant=MEMBER) as client:
        resp = await client.get("/api/v1/integrations/presets")
        assert resp.status_code == 200
        items = {p["id"]: p for p in resp.json()["items"]}
    github, gmail = items["github"], items["gmail"]
    assert github["server"]["auth_type"] == "oauth"
    assert github["oauth"]["endpoints"] is None
    assert "repo" in github["oauth"]["scopes"]
    assert gmail["oauth"]["endpoints"]["issuer"] == "https://accounts.google.com"
    assert gmail["oauth"]["extra_authorize_params"]["access_type"] == "offline"
    # Both presets know how to answer "connected as WHO".
    assert github["oauth"]["identity"]["label_field"] == "login"
    assert gmail["oauth"]["identity"]["label_field"] == "email"
    assert any("RESTRICTED" in note for note in gmail["setup_notes"])

    # The slice-10 follow-up trio. Drive rides the same Google client pattern
    # as Gmail (manual endpoints + the visible refresh-token ask); Slack and
    # Asana are MCP-native, so their endpoints are discovered, never typed.
    assert len(items) == 5
    gdrive, slack, asana = items["gdrive"], items["slack"], items["asana"]
    assert gdrive["oauth"]["endpoints"]["issuer"] == "https://accounts.google.com"
    assert gdrive["oauth"]["extra_authorize_params"]["access_type"] == "offline"
    assert any(scope.endswith("drive.readonly") for scope in gdrive["oauth"]["scopes"])
    assert slack["oauth"]["endpoints"] is None
    assert asana["oauth"]["endpoints"] is None
    # Asana's single full-access scope; granular scopes are rejected upstream.
    assert asana["oauth"]["scopes"] == ["default"]
    assert slack["oauth"]["identity"]["label_field"] == "user"
    # Asana answers inside a data envelope; the dotted field walks it.
    assert asana["oauth"]["identity"]["label_field"] == "data.name"


async def test_saved_test_resolves_the_oauth_bearer(db_sessionmaker, monkeypatch) -> None:
    # The saved-server test of an oauth row resolves the grant's live bearer
    # through the SAME helper the chat path uses and hands it to the probe -
    # a connected server must test exactly as it would list.
    import app.api.v1.integrations as integrations_router
    from app.schemas.integrations import ConnectionTestResult

    async def fake_reveal(row, settings, tenant, *, sessionmaker, cipher):
        return "at-live", None

    captured: dict = {}

    async def fake_test(settings, tenant, server, store, *, override_credential, oauth_bearer=None):
        captured["oauth_bearer"] = oauth_bearer
        return ConnectionTestResult(ok=True)

    monkeypatch.setattr(integrations_router, "reveal_oauth_bearer", fake_reveal)
    monkeypatch.setattr(
        integrations_router.integrations_service, "test_saved_connection", fake_test
    )
    async with integrations_client(db_sessionmaker) as client:
        created = await _create(client, auth_type="oauth")
        resp = await client.post(
            f"{SERVERS}/{created['id']}/test-connection", json={"credential": None}
        )
        assert resp.status_code == 200
    assert captured["oauth_bearer"] == "at-live"


async def test_saved_test_answers_200_when_the_grant_is_unusable(
    db_sessionmaker, monkeypatch
) -> None:
    # A corrupt grant is exactly what an admin clicks Test to learn: it must
    # come back as a classified diagnostic, never a 500 - the endpoint's
    # always-200 contract. A bundle that will not PARSE is terminal, so the
    # answer names the only cure there is, reconnecting.
    import app.api.v1.integrations as integrations_router
    from app.services.tools.mcp_sources import WorkspaceCredentialRevealError

    async def raising_reveal(row, settings, tenant, *, sessionmaker, cipher):
        raise WorkspaceCredentialRevealError("sealed grant did not decrypt", transient=False)

    monkeypatch.setattr(integrations_router, "reveal_oauth_bearer", raising_reveal)
    async with integrations_client(db_sessionmaker) as client:
        created = await _create(client, auth_type="oauth")
        resp = await client.post(
            f"{SERVERS}/{created['id']}/test-connection", json={"credential": None}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "auth_rejected"
    assert body["error"]["detail"] == "reconnect needed"


async def test_saved_test_tells_a_transient_grant_failure_apart(
    db_sessionmaker, monkeypatch
) -> None:
    # The counterpart, and the reason the flag exists: a provider's token
    # endpoint being down for five minutes must NOT read as a dead grant. An
    # admin who believes that clicks Disconnect and revokes a perfectly healthy
    # refresh token upstream - a destructive action one click from a false
    # diagnosis. Collapse `transient` back into a single verdict and this test
    # sees "auth_rejected"/"reconnect needed" for an outage.
    import app.api.v1.integrations as integrations_router
    from app.services.tools.mcp_sources import WorkspaceCredentialRevealError

    async def raising_reveal(row, settings, tenant, *, sessionmaker, cipher):
        raise WorkspaceCredentialRevealError("token endpoint unreachable")

    monkeypatch.setattr(integrations_router, "reveal_oauth_bearer", raising_reveal)
    async with integrations_client(db_sessionmaker) as client:
        created = await _create(client, auth_type="oauth")
        resp = await client.post(
            f"{SERVERS}/{created['id']}/test-connection", json={"credential": None}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "unreachable"
    assert body["error"]["detail"] == "renewal failed - retry"


async def test_saved_test_of_a_fresh_grant_sends_its_bearer_end_to_end(
    db_sessionmaker, monkeypatch
) -> None:
    # No monkeypatched reveal here: a REAL sealed grant, revealed by the real
    # helper (its own sessions coexisting with the request session), reaches
    # the probe as the Authorization header. Fresh token = no renewal write.
    from datetime import timedelta

    from app.models import utcnow
    from app.providers.secrets import SecretScope, build_cipher
    from app.services import integrations as integrations_service_mod
    from app.services.oauth.bundle import OAuthTokenBundle, oauth_token_secret_name
    from app.services.secret_store import SecretStore

    settings = _settings(master_key=MASTER_KEY)
    cipher = build_cipher(MASTER_KEY)
    captured: dict = {}

    async def spy_probe(settings, tenant, *, base_url, headers, source=None):
        from app.schemas.integrations import ConnectionTestResult

        captured["headers"] = headers
        return ConnectionTestResult(ok=True)

    monkeypatch.setattr(integrations_service_mod, "_probe", spy_probe)
    async with integrations_client(db_sessionmaker, settings=settings) as client:
        created = await _create(client, auth_type="oauth")
        bundle = OAuthTokenBundle(
            access_token="at-live-123",
            refresh_token="rt-1",
            expires_at=utcnow() + timedelta(hours=1),
            granted_scope="default",
            token_endpoint="https://as.example/token",
            client_id="cid",
            issuer="https://as.example",
        )
        async with db_sessionmaker() as session:
            admin = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="admin")
            await SecretStore(session, admin, cipher).store(
                oauth_token_secret_name(created["id"], created["base_url"]),
                bundle.seal(),
                scope=SecretScope.WORKSPACE,
            )
            await session.commit()
        resp = await client.post(
            f"{SERVERS}/{created['id']}/test-connection", json={"credential": None}
        )
    assert resp.status_code == 200
    assert captured["headers"] == {"Authorization": "Bearer at-live-123"}


async def test_a_per_user_oauth_row_is_rejected(db_sessionmaker) -> None:
    # The consent flow seals ONE workspace grant per server, so a per-user
    # OAuth row would pass the member gate while every reveal hit the shared
    # grant. Rejected at create (422) AND at the resulting patch state (400);
    # the guard leaves with the Phase 8b per-user flow.
    async with integrations_client(db_sessionmaker) as client:
        bad = await client.post(
            SERVERS, json={**CREATE, "auth_type": "oauth", "credential_scope": "user"}
        )
        assert bad.status_code == 422

        created = await _create(client, auth_type="oauth")
        patched = await client.patch(
            f"{SERVERS}/{created['id']}", json={"credential_scope": "user"}
        )
        assert patched.status_code == 400
        # The refused combination did not stick.
        got = await client.get(f"{SERVERS}/{created['id']}")
        assert got.json()["credential_scope"] == "workspace"


async def test_a_per_user_auth_none_row_is_rejected(db_sessionmaker) -> None:
    # An auth-none source attaches no credential for anyone, so "per-user"
    # would promise a fail-closed per-member gate that no call path enforces.
    # Rejected at create (422) and at the RESULTING patch state (400) - both
    # the rescope of a none row and the auth flip of a user-scoped row, since
    # either edit mints the same dead combination.
    async with integrations_client(db_sessionmaker) as client:
        bad = await client.post(
            SERVERS, json={**CREATE, "auth_type": "none", "credential_scope": "user"}
        )
        assert bad.status_code == 422

        none_row = await client.post(SERVERS, json={**CREATE, "name": "open", "auth_type": "none"})
        assert none_row.status_code == 201
        rescoped = await client.patch(
            f"{SERVERS}/{none_row.json()['id']}", json={"credential_scope": "user"}
        )
        assert rescoped.status_code == 400

        user_row = await client.post(
            SERVERS, json={**CREATE, "name": "personal", "credential_scope": "user"}
        )
        assert user_row.status_code == 201
        flipped = await client.patch(
            f"{SERVERS}/{user_row.json()['id']}", json={"auth_type": "none"}
        )
        assert flipped.status_code == 400
        # Neither refused edit stuck.
        got = await client.get(f"{SERVERS}/{none_row.json()['id']}")
        assert got.json()["credential_scope"] == "workspace"
        got = await client.get(f"{SERVERS}/{user_row.json()['id']}")
        assert got.json()["auth_type"] == "header"


async def test_create_round_trips_config_extras(db_sessionmaker) -> None:
    # The UI stamps the creating preset's id into config so a card can rebuild
    # its prefill after a remount; the schema must pass unknown keys through
    # untouched (only disabled_tools is typed).
    async with integrations_client(db_sessionmaker) as client:
        created = await _create(client, config={"preset_id": "github"})
        assert created["config"] == {"preset_id": "github"}
        got = await client.get(f"{SERVERS}/{created['id']}")
        assert got.json()["config"] == {"preset_id": "github"}


async def test_the_redirect_uri_endpoint_reports_the_literal_callback(db_sessionmaker) -> None:
    # The UI shows this exact string for the operator to paste into a
    # provider's app registration, so it must be the same value the flow will
    # send (base with any trailing slash stripped, plus the callback path) -
    # and readable by any member, like the presets it sits beside.
    settings = Settings(
        _env_file=None,
        model_provider="fake",
        oauth_redirect_base_url="https://chat.example.com/",
    )
    async with integrations_client(db_sessionmaker, settings=settings, tenant=MEMBER) as client:
        resp = await client.get("/api/v1/integrations/oauth/redirect-uri")
        assert resp.status_code == 200
        expected = "https://chat.example.com/api/v1/integrations/oauth/callback"
        assert resp.json() == {"redirect_uri": expected}


async def test_connect_carries_extra_params_and_refuses_shadowing(db_sessionmaker) -> None:
    key = base64.b64encode(b"k" * 32).decode()
    async with integrations_client(db_sessionmaker, settings=_settings(master_key=key)) as client:
        created = await _create(client)
        body = {
            "client_id": "cid",
            "scopes": ["repo"],
            "endpoints": {
                "issuer": "https://as.example",
                "authorization_endpoint": "https://as.example/authorize",
                "token_endpoint": "https://as.example/token",
            },
            "extra_authorize_params": {"access_type": "offline"},
        }
        ok = await client.post(f"{SERVERS}/{created['id']}/oauth/connect", json=body)
        assert ok.status_code == 200
        assert "access_type=offline" in ok.json()["authorization_url"]

        # A key the protocol owns is a clean 422 at the boundary, never a 500 -
        # and a case/whitespace variant of one is the same attempt.
        for key in ("redirect_uri", "Redirect_Uri"):
            body["extra_authorize_params"] = {key: "https://evil.example/cb"}
            shadowed = await client.post(f"{SERVERS}/{created['id']}/oauth/connect", json=body)
            assert shadowed.status_code == 422


async def test_disconnect_clears_the_connected_account_label(db_sessionmaker) -> None:
    # The label describes a grant; disconnecting deletes the grant, so the
    # label must go with it (other config keys survive).
    key = base64.b64encode(b"k" * 32).decode()
    async with integrations_client(db_sessionmaker, settings=_settings(master_key=key)) as client:
        created = await _create(client, name="gh", auth_type="oauth")
        patched = await client.patch(
            f"{SERVERS}/{created['id']}",
            json={"config": {"connected_account": "octocat", "disabled_tools": []}},
        )
        assert patched.status_code == 200
        resp = await client.post(f"{SERVERS}/{created['id']}/oauth/disconnect")
        assert resp.status_code == 204
        got = await client.get(f"{SERVERS}/{created['id']}")
        assert "connected_account" not in got.json()["config"]
        assert "disabled_tools" in got.json()["config"]


async def test_deleting_a_server_erases_both_of_its_oauth_secrets(db_sessionmaker) -> None:
    # The finding this whole change exists for. Deleting an integration is what
    # an admin does when offboarding or when a grant is suspected compromised,
    # and the secrets are named from (server id, base_url) - so once the row is
    # gone, a surviving grant is not merely live upstream, it is unreachable
    # from every API and UI this app has. Both names must go: disconnect alone
    # takes the token bundle and leaves the client registration behind.
    from app.providers.secrets import SecretScope, build_cipher
    from app.services.oauth.bundle import oauth_client_secret_name, oauth_token_secret_name
    from app.services.secret_store import SecretStore

    settings = _settings(master_key=MASTER_KEY)
    cipher = build_cipher(MASTER_KEY)
    admin = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="admin")

    async with integrations_client(db_sessionmaker, settings=settings) as client:
        created = await _create(client, auth_type="oauth")
        token_name = oauth_token_secret_name(created["id"], created["base_url"])
        client_name = oauth_client_secret_name(created["id"], created["base_url"])
        async with db_sessionmaker() as session:
            store = SecretStore(session, admin, cipher)
            # No revocation endpoint is sealed, so nothing dials out; what is
            # under test is the local erasure, which is the part that must be
            # unconditional.
            await store.store(token_name, "sealed-grant", scope=SecretScope.WORKSPACE)
            await store.store(client_name, "cs-1", scope=SecretScope.WORKSPACE)
            await session.commit()

        deleted = await client.delete(f"{SERVERS}/{created['id']}")
        assert deleted.status_code == 204

    async with db_sessionmaker() as session:
        store = SecretStore(session, admin, cipher)
        assert await store.get(token_name, scope=SecretScope.WORKSPACE) is None
        assert await store.get(client_name, scope=SecretScope.WORKSPACE) is None


async def test_deleting_a_server_survives_an_unusable_master_key(db_sessionmaker) -> None:
    # The deletion must not depend on the crypto working. With no master key
    # the reveals fail, so nothing can be revoked upstream - but the row and
    # both secret rows still have to go, or a lost key would leave undeletable
    # integrations behind forever.
    from app.providers.secrets import SecretScope, build_cipher
    from app.services.oauth.bundle import oauth_client_secret_name, oauth_token_secret_name
    from app.services.secret_store import SecretStore

    cipher = build_cipher(MASTER_KEY)
    admin = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="admin")

    async with integrations_client(db_sessionmaker, settings=_settings(master_key=MASTER_KEY)) as c:
        created = await _create(c, auth_type="oauth")
        token_name = oauth_token_secret_name(created["id"], created["base_url"])
        client_name = oauth_client_secret_name(created["id"], created["base_url"])
        async with db_sessionmaker() as session:
            store = SecretStore(session, admin, cipher)
            await store.store(token_name, "sealed-grant", scope=SecretScope.WORKSPACE)
            await store.store(client_name, "cs-1", scope=SecretScope.WORKSPACE)
            await session.commit()

    # Same rows, a deployment that has since lost its key.
    async with integrations_client(db_sessionmaker, settings=_settings()) as keyless:
        deleted = await keyless.delete(f"{SERVERS}/{created['id']}")
        assert deleted.status_code == 204
        assert (await keyless.get(f"{SERVERS}/{created['id']}")).status_code == 404

    async with db_sessionmaker() as session:
        store = SecretStore(session, admin, cipher)
        assert await store.get(token_name, scope=SecretScope.WORKSPACE) is None
        assert await store.get(client_name, scope=SecretScope.WORKSPACE) is None


# --- env-seeded rows -------------------------------------------------------

CHASSIS_URL = "http://chassis.internal:8765/mcp"
DOCGEN_URL = "http://doc-gen:8765/docgen/mcp"


async def test_deleting_an_env_seeded_server_is_refused_and_says_why(db_sessionmaker) -> None:
    """A delete the next restart would undo is refused, not faked.

    The bootstrap is get-or-create by (tenant, name), so a row the deployment
    still names comes back at the next boot - with its credential and its tool
    toggles gone. The refusal also has to land BEFORE any destructive step: an
    OAuth grant revoked on the way to a 409 is unrecoverable damage done by a
    request that failed, so both sealed secrets are asserted intact afterwards.
    """
    from app.providers.secrets import SecretScope, build_cipher
    from app.services.oauth.bundle import oauth_client_secret_name, oauth_token_secret_name
    from app.services.secret_store import SecretStore

    settings = _settings(master_key=MASTER_KEY, mcp_chassis_url=CHASSIS_URL)
    cipher = build_cipher(MASTER_KEY)
    admin = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="admin")

    async with integrations_client(db_sessionmaker, settings=settings) as client:
        created = await _create(client, name="mcp-chassis", base_url=CHASSIS_URL, auth_type="oauth")
        token_name = oauth_token_secret_name(created["id"], created["base_url"])
        client_name = oauth_client_secret_name(created["id"], created["base_url"])
        async with db_sessionmaker() as session:
            store = SecretStore(session, admin, cipher)
            await store.store(token_name, "sealed-grant", scope=SecretScope.WORKSPACE)
            await store.store(client_name, "cs-1", scope=SecretScope.WORKSPACE)
            await session.commit()

        refused = await client.delete(f"{SERVERS}/{created['id']}")
        assert refused.status_code == 409
        detail = refused.json()["detail"]
        # It names the row, says why the delete would not stick, and points at
        # the switch that does stick.
        assert "mcp-chassis" in detail
        assert "restart" in detail
        assert "Turn it off" in detail

        # Nothing was half-done: the row is still there and still readable.
        assert (await client.get(f"{SERVERS}/{created['id']}")).status_code == 200

    async with db_sessionmaker() as session:
        store = SecretStore(session, admin, cipher)
        assert await store.get(token_name, scope=SecretScope.WORKSPACE) is not None
        assert await store.get(client_name, scope=SecretScope.WORKSPACE) is not None


async def test_a_seed_named_row_becomes_deletable_once_the_env_stops_naming_it(
    db_sessionmaker,
) -> None:
    """The gate is computed from the live settings, so it opens by itself.

    No tombstone and no column: the same row, read by a deployment that no
    longer configures that sidecar, is an ordinary row - nothing would recreate
    it, so deleting it is honest and goes through.
    """
    async with integrations_client(
        db_sessionmaker, settings=_settings(mcp_chassis_url=CHASSIS_URL)
    ) as configured:
        created = await _create(configured, name="mcp-chassis", base_url=CHASSIS_URL)
        assert (await configured.delete(f"{SERVERS}/{created['id']}")).status_code == 409

    async with integrations_client(db_sessionmaker, settings=_settings()) as unconfigured:
        assert (await unconfigured.delete(f"{SERVERS}/{created['id']}")).status_code == 204
        assert (await unconfigured.get(f"{SERVERS}/{created['id']}")).status_code == 404


async def test_the_listing_marks_which_rows_the_deployment_seeds(db_sessionmaker) -> None:
    """A card cannot warn about a delete it does not know will be refused.

    The flag rides on every read for the same reason ``credential_state`` does:
    otherwise a client learns it one failed mutation at a time.
    """
    settings = _settings(mcp_chassis_url=CHASSIS_URL, docgen_mcp_url=DOCGEN_URL)
    async with integrations_client(db_sessionmaker, settings=settings) as client:
        chassis = await _create(client, name="mcp-chassis", base_url=CHASSIS_URL)
        docgen = await _create(client, name="doc-gen", base_url=DOCGEN_URL)
        hand_added = await _create(client)
        # NAME is the resurrection key, not the URL: a second row pointed at the
        # seeded URL under another name is nobody's bootstrap row.
        alias = await _create(client, name="chassis-copy", base_url=CHASSIS_URL)

        seeded = {row["id"]: row["seeded"] for row in (await client.get(SERVERS)).json()["items"]}
        assert seeded[chassis["id"]] is True
        assert seeded[docgen["id"]] is True
        assert seeded[hand_added["id"]] is False
        assert seeded[alias["id"]] is False
        # The single read answers identically - one serializer, no drift.
        assert (await client.get(f"{SERVERS}/{chassis['id']}")).json()["seeded"] is True


async def test_the_docgen_env_key_reports_as_a_deployment_key(db_sessionmaker) -> None:
    """The credential state has to say what a call actually sends, for EVERY sidecar.

    This readout knew only the chassis key while the chat path had grown a
    doc-gen branch, so a deployment that set ``DOCGEN_API_KEY`` sent it on every
    doc-gen call while the card reported the row as having no credential. Both
    now ask one shared helper.
    """
    configured = {"master_key": MASTER_KEY, "docgen_mcp_url": DOCGEN_URL}
    async with integrations_client(
        db_sessionmaker, settings=_settings(**configured, docgen_api_key="docgen-key")
    ) as with_key:
        row = await _create(with_key, name="doc-gen", base_url=DOCGEN_URL)
        assert (await with_key.get(f"{SERVERS}/{row['id']}")).json()["credential_state"] == (
            "deployment_key"
        )
        # The key belongs to ONE url; an unrelated server never inherits it.
        other = await _create(with_key, name="elsewhere", base_url="https://other.example/mcp")
        assert (await with_key.get(f"{SERVERS}/{other['id']}")).json()["credential_state"] == (
            "missing"
        )

    # The same row, read by a deployment that never set the key: it needs a
    # credential and has none, which is exactly what the calls would do.
    async with integrations_client(db_sessionmaker, settings=_settings(**configured)) as no_key:
        assert (await no_key.get(f"{SERVERS}/{row['id']}")).json()["credential_state"] == "missing"
