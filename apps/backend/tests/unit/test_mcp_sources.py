"""McpSourceProvider: per-tenant tool sources from stored rows (slice 6b PR2).

These drive the provider over the test sqlite session and inspect the sources it
builds - no network. They pin the four properties the design rests on: sources
reflect the enabled rows (name-ordered), the env server seeds once, a source is
memoized per (tenant, server) so its listing cache survives, and - the security
one - two tenants NEVER share a source instance even for an identical name+URL.
"""

from __future__ import annotations

import base64
from datetime import timedelta
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.models import McpAuthType, McpCredentialScope, McpServer, utcnow
from app.providers.secrets import SecretScope, build_cipher
from app.repositories.mcp_servers import McpServerRepository
from app.services.oauth.bundle import (
    OAuthTokenBundle,
    oauth_token_secret_name,
)
from app.services.oauth.flow import OAuthFlowError, OAuthRefreshDenied
from app.services.secret_store import SecretStore
from app.services.tools.mcp_credentials import PerUserCredentialResolver
from app.services.tools.mcp_sources import (
    McpSourceProvider,
    WorkspaceCredentialRevealError,
    _reveal_workspace_credential,
    build_source_for_row,
    credential_secret_name,
    env_seed_names,
)

_KEY = base64.b64encode(b"m" * 32).decode()
CIPHER = build_cipher(_KEY)

pytestmark = pytest.mark.anyio

TENANT_A = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="admin")
TENANT_B = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="admin")


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, model_provider="fake", **overrides)


async def _add_row(
    db_sessionmaker,
    tenant: TenantContext,
    *,
    name: str,
    base_url: str = "http://x/mcp",
    enabled: bool = True,
    scope: str = McpCredentialScope.WORKSPACE.value,
    auth_type: str = McpAuthType.HEADER.value,
) -> None:
    async with db_sessionmaker() as session:
        repo = McpServerRepository(session, tenant)
        repo.add(
            McpServer(
                tenant_id=tenant.tenant_id,
                name=name,
                base_url=base_url,
                enabled=enabled,
                credential_scope=scope,
                auth_type=auth_type,
            )
        )
        await session.commit()


def _names(sources) -> list[str]:
    """The server name each built (CachingToolSource -> McpToolSource) carries."""
    return [source._inner._config.name for source in sources]


async def _get_row(db_sessionmaker, tenant: TenantContext, name: str) -> McpServer:
    """Load a server row by name (for its id in an invalidation test)."""
    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, tenant).get_by_name(name)
    assert row is not None
    return row


async def _store_workspace_secret(
    db_sessionmaker, tenant: TenantContext, name: str, value: str
) -> None:
    """Seal a WORKSPACE secret under ``name`` - a credential connected via the API."""
    async with db_sessionmaker() as session:
        await SecretStore(session, tenant, CIPHER).store(name, value, scope=SecretScope.WORKSPACE)
        await session.commit()


def _auth(source) -> str | None:
    """The Authorization header the built (CachingToolSource -> McpToolSource) carries."""
    return source._inner._config.headers.get("Authorization")


async def test_sources_reflect_enabled_rows_in_name_order(db_sessionmaker) -> None:
    await _add_row(db_sessionmaker, TENANT_A, name="beta")
    await _add_row(db_sessionmaker, TENANT_A, name="alpha")
    await _add_row(db_sessionmaker, TENANT_A, name="gamma", enabled=False)

    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    sources = await provider.sources_for(TENANT_A)

    assert _names(sources) == ["alpha", "beta"]


async def test_no_server_configured_yields_no_sources(db_sessionmaker) -> None:
    # No env chassis and an empty table: no seed, no rows, no sources - the
    # honest "no MCP deployment".
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    assert await provider.sources_for(TENANT_A) == []


async def test_env_server_is_seeded_once(db_sessionmaker) -> None:
    provider = McpSourceProvider(
        _settings(mcp_chassis_url="http://chassis/mcp"), sessionmaker=db_sessionmaker
    )
    first = await provider.sources_for(TENANT_A)
    # Seeded under the credential name so a user-scoped call's secret lookup lines
    # up with what the user connected under.
    assert _names(first) == ["mcp-chassis"]

    # A second turn does not duplicate the seed row.
    await provider.sources_for(TENANT_A)
    async with db_sessionmaker() as session:
        rows = await McpServerRepository(session, TENANT_A).list_enabled()
    assert [row.name for row in rows] == ["mcp-chassis"]


async def test_seed_scope_follows_the_per_user_flag(db_sessionmaker) -> None:
    # MCP_PER_USER_CREDENTIALS on -> the seeded server is user-scoped, so its
    # source wires a fail-closed per-user resolver.
    provider = McpSourceProvider(
        _settings(mcp_chassis_url="http://chassis/mcp", mcp_per_user_credentials=True),
        sessionmaker=db_sessionmaker,
    )
    sources = await provider.sources_for(TENANT_A)
    resolver = sources[0]._inner._credential_resolver
    assert isinstance(resolver, PerUserCredentialResolver)
    assert resolver._required is True


async def test_the_docgen_sidecar_is_seeded_alongside_the_chassis(db_sessionmaker) -> None:
    # Two configured sidecars, two rows: each keeps its own name, credential and
    # tool toggles. One merged registration would force a single credential
    # scope across unrelated servers.
    provider = McpSourceProvider(
        _settings(
            mcp_chassis_url="http://chassis/mcp",
            docgen_mcp_url="http://doc-gen:8765/docgen/mcp",
        ),
        sessionmaker=db_sessionmaker,
    )
    assert _names(await provider.sources_for(TENANT_A)) == ["doc-gen", "mcp-chassis"]

    # Seeding stays once-per-tenant across turns for BOTH rows.
    await provider.sources_for(TENANT_A)
    async with db_sessionmaker() as session:
        rows = await McpServerRepository(session, TENANT_A).list_enabled()
    assert [row.name for row in rows] == ["doc-gen", "mcp-chassis"]


async def test_the_docgen_sidecar_seeds_without_a_chassis(db_sessionmaker) -> None:
    # The sidecars are independent: a deployment that wants document generation
    # and no datastore wrapper gets exactly the one row.
    provider = McpSourceProvider(
        _settings(docgen_mcp_url="http://doc-gen:8765/docgen/mcp"), sessionmaker=db_sessionmaker
    )
    assert _names(await provider.sources_for(TENANT_A)) == ["doc-gen"]


async def test_two_seeds_sharing_a_name_are_warned_about(
    db_sessionmaker, caplog: pytest.LogCaptureFixture
) -> None:
    # Rows are unique per (tenant, name) and ensure_seed is get-or-create, so a
    # collided name silently registers ONE server and drops the other. Nothing
    # raises, nothing 500s - the second sidecar's tools are just never offered,
    # which is precisely the class of misconfiguration that hides for months.
    provider = McpSourceProvider(
        _settings(
            mcp_chassis_url="http://chassis/mcp",
            docgen_mcp_url="http://doc-gen:8765/docgen/mcp",
            docgen_credential_name="mcp-chassis",
        ),
        sessionmaker=db_sessionmaker,
    )

    with caplog.at_level("WARNING", logger="app.services.tools.mcp_sources"):
        sources = await provider.sources_for(TENANT_A)

    assert _names(sources) == ["mcp-chassis"]
    assert "share the name 'mcp-chassis'" in caplog.text


async def test_the_docgen_seed_stays_workspace_scoped_under_per_user_credentials(
    db_sessionmaker,
) -> None:
    # MCP_PER_USER_CREDENTIALS is about upstreams a USER can connect an account
    # to. doc-gen reaches no upstream, so it must not inherit the flag: a
    # user-scoped row wires a fail-closed resolver, and every document call
    # would fail for want of a credential nobody can connect.
    provider = McpSourceProvider(
        _settings(
            mcp_chassis_url="http://chassis/mcp",
            docgen_mcp_url="http://doc-gen:8765/docgen/mcp",
            mcp_per_user_credentials=True,
        ),
        sessionmaker=db_sessionmaker,
    )
    built = await provider.sources_for(TENANT_A)
    sources = dict(zip(_names(built), built, strict=True))

    assert isinstance(sources["mcp-chassis"]._inner._credential_resolver, PerUserCredentialResolver)
    assert sources["doc-gen"]._inner._credential_resolver is None
    async with db_sessionmaker() as session:
        rows = {
            row.name: row for row in await McpServerRepository(session, TENANT_A).list_enabled()
        }
    assert rows["doc-gen"].credential_scope == McpCredentialScope.WORKSPACE.value
    assert rows["mcp-chassis"].credential_scope == McpCredentialScope.USER.value


async def test_seed_names_are_exactly_the_rows_seeding_creates(db_sessionmaker) -> None:
    # The Integrations API refuses to delete a row whose NAME is in
    # env_seed_names, on the grounds that seeding brings it back. That reasoning
    # only holds while the set it consults is the set of names seeding actually
    # writes, so the two are asserted against each other here rather than
    # against a literal - a change to either side that parts them fails.
    settings = _settings(
        mcp_chassis_url="http://chassis/mcp",
        docgen_mcp_url="http://doc-gen:8765/docgen/mcp",
    )
    provider = McpSourceProvider(settings, sessionmaker=db_sessionmaker)
    await provider.sources_for(TENANT_A)

    async with db_sessionmaker() as session:
        rows = await McpServerRepository(session, TENANT_A).list_enabled()
    assert {row.name for row in rows} == env_seed_names(settings)


def test_no_configured_sidecar_seeds_no_names() -> None:
    # The empty case is the one that decides whether an ordinary hand-added row
    # is deletable: with nothing configured, nothing is protected.
    assert env_seed_names(_settings()) == frozenset()


async def test_workspace_scoped_row_wires_no_resolver(db_sessionmaker) -> None:
    await _add_row(db_sessionmaker, TENANT_A, name="ws", scope=McpCredentialScope.WORKSPACE.value)
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    sources = await provider.sources_for(TENANT_A)
    # A workspace server uses the shared header for every call - no resolver.
    assert sources[0]._inner._credential_resolver is None


async def test_a_source_is_memoized_per_tenant_server(db_sessionmaker) -> None:
    await _add_row(db_sessionmaker, TENANT_A, name="s")
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    first = await provider.sources_for(TENANT_A)
    second = await provider.sources_for(TENANT_A)
    # The SAME CachingToolSource instance across turns - that is what keeps the
    # remote listing cache warm rather than dialling the sidecar every turn.
    assert first[0] is second[0]


async def test_two_tenants_never_share_a_source_instance(db_sessionmaker) -> None:
    # The security property: even with an IDENTICAL server name and URL, each
    # tenant gets its own source instance (the memo key includes tenant_id), so
    # one tenant's cached listing can never be served to another.
    await _add_row(db_sessionmaker, TENANT_A, name="shared", base_url="http://same/mcp")
    await _add_row(db_sessionmaker, TENANT_B, name="shared", base_url="http://same/mcp")
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)

    a = await provider.sources_for(TENANT_A)
    b = await provider.sources_for(TENANT_B)

    assert _names(a) == ["shared"]
    assert _names(b) == ["shared"]
    assert a[0] is not b[0]


async def test_registry_read_failure_degrades_to_no_tools(db_sessionmaker) -> None:
    # MCP is one source of tools: a registry-read failure must offer no tools,
    # never take the turn down.
    class BoomSessionmaker:
        def __call__(self):
            raise RuntimeError("db down")

    provider = McpSourceProvider(_settings(), sessionmaker=BoomSessionmaker())
    assert await provider.sources_for(TENANT_A) == []


async def test_rescoping_a_row_rebuilds_the_source_fail_closed(db_sessionmaker) -> None:
    # credential_scope is in the memo key, so an operator rescoping a live row
    # workspace -> user in the DB gets a NEW source wired FAIL-CLOSED on the next
    # turn, never the stale shared-header source. This locks credential_scope into
    # the key: a future key simplification that dropped it would fail here loudly
    # instead of silently bypassing fail-closed (a credential-scope-bypass leak).
    await _add_row(db_sessionmaker, TENANT_A, name="s", scope=McpCredentialScope.WORKSPACE.value)
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    before = await provider.sources_for(TENANT_A)
    assert before[0]._inner._credential_resolver is None

    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT_A).get_by_name("s")
        row.credential_scope = McpCredentialScope.USER.value
        session.add(row)
        await session.commit()

    after = await provider.sources_for(TENANT_A)
    assert after[0] is not before[0]
    resolver = after[0]._inner._credential_resolver
    assert isinstance(resolver, PerUserCredentialResolver)
    assert resolver._required is True


async def test_editing_a_rows_base_url_rebuilds_the_source(db_sessionmaker) -> None:
    # base_url is in the memo key too: re-pointing a live row yields a new source
    # carrying the new URL, not the stale cached one.
    await _add_row(db_sessionmaker, TENANT_A, name="s", base_url="http://old/mcp")
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    before = await provider.sources_for(TENANT_A)
    assert before[0]._inner._config.url == "http://old/mcp"

    async with db_sessionmaker() as session:
        row = await McpServerRepository(session, TENANT_A).get_by_name("s")
        row.base_url = "http://new/mcp"
        session.add(row)
        await session.commit()

    after = await provider.sources_for(TENANT_A)
    assert after[0] is not before[0]
    assert after[0]._inner._config.url == "http://new/mcp"


class _CommitFailsOnceSession:
    """Delegates to a real session but makes the FIRST commit raise once."""

    def __init__(self, real, state: dict) -> None:
        self._real = real
        self._state = state

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    async def commit(self) -> None:
        if not self._state["failed"]:
            self._state["failed"] = True
            raise RuntimeError("seed commit failed")
        await self._real.commit()


class _FlakySeedCM:
    def __init__(self, inner_cm, state: dict) -> None:
        self._inner_cm = inner_cm
        self._state = state

    async def __aenter__(self):
        return _CommitFailsOnceSession(await self._inner_cm.__aenter__(), self._state)

    async def __aexit__(self, *exc):
        return await self._inner_cm.__aexit__(*exc)


class _FlakySeedSessionmaker:
    """A sessionmaker whose first-produced session fails its commit exactly once."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._state = {"failed": False}

    def __call__(self):
        return _FlakySeedCM(self._inner(), self._state)


async def test_a_seed_commit_failure_is_retried_not_marked(db_sessionmaker) -> None:
    # The tenant is marked seeded ONLY after the seed's commit succeeds. A
    # transient commit failure on the first turn must leave the tenant un-marked
    # so the next turn RETRIES - never permanently deprive it of its bootstrap
    # server (which is what moving the _seeded.add before the commit would do).
    flaky = _FlakySeedSessionmaker(db_sessionmaker)
    provider = McpSourceProvider(
        _settings(mcp_chassis_url="http://chassis/mcp"), sessionmaker=flaky
    )
    # First turn: the seed's commit fails -> no tools, and the tenant is NOT
    # marked seeded.
    assert await provider.sources_for(TENANT_A) == []
    assert TENANT_A.tenant_id not in provider._seeded
    # Next turn: the seed succeeds and the bootstrap server appears.
    assert _names(await provider.sources_for(TENANT_A)) == ["mcp-chassis"]


async def test_invalidate_drops_the_cached_source_so_it_rebuilds(db_sessionmaker) -> None:
    # An admin edit (config/enabled/delete) keeps the memo key, so the write path
    # must invalidate explicitly; the next sources_for then rebuilds a NEW source.
    await _add_row(db_sessionmaker, TENANT_A, name="s")
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    before = await provider.sources_for(TENANT_A)
    server_id = (await _get_row(db_sessionmaker, TENANT_A, "s")).id

    provider.invalidate(TENANT_A.tenant_id, server_id)

    after = await provider.sources_for(TENANT_A)
    assert after[0] is not before[0]


async def test_invalidate_is_scoped_to_one_tenant_server(db_sessionmaker) -> None:
    # Invalidating one server leaves another server's - and another tenant's -
    # memoized sources warm, so an edit does not cold-start the whole registry.
    await _add_row(db_sessionmaker, TENANT_A, name="keep")
    await _add_row(db_sessionmaker, TENANT_A, name="drop")
    await _add_row(db_sessionmaker, TENANT_B, name="other")
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    a_before = await provider.sources_for(TENANT_A)
    b_before = await provider.sources_for(TENANT_B)
    drop_id = (await _get_row(db_sessionmaker, TENANT_A, "drop")).id

    provider.invalidate(TENANT_A.tenant_id, drop_id)

    a_after = await provider.sources_for(TENANT_A)
    b_after = await provider.sources_for(TENANT_B)
    # "keep" (a_after[1]) and TENANT_B's source survive; only "drop" rebuilt.
    assert a_after[1] is a_before[1]  # keep
    assert a_after[0] is not a_before[0]  # drop
    assert b_after[0] is b_before[0]  # other tenant untouched


async def test_invalidate_unknown_is_a_no_op(db_sessionmaker) -> None:
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    # No cache entries yet; invalidating anything must not raise.
    provider.invalidate(TENANT_A.tenant_id, uuid4())


async def test_build_source_for_row_wires_disabled_tools_from_config(db_sessionmaker) -> None:
    row = McpServer(
        tenant_id=TENANT_A.tenant_id,
        name="s",
        base_url="http://x/mcp",
        config={"disabled_tools": ["create_record", "delete_record"]},
    )
    source, _ = await build_source_for_row(
        row, _settings(), TENANT_A, sessionmaker=db_sessionmaker, cipher=None
    )
    assert source._inner._config.disabled_tools == frozenset({"create_record", "delete_record"})


async def test_build_source_for_row_tolerates_a_malformed_disabled_tools(db_sessionmaker) -> None:
    # config is an operator-editable blob: a non-list disabled_tools must not
    # crash a turn - it degrades to "offer everything".
    row = McpServer(
        tenant_id=TENANT_A.tenant_id,
        name="s",
        base_url="http://x/mcp",
        config={"disabled_tools": "create_record"},
    )
    source, _ = await build_source_for_row(
        row, _settings(), TENANT_A, sessionmaker=db_sessionmaker, cipher=None
    )
    assert source._inner._config.disabled_tools == frozenset()


async def test_the_env_key_backs_only_the_chassis_url(db_sessionmaker) -> None:
    # The core security improvement of this slice: the env MCP_CHASSIS_API_KEY
    # belongs to MCP_CHASSIS_URL, so it backs the seeded chassis but must NOT leak
    # onto an unrelated server an admin later registers - that server gets no
    # shared header until a credential is connected for it.
    await _add_row(db_sessionmaker, TENANT_A, name="other", base_url="http://other/mcp")
    provider = McpSourceProvider(
        _settings(
            secret_master_key=_KEY,
            mcp_chassis_url="http://chassis/mcp",
            mcp_chassis_api_key="chassis-key",
        ),
        sessionmaker=db_sessionmaker,
    )
    sources = await provider.sources_for(TENANT_A)
    by_name = dict(zip(_names(sources), sources, strict=True))

    assert _auth(by_name["mcp-chassis"]) == "Bearer chassis-key"
    assert _auth(by_name["other"]) is None


async def test_each_env_key_backs_only_its_own_sidecar(db_sessionmaker) -> None:
    # Two env-configured sidecars, two env keys. Each key is bound to the URL it
    # belongs to, so doc-gen's key can never be sent to the chassis (or to an
    # unrelated server an admin registers later) and the chassis's can never be
    # sent to doc-gen. Collapse the two branches into one and this fails.
    await _add_row(db_sessionmaker, TENANT_A, name="other", base_url="http://other/mcp")
    provider = McpSourceProvider(
        _settings(
            secret_master_key=_KEY,
            mcp_chassis_url="http://chassis/mcp",
            mcp_chassis_api_key="chassis-key",
            docgen_mcp_url="http://doc-gen:8765/docgen/mcp",
            docgen_api_key="docgen-key",
        ),
        sessionmaker=db_sessionmaker,
    )
    built = await provider.sources_for(TENANT_A)
    by_name = dict(zip(_names(built), built, strict=True))

    assert _auth(by_name["mcp-chassis"]) == "Bearer chassis-key"
    assert _auth(by_name["doc-gen"]) == "Bearer docgen-key"
    assert _auth(by_name["other"]) is None


async def test_a_stored_credential_still_overrides_the_docgen_env_key(db_sessionmaker) -> None:
    # The env key is a BOOTSTRAP: once an admin connects a credential for the
    # row through the Integrations API, that credential wins - same rule the
    # chassis follows, so the second sidecar cannot become an exception to it.
    await _add_row(
        db_sessionmaker, TENANT_A, name="doc-gen", base_url="http://doc-gen:8765/docgen/mcp"
    )
    row = await _get_row(db_sessionmaker, TENANT_A, "doc-gen")
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, credential_secret_name(row.id, row.base_url), "connected-token"
    )
    provider = McpSourceProvider(
        _settings(
            secret_master_key=_KEY,
            docgen_mcp_url="http://doc-gen:8765/docgen/mcp",
            docgen_api_key="docgen-key",
        ),
        sessionmaker=db_sessionmaker,
    )

    sources = await provider.sources_for(TENANT_A)

    assert _auth(sources[0]) == "Bearer connected-token"


async def test_auth_type_none_sends_no_header_despite_a_stored_credential(db_sessionmaker) -> None:
    # The mutation guard for the auth_type branch: this row has a workspace
    # credential CONNECTED (the test below proves that alone becomes the
    # header), yet its declared ``auth_type="none"`` means NOTHING is attached.
    # Delete the none branch in build_source_for_row and this fails with the
    # stored token reappearing as a bearer header.
    await _add_row(
        db_sessionmaker,
        TENANT_A,
        name="open",
        base_url="http://open/mcp",
        auth_type=McpAuthType.NONE.value,
    )
    row = await _get_row(db_sessionmaker, TENANT_A, "open")
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, credential_secret_name(row.id, row.base_url), "stored-token"
    )
    provider = McpSourceProvider(_settings(secret_master_key=_KEY), sessionmaker=db_sessionmaker)

    sources = await provider.sources_for(TENANT_A)

    assert _auth(sources[0]) is None


async def test_auth_type_none_wires_no_resolver_even_on_a_user_scoped_row(db_sessionmaker) -> None:
    # ``none`` beats the scope: an unauthenticated upstream takes no per-user
    # credential either, so no resolver is wired and calls cannot fail closed on
    # a credential the server never needed.
    await _add_row(
        db_sessionmaker,
        TENANT_A,
        name="open",
        scope=McpCredentialScope.USER.value,
        auth_type=McpAuthType.NONE.value,
    )
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    sources = await provider.sources_for(TENANT_A)
    assert sources[0]._inner._credential_resolver is None


async def test_oauth_row_sends_the_bundles_access_token(db_sessionmaker) -> None:
    # An oauth row's ONLY credential source is the sealed grant: its access
    # token becomes the bearer, and a stored pasted credential is ignored.
    from datetime import UTC, datetime

    await _add_row(
        db_sessionmaker,
        TENANT_A,
        name="gh",
        base_url="http://gh/mcp",
        auth_type=McpAuthType.OAUTH.value,
    )
    row = await _get_row(db_sessionmaker, TENANT_A, "gh")
    bundle = OAuthTokenBundle(
        access_token="oauth-at",
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        granted_scope="repo",
        token_endpoint="https://as.example/token",
        client_id="cid",
        issuer="https://as.example",
    )
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, oauth_token_secret_name(row.id, row.base_url), bundle.seal()
    )
    # A stray pasted credential under the plain name must NOT win.
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, credential_secret_name(row.id, row.base_url), "pasted"
    )
    provider = McpSourceProvider(_settings(secret_master_key=_KEY), sessionmaker=db_sessionmaker)
    sources = await provider.sources_for(TENANT_A)
    assert _auth(sources[0]) == "Bearer oauth-at"


async def test_a_tampered_oauth_bundle_degrades_without_caching(db_sessionmaker) -> None:
    # Garbage under the bundle's name is a state a reconnect can heal, so the
    # build must raise the no-cache signal rather than bake "no header" into
    # a memoized source for the life of the process.
    from app.services.tools.mcp_sources import (
        WorkspaceCredentialRevealError,
        build_source_for_row,
    )

    await _add_row(db_sessionmaker, TENANT_A, name="gh", auth_type=McpAuthType.OAUTH.value)
    row = await _get_row(db_sessionmaker, TENANT_A, "gh")
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, oauth_token_secret_name(row.id, row.base_url), "not-a-bundle"
    )
    with pytest.raises(WorkspaceCredentialRevealError):
        await build_source_for_row(
            row,
            _settings(secret_master_key=_KEY),
            TENANT_A,
            sessionmaker=db_sessionmaker,
            cipher=CIPHER,
        )


async def test_oauth_row_without_a_grant_goes_bare(db_sessionmaker) -> None:
    # Not connected: calls go out with no header and fail honestly upstream -
    # never a fallback to a pasted credential or the env bootstrap.
    await _add_row(
        db_sessionmaker,
        TENANT_A,
        name="gh",
        auth_type=McpAuthType.OAUTH.value,
    )
    provider = McpSourceProvider(_settings(secret_master_key=_KEY), sessionmaker=db_sessionmaker)
    sources = await provider.sources_for(TENANT_A)
    assert _auth(sources[0]) is None
    assert sources[0]._inner._credential_resolver is None


async def test_unknown_auth_type_fails_closed_with_no_header(db_sessionmaker) -> None:
    # A value this build does not implement (a partial upgrade, a hand-edited
    # row) must not fall through to the bearer path: forwarding the credential
    # to an upstream that negotiates its own auth would leak it silently. Pins
    # header-vs-everything-else, not none-vs-everything-else.
    await _add_row(
        db_sessionmaker, TENANT_A, name="odd", base_url="http://odd/mcp", auth_type="mystery"
    )
    row = await _get_row(db_sessionmaker, TENANT_A, "odd")
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, credential_secret_name(row.id, row.base_url), "stored-token"
    )
    provider = McpSourceProvider(_settings(secret_master_key=_KEY), sessionmaker=db_sessionmaker)

    sources = await provider.sources_for(TENANT_A)

    assert _auth(sources[0]) is None


async def test_auth_type_none_source_keeps_its_deny_list(db_sessionmaker) -> None:
    # The unauthenticated branch shares _assemble_source with the header path,
    # so the non-auth knobs must survive it - a none row's per-tool deny-list
    # still filters, or "unauthenticated" would quietly widen the offered set.
    async with db_sessionmaker() as session:
        McpServerRepository(session, TENANT_A).add(
            McpServer(
                tenant_id=TENANT_A.tenant_id,
                name="open",
                base_url="http://open/mcp",
                auth_type=McpAuthType.NONE.value,
                config={"disabled_tools": ["create_record"]},
            )
        )
        await session.commit()
    provider = McpSourceProvider(_settings(), sessionmaker=db_sessionmaker)
    sources = await provider.sources_for(TENANT_A)
    assert sources[0]._inner._config.disabled_tools == frozenset({"create_record"})


async def test_a_stored_workspace_credential_becomes_the_header(db_sessionmaker) -> None:
    # A credential connected via the Integrations API (sealed under the server's
    # stable-identity key) becomes that server's shared header - the whole point of
    # this slice. No env chassis key is configured, so this is the stored path only.
    await _add_row(db_sessionmaker, TENANT_A, name="ws", base_url="http://ws/mcp")
    row = await _get_row(db_sessionmaker, TENANT_A, "ws")
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, credential_secret_name(row.id, row.base_url), "stored-token"
    )
    provider = McpSourceProvider(_settings(secret_master_key=_KEY), sessionmaker=db_sessionmaker)

    sources = await provider.sources_for(TENANT_A)

    assert _auth(sources[0]) == "Bearer stored-token"


async def test_a_stored_credential_overrides_the_env_key_for_the_chassis(db_sessionmaker) -> None:
    # A stored credential wins over the env bootstrap key even for the chassis, so
    # an admin can rotate the chassis credential from the UI. Store it, invalidate,
    # and the rebuilt source carries the stored value, not the env one.
    provider = McpSourceProvider(
        _settings(
            secret_master_key=_KEY,
            mcp_chassis_url="http://chassis/mcp",
            mcp_chassis_api_key="env-key",
        ),
        sessionmaker=db_sessionmaker,
    )
    first = await provider.sources_for(TENANT_A)
    assert _auth(first[0]) == "Bearer env-key"  # env key backs the seed until connected

    row = await _get_row(db_sessionmaker, TENANT_A, "mcp-chassis")
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, credential_secret_name(row.id, row.base_url), "stored-token"
    )
    provider.invalidate(TENANT_A.tenant_id, row.id)

    rebuilt = await provider.sources_for(TENANT_A)
    assert _auth(rebuilt[0]) == "Bearer stored-token"


async def test_without_a_master_key_the_chassis_still_uses_the_env_key(db_sessionmaker) -> None:
    # SECRET_MASTER_KEY unset (cipher None): the stored-credential reveal is skipped
    # entirely, and the seeded chassis still works via the env bootstrap key -
    # zero-config bootstrap is preserved.
    provider = McpSourceProvider(
        _settings(mcp_chassis_url="http://chassis/mcp", mcp_chassis_api_key="env-key"),
        sessionmaker=db_sessionmaker,
    )
    sources = await provider.sources_for(TENANT_A)
    assert _auth(sources[0]) == "Bearer env-key"


class _FailingExecuteSession:
    """Delegates to a real session but raises on ``execute`` - a DB hiccup on the
    workspace-credential reveal query, not on the earlier ``list_enabled``."""

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    async def execute(self, *args, **kwargs):
        raise RuntimeError("reveal query failed")


class _FailingExecuteCM:
    def __init__(self, inner_cm) -> None:
        self._inner_cm = inner_cm

    async def __aenter__(self):
        return _FailingExecuteSession(await self._inner_cm.__aenter__())

    async def __aexit__(self, *exc):
        return await self._inner_cm.__aexit__(*exc)


class _FailingExecuteSessionmaker:
    """Every produced session raises on execute - an always-down reveal."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __call__(self):
        return _FailingExecuteCM(self._inner())


class _RevealFailsOnceSessionmaker:
    """The first reveal session (the 2nd overall, after list_enabled) fails once.

    So a provider's turn 1 sees ``list_enabled`` succeed and the workspace reveal
    fail (the exact new cold-build failure path), and turn 2 retries cleanly -
    proving a degraded build is not memoized.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._state = {"list_seen": False, "failed": False}

    def __call__(self):
        cm = self._inner()
        if not self._state["list_seen"]:
            self._state["list_seen"] = True
            return cm  # the first session is list_enabled: let it through
        if not self._state["failed"]:
            self._state["failed"] = True
            return _FailingExecuteCM(cm)  # the reveal session: fail it once
        return cm


async def test_a_transient_reveal_failure_is_not_cached_and_retries(db_sessionmaker) -> None:
    # The header now comes from a fallible DB reveal, so a degraded build (the
    # reveal ERRORED, not merely 'not connected') must NOT be memoized - else a
    # momentary DB hiccup would bake a wrong/absent header for the life of the
    # process (the listing error-TTL re-lists through the same source and never
    # re-derives a baked header). The server is skipped this turn, retried next.
    await _add_row(db_sessionmaker, TENANT_A, name="crm", base_url="http://crm/mcp")
    row = await _get_row(db_sessionmaker, TENANT_A, "crm")
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, credential_secret_name(row.id, row.base_url), "stored-token"
    )
    provider = McpSourceProvider(
        _settings(secret_master_key=_KEY),
        sessionmaker=_RevealFailsOnceSessionmaker(db_sessionmaker),
    )

    # Turn 1: list_enabled succeeds but the reveal errors -> crm is skipped, the
    # turn does not crash, and nothing is cached.
    assert await provider.sources_for(TENANT_A) == []
    assert provider._cache == {}

    # Turn 2: the reveal succeeds -> crm is built, cached, and carries the header.
    sources = await provider.sources_for(TENANT_A)
    assert _auth(sources[0]) == "Bearer stored-token"


async def test_reveal_raises_on_error_but_returns_none_when_not_connected(db_sessionmaker) -> None:
    # The contract the no-stale-header fix rests on: a not-connected credential is
    # a cacheable None, but a reveal that ERRORS raises so the caller declines to
    # cache the degraded build.
    name = credential_secret_name(uuid4(), "http://x/mcp")
    assert (
        await _reveal_workspace_credential(
            TENANT_A, sessionmaker=db_sessionmaker, cipher=CIPHER, name=name
        )
        is None
    )
    with pytest.raises(WorkspaceCredentialRevealError) as raised:
        await _reveal_workspace_credential(
            TENANT_A,
            sessionmaker=_FailingExecuteSessionmaker(db_sessionmaker),
            cipher=CIPHER,
            name=name,
        )
    # A database that would not answer is the definition of "try again", so the
    # verdict must not be the one that tells an admin to reconnect.
    assert raised.value.transient is True


def test_credential_secret_name_is_bound_to_id_and_url() -> None:
    # The credential key is the leak-prevention mechanism: it binds a stored
    # credential to the server's STABLE id AND its url, not its mutable name.
    a, b = uuid4(), uuid4()
    url1, url2 = "http://one/mcp", "http://two/mcp"
    # A different server (a delete + recreate under the same name gets a new id)
    # -> a different key, so the new server cannot inherit the old credential.
    assert credential_secret_name(a, url1) != credential_secret_name(b, url1)
    # A re-pointed server (same id, new url) -> a different key, so the credential
    # is orphaned and the user reconnects for the new endpoint.
    assert credential_secret_name(a, url1) != credential_secret_name(a, url2)
    # Stable for the same (id, url), so storing and revealing agree.
    assert credential_secret_name(a, url1) == credential_secret_name(a, url1)


# --- oauth renewal (the refresh slice) --------------------------------------


def _grant(
    *,
    expires_in: float,
    access_token: str = "oauth-at",
    refresh_token: str | None = "rt",
    needs_reconnect: bool = False,
) -> OAuthTokenBundle:
    """A sealed-grant bundle whose expiry is ``expires_in`` seconds from now."""
    return OAuthTokenBundle(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=utcnow() + timedelta(seconds=expires_in),
        granted_scope="repo",
        token_endpoint="https://as.example/token",
        client_id="cid",
        needs_reconnect=needs_reconnect,
        issuer="https://as.example",
    )


async def _read_grant(db_sessionmaker, name: str) -> OAuthTokenBundle:
    """Read back the sealed grant in a FRESH session (so a commit is proven)."""
    async with db_sessionmaker() as session:
        sealed = await SecretStore(session, TENANT_A, CIPHER).reveal(
            name, scope=SecretScope.WORKSPACE
        )
    return OAuthTokenBundle.unseal(sealed)


async def _oauth_row(db_sessionmaker) -> tuple[McpServer, str]:
    """An oauth-typed row and the secret name its grant lives under."""
    await _add_row(db_sessionmaker, TENANT_A, name="gh", auth_type=McpAuthType.OAUTH.value)
    row = await _get_row(db_sessionmaker, TENANT_A, "gh")
    return row, oauth_token_secret_name(row.id, row.base_url)


async def test_an_expiring_oauth_token_is_renewed_on_build(db_sessionmaker, monkeypatch) -> None:
    # Within the refresh skew, the build renews the grant in place and the
    # source carries the NEW bearer; the read-back in a fresh session proves
    # the renewal was COMMITTED, not left to roll back.
    row, name = await _oauth_row(db_sessionmaker)
    await _store_workspace_secret(db_sessionmaker, TENANT_A, name, _grant(expires_in=30).seal())

    async def fake_refresh(server, settings, tenant, *, store, bundle, http_client=None):
        renewed = bundle.model_copy(
            update={"access_token": "renewed-at", "expires_at": utcnow() + timedelta(hours=1)}
        )
        await store.store(name, renewed.seal(), scope=SecretScope.WORKSPACE)
        return renewed

    monkeypatch.setattr("app.services.oauth.flow.refresh_grant", fake_refresh)
    provider = McpSourceProvider(_settings(secret_master_key=_KEY), sessionmaker=db_sessionmaker)
    sources = await provider.sources_for(TENANT_A)
    assert _auth(sources[0]) == "Bearer renewed-at"
    assert (await _read_grant(db_sessionmaker, name)).access_token == "renewed-at"


async def test_a_needs_reconnect_grant_goes_bare_without_renewing(
    db_sessionmaker, monkeypatch
) -> None:
    # The marker is what stops a dead grant being hammered every turn: no
    # renewal attempt, no header - a stable "not connected" until a human
    # reconnects (which invalidates the cached source).
    row, name = await _oauth_row(db_sessionmaker)
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, name, _grant(expires_in=-100, needs_reconnect=True).seal()
    )

    async def explode(*args, **kwargs):
        raise AssertionError("a needs_reconnect grant must never be renewed")

    monkeypatch.setattr("app.services.oauth.flow.refresh_grant", explode)
    provider = McpSourceProvider(_settings(secret_master_key=_KEY), sessionmaker=db_sessionmaker)
    sources = await provider.sources_for(TENANT_A)
    assert _auth(sources[0]) is None


async def test_an_expired_grant_with_no_refresh_token_is_marked_dead(db_sessionmaker) -> None:
    # No renewal path exists, so the build marks the grant itself: the NEXT
    # turn short-circuits on needs_reconnect instead of re-deciding.
    row, name = await _oauth_row(db_sessionmaker)
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, name, _grant(expires_in=-100, refresh_token=None).seal()
    )
    provider = McpSourceProvider(_settings(secret_master_key=_KEY), sessionmaker=db_sessionmaker)
    sources = await provider.sources_for(TENANT_A)
    assert _auth(sources[0]) is None
    assert (await _read_grant(db_sessionmaker, name)).needs_reconnect is True


async def test_a_denied_renewal_goes_bare_instead_of_degrading(
    db_sessionmaker, monkeypatch
) -> None:
    # Denial is a STABLE outcome (the marker is stored), so the build returns a
    # bare source rather than raising the no-cache signal - unlike a transient
    # failure, there is nothing a retry could heal.
    row, name = await _oauth_row(db_sessionmaker)
    await _store_workspace_secret(db_sessionmaker, TENANT_A, name, _grant(expires_in=30).seal())

    async def deny(server, settings, tenant, *, store, bundle, http_client=None):
        dead = bundle.model_copy(update={"needs_reconnect": True})
        await store.store(name, dead.seal(), scope=SecretScope.WORKSPACE)
        raise OAuthRefreshDenied("the refresh was denied with HTTP 400")

    monkeypatch.setattr("app.services.oauth.flow.refresh_grant", deny)
    source, stale_at = await build_source_for_row(
        row,
        _settings(secret_master_key=_KEY),
        TENANT_A,
        sessionmaker=db_sessionmaker,
        cipher=CIPHER,
    )
    assert _auth(source) is None
    assert stale_at is None
    # The marker the fake stored under the flow's contract was committed.
    assert (await _read_grant(db_sessionmaker, name)).needs_reconnect is True


async def test_a_transient_renewal_failure_degrades_without_caching(
    db_sessionmaker, monkeypatch
) -> None:
    # A provider outage mid-renewal may clear, so the build must raise the
    # no-cache signal - baking a bare source would outlive the outage.
    row, name = await _oauth_row(db_sessionmaker)
    await _store_workspace_secret(db_sessionmaker, TENANT_A, name, _grant(expires_in=30).seal())

    async def flaky(*args, **kwargs):
        raise OAuthFlowError("the refresh failed with HTTP 503")

    monkeypatch.setattr("app.services.oauth.flow.refresh_grant", flaky)
    with pytest.raises(WorkspaceCredentialRevealError) as raised:
        await build_source_for_row(
            row,
            _settings(secret_master_key=_KEY),
            TENANT_A,
            sessionmaker=db_sessionmaker,
            cipher=CIPHER,
        )
    # TRANSIENT, and the flag is the point rather than a detail: an outage
    # reported as a dead grant sends an admin to Disconnect, which revokes a
    # perfectly healthy refresh token upstream. Flip this raise to terminal and
    # the Test-Connection wording flips with it.
    assert raised.value.transient is True


async def test_an_unparseable_grant_is_terminal_not_transient(db_sessionmaker) -> None:
    # The one case nothing can retry its way out of: bytes that are not a
    # bundle stay not-a-bundle, so this must be the terminal verdict that tells
    # a human to reconnect. It is the only raise site that may be terminal -
    # everything else here is a failure that may clear on its own.
    row, name = await _oauth_row(db_sessionmaker)
    await _store_workspace_secret(db_sessionmaker, TENANT_A, name, "not-a-sealed-bundle")

    with pytest.raises(WorkspaceCredentialRevealError) as raised:
        await build_source_for_row(
            row,
            _settings(secret_master_key=_KEY),
            TENANT_A,
            sessionmaker=db_sessionmaker,
            cipher=CIPHER,
        )
    assert raised.value.transient is False


async def test_a_cached_oauth_source_expires_with_its_token(db_sessionmaker, monkeypatch) -> None:
    # The memo must die WITH the token: while fresh it is served (a re-sealed
    # grant is NOT picked up - that is what invalidate is for), and past the
    # staleness horizon the same key rebuilds and picks up the current grant.
    row, name = await _oauth_row(db_sessionmaker)
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, name, _grant(expires_in=300, access_token="token-a").seal()
    )
    provider = McpSourceProvider(_settings(secret_master_key=_KEY), sessionmaker=db_sessionmaker)
    assert _auth((await provider.sources_for(TENANT_A))[0]) == "Bearer token-a"

    # Replace the grant at rest; the memoized source still serves token-a.
    await _store_workspace_secret(
        db_sessionmaker, TENANT_A, name, _grant(expires_in=3600, access_token="token-b").seal()
    )
    assert _auth((await provider.sources_for(TENANT_A))[0]) == "Bearer token-a"

    # Cross the horizon (expiry minus skew): the rebuild sees the new grant.
    real_now = utcnow()
    monkeypatch.setattr(
        "app.services.tools.mcp_sources.utcnow", lambda: real_now + timedelta(seconds=250)
    )
    assert _auth((await provider.sources_for(TENANT_A))[0]) == "Bearer token-b"
