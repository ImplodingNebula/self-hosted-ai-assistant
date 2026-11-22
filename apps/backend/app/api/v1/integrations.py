"""Integrations endpoints: register MCP servers and connect their credentials.

The write surface over the per-tenant ``mcp_servers`` registry (PR2 is the read
side). Routers stay HTTP-only: data access goes through the tenant-scoped
``McpServerRepository`` and the ``SecretStore`` seam, so a foreign-tenant id is
indistinguishable from a missing one (404), and a malformed id is a 404 too.

Registration is admin-gated and SSRF-adjacent (an admin can point the backend at
an internal URL), so every mutation is recorded in the audit log with the actor -
including the ``base_url`` on create and re-point, the one place the "name the
server, never the URL" rule is deliberately relaxed, because recording *where* an
admin pointed the backend is the whole point of that trail. A credential VALUE is
never logged and never returned.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import (
    get_current_tenant,
    get_mcp_source_provider,
    get_secret_cipher,
    get_secret_store,
    get_session,
    get_sessionmaker,
    get_settings,
    require_role,
)
from app.core.config import Settings
from app.core.errors import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ServiceUnavailableError,
)
from app.core.tenancy import TenantContext
from app.models import McpAuthType, McpCredentialScope
from app.providers.secrets import SecretScope, SecretStoreUnavailableError
from app.providers.secrets.cipher import SecretCipher
from app.repositories.mcp_servers import McpServerRepository
from app.repositories.oauth_authorizations import OAuthAuthorizationRepository
from app.schemas.integrations import (
    ConnectionTestError,
    ConnectionTestRequest,
    ConnectionTestResult,
    CredentialStatus,
    CredentialWrite,
    IntegrationOAuthConnectRequest,
    IntegrationOAuthConnectResponse,
    IntegrationOAuthRedirectUriOut,
    IntegrationPresetListResponse,
    IntegrationServerCreate,
    IntegrationServerListResponse,
    IntegrationServerOut,
    IntegrationServerUpdate,
    ServerConnectionTestRequest,
)
from app.services import integrations as integrations_service
from app.services.integrations_presets import PRESETS
from app.services.oauth import flow as oauth_flow
from app.services.oauth.authorize import OAuthEndpoints
from app.services.oauth.flow import OAuthFlowError
from app.services.secret_store import SecretStore
from app.services.tools.mcp_sources import (
    McpSourceProvider,
    WorkspaceCredentialRevealError,
    credential_secret_name,
    env_bootstrap_key_for,
    env_seed_names,
    reveal_oauth_bearer,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["integrations"])

_NOT_FOUND_DETAIL = "MCP server not found."


def _storage_unavailable(consequence: str) -> ServiceUnavailableError:
    """The 503 for an unset ``SECRET_MASTER_KEY``, with the operator's remedy.

    One builder so every path says the same thing. The remedy belongs IN the
    message: an unset key is a deployment gap that nobody can fix from the page
    reporting it, and a message that stops at the variable's name reads as an
    instruction to go hunting.
    """
    return ServiceUnavailableError(
        f"Credential storage is unavailable: this deployment has no SECRET_MASTER_KEY "
        f"set, so {consequence}. Set that variable in the deployment environment "
        f"(a base64 32-byte key) and restart the backend, then try again."
    )


SessionDep = Annotated[AsyncSession, Depends(get_session)]
TenantDep = Annotated[TenantContext, Depends(get_current_tenant)]
AdminDep = Annotated[TenantContext, Depends(require_role("admin"))]
StoreDep = Annotated[SecretStore, Depends(get_secret_store)]
ProviderDep = Annotated[McpSourceProvider | None, Depends(get_mcp_source_provider)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionmakerDep = Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)]
CipherDep = Annotated[SecretCipher | None, Depends(get_secret_cipher)]


def _parse_id(raw: str) -> UUID:
    """Parse a path id, treating a malformed value as an absent resource (404)."""
    try:
        return UUID(raw)
    except ValueError as exc:
        raise NotFoundError(_NOT_FOUND_DETAIL) from exc


def _is_env_seeded(server, settings: Settings) -> bool:
    """Whether the deployment's env currently bootstraps a row of this NAME.

    Computed, never stored. The seeding path is get-or-create by (tenant, name),
    so the name - not the id, not the URL - is what brings a row back after a
    restart. Keeping the check computed is what makes it self-healing in both
    directions: drop the variable and the row becomes ordinary, and a row the
    env has never named is never touched by this.
    """
    return server.name in env_seed_names(settings)


async def _credential_state(
    server, *, oauth_connected: bool, store: SecretStore, settings: Settings
) -> str:
    """Whether this row can authenticate, in one answer across all three auth types.

    A stored credential wins; absent one, the env bootstrap question is asked of
    :func:`~app.services.tools.mcp_sources.env_bootstrap_key_for` - the SAME
    function the chat path resolves its header with, not a copy of its rules. A
    copy is what this was, and it drifted: the chat path grew a second sidecar's
    key and the copy did not, so a deployment setting ``DOCGEN_API_KEY`` sent
    that key on every call while this card read "missing". A status that
    disagrees with what a call actually sends is worse than no status at all.

    Reads metadata only - existence, never plaintext - so it needs no master key
    and cannot leak a credential into a listing.
    """
    if server.auth_type == McpAuthType.NONE.value:
        return "not_required"
    if server.auth_type == McpAuthType.OAUTH.value:
        return "connected" if oauth_connected else "missing"
    if await store.get(_credential_name(server), scope=_scope_for(server)) is not None:
        return "connected"
    if env_bootstrap_key_for(server.base_url, settings) is not None:
        return "deployment_key"
    return "missing"


async def _serialize(
    session: AsyncSession,
    tenant: TenantContext,
    server,
    *,
    store: SecretStore,
    settings: Settings,
) -> IntegrationServerOut:
    """The wire shape, with grant status (oauth rows), credential state and seeded-ness.

    ``seeded`` is computed here, against the CURRENT settings, so every read
    answers it the same way the delete gate does - see :func:`delete_server`.
    """
    meta = (
        await integrations_service.oauth_grant_metadata(session, tenant, server)
        if server.auth_type == McpAuthType.OAUTH.value
        else None
    )
    return IntegrationServerOut.from_entity(
        server,
        oauth_connected=meta is not None,
        oauth_connected_at=None if meta is None else meta.updated_at,
        credential_state=await _credential_state(
            server, oauth_connected=meta is not None, store=store, settings=settings
        ),
        seeded=_is_env_seeded(server, settings),
    )


async def _load(repo: McpServerRepository, raw_id: str):
    """Load the tenant's server by path id, or raise 404."""
    server = await repo.get(_parse_id(raw_id))
    if server is None:
        raise NotFoundError(_NOT_FOUND_DETAIL)
    return server


def _scope_for(server) -> SecretScope:
    """The secret scope a server's credential lives in, from its credential_scope."""
    if server.credential_scope == McpCredentialScope.USER.value:
        return SecretScope.USER
    return SecretScope.WORKSPACE


def _credential_name(server) -> str:
    """The SecretStore name this server's credential is keyed under (id + url).

    Keyed on the server's stable identity, not its mutable display name, so a
    recreated or re-pointed server cannot inherit a prior server's credential -
    see :func:`credential_secret_name`. Storing, reading, and deleting a
    credential all derive it the same way from the current row.
    """
    return credential_secret_name(server.id, server.base_url)


def _require_credential_access(tenant: TenantContext, server) -> None:
    """Gate a credential op by the server's scope.

    A WORKSPACE credential is shared by the whole tenant, so only an admin may
    set or clear it. A USER credential is the acting member's own (the SecretStore
    scopes it to their user_id), so any member may manage theirs.
    """
    if server.credential_scope == McpCredentialScope.WORKSPACE.value and tenant.role != "admin":
        raise ForbiddenError("Only an admin may manage a workspace server's shared credential.")


def _invalidate(provider: McpSourceProvider | None, tenant: TenantContext, server_id: UUID) -> None:
    """Drop the cached source for a server so the next turn rebuilds it."""
    if provider is not None:
        provider.invalidate(tenant.tenant_id, server_id)


async def _forget_oauth_secrets(
    server, settings: Settings, tenant: TenantContext, store: SecretStore
) -> None:
    """Revoke and erase a server's OAuth secrets, never blocking the caller's write.

    Used where a server's IDENTITY stops being the one its grant consented to -
    a delete, a re-point, a move off ``oauth``. Both secret names derive from
    (id, base_url), so after such a change nothing can name them again: the
    grant would live on at the provider with no API or UI able to list, revoke
    or delete it.

    Every failure is swallowed deliberately. Revocation is best effort, a
    missing ``SECRET_MASTER_KEY`` makes the reveals fail, and neither may leave
    an undeletable row or an unmodifiable server - the local deletes inside
    :func:`~app.services.oauth.flow.forget_grant` need no cipher and run
    regardless.
    """
    try:
        await oauth_flow.forget_grant(server, settings, tenant, store=store)
    except Exception:  # noqa: BLE001 - the local delete must not be blocked by any of it
        logger.warning(
            "forgetting the OAuth grant for server %s failed; continuing with the "
            "local change (the grant may still be live upstream)",
            server.id,
            exc_info=True,
        )


def _audit(
    tenant: TenantContext,
    action: str,
    *,
    name: str,
    server_id: UUID,
    base_url: str | None = None,
) -> None:
    """Record one mutating action with its actor. Never the credential value."""
    logger.info(
        "integration %s by tenant=%s user=%s: server=%r id=%s%s",
        action,
        tenant.tenant_id,
        tenant.user_id,
        name,
        server_id,
        f" base_url={base_url}" if base_url else "",
    )


@router.get("/integrations/servers")
async def list_servers(
    session: SessionDep,
    tenant: TenantDep,
    cipher: CipherDep,
    store: StoreDep,
    settings: SettingsDep,
) -> IntegrationServerListResponse:
    """List the tenant's registered servers (enabled and disabled), name-ordered.

    Carries whether this deployment can seal credentials at all, so a client can
    say so BEFORE an admin types one - see the schema's note.
    """
    rows = await McpServerRepository(session, tenant).list_all()
    return IntegrationServerListResponse(
        items=[await _serialize(session, tenant, r, store=store, settings=settings) for r in rows],
        credential_storage_available=cipher is not None,
    )


@router.post("/integrations/servers/test-connection")
async def test_server_connection(
    payload: ConnectionTestRequest,
    admin: AdminDep,
    settings: SettingsDep,
) -> ConnectionTestResult:
    """Probe a server (initialize + tools/list ONLY) and report its tools or an error.

    Always 200: a target being down is a diagnostic result carried in
    ``error.code``, not a fault of this API. Never runs a tool. Admin-gated: a
    handshake reaches an arbitrary URL, so it is as SSRF-adjacent as registration.
    """
    return await integrations_service.test_connection(settings, admin, payload)


@router.get("/integrations/servers/{server_id}")
async def get_server(
    server_id: str,
    session: SessionDep,
    tenant: TenantDep,
    store: StoreDep,
    settings: SettingsDep,
) -> IntegrationServerOut:
    """Return one registered server's metadata (never its credential)."""
    server = await _load(McpServerRepository(session, tenant), server_id)
    return await _serialize(session, tenant, server, store=store, settings=settings)


@router.post("/integrations/servers", status_code=201)
async def create_server(
    payload: IntegrationServerCreate,
    response: Response,
    session: SessionDep,
    admin: AdminDep,
    store: StoreDep,
    settings: SettingsDep,
) -> IntegrationServerOut:
    """Register a new MCP server. Admin-gated; a duplicate name is a 409."""
    repo = McpServerRepository(session, admin)
    server = await repo.create(
        name=payload.name,
        base_url=payload.base_url,
        credential_scope=payload.credential_scope.value,
        auth_type=payload.auth_type.value,
        config=payload.config,
    )
    await repo.commit()
    _audit(admin, "create", name=server.name, server_id=server.id, base_url=server.base_url)
    response.headers["Location"] = f"/api/v1/integrations/servers/{server.id}"
    # Through the same serializer as every other read: a fresh row still has a
    # credential state to report (a chassis-URL row is born deployment-key
    # backed), and one code path means the field cannot drift between endpoints.
    return await _serialize(session, admin, server, store=store, settings=settings)


@router.patch("/integrations/servers/{server_id}")
async def update_server(
    server_id: str,
    payload: IntegrationServerUpdate,
    session: SessionDep,
    admin: AdminDep,
    provider: ProviderDep,
    store: StoreDep,
    settings: SettingsDep,
) -> IntegrationServerOut:
    """Patch a server (enable/disable, re-point, rescope, edit config).

    An empty body is a no-op. Any real change invalidates the cached source so the
    next turn rebuilds from the committed row.
    """
    repo = McpServerRepository(session, admin)
    server = await _load(repo, server_id)
    provided = payload.model_dump(exclude_unset=True)
    # Decided on the RESULTING combination, and BEFORE anything is mutated or
    # revoked below - a refused patch must leave the grant exactly as it was.
    resulting_auth = (
        McpAuthType(provided["auth_type"]).value
        if provided.get("auth_type") is not None
        else server.auth_type
    )
    resulting_scope = (
        McpCredentialScope(provided["credential_scope"]).value
        if provided.get("credential_scope") is not None
        else server.credential_scope
    )
    if (
        resulting_auth == McpAuthType.OAUTH.value
        and resulting_scope == McpCredentialScope.USER.value
    ):
        # Same guard as create: an OAuth grant is workspace-scoped until real
        # login (Phase 8b), and a per-user row would pass the member gate while
        # every reveal hit the shared grant.
        raise BadRequestError(
            "Per-user OAuth connections are not available yet: an OAuth "
            "server's grant is workspace-scoped until real login lands."
        )
    if (
        resulting_auth == McpAuthType.NONE.value
        and resulting_scope == McpCredentialScope.USER.value
    ):
        # Same guard as create, and permanent: an auth-none source attaches no
        # credential for anyone, so a per-user row would promise a fail-closed
        # per-member gate that no call path enforces.
        raise BadRequestError(
            "A per-user scope needs a per-user credential: a server that "
            "takes no credential is shared by nature."
        )
    # An OAuth grant is welded to the (id, base_url) pair it consented to, so a
    # re-point - or a move off oauth - makes it unreachable by name forever
    # after. Revoke and erase it FIRST, while the row still names it: the
    # alternative is a live third-party refresh token no endpoint can see. The
    # card already reports such a server as disconnected (the derived secret
    # name no longer resolves), so this changes no visible behaviour - it makes
    # the disconnection true upstream as well as locally.
    repointing = provided.get("base_url") is not None and provided["base_url"] != server.base_url
    if server.auth_type == McpAuthType.OAUTH.value and (
        repointing or resulting_auth != McpAuthType.OAUTH.value
    ):
        await _forget_oauth_secrets(server, settings, admin, store)
    changed = False
    repointed = False
    if provided.get("enabled") is not None:
        server.enabled = provided["enabled"]
        changed = True
    if provided.get("base_url") is not None:
        server.base_url = provided["base_url"]
        changed = repointed = True
    if provided.get("credential_scope") is not None:
        server.credential_scope = McpCredentialScope(provided["credential_scope"]).value
        changed = True
    if provided.get("auth_type") is not None:
        server.auth_type = McpAuthType(provided["auth_type"]).value
        changed = True
    if provided.get("config") is not None:
        # Wholesale replace, by contract. Callers must send the FULL blob: the
        # row also carries backend-written keys (connected_account) and the
        # UI-written preset_id, and a partial PATCH here silently drops them.
        server.config = provided["config"]
        changed = True
    if changed:
        await repo.commit()
        _invalidate(provider, admin, server.id)
        _audit(
            admin,
            "update",
            name=server.name,
            server_id=server.id,
            base_url=server.base_url if repointed else None,
        )
    return await _serialize(session, admin, server, store=store, settings=settings)


@router.delete("/integrations/servers/{server_id}", status_code=204)
async def delete_server(
    server_id: str,
    session: SessionDep,
    admin: AdminDep,
    store: StoreDep,
    provider: ProviderDep,
    settings: SettingsDep,
) -> Response:
    """Delete a server, its shared workspace credential, and any OAuth grant.

    A row the deployment's environment currently seeds is REFUSED with a 409
    instead, because deleting it would not last: the bootstrap is get-or-create
    by name, so the row returns at the next restart with its credential and its
    tool toggles gone. A button that appears to work and quietly undoes itself
    is worse than one that explains the constraint, and the durable off switch
    already exists - ``enabled`` survives every restart precisely because
    seeding never overwrites an existing row. This is the config-owned carve-out
    of the disclose-and-migrate rule: nothing here can repair the request, since
    what recreates the row lives in the deployment's environment, not in the
    request. The refusal comes FIRST, before any destructive step - a delete
    that is going to be refused must not revoke an OAuth grant on its way out.

    The row and its workspace secrets go in one transaction. An OAuth-connected
    server is REVOKED upstream first (best effort) and both of its sealed
    secrets - the token bundle and the client registration - are deleted with
    it: they are separate rows with no foreign key to ``mcp_servers``, keyed on
    a name derivable only from the id and URL that are about to vanish, so
    leaving them would strand a live third-party refresh token that no API or
    UI could ever list, let alone revoke. A dead provider or a missing
    ``SECRET_MASTER_KEY`` is logged and does not block the delete - an
    undeletable row would be the worse failure.

    User credentials across members are left as harmless orphans, and here they
    genuinely are: every credential is keyed on the server's stable id (see
    :func:`credential_secret_name`), so a later server - even one recreated under
    the same name - gets a fresh id and can never derive the orphaned name, and
    the SecretStore reaches only the acting principal's rows regardless.
    """
    repo = McpServerRepository(session, admin)
    server = await _load(repo, server_id)
    if _is_env_seeded(server, settings):
        raise ConflictError(
            f"{server.name!r} is seeded from this deployment's configuration, so "
            "deleting it would only last until the next restart. Turn it off to "
            "withhold its tools - that survives restarts - or stop configuring it "
            "in the deployment environment to make it deletable."
        )
    name, server_uuid = server.name, server.id
    workspace_secret = _credential_name(server)
    # Before the row goes: it is the only thing that can still name the grant.
    # Unconditional, not gated on auth_type - a row hand-edited off oauth, or
    # switched in an earlier patch, would otherwise strand its secrets here.
    await _forget_oauth_secrets(server, settings, admin, store)
    await repo.delete(server)
    await store.delete(workspace_secret, scope=SecretScope.WORKSPACE)
    await repo.commit()
    _invalidate(provider, admin, server_uuid)
    _audit(admin, "delete", name=name, server_id=server_uuid)
    return Response(status_code=204)


@router.put("/integrations/servers/{server_id}/credential")
async def set_credential(
    server_id: str,
    payload: CredentialWrite,
    session: SessionDep,
    tenant: TenantDep,
    store: StoreDep,
    provider: ProviderDep,
) -> CredentialStatus:
    """Seal a server's credential (write-only). Returns metadata + last4, never plaintext.

    Scope follows the server: a workspace server's shared credential (admin only)
    or the acting member's own for a user-scoped server. 503 when
    ``SECRET_MASTER_KEY`` is unset - never plaintext, never a throwaway key.
    """
    repo = McpServerRepository(session, tenant)
    server = await _load(repo, server_id)
    _require_credential_access(tenant, server)
    scope = _scope_for(server)
    try:
        metadata = await store.store(_credential_name(server), payload.value, scope=scope)
    except SecretStoreUnavailableError as exc:
        raise _storage_unavailable("credentials cannot be sealed") from exc
    await session.commit()
    # A workspace credential feeds the shared-header path, so a rebuild must pick
    # it up; a user credential is revealed per call, so no rebuild is needed.
    if scope is SecretScope.WORKSPACE:
        _invalidate(provider, tenant, server.id)
    _audit(tenant, "set_credential", name=server.name, server_id=server.id)
    return CredentialStatus.from_metadata(metadata)


@router.get("/integrations/servers/{server_id}/credential")
async def get_credential(
    server_id: str,
    session: SessionDep,
    tenant: TenantDep,
    store: StoreDep,
) -> CredentialStatus:
    """Return whether a server has a credential, plus its metadata hint - no plaintext."""
    repo = McpServerRepository(session, tenant)
    server = await _load(repo, server_id)
    _require_credential_access(tenant, server)
    metadata = await store.get(_credential_name(server), scope=_scope_for(server))
    return (
        CredentialStatus.not_connected()
        if metadata is None
        else CredentialStatus.from_metadata(metadata)
    )


@router.delete("/integrations/servers/{server_id}/credential", status_code=204)
async def delete_credential(
    server_id: str,
    session: SessionDep,
    tenant: TenantDep,
    store: StoreDep,
    provider: ProviderDep,
) -> Response:
    """Clear a server's credential (scope-dependent gate, like the write)."""
    repo = McpServerRepository(session, tenant)
    server = await _load(repo, server_id)
    _require_credential_access(tenant, server)
    scope = _scope_for(server)
    await store.delete(_credential_name(server), scope=scope)
    await session.commit()
    if scope is SecretScope.WORKSPACE:
        _invalidate(provider, tenant, server.id)
    _audit(tenant, "delete_credential", name=server.name, server_id=server.id)
    return Response(status_code=204)


@router.post("/integrations/servers/{server_id}/test-connection")
async def test_saved_server_connection(
    server_id: str,
    payload: ServerConnectionTestRequest,
    session: SessionDep,
    tenant: TenantDep,
    store: StoreDep,
    settings: SettingsDep,
    sessionmaker: SessionmakerDep,
    cipher: CipherDep,
) -> ConnectionTestResult:
    """Re-test a registered server using its STORED credential (or a one-shot override).

    Unlike the pre-save ``POST /servers/test-connection`` (an ad-hoc URL + typed
    key), this probes a saved row with the credential the chat path would send, so a
    connected-and-healthy server no longer reports ``auth_rejected`` on a
    credential-less probe. An ``oauth`` row resolves its grant's live bearer through
    the SAME helper the chat path uses (renewing a stale token in place), so a
    connected server tests exactly as it would list. Scope-gated like the credential
    ops (a workspace server's shared credential is admin-only; a user-scoped server
    tests the acting member's own). Always 200 - a down target is a diagnostic in
    ``error.code``, never a credential value. The handshake never runs a tool.
    """
    repo = McpServerRepository(session, tenant)
    server = await _load(repo, server_id)
    _require_credential_access(tenant, server)
    oauth_bearer: str | None = None
    if server.auth_type == McpAuthType.OAUTH.value:
        try:
            oauth_bearer, _ = await reveal_oauth_bearer(
                server, settings, tenant, sessionmaker=sessionmaker, cipher=cipher
            )
        except WorkspaceCredentialRevealError as exc:
            # A corrupt/tampered grant or a renewal the provider refused: exactly
            # what an admin clicks Test to learn, so it is a classified
            # diagnostic - the endpoint's always-200 contract holds, and the
            # detail names the STATE, never the exception's message (which
            # embeds URLs that may embed credentials - see _classify).
            #
            # Transient and terminal must not read alike. "Reconnect needed" one
            # click from Disconnect turns a five-minute provider outage into a
            # revoked, healthy refresh token and a fresh consent to recover it.
            logger.warning("oauth grant unusable for a test of server %s", server.id, exc_info=True)
            error = (
                ConnectionTestError(code="unreachable", detail="renewal failed - retry")
                if exc.transient
                else ConnectionTestError(code="auth_rejected", detail="reconnect needed")
            )
            return ConnectionTestResult(ok=False, error=error)
    return await integrations_service.test_saved_connection(
        settings,
        tenant,
        server,
        store,
        override_credential=payload.credential,
        oauth_bearer=oauth_bearer,
    )


# --- OAuth connect / callback / disconnect ---------------------------------


def _require_cipher(cipher: SecretCipher | None) -> SecretCipher:
    """The flow needs the secret cipher end to end; unset key = 503, like credentials."""
    if cipher is None:
        raise _storage_unavailable("an OAuth grant cannot be stored")
    return cipher


@router.post("/integrations/servers/{server_id}/oauth/connect")
async def oauth_connect(
    server_id: str,
    payload: IntegrationOAuthConnectRequest,
    session: SessionDep,
    admin: AdminDep,
    store: StoreDep,
    cipher: CipherDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> IntegrationOAuthConnectResponse:
    """Start a workspace OAuth consent; answer with the provider URL to open.

    Admin-gated like every registry write. With ``endpoints`` supplied the
    request is the generic-provider path (no discovery, no resource binding);
    without, the authorization server is discovered from the MCP server
    itself. Nothing here runs a tool or stores a token - the callback does the
    storing, and abandoning the returned URL leaves only an expiring row.
    """
    repo = McpServerRepository(session, admin)
    server = await _load(repo, server_id)
    if server.credential_scope == McpCredentialScope.USER.value:
        # The third write path into the forbidden combination (create and patch
        # already refuse it), and the only one that would mint a real grant:
        # completing this consent flips auth_type to oauth, after which every
        # member's calls run under one admin-sealed WORKSPACE grant while the
        # card claims per-user credentials - and the row's credential endpoints
        # drop from admin-only to any-member. Same message as the patch guard.
        raise BadRequestError(
            "Per-user OAuth connections are not available yet: an OAuth "
            "server's grant is workspace-scoped until real login lands."
        )
    endpoints = None
    if payload.endpoints is not None:
        try:
            endpoints = OAuthEndpoints(
                issuer=payload.endpoints.issuer,
                authorization_endpoint=payload.endpoints.authorization_endpoint,
                token_endpoint=payload.endpoints.token_endpoint,
                revocation_endpoint=payload.endpoints.revocation_endpoint,
                send_resource=False,
            )
        except ValidationError as exc:
            raise BadRequestError(f"invalid endpoints: {exc.errors()[0].get('msg')}") from exc
    try:
        url = await oauth_flow.start_connect(
            server,
            settings,
            admin,
            repo=OAuthAuthorizationRepository(session, admin),
            store=store,
            cipher=_require_cipher(cipher),
            client_id=payload.client_id,
            client_secret=payload.client_secret,
            scopes=payload.scopes,
            endpoints=endpoints,
            extra_authorize_params=payload.extra_authorize_params,
            userinfo_endpoint=None
            if payload.identity is None
            else payload.identity.userinfo_endpoint,
            userinfo_label_field=None if payload.identity is None else payload.identity.label_field,
        )
    except OAuthFlowError as exc:
        raise BadRequestError(str(exc)) from exc
    _audit(admin, "oauth_connect_started", name=server.name, server_id=server.id)
    return IntegrationOAuthConnectResponse(authorization_url=url)


@router.get("/integrations/presets")
async def list_presets(tenant: TenantDep) -> IntegrationPresetListResponse:
    """The curated presets: prefill values plus setup notes, display-ordered.

    Static, read-only metadata - no credential, no tenant data - so any member
    may read it; connecting anything still goes through the admin-gated
    routes above.
    """
    return IntegrationPresetListResponse(items=list(PRESETS))


@router.get("/integrations/oauth/redirect-uri")
async def get_oauth_redirect_uri(
    tenant: TenantDep, settings: SettingsDep
) -> IntegrationOAuthRedirectUriOut:
    """The literal redirect URI every consent this deployment starts will use.

    Providers only deliver the authorization code to a URL registered on the
    app, so the operator must paste this exact value into the provider's app
    settings BEFORE connecting. Deployment config (`OAUTH_REDIRECT_BASE_URL`
    plus the callback path), never a secret - any member may read it.
    """
    return IntegrationOAuthRedirectUriOut(redirect_uri=oauth_flow.redirect_uri(settings))


@router.get("/integrations/oauth/callback")
async def oauth_callback(
    session: SessionDep,
    tenant: TenantDep,
    store: StoreDep,
    cipher: CipherDep,
    provider: ProviderDep,
    settings: Annotated[Settings, Depends(get_settings)],
    state: str = "",
    code: str = "",
    error: str = "",
) -> RedirectResponse:
    """The browser's return leg: consume the state, exchange, and go home.

    Always answers a 302 to the Integrations page with NO query string - the
    authorization code must not ride into another access log, and the page
    reads connection status from the API rather than the URL. A denial or a
    failed exchange still consumes the state (single use), logs the reason,
    and redirects; the card simply stays disconnected.
    """
    home = f"{settings.oauth_redirect_base_url.rstrip('/')}/console/integrations"
    if not state:
        return RedirectResponse(home, status_code=302)
    if error or not code:
        # The user denied (or the provider errored): burn the state so the
        # flow cannot be resumed with an injected code later.
        repo = OAuthAuthorizationRepository(session, tenant)
        await repo.consume(state)
        await repo.commit()
        logger.info("oauth callback ended without a code (error=%r)", error or None)
        return RedirectResponse(home, status_code=302)
    if cipher is None:
        # A keyless deployment cannot hold pending state, so nothing real can
        # complete - but the browser still gets the clean redirect the
        # contract promises, never a 503 describing the deployment's config.
        logger.warning("oauth callback arrived with SECRET_MASTER_KEY unset; redirecting")
        return RedirectResponse(home, status_code=302)
    try:
        server_id = await oauth_flow.complete_callback(
            state=state,
            code=code,
            settings=settings,
            tenant=tenant,
            repo=OAuthAuthorizationRepository(session, tenant),
            servers=McpServerRepository(session, tenant),
            store=store,
            cipher=cipher,
        )
    except OAuthFlowError as exc:
        logger.warning("oauth callback failed: %s", exc)
        return RedirectResponse(home, status_code=302)
    _invalidate(provider, tenant, server_id)
    _audit(tenant, "oauth_connected", name="", server_id=server_id)
    return RedirectResponse(home, status_code=302)


@router.post("/integrations/servers/{server_id}/oauth/disconnect", status_code=204)
async def oauth_disconnect(
    server_id: str,
    session: SessionDep,
    admin: AdminDep,
    store: StoreDep,
    provider: ProviderDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Revoke the server's grant (best effort) and delete the sealed bundle."""
    repo = McpServerRepository(session, admin)
    server = await _load(repo, server_id)
    await oauth_flow.disconnect(server, settings, admin, store=store)
    if "connected_account" in server.config:
        # The label describes a grant that no longer exists.
        server.config = {k: v for k, v in server.config.items() if k != "connected_account"}
    await repo.commit()
    _invalidate(provider, admin, server.id)
    _audit(admin, "oauth_disconnected", name=server.name, server_id=server.id)
