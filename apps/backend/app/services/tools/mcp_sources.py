"""The DB-backed MCP source registry: per-tenant tool sources from stored rows.

Slice 6b PR2 makes ``mcp_servers`` the live source of MCP tool sources. Where
slice 4 built ONE source from the ``MCP_CHASSIS_URL`` env var for the whole
process, this reads the acting tenant's enabled ``mcp_servers`` rows and builds
one listing-cached source per row - so a per-tenant registry, edited in the DB
(and, from PR3, the Integrations UI), drives what tools a turn can reach.

Two things keep this correct and cheap:

- **Tenant isolation is structural.** Each built :class:`CachingToolSource` is
  memoized under a key that INCLUDES ``tenant_id``, so two tenants never share a
  source instance and one tenant's cached listing can never be served to another.
  The cache inside :class:`CachingToolSource` stays safe UNKEYED precisely
  because each instance is dedicated to one ``(tenant, server)`` and a LISTING is
  identity-independent (it uses the workspace header) - see that class's security
  note.
- **The env server is a bootstrap SEED, not the source of truth.** On a tenant's
  first use its ``MCP_CHASSIS_URL`` is upserted into ``mcp_servers`` once
  (idempotent, non-destructive), so a fresh env-only deployment works with zero
  config while the DB row becomes authoritative thereafter.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from app.models import McpAuthType, McpCredentialScope, utcnow
from app.providers.mcp.source import McpServerConfig, McpToolSource
from app.providers.secrets import (
    SecretError,
    SecretNotFoundError,
    SecretScope,
    build_cipher,
)
from app.providers.tools.base import ToolSource
from app.repositories.mcp_servers import McpServerRepository
from app.services.oauth import flow as oauth_flow
from app.services.oauth.bundle import OAuthTokenBundle, oauth_token_secret_name
from app.services.oauth.flow import OAuthFlowError, OAuthRefreshDenied
from app.services.oauth.locks import grant_lock
from app.services.secret_store import SecretStore
from app.services.tools.mcp_credentials import PerUserCredentialResolver
from app.services.tools.naming import credential_secret_name
from app.services.tools.sources import CachingToolSource

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.core.config import Settings
    from app.core.tenancy import TenantContext
    from app.models import McpServer
    from app.providers.secrets import SecretCipher

logger = logging.getLogger(__name__)

# Renew an OAuth access token this long before it actually expires, so a token
# that is valid NOW is still valid when the tool call it authorizes lands.
_OAUTH_REFRESH_SKEW = timedelta(seconds=60)

# One renewal in flight per sealed grant: concurrent turns hitting the same
# expiring token would otherwise race the token endpoint, and a provider that
# ROTATES refresh tokens invalidates the loser's copy. The registry lives in
# app.services.oauth.locks because an admin's Disconnect must take the SAME
# lock, and that path cannot import this module (this one imports the flow).


def _disabled_tools(config: dict) -> frozenset[str]:
    """The per-tool deny-list from a server row's config, parsed defensively.

    ``config`` is an operator-editable JSON blob, so a malformed value must not
    crash a turn: anything but a list of strings yields an empty deny-list (offer
    everything) rather than raising.
    """
    raw = config.get("disabled_tools")
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(item for item in raw if isinstance(item, str))


# --- env bootstrap: what deployment config seeds, and what it backs ---------
#
# Both questions are answered here, module level and pure, because both have a
# second caller outside the chat path: the Integrations API has to report which
# rows the deployment recreates on restart, and which rows an env key already
# authenticates. Answering either question a second time elsewhere is how the
# credential-state readout drifted from what a call actually sends.


def env_seeds(settings: Settings) -> list[tuple[str, str, str]]:
    """The (name, base_url, credential_scope) rows the env asks to bootstrap.

    Each configured sidecar contributes one row. The chassis follows
    ``MCP_PER_USER_CREDENTIALS``, because it fronts an upstream system whose
    account a user can connect. doc-gen never does: it renders what it is
    handed and reaches nothing, so there is no per-user account to connect
    and a USER scope would only make every caller fail closed.

    Args:
        settings: The deployment's configuration.

    Returns:
        One tuple per env-configured sidecar, in seeding order.
    """
    seeds: list[tuple[str, str, str]] = []
    if settings.mcp_chassis_url:
        chassis_scope = (
            McpCredentialScope.USER.value
            if settings.mcp_per_user_credentials
            else McpCredentialScope.WORKSPACE.value
        )
        # Server name == the credential name, so a user-scoped call's secret
        # lookup (keyed on the server name) matches what the user connected
        # under - continuous with the PR1 env path.
        seeds.append(
            (settings.mcp_chassis_credential_name, settings.mcp_chassis_url, chassis_scope)
        )
    if settings.docgen_mcp_url:
        seeds.append(
            (
                settings.docgen_credential_name,
                settings.docgen_mcp_url,
                McpCredentialScope.WORKSPACE.value,
            )
        )
    return seeds


def env_seed_names(settings: Settings) -> frozenset[str]:
    """The registry names this deployment's env currently bootstraps.

    NAME is the resurrection key: ``McpServerRepository.ensure_seed`` is
    get-or-create by (tenant, name), so a row whose name is in this set comes
    back at the next restart no matter what happened to it. Computed from the
    live settings and never stored, so it self-heals - unset the variable and
    the row becomes an ordinary one.

    Args:
        settings: The deployment's configuration.

    Returns:
        The seeded names, empty when no sidecar is configured.
    """
    return frozenset(name for name, _, _ in env_seeds(settings))


def env_bootstrap_key_for(base_url: str, settings: Settings) -> str | None:
    """The env bootstrap credential that backs ``base_url``, if any.

    Each env key belongs to ONE url - the sidecar it was configured for - so it
    can never be handed to an unrelated upstream an admin later registers. The
    branches are written out rather than folded into a lookup table so the
    binding between each key and its URL stays visible at the point of use.

    Args:
        base_url: The registered server's URL, compared verbatim.
        settings: The deployment's configuration.

    Returns:
        The bearer value this URL authenticates with, or None when no env key
        backs it.
    """
    if settings.mcp_chassis_api_key and base_url == settings.mcp_chassis_url:
        return settings.mcp_chassis_api_key
    if settings.docgen_api_key and base_url == settings.docgen_mcp_url:
        return settings.docgen_api_key
    return None


class WorkspaceCredentialRevealError(Exception):
    """A server's stored workspace-credential reveal ERRORED (not merely absent).

    Raised by :func:`_reveal_workspace_credential` on a decrypt/tamper failure (the
    SecretStore AAD tripwire, or a corrupt row) or a DB hiccup - a state that MAY
    clear - so :meth:`McpSourceProvider.sources_for` DECLINES TO CACHE the degraded
    build and retries the reveal next turn, rather than baking a wrong/absent
    Authorization header into a memoized source for the life of the process (the
    listing error-TTL re-lists through the SAME wrapped source, so it never
    re-derives a baked header). A plainly not-connected credential is NOT this: it
    returns None and is cacheable. Carries only the secret/server name - never a URL
    or the secret - so it is safe to log.

    ``transient`` separates "try again shortly" from "a human must reconnect",
    and the distinction is one click from a destructive action: an admin told
    that a five-minute provider outage means the grant is dead will click
    Disconnect and revoke a perfectly healthy refresh token upstream. Only a
    bundle that failed to PARSE is terminal - nothing retries its way out of
    that; a failed reveal or a refused renewal may well clear on its own, so
    they default to transient.
    """

    def __init__(self, name: str, *, transient: bool = True) -> None:
        """Record the server/secret name the reveal failed for, and its kind."""
        super().__init__(f"revealing the workspace credential {name!r} failed")
        self.secret_name = name
        self.transient = transient


async def _reveal_workspace_credential(
    tenant: TenantContext,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    cipher: SecretCipher,
    name: str,
) -> str | None:
    """Reveal a server's stored WORKSPACE credential.

    Returns the plaintext when connected, or None when the server has none
    connected (the common case) OR the master key is unusable - both STABLE states
    the caller may safely cache while falling back to the env bootstrap credential.

    Raises:
        WorkspaceCredentialRevealError: The reveal ERRORED - a decrypt/tamper
            failure (the AAD tripwire) or a DB hiccup, a state that may clear. The
            caller must NOT cache the resulting build; retrying next turn is what
            keeps a momentary failure from baking a wrong/absent header for the life
            of the process. Logged at WARNING for audit.
    """
    try:
        async with sessionmaker() as session:
            store = SecretStore(session, tenant, cipher)
            return await store.reveal(name, scope=SecretScope.WORKSPACE)
    except SecretNotFoundError:
        # No workspace credential connected for this server: expected, cacheable.
        return None
    except SecretError:
        # A secret-layer configuration problem (an unusable key surfacing on the
        # reveal): a STABLE state within the process, so degrade to the env
        # bootstrap credential and let the build cache. Not a bug, no traceback.
        return None
    except Exception as exc:
        # A decrypt/tamper failure (the AAD tripwire) or a DB hiccup: a state that
        # MAY clear, so signal it rather than paper over it with a cached
        # wrong/absent header. The caller declines to cache and retries next turn.
        logger.warning(
            "revealing the workspace MCP credential %r failed; the build is "
            "degraded and will not be cached (the reveal retries next turn)",
            name,
            exc_info=True,
        )
        raise WorkspaceCredentialRevealError(name, transient=True) from exc


async def _load_bundle(
    name: str,
    tenant: TenantContext,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    cipher: SecretCipher,
) -> OAuthTokenBundle | None:
    """Reveal and parse one sealed OAuth bundle; None when none is connected.

    Raises:
        WorkspaceCredentialRevealError: The reveal ERRORED (decrypt/tamper, a DB
            hiccup) or the bundle did not parse - degraded states the caller
            must not cache; a reconnect or the next turn heals them.
    """
    try:
        async with sessionmaker() as session:
            sealed = await SecretStore(session, tenant, cipher).reveal(
                name, scope=SecretScope.WORKSPACE
            )
    except SecretNotFoundError:
        return None
    except SecretError:
        return None
    except Exception as exc:
        logger.warning(
            "revealing the OAuth bundle %r failed; the build is degraded and "
            "will not be cached (the reveal retries next turn)",
            name,
            exc_info=True,
        )
        raise WorkspaceCredentialRevealError(name, transient=True) from exc
    try:
        return OAuthTokenBundle.unseal(sealed)
    except Exception as exc:
        # A tampered or malformed bundle: reconnecting replaces it, so do not
        # bake the broken state into a cached source. TERMINAL - no retry can
        # turn unparseable bytes into a grant.
        logger.warning("the sealed OAuth bundle %r did not parse; not caching", name)
        raise WorkspaceCredentialRevealError(name, transient=False) from exc


async def reveal_oauth_bearer(
    row: McpServer,
    settings: Settings,
    tenant: TenantContext,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    cipher: SecretCipher | None,
) -> tuple[str | None, datetime | None]:
    """The access token for an OAuth row, and the instant it stops being fresh.

    ``(None, None)`` when nothing is connected, the master key is unusable, or
    the grant is marked ``needs_reconnect`` - all stable, cacheable states in
    which the server is simply "not connected" (its calls go out bare and fail
    honestly upstream; a reconnect invalidates the cached source). A fresh
    token returns with its staleness horizon (``expires_at`` minus the refresh
    skew) so the provider's memo expires WITH the token instead of serving a
    baked-in bearer past its death. At or past the horizon the grant is renewed
    in place - single-flight per grant, re-checking under the lock so a
    concurrent turn's renewal is reused rather than raced. A grant that has
    expired with no refresh token is marked ``needs_reconnect`` here, so no
    later turn retries a dead grant.

    Raises:
        WorkspaceCredentialRevealError: The reveal errored, the bundle did not
            parse, or a renewal failed TRANSIENTLY - states that may clear, so
            the degraded build is not cached and the next turn retries.
    """
    if cipher is None:
        return None, None
    name = oauth_token_secret_name(row.id, row.base_url)
    bundle = await _load_bundle(name, tenant, sessionmaker=sessionmaker, cipher=cipher)
    if bundle is None or bundle.needs_reconnect:
        return None, None
    stale_at = bundle.expires_at - _OAUTH_REFRESH_SKEW
    if utcnow() < stale_at:
        return bundle.access_token, stale_at
    async with grant_lock(name):
        # Re-read under the lock: a concurrent turn may have renewed already.
        bundle = await _load_bundle(name, tenant, sessionmaker=sessionmaker, cipher=cipher)
        if bundle is None or bundle.needs_reconnect:
            return None, None
        stale_at = bundle.expires_at - _OAUTH_REFRESH_SKEW
        if utcnow() < stale_at:
            return bundle.access_token, stale_at
        async with sessionmaker() as session:
            store = SecretStore(session, tenant, cipher)
            if not bundle.refresh_token:
                # Expired with no renewal path: dead until a human reconnects.
                # Update-only, like every other renewal-path write: a disconnect
                # that landed since the read must not be undone by a marker.
                dead = bundle.model_copy(update={"needs_reconnect": True})
                await store.store_existing_only(name, dead.seal(), scope=SecretScope.WORKSPACE)
                await session.commit()
                logger.warning(
                    "the OAuth grant for server %s expired with no refresh token; reconnect needed",
                    row.id,
                )
                return None, None
            try:
                renewed = await oauth_flow.refresh_grant(
                    row, settings, tenant, store=store, bundle=bundle
                )
            except OAuthRefreshDenied:
                # refresh_grant stored the needs_reconnect marker; persist it.
                await session.commit()
                return None, None
            except OAuthFlowError as exc:
                logger.warning(
                    "renewing the OAuth grant %r failed transiently; the build "
                    "is degraded and will not be cached (the renewal retries "
                    "next turn)",
                    name,
                )
                raise WorkspaceCredentialRevealError(name, transient=True) from exc
            await session.commit()
            return renewed.access_token, renewed.expires_at - _OAUTH_REFRESH_SKEW


async def _resolve_workspace_bearer(
    row: McpServer,
    settings: Settings,
    tenant: TenantContext,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    cipher: SecretCipher | None,
) -> str | None:
    """The shared workspace bearer a server LISTS (and, if workspace-scoped, CALLS) with.

    A stored per-server workspace credential (sealed via the Integrations API under
    the server's stable-identity key) is authoritative and overrides the env
    bootstrap key even for the seeded chassis. Absent one, an env bootstrap key
    backs ONLY a server pointed at the URL that key belongs to (see
    :func:`env_bootstrap_key_for`), so it can never be handed to an unrelated
    upstream an admin later registers. Any other server gets no shared header until
    a credential is connected. Caller-agnostic (a WORKSPACE reveal ignores
    ``user_id``), which is what keeps the built source's listing cache shareable
    across a tenant's users.
    """
    if cipher is not None:
        stored = await _reveal_workspace_credential(
            tenant,
            sessionmaker=sessionmaker,
            cipher=cipher,
            name=credential_secret_name(row.id, row.base_url),
        )
        if stored is not None:
            return stored
    return env_bootstrap_key_for(row.base_url, settings)


def _assemble_source(
    row: McpServer,
    settings: Settings,
    *,
    headers: dict[str, str],
    resolver: PerUserCredentialResolver | None = None,
) -> ToolSource:
    """Wrap a row in the (CachingToolSource -> McpToolSource) pair every path shares.

    One construction site, so the auth branches in :func:`build_source_for_row`
    cannot drift apart on the non-auth knobs (the per-tool deny-list, the
    transport timeouts, the listing TTLs).
    """
    return CachingToolSource(
        McpToolSource(
            McpServerConfig(
                name=row.name,
                url=row.base_url,
                headers=headers,
                disabled_tools=_disabled_tools(row.config),
            ),
            timeout_seconds=settings.mcp_timeout_seconds,
            list_timeout_seconds=settings.mcp_list_timeout_seconds,
            credential_resolver=resolver,
        ),
        ttl_seconds=settings.mcp_list_cache_seconds,
        error_ttl_seconds=settings.mcp_list_cache_error_seconds,
    )


async def build_source_for_row(
    row: McpServer,
    settings: Settings,
    tenant: TenantContext,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    cipher: SecretCipher | None,
) -> tuple[ToolSource, datetime | None]:
    """Build one listing-cached tool source for a registered server row.

    Returns the source and its staleness horizon: the instant a memo of this
    build must be treated as expired and rebuilt. Only an OAuth row has one
    (its bearer dies with the token; the rebuild renews it) - every other
    build returns None, meaning "fresh until invalidated".

    The server's static header is its stored workspace credential (sealed via the
    Integrations API), falling back to the env ``MCP_CHASSIS_API_KEY`` only for a
    server pointed at ``MCP_CHASSIS_URL`` - the one server that key belongs to. That
    header is used for LISTING always and, on a ``workspace`` row, for CALLS too. A
    ``credential_scope=user`` row additionally wires a per-user resolver bound
    ``required=True``, so each CALL carries the acting user's own credential and
    FAILS CLOSED when the user has connected none - never falling back to the shared
    header. A ``workspace`` row wires no resolver and every call uses the shared
    header.

    Building reveals the workspace credential in a short-lived DB session, but opens
    no connection to the MCP upstream (the source dials that on first use). A reveal
    that ERRORS - as opposed to a not-connected None - propagates as
    :class:`WorkspaceCredentialRevealError`, so :meth:`McpSourceProvider.sources_for`
    can decline to cache the degraded build and retry next turn.

    Args:
        row: The enabled ``mcp_servers`` row to adapt.
        settings: The active settings (env bootstrap credential, timeouts, TTLs).
        tenant: The acting tenant, for the workspace-credential reveal. Its user_id
            is irrelevant to a WORKSPACE reveal (owner == "workspace"), so the built
            source stays identity-independent and its listing cache shareable.
        sessionmaker: The process sessionmaker, for the workspace reveal and a
            user-scoped row's per-call credential reveal.
        cipher: The process secret cipher; None (no/malformed master key) skips the
            stored-credential reveal - the env bootstrap still applies - and fails a
            user-scoped row's calls closed.
    """
    if row.auth_type == McpAuthType.OAUTH.value:
        if row.credential_scope == McpCredentialScope.USER.value:
            # The forbidden pair. Three write paths refuse to MINT it, but this
            # is the read path, and it decides what actually goes on the wire -
            # so it refuses to HONOUR it too, for a row that arrived by any
            # other route (direct SQL, a seeder, a half-applied migration).
            # The grant is workspace-scoped; serving it to whichever member's
            # turn this is would spend one account's authority as if it were
            # theirs, while the card claims a per-user credential.
            logger.warning(
                "MCP server %r declares oauth with per-user credentials, which is not "
                "supported; attaching no credential",
                row.name,
            )
            return _assemble_source(row, settings, headers={}), None
        # The consent flow's sealed grant is this row's ONLY credential source:
        # no stored pasted credential, no env bootstrap, no per-user resolver.
        # Not connected -> calls go out bare and fail honestly upstream.
        bearer, stale_at = await reveal_oauth_bearer(
            row, settings, tenant, sessionmaker=sessionmaker, cipher=cipher
        )
        headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
        return _assemble_source(row, settings, headers=headers), stale_at
    if row.auth_type != McpAuthType.HEADER.value:
        if row.auth_type != McpAuthType.NONE.value:
            # A mechanism this build does not implement (a partial upgrade, a
            # hand-edited row). Fail CLOSED: forwarding the bearer to an
            # upstream that negotiates its own auth would leak it, silently.
            logger.warning(
                "MCP server %r declares unknown auth_type %r; attaching no credential",
                row.name,
                row.auth_type,
            )
        # An unauthenticated upstream: attach NOTHING - not a stored workspace
        # credential, not the env bootstrap, not a per-user resolver - even when
        # a credential exists in the store. The row's declared auth is the
        # authority; skipping the reveal also means a broken cipher cannot fail
        # a server that never needed a credential in the first place.
        return _assemble_source(row, settings, headers={}), None
    headers: dict[str, str] = {}
    bearer = await _resolve_workspace_bearer(
        row, settings, tenant, sessionmaker=sessionmaker, cipher=cipher
    )
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    resolver = None
    if row.credential_scope == McpCredentialScope.USER.value:
        # Per-user, per-call credential; required=True so a user who has connected
        # none gets no call (a URL-free tool error), never the shared header. The
        # secret is keyed on the server's stable id + url (see
        # credential_secret_name), so a recreated or re-pointed server cannot
        # inherit a prior server's per-user credentials.
        resolver = PerUserCredentialResolver(
            sessionmaker=sessionmaker,
            cipher=cipher,
            secret_name=credential_secret_name(row.id, row.base_url),
            required=True,
        )
    return _assemble_source(row, settings, headers=headers, resolver=resolver), None


class McpSourceProvider:
    """Builds and memoizes a tenant's MCP tool sources from stored rows.

    Held once per process (like the pre-PR2 source list). ``sources_for`` reads a
    tenant's enabled ``mcp_servers`` rows, seeding the env server on first use,
    and returns one memoized :class:`CachingToolSource` per row - so the remote
    listing cache survives across turns while the set of servers stays per-tenant.

    MCP is optional: any failure reading the registry degrades to "no MCP tools
    this turn" rather than taking the turn down, the same contract the sources
    themselves keep.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
    ) -> None:
        """Bind the provider to the settings and the process sessionmaker."""
        self._settings = settings
        self._sessionmaker = sessionmaker
        # Tenants whose env server has been seeded this process (at most one seed
        # per tenant per process; a retry stays cheap if a seed ever fails).
        self._seeded: set[UUID] = set()
        # (tenant_id, server_id, base_url, credential_scope, auth_type) -> the
        # source and its staleness horizon (None = fresh until invalidated; an
        # OAuth build carries its bearer's expiry-minus-skew, so the memo dies
        # WITH the token and the rebuild renews it). The key carries tenant_id
        # so no instance is ever shared across tenants; a server whose URL,
        # scope, or auth mechanism is edited in the DB gets a NEW key (the old
        # entry is orphaned until process restart - PR3's write API
        # invalidates). auth_type is in the key for the same reason
        # credential_scope is: both flip what credential a source sends, so an
        # edit re-keys and self-heals even where no invalidate call runs.
        self._cache: dict[tuple[UUID, UUID, str, str, str], tuple[ToolSource, datetime | None]] = {}
        # The cipher is built lazily and at most once. Both the per-server
        # workspace-credential reveal and a user-scoped row's per-call reveal need
        # it, and a malformed SECRET_MASTER_KEY must not break the (common) env-only
        # path. None when unset/malformed -> stored credentials cannot be revealed
        # (env bootstrap still applies) and user-scoped calls fail closed, which is
        # the safe direction.
        self._cipher_built = False
        self._cipher: SecretCipher | None = None

    def _process_cipher(self) -> SecretCipher | None:
        """Return the process cipher, building it once; None if unset/unusable.

        Serves both the per-server workspace-credential reveal and a user-scoped
        row's per-call resolver. A malformed key logs once and yields None, so the
        env-only bootstrap path keeps working and user-scoped calls fail closed.
        """
        if not self._cipher_built:
            self._cipher_built = True
            try:
                self._cipher = build_cipher(self._settings.secret_master_key)
            except Exception:
                # A malformed key must not break tool listing; stored credentials
                # then simply cannot be revealed (env bootstrap still applies) and
                # user-scoped calls fail closed, which is the safe direction.
                logger.warning(
                    "SECRET_MASTER_KEY is set but unusable; stored MCP credentials "
                    "cannot be revealed (env bootstrap still applies; user-scoped "
                    "servers fail closed) until it is fixed",
                    exc_info=True,
                )
                self._cipher = None
        return self._cipher

    async def sources_for(self, tenant: TenantContext) -> list[ToolSource]:
        """Return the tenant's enabled MCP tool sources, seeding the env server.

        Degrades to an empty list on any registry-read failure: MCP is one source
        of tools, and its unavailability must not kill the turn.
        """
        try:
            await self._seed_if_needed(tenant)
            async with self._sessionmaker() as session:
                rows = await McpServerRepository(session, tenant).list_enabled()
        except Exception:
            logger.warning(
                "reading the MCP server registry failed for tenant %s; offering no "
                "MCP tools this turn",
                tenant.tenant_id,
                exc_info=True,
            )
            return []
        sources: list[ToolSource] = []
        for row in rows:
            key = (tenant.tenant_id, row.id, row.base_url, row.credential_scope, row.auth_type)
            source: ToolSource | None = None
            cached = self._cache.get(key)
            if cached is not None:
                source, stale_at = cached
                if stale_at is not None and utcnow() >= stale_at:
                    # The memo's baked bearer is at or past its refresh skew:
                    # rebuild, which renews the grant (or degrades honestly).
                    # The stale entry stays in place until a successful rebuild
                    # replaces it, so a transient renewal failure below simply
                    # retries here next turn.
                    source = None
            if source is None:
                try:
                    # The cipher backs both the workspace-credential reveal (every
                    # row) and a user-scoped row's per-call resolver; built once.
                    source, stale_at = await build_source_for_row(
                        row,
                        self._settings,
                        tenant,
                        sessionmaker=self._sessionmaker,
                        cipher=self._process_cipher(),
                    )
                except Exception:
                    # A degraded build - a workspace-credential reveal that ERRORED
                    # (not a legitimately not-connected None), or any other build
                    # failure. Skip this server for THIS turn and do NOT cache it,
                    # so the next turn retries with a healthy reveal instead of
                    # serving a memoized wrong/absent header. MCP is one source of
                    # tools; one server's failure must not take the whole turn down.
                    logger.warning(
                        "building the MCP source for server %s failed; offering it "
                        "no tools this turn",
                        row.id,
                        exc_info=True,
                    )
                    continue
                self._cache[key] = (source, stale_at)
            sources.append(source)
        return sources

    def invalidate(self, tenant_id: UUID, server_id: UUID) -> None:
        """Drop every memoized source for one (tenant, server).

        Called after an admin edits/deletes a server or writes its workspace
        credential, so the next ``sources_for`` rebuilds from the committed row
        instead of serving a stale ``CachingToolSource``. It sweeps by the first
        two key elements, so it also clears entries orphaned by a prior
        base_url/scope edit (which produced a differently-keyed instance for the
        same server). A synchronous dict rebuild - no await, no lock - so it is
        atomic against concurrent turns and costs only the tenant's cached count.

        A ``config``/``enabled`` change or a delete keeps the same key and so
        relies on this; a base_url/scope change already re-keys on its own, and a
        USER credential is revealed per call (no rebuild needed). In a
        multi-worker deployment this clears only THIS process's cache; other
        workers pick the change up on their next listing-cache expiry.
        """
        self._cache = {
            key: entry
            for key, entry in self._cache.items()
            if not (key[0] == tenant_id and key[1] == server_id)
        }

    def _env_seeds(self) -> list[tuple[str, str, str]]:
        """This deployment's seed rows, plus the collision warning seeding owes.

        The rows themselves come from :func:`env_seeds`. The warning stays on
        this path rather than moving with them: it describes what SEEDING does
        with a collided name, and the API asks the same question on every
        listing, where the log line would be pure noise.
        """
        seeds = env_seeds(self._settings)
        names = [name for name, _, _ in seeds]
        if len(set(names)) != len(names):
            # Rows are unique per (tenant, name), and ensure_seed is get-or-
            # create - so two seeds sharing a name silently collapse into ONE
            # row pointing at whichever URL was seeded first, and the other
            # sidecar simply never appears. Nothing errors, so say it out loud.
            logger.warning(
                "two env-configured MCP servers share the name %r; only the first "
                "is registered - give each its own *_CREDENTIAL_NAME",
                sorted(name for name in names if names.count(name) > 1)[0],
            )
        return seeds

    async def _seed_if_needed(self, tenant: TenantContext) -> None:
        """Upsert the env servers into this tenant's rows, once per process.

        Non-destructive (``ensure_seed`` never overwrites an existing row) and
        idempotent. A seed failure is logged and left un-marked so the next turn
        retries rather than permanently losing a bootstrap row.
        """
        if tenant.tenant_id in self._seeded:
            return
        seeds = self._env_seeds()
        if not seeds:
            return
        for name, base_url, scope in seeds:
            # One session PER seed: ensure_seed rolls its session back when it
            # loses the insert race, and a shared session would take a sibling
            # seed's pending insert down with it - leaving that server unseeded
            # for the life of the process, since the tenant is marked below.
            async with self._sessionmaker() as session:
                repo = McpServerRepository(session, tenant)
                await repo.ensure_seed(
                    name=name,
                    base_url=base_url,
                    credential_scope=scope,
                    config={"seeded_from_env": True},
                )
                await session.commit()
        self._seeded.add(tenant.tenant_id)
