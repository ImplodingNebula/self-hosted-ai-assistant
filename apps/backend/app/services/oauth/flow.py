"""The connect flow: start a consent, complete the callback, disconnect.

This is the wiring layer over the pure pieces - discovery, the authorize
builders, and the sealed bundle - plus the one stateful artifact the flow
needs: a pending :class:`~app.models.OAuthAuthorization` row between "Connect"
and the callback. Routers call these three functions and do HTTP only.

Trust properties, restated where they are enforced:

- ``state`` is single-use (the repository DELETES on consume) and expiring;
  an unknown, replayed, or expired state is one indistinguishable failure.
- The PKCE verifier is stored SEALED with the secret cipher, AAD-bound to the
  tenant and state, so a database read alone cannot complete an exchange.
- The token exchange repeats the EXACT ``redirect_uri`` the consent used, and
  sends the ``resource`` binding whenever the endpoints declare it.
- The sealed bundle is workspace-scoped: this flow is the workspace-credential
  consent (the per-user flow waits for real login; see the slice plan).

Residual risk, stated plainly: with no login there is no session cookie to
bind ``state`` to the browser that started the flow, so ``state`` alone
carries the full CSRF weight (256 random bits, single-use, expiring). An
attacker who obtained a victim's state could complete the exchange with THEIR
code - binding the attacker's own account as the workspace grant, a
self-downgrade rather than a theft. OIDC (Phase 8b) adds the session binding.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import timedelta
from typing import TYPE_CHECKING

import httpx

from app.models import McpAuthType, McpCredentialScope, utcnow
from app.providers.secrets import SecretScope
from app.repositories.oauth_authorizations import OAuthAuthorizationRepository
from app.services.oauth.authorize import (
    OAuthEndpoints,
    build_authorization_url,
    generate_pkce_verifier,
    generate_state,
    pkce_challenge,
)
from app.services.oauth.bundle import (
    OAuthTokenBundle,
    oauth_client_secret_name,
    oauth_token_secret_name,
    validate_endpoint,
)
from app.services.oauth.discovery import (
    DiscoveryError,
    discover_authorization_server,
    registered_host_is_public,
    require_fetchable,
)
from app.services.oauth.fetch import OversizedResponseError, read_capped, read_json_capped
from app.services.oauth.locks import grant_lock

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from uuid import UUID

    from app.core.config import Settings
    from app.core.tenancy import TenantContext
    from app.models import McpServer
    from app.providers.secrets import SecretCipher
    from app.repositories.mcp_servers import McpServerRepository
    from app.services.secret_store import SecretStore

logger = logging.getLogger(__name__)

# When a token response declares no expiry, the bundle still records an
# absolute horizon so the refresh path has a number to compare - a year, far
# beyond any refresh cadence, standing in for "the provider did not say".
_NO_EXPIRY_HORIZON = timedelta(days=365)

# Refresh error codes that mean the GRANT ITSELF is dead (re-consent needed),
# judged from the response body per RFC 6749 section 5.2 - never from the
# status line, which providers use loosely ("bad_refresh_token" is GitHub's
# spelling, answered with HTTP 200). Anything else - invalid_client, a bare
# 4xx, an outage - is transient: marking a healthy grant dead costs a human
# re-consent, so the terminal verdict must be earned by an explicit code.
_TERMINAL_REFRESH_ERRORS = frozenset({"invalid_grant", "bad_refresh_token"})

_CALLBACK_PATH = "/api/v1/integrations/oauth/callback"

# The endpoints snapshot key recording whether this consent's outbound targets
# must resolve to public addresses. Computed ONCE, from the registered server's
# host, at the moment the admin started the consent; the callback re-applies it
# rather than re-deciding, so the rule cannot drift between the two legs.
_REQUIRE_PUBLIC_KEY = "require_public_endpoints"

# Per-user OAuth is not representable yet: the consent seals ONE workspace
# grant per server, so a credential_scope=user row would drop its credential
# endpoints to any member while every reveal still hit the shared grant. Same
# wording as the schema and PATCH guards, so the UI reads one message.
_PER_USER_OAUTH_REFUSED = (
    "per-user OAuth connections are not available yet: an OAuth "
    "server's grant is workspace-scoped until real login lands."
)


class OAuthFlowError(Exception):
    """A connect/callback/disconnect step failed, with a loggable reason.

    Carries only reason phrases and already-public URLs - never a code, token,
    or verifier - so it is safe to log and to show an admin verbatim.
    """


class OAuthRefreshDenied(OAuthFlowError):
    """The authorization server refused the refresh token terminally.

    Raised by :func:`refresh_grant` AFTER the sealed bundle has been re-stored
    with ``needs_reconnect`` set, so a caller treats it as "stop renewing until
    a human reconnects" - distinct from a transient :class:`OAuthFlowError`,
    which stores nothing and is worth retrying next turn.
    """


def _verifier_aad(tenant_id: UUID, state: str) -> bytes:
    """The AAD welding a sealed verifier to its tenant and state row."""
    return f"oauth-authz:{tenant_id}:{state}".encode()


def _expiry_from(payload: dict) -> datetime:
    """The absolute expiry a token response implies.

    bool is an int subclass in Python: a provider answering JSON ``true`` would
    otherwise read as "1 second", so only real numbers count. A response that
    declares no (positive, numeric) expiry gets the standing horizon.
    """
    expires_in = payload.get("expires_in")
    numeric = isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool)
    if numeric and expires_in > 0:
        return utcnow() + timedelta(seconds=float(expires_in))
    return utcnow() + _NO_EXPIRY_HORIZON


def redirect_uri(settings: Settings) -> str:
    """The callback URL registered with providers, from the configured origin."""
    return f"{settings.oauth_redirect_base_url.rstrip('/')}{_CALLBACK_PATH}"


async def start_connect(
    server: McpServer,
    settings: Settings,
    tenant: TenantContext,
    *,
    repo: OAuthAuthorizationRepository,
    store: SecretStore,
    cipher: SecretCipher,
    client_id: str,
    client_secret: str | None,
    scopes: list[str],
    endpoints: OAuthEndpoints | None,
    extra_authorize_params: Mapping[str, str] | None = None,
    userinfo_endpoint: str | None = None,
    userinfo_label_field: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> str:
    """Begin a workspace OAuth consent for a server; return the consent URL.

    ``endpoints=None`` means the MCP-native path: discover the authorization
    server from the MCP server itself and bind the grant to it with the
    ``resource`` parameter. Supplied endpoints (the presets / generic-provider
    path) skip discovery and send no ``resource``. ``extra_authorize_params``
    are operator-visible provider extensions (Google's ``access_type=offline``)
    forwarded to the consent URL only - the token exchange never sees them.
    ``userinfo_endpoint``/``userinfo_label_field`` name the provider's
    identity lookup for the "Connected as" label; validated here (https or
    loopback - the callback sends the fresh ACCESS TOKEN there) and stored in
    the pending row's endpoints snapshot for the callback to use.

    Every outbound target this consent will later reach - the token,
    revocation and identity endpoints - is additionally held to the discovery
    walk's public-host anchor: if the ADMIN-REGISTERED server resolves public,
    none of them may be private. The verdict is recorded in the snapshot so the
    callback applies the same rule without re-deciding.

    Raises:
        OAuthFlowError: Discovery failed, an endpoint was refused, or the row
            is credential_scope=user - nothing was stored in any case.
    """
    if server.credential_scope == McpCredentialScope.USER.value:
        # The create and PATCH guards reject this combination; connecting is
        # the third write path into it, and the one that would actually mint
        # the misleading grant.
        raise OAuthFlowError(_PER_USER_OAUTH_REFUSED)
    # The trust anchor for every derived target below, read once from what the
    # admin registered.
    registered_public = await registered_host_is_public(server.base_url)
    # The RFC 8707 binding stays None on the supplied-endpoints path. On the
    # MCP-native path it is the DECLARED identifier from the metadata walk,
    # never a recomputation from the base URL: the authorization server
    # issues tokens for the identifier it published (Asana's covers its
    # connection URL as a path prefix), and the two need not be byte-equal.
    resource: str | None = None
    if endpoints is None:
        client = http_client or httpx.AsyncClient(timeout=settings.mcp_list_timeout_seconds)
        try:
            metadata, resource = await discover_authorization_server(server.base_url, client=client)
        except DiscoveryError as exc:
            raise OAuthFlowError(f"discovery failed: {exc}") from exc
        finally:
            if http_client is None:
                await client.aclose()
        endpoints = OAuthEndpoints.from_metadata(metadata)

    # Discovery guards what IT derived; this covers the operator-supplied path
    # too, so one rule holds whichever way the endpoints arrived.
    try:
        await require_fetchable(
            endpoints.token_endpoint, registered_public=registered_public, what="token endpoint"
        )
        # A browser redirect target rather than a server-side fetch, so the
        # exposure is steering the ADMIN's browser, not exfiltrating a secret
        # from this process. Anchored anyway: discovery already holds derived
        # endpoints to this bar, and the comment above promises one rule.
        await require_fetchable(
            endpoints.authorization_endpoint,
            registered_public=registered_public,
            what="authorization endpoint",
        )
        if endpoints.revocation_endpoint:
            await require_fetchable(
                endpoints.revocation_endpoint,
                registered_public=registered_public,
                what="revocation endpoint",
            )
    except DiscoveryError as exc:
        raise OAuthFlowError(f"an endpoint was refused: {exc}") from exc

    endpoints_snapshot = endpoints.model_dump()
    endpoints_snapshot[_REQUIRE_PUBLIC_KEY] = registered_public
    if userinfo_endpoint and userinfo_label_field:
        # The callback will send the ACCESS TOKEN to this URL, so it gets the
        # same hygiene bar as the token endpoint - and it rides the snapshot
        # blob (schema-free JSON) rather than growing the row.
        try:
            validate_endpoint(userinfo_endpoint)
        except ValueError as exc:
            raise OAuthFlowError(f"the identity endpoint failed validation: {exc}") from exc
        try:
            await require_fetchable(
                userinfo_endpoint, registered_public=registered_public, what="identity endpoint"
            )
        except DiscoveryError as exc:
            raise OAuthFlowError(f"the identity endpoint was refused: {exc}") from exc
        endpoints_snapshot["userinfo_endpoint"] = userinfo_endpoint
        endpoints_snapshot["userinfo_label_field"] = userinfo_label_field

    state = generate_state()
    verifier = generate_pkce_verifier()
    if client_secret:
        # Stored under the server's stable identity, separate from the token
        # bundle, so revoking a grant never destroys the registration.
        await store.store(
            oauth_client_secret_name(server.id, server.base_url),
            client_secret,
            scope=SecretScope.WORKSPACE,
        )
    await repo.create(
        mcp_server_id=server.id,
        state=state,
        code_verifier_sealed=cipher.encrypt(verifier, aad=_verifier_aad(tenant.tenant_id, state)),
        client_id=client_id,
        redirect_uri=redirect_uri(settings),
        resource=resource,
        endpoints=endpoints_snapshot,
        scopes=" ".join(scopes),
        created_by=tenant.user_id,
        expires_at=utcnow() + timedelta(seconds=settings.oauth_authorization_ttl_seconds),
    )
    await repo.commit()
    logger.info(
        "oauth connect started: tenant=%s user=%s server=%s issuer=%s",
        tenant.tenant_id,
        tenant.user_id,
        server.id,
        endpoints.issuer,
    )
    return build_authorization_url(
        endpoints,
        client_id=client_id,
        redirect_uri=redirect_uri(settings),
        scopes=scopes,
        state=state,
        code_challenge=pkce_challenge(verifier),
        resource=resource,
        extra_params=extra_authorize_params,
    )


# The account label is a display string; cap it so a hostile provider cannot
# balloon the server's config row.
_MAX_ACCOUNT_LABEL = 200

# The label lookup is a display nicety with its own short deadline: a slow
# identity endpoint must not hold the callback (or the admin's browser) to
# the full MCP timeout.
_IDENTITY_TIMEOUT_SECONDS = 5.0


async def _fetch_account_label(
    endpoints_snapshot: dict,
    access_token: str,
    settings: Settings,
    *,
    registered_public: bool,
    http_client: httpx.AsyncClient | None = None,
) -> str | None:
    """Ask the provider who the connected account is, best effort.

    None on every failure - no identity configured, a non-200, a non-JSON
    body, a missing or non-string field - because the label is a nicety and
    the callback must never fail over it. The endpoint was validated at
    ``start_connect``; re-checked here (defense in depth, the snapshot is a
    plain JSON blob) before the access token is sent to it - both the scheme
    rule and the public-host anchor, which the CALLER recomputes from the
    registered URL rather than reading back out of the mutable snapshot,
    because this is a server-side request carrying a live access token. A
    dotted field walks
    nested objects (Asana wraps ``/users/me`` in a ``data`` envelope, so its
    label field is ``data.name``).
    """
    url = endpoints_snapshot.get("userinfo_endpoint")
    field = endpoints_snapshot.get("userinfo_label_field")
    if not isinstance(url, str) or not isinstance(field, str) or not url or not field:
        return None
    try:
        validate_endpoint(url)
        await require_fetchable(url, registered_public=registered_public, what="identity endpoint")
    except (ValueError, DiscoveryError):
        return None
    client = http_client or httpx.AsyncClient(timeout=_IDENTITY_TIMEOUT_SECONDS)
    try:
        async with client.stream(
            "GET",
            url,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            follow_redirects=False,
        ) as response:
            if response.status_code != 200:
                return None
            value: object = await read_json_capped(response, what="identity document")
        for part in field.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
    except Exception:  # noqa: BLE001 - a label lookup must never fail the callback
        logger.debug("the identity lookup failed; connecting without a label", exc_info=True)
        return None
    finally:
        if http_client is None:
            await client.aclose()
    if not isinstance(value, str):
        return None
    # Display hardening: drop control and format characters (newlines, RTL
    # overrides, zero-width joiners) a hostile provider could spoof the card
    # line with; React already escapes HTML.
    value = "".join(ch for ch in value if ch.isprintable()).strip()
    if not value:
        return None
    return value[:_MAX_ACCOUNT_LABEL]


async def complete_callback(
    *,
    state: str,
    code: str,
    settings: Settings,
    tenant: TenantContext,
    repo: OAuthAuthorizationRepository,
    servers: McpServerRepository,
    store: SecretStore,
    cipher: SecretCipher,
    http_client: httpx.AsyncClient | None = None,
) -> UUID:
    """Consume a pending consent, exchange the code, and seal the grant.

    Returns the connected server's id (for the audit log). On any failure the
    pending row is ALREADY GONE - consume-then-act is what makes a replayed
    callback inert - and nothing is stored.

    Raises:
        OAuthFlowError: The state is unknown/expired/replayed, the server row
            is gone or became credential_scope=user while the consent was open,
            the exchange failed, or the response was malformed.
    """
    row = await repo.consume(state)
    await repo.commit()
    if row is None:
        raise OAuthFlowError("unknown, expired, or already-used state")
    server = await servers.get(row.mcp_server_id)
    if server is None:
        raise OAuthFlowError("the server this consent was for no longer exists")
    if server.credential_scope == McpCredentialScope.USER.value:
        # Re-checked here, not just at start: the row is editable while the
        # consent is open, and this is the step that flips auth_type to oauth -
        # the forbidden combination would be minted right here.
        raise OAuthFlowError(_PER_USER_OAUTH_REFUSED)
    try:
        verifier = cipher.decrypt(
            row.code_verifier_sealed, aad=_verifier_aad(tenant.tenant_id, state)
        )
    except Exception as exc:
        raise OAuthFlowError("the sealed verifier could not be opened") from exc

    client_secret = None
    try:
        client_secret = await store.reveal(
            oauth_client_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
        )
    except Exception:  # noqa: BLE001 - a public client simply has no secret stored
        client_secret = None

    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": row.redirect_uri,
        "client_id": row.client_id,
        "code_verifier": verifier,
    }
    if client_secret:
        form["client_secret"] = client_secret
    # Truthiness, matching the refresh path: an empty-string binding must
    # never ride a token request as a real parameter.
    if row.resource:
        form["resource"] = row.resource
    token_endpoint = str(row.endpoints.get("token_endpoint", ""))
    # RECOMPUTED from the registered URL, never read back from the snapshot.
    # The snapshot is a plain mutable JSON blob, so a row written before this
    # key existed - or with it dropped by any later edit - answers False, which
    # silently disables the guard on precisely the leg it exists for. The
    # stored key stays as a record of what was decided at start; the decision
    # enforced here is made fresh.
    require_public = await registered_host_is_public(server.base_url)
    revocation_endpoint = row.endpoints.get("revocation_endpoint")
    try:
        # Defense in depth: the snapshot was validated at start, but the form
        # below carries the client secret and the code - re-check the target
        # before POSTing them anywhere, mirroring what disconnect gets for
        # free from the bundle's own validators. The scheme rule alone permits
        # any https host, so the public-host anchor recorded at start rides
        # along: it is what keeps an attacker-named internal target out of a
        # request carrying the client secret, the code and the verifier.
        validate_endpoint(token_endpoint)
        await require_fetchable(
            token_endpoint, registered_public=require_public, what="token endpoint"
        )
        if revocation_endpoint:
            # Sealed into the bundle below and POSTed to on disconnect, so it
            # gets the same check before it becomes durable.
            validate_endpoint(str(revocation_endpoint))
            await require_fetchable(
                str(revocation_endpoint),
                registered_public=require_public,
                what="revocation endpoint",
            )
    except (ValueError, DiscoveryError) as exc:
        raise OAuthFlowError("a snapshotted endpoint failed validation") from exc
    client = http_client or httpx.AsyncClient(timeout=settings.mcp_timeout_seconds)
    try:
        # Streamed so the body passes under a byte cap: the token endpoint was
        # named by the server's own metadata, so its size is the remote side's
        # choice, and this process also serves every live chat stream.
        async with client.stream(
            "POST",
            token_endpoint,
            data=form,
            # GitHub answers form-encoded unless asked for JSON; harmless for
            # spec-shaped servers, which return JSON regardless.
            headers={"Accept": "application/json"},
            follow_redirects=False,
        ) as response:
            status = response.status_code
            if status != 200:
                raise OAuthFlowError(f"the token exchange failed with HTTP {status}")
            body = await read_capped(response, what="token response")
    except OversizedResponseError as exc:
        raise OAuthFlowError("the token response was oversized and was refused") from exc
    except httpx.HTTPError as exc:
        raise OAuthFlowError(f"the token exchange could not reach {token_endpoint}") from exc
    finally:
        if http_client is None:
            await client.aclose()
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise OAuthFlowError("the token response was not JSON") from exc
    access_token = payload.get("access_token")
    if not access_token or not isinstance(access_token, str):
        raise OAuthFlowError("the token response carried no access token")

    bundle = OAuthTokenBundle(
        access_token=access_token,
        refresh_token=payload.get("refresh_token") or None,
        expires_at=_expiry_from(payload),
        # RFC 6749: an absent scope means "as requested".
        granted_scope=str(payload.get("scope") or row.scopes),
        token_endpoint=token_endpoint,
        revocation_endpoint=row.endpoints.get("revocation_endpoint"),
        client_id=row.client_id,
        # The RFC 8707 binding the consent carried; refresh repeats it so a
        # renewed token keeps the same audience.
        resource=row.resource,
        issuer=str(row.endpoints.get("issuer", "")),
    )
    await store.store(
        oauth_token_secret_name(server.id, server.base_url),
        bundle.seal(),
        scope=SecretScope.WORKSPACE,
    )
    server.auth_type = McpAuthType.OAUTH.value
    # The grant is made durable BEFORE the identity lookup: a slow provider
    # must not hold the flushed token write (or the admin's redirect) open
    # across network I/O for a display nicety.
    await servers.commit()
    # "Connected as <account>": stored in the server's non-secret config so
    # the card can show it without ever decrypting the grant. Best effort - a
    # failed lookup connects without a label, never fails the callback.
    label = await _fetch_account_label(
        row.endpoints,
        access_token,
        settings,
        registered_public=require_public,
        http_client=http_client,
    )
    if label:
        server.config = {**server.config, "connected_account": label}
        await servers.commit()
    logger.info(
        "oauth connect completed: tenant=%s user=%s server=%s issuer=%s scope=%r",
        tenant.tenant_id,
        tenant.user_id,
        server.id,
        bundle.issuer,
        bundle.granted_scope,
    )
    return server.id


async def refresh_grant(
    server: McpServer,
    settings: Settings,
    tenant: TenantContext,
    *,
    store: SecretStore,
    bundle: OAuthTokenBundle,
    http_client: httpx.AsyncClient | None = None,
) -> OAuthTokenBundle:
    """Renew a server's access token with its refresh token and re-seal the grant.

    Called on demand from the tool path when the sealed bundle is at or past its
    refresh skew - there is no background renewal. On success the NEW bundle has
    been stored through ``store`` (the caller owns the session commit) and the
    renewal is logged, which is what keeps the grant-status "last updated" line
    honest. Per RFC 6749 section 6: a response may rotate the refresh token (the
    new one replaces the old), omit it (the old one is kept), or omit the scope
    (unchanged).

    The token endpoint is re-anchored before the POST, not merely re-validated.
    The bundle's own field validator ran on unseal, but that is the SCHEME rule
    (https, or http on loopback), which permits any public https host - and a
    grant sealed before the public-host anchor existed carries whatever endpoint
    its metadata named at the time, forever. This request carries the client
    secret and the refresh token, so it gets the same anchor the consent did.

    Both writes are UPDATE-ONLY. A disconnect that landed while this renewal
    was in flight leaves no row, and re-inserting one would resurrect a grant -
    holding freshly rotated, still-live tokens - that the admin believes is
    severed. The write simply becomes a no-op, in any worker process.

    Raises:
        OAuthRefreshDenied: The authorization server declared the refresh token
            DEAD (an ``invalid_grant``-class error code in the body - expired,
            revoked, or rotated away). The bundle has been re-stored with
            ``needs_reconnect=True`` BEFORE this raises, so the tool path stops
            renewing until a human reconnects.
        OAuthFlowError: A transient failure (network, non-JSON, any other
            error status) - nothing was stored, and the next turn retries.
    """
    if not bundle.refresh_token:
        raise OAuthFlowError("the grant has no refresh token")
    try:
        await require_fetchable(
            bundle.token_endpoint,
            registered_public=await registered_host_is_public(server.base_url),
            what="token endpoint",
        )
    except DiscoveryError as exc:
        raise OAuthFlowError("the sealed token endpoint was refused") from exc
    form = {
        "grant_type": "refresh_token",
        "refresh_token": bundle.refresh_token,
        "client_id": bundle.client_id,
    }
    if bundle.resource:
        # RFC 8707: the binding rides every token request, or the renewed
        # token could come back scoped to a different audience.
        form["resource"] = bundle.resource
    # A public client simply has no secret stored; anything else it could
    # raise (an unusable key) equally means "send the form without one".
    with contextlib.suppress(Exception):
        form["client_secret"] = await store.reveal(
            oauth_client_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
        )
    client = http_client or httpx.AsyncClient(timeout=settings.mcp_timeout_seconds)
    try:
        async with client.stream(
            "POST",
            bundle.token_endpoint,
            data=form,
            headers={"Accept": "application/json"},
            follow_redirects=False,
        ) as response:
            status = response.status_code
            body = await read_capped(response, what="refresh response")
    except OversizedResponseError as exc:
        raise OAuthFlowError("the refresh response was oversized and was refused") from exc
    except httpx.HTTPError as exc:
        raise OAuthFlowError("the refresh could not reach the token endpoint") from exc
    finally:
        if http_client is None:
            await client.aclose()
    try:
        payload = json.loads(body)
    except ValueError:
        payload = None
    error_code = payload.get("error") if isinstance(payload, dict) else None
    if error_code in _TERMINAL_REFRESH_ERRORS:
        # The refresh token itself is dead: retrying cannot help, a human must
        # re-consent. Mark that ONCE so no later turn hammers a dead grant.
        dead = bundle.model_copy(update={"needs_reconnect": True})
        await store.store_existing_only(
            oauth_token_secret_name(server.id, server.base_url),
            dead.seal(),
            scope=SecretScope.WORKSPACE,
        )
        logger.warning(
            "oauth refresh denied (%s, HTTP %s): tenant=%s server=%s; reconnect needed",
            error_code,
            status,
            tenant.tenant_id,
            server.id,
        )
        raise OAuthRefreshDenied(f"the refresh was denied: {error_code}")
    if status != 200:
        raise OAuthFlowError(f"the refresh failed with HTTP {status}")
    if not isinstance(payload, dict):
        raise OAuthFlowError("the refresh response was not JSON")
    access_token = payload.get("access_token")
    if not access_token or not isinstance(access_token, str):
        raise OAuthFlowError("the refresh response carried no access token")
    # A non-string refresh_token or scope from a misbehaving provider must not
    # be sealed: model_copy skips validation, so a bad value would brick the
    # bundle on the NEXT unseal. Keep the known-good prior value instead.
    new_refresh = payload.get("refresh_token")
    new_scope = payload.get("scope")
    renewed = bundle.model_copy(
        update={
            "access_token": access_token,
            "refresh_token": new_refresh
            if isinstance(new_refresh, str) and new_refresh
            else bundle.refresh_token,
            "expires_at": _expiry_from(payload),
            "granted_scope": new_scope
            if isinstance(new_scope, str) and new_scope
            else bundle.granted_scope,
            "needs_reconnect": False,
        }
    )
    stored = await store.store_existing_only(
        oauth_token_secret_name(server.id, server.base_url),
        renewed.seal(),
        scope=SecretScope.WORKSPACE,
    )
    if not stored:
        # A disconnect won the race: the grant is gone and stays gone. The
        # renewed token still returns to the in-flight caller (revoked upstream
        # by that disconnect, so it is short-lived and unusable soon), but
        # nothing durable is left behind.
        logger.warning(
            "the OAuth grant for server %s was disconnected mid-renewal; the "
            "renewed token was not stored",
            server.id,
        )
    logger.info(
        "oauth token renewed: tenant=%s server=%s issuer=%s",
        tenant.tenant_id,
        server.id,
        bundle.issuer,
    )
    return renewed


async def disconnect(
    server: McpServer,
    settings: Settings,
    tenant: TenantContext,
    *,
    store: SecretStore,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Revoke a server's grant (best effort) and delete the sealed bundle.

    Returns True when a bundle existed. Revocation is best-effort by design
    (RFC 7009 servers vary; some providers have none, and one that is simply
    down must not make a grant undeletable): the LOCAL deletion is the part
    that must succeed, and it happens regardless. ``auth_type`` stays
    ``oauth`` - the server still authenticates that way, it is simply no
    longer connected, and the ``:oauth-client`` registration survives so the
    same client can grant again (:func:`forget_grant` is the variant for when
    the server itself is going away).

    Runs under the grant's shared lock, so a renewal on a concurrent turn
    cannot interleave its read and its write around this delete; the renewal's
    update-only write is what covers the multi-process case, where no
    in-memory lock reaches.
    """
    name = oauth_token_secret_name(server.id, server.base_url)
    async with grant_lock(name):
        try:
            sealed = await store.reveal(name, scope=SecretScope.WORKSPACE)
            bundle = OAuthTokenBundle.unseal(sealed)
        except Exception:  # noqa: BLE001 - nothing connected (or unreadable): nothing to revoke
            bundle = None
        revocable = bundle is not None and bool(bundle.revocation_endpoint)
        if revocable:
            # Anchored like every other server-side leg: this POST carries a live
            # token and the client secret, and the endpoint came from metadata a
            # remote server served - possibly before that anchor existed. A
            # refusal SKIPS the call (it is best effort anyway); the local delete
            # below is not optional and happens either way.
            try:
                await require_fetchable(
                    str(bundle.revocation_endpoint),
                    registered_public=await registered_host_is_public(server.base_url),
                    what="revocation endpoint",
                )
            except DiscoveryError:
                logger.warning(
                    "sealed revocation endpoint for server %s failed the public-host "
                    "anchor; skipping the revocation call and deleting locally",
                    server.id,
                )
                revocable = False
        if bundle is not None and revocable:
            form = {
                "token": bundle.refresh_token or bundle.access_token,
                "client_id": bundle.client_id,
            }
            try:
                client_secret = await store.reveal(
                    oauth_client_secret_name(server.id, server.base_url),
                    scope=SecretScope.WORKSPACE,
                )
                form["client_secret"] = client_secret
            except Exception:  # noqa: BLE001 - public client
                pass
            client = http_client or httpx.AsyncClient(timeout=settings.mcp_timeout_seconds)
            try:
                # Streamed and never read: the answer to a revocation carries
                # nothing this needs, so there is no reason to buffer whatever
                # the provider decides to send.
                async with client.stream(
                    "POST", bundle.revocation_endpoint, data=form, follow_redirects=False
                ):
                    pass
            except Exception:  # noqa: BLE001 - ANY revocation failure is survivable; the
                # local delete below is not optional. A provider that is down,
                # unresolvable, or answering nonsense must never leave a grant
                # (or the row that owns it) undeletable.
                logger.warning(
                    "oauth revocation call failed for server %s; the local grant "
                    "is deleted regardless",
                    server.id,
                )
            finally:
                if http_client is None:
                    await client.aclose()
        deleted = await store.delete(name, scope=SecretScope.WORKSPACE)
    logger.info(
        "oauth disconnected: tenant=%s user=%s server=%s had_bundle=%s",
        tenant.tenant_id,
        tenant.user_id,
        server.id,
        deleted,
    )
    return deleted


async def forget_grant(
    server: McpServer,
    settings: Settings,
    tenant: TenantContext,
    *,
    store: SecretStore,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Erase every OAuth secret a server owns, revoking the grant on the way out.

    Called when the server row is being DELETED or when its identity is about
    to change (a re-point, or a move off ``oauth``). :func:`disconnect` is not
    enough on its own: it deletes the ``:oauth-token`` bundle only, leaving the
    ``:oauth-client`` registration behind under a name derivable solely from
    the old id and URL - unreachable from any API or UI, and pointing at a
    client the admin believes is gone.

    Failure-tolerant on purpose. The revoke is best effort (see
    :func:`disconnect`) and the whole call is safe with ``SECRET_MASTER_KEY``
    unset: the reveals fail, so nothing is revoked, but both deletes still run
    - they need no cipher. A caller must be able to delete a row even when the
    provider is unreachable and the key is gone.

    The two deletes are ordered but not conditional: whatever the grant delete
    does, the client-secret delete still runs. Letting the first failure skip
    the second would strand precisely the secret this function exists to
    remove, which is the state it was written to prevent.
    """
    try:
        await disconnect(server, settings, tenant, store=store, http_client=http_client)
    finally:
        await store.delete(
            oauth_client_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
        )
    logger.info(
        "oauth secrets forgotten: tenant=%s user=%s server=%s",
        tenant.tenant_id,
        tenant.user_id,
        server.id,
    )
