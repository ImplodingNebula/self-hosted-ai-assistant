"""Integrations service: the MCP test-connection handshake.

Server CRUD and credential writes are thin enough to live in the router (they
are one repository/SecretStore call plus a commit, like the conversations
router). This module holds the one piece with real orchestration: probing a
server with an ``initialize`` + ``tools/list`` handshake and classifying the
outcome for the UI. It NEVER calls a tool - a test only discovers what a server
offers, it does not exercise it.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import httpx

from app.models import McpAuthType, McpCredentialScope
from app.providers.mcp.source import McpServerConfig, McpToolSource
from app.providers.secrets import SecretScope
from app.providers.tools.base import ToolContext
from app.schemas.integrations import (
    ConnectionTestError,
    ConnectionTestRequest,
    ConnectionTestResult,
    ToolDescriptorOut,
)
from app.services.oauth.bundle import oauth_token_secret_name
from app.services.secret_store import SecretStore
from app.services.tools.mcp_sources import credential_secret_name, env_bootstrap_key_for

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.config import Settings
    from app.core.tenancy import TenantContext
    from app.models import McpServer
    from app.providers.secrets import SecretMetadata
    from app.providers.tools.base import Tool

logger = logging.getLogger(__name__)


def _auth_header(settings: Settings, *, base_url: str, credential: str | None) -> dict[str, str]:
    """The Authorization header for a handshake.

    An explicit ``credential`` wins (a typed one-shot on the pre-save test, or a
    SAVED server's revealed stored credential); absent one, the env bootstrap
    question goes to :func:`~app.services.tools.mcp_sources.env_bootstrap_key_for`
    - the SAME function the chat path resolves its header with, so a key can
    never be sent to a URL it does not belong to, and a probe can never disagree
    with a real call about which sidecars are backed by deployment config.
    Never reads the SecretStore itself: the caller supplies the credential.
    """
    if credential is not None:
        return {"Authorization": f"Bearer {credential}"}
    bootstrap = env_bootstrap_key_for(base_url, settings)
    return {} if bootstrap is None else {"Authorization": f"Bearer {bootstrap}"}


def _leaf(exc: BaseException) -> BaseException:
    """Unwrap an exception group to its first leaf (the SDK runs a task group)."""
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


def _classify(exc: BaseException) -> ConnectionTestError:
    """Map a handshake failure to a UI code, with a message-free ``detail``.

    The detail never carries the exception's message: an httpx error's text
    embeds the request URL, which may embed a credential. A status failure
    reports the HTTP status code (never secret, and far more actionable than
    the exception class name); everything else reports the leaf TYPE name -
    the unknown bucket adds a next step, because a bare type name there reads
    as noise (see below).
    """
    leaf = _leaf(exc)
    if isinstance(leaf, httpx.HTTPStatusError):
        status = leaf.response.status_code
        if 300 <= status < 400:
            # The dial deliberately does not follow redirects (a 302 is a server
            # re-pointing a credential-bearing call at an address nobody
            # approved), so this is a CONFIGURED URL problem, not a dead server -
            # and without saying so it reads as one. The hint carries no URL: the
            # Location header is the remote side's text.
            return ConnectionTestError(
                code="protocol_mismatch",
                detail=f"HTTP {status} - the server redirected; register its final URL",
            )
        code = "auth_rejected" if status in (401, 403) else "protocol_mismatch"
        return ConnectionTestError(code=code, detail=f"HTTP {status}")
    if isinstance(leaf, httpx.TimeoutException):
        code = "timeout"
    elif isinstance(leaf, httpx.TransportError | ConnectionError | OSError):
        code = "unreachable"
    elif isinstance(leaf, json.JSONDecodeError | ValueError):
        code = "protocol_mismatch"
    else:
        # The MCP SDK wraps a mid-handshake death (a deadline expiring, a
        # stream closing under it) in its own error type, so the bare type
        # name reads as noise ("McpError" explained nothing to the person who
        # hit it - the server was merely cold). Name the phase and the likely
        # next step; still no message text, which may embed a URL and so a
        # credential.
        return ConnectionTestError(
            code="unknown",
            detail=f"{type(leaf).__name__} - the handshake did not complete; "
            "a server that is still starting may need a retry",
        )
    return ConnectionTestError(code=code, detail=type(leaf).__name__)


async def _probe(
    settings: Settings,
    tenant: TenantContext,
    *,
    base_url: str,
    headers: dict[str, str],
    source: McpToolSource | None = None,
) -> ConnectionTestResult:
    """Handshake a server at ``base_url`` with ``headers`` and report its tools.

    Builds a one-off source and runs the RAISING listing variant
    (:meth:`McpToolSource.list_tools_or_raise`), so a reachable-but-empty server is
    told apart from an unreachable/auth-rejected/protocol-mismatched one - a
    distinction the chat path's swallow-to-empty listing erases. Builds NO
    ``disabled_tools`` filter: a test shows the full advertised set so an admin can
    decide what to toggle. ``source`` is injectable for hermetic tests.
    """
    if source is None:
        source = McpToolSource(
            McpServerConfig(name="(test)", url=base_url, headers=headers),
            timeout_seconds=settings.mcp_timeout_seconds,
            # The turn's listing budget (mcp_list_timeout_seconds, 5s) fails
            # fast because discovery there is latency nobody asked for. This
            # probe IS the ask - a human clicked Test - so it borrows the CALL
            # budget instead: an upstream still warming up that needs seconds
            # per request is healthy, not down, and a deadline tuned for the
            # turn fails it mid-handshake (measured live: a cold local server
            # answered each handshake request in ~5s and died twice at ~5.5s).
            list_timeout_seconds=settings.mcp_timeout_seconds,
        )
    context = ToolContext(tenant_id=str(tenant.tenant_id), user_id=str(tenant.user_id))
    try:
        tools: list[Tool] = await source.list_tools_or_raise(context)
    except Exception as exc:
        return ConnectionTestResult(ok=False, error=_classify(exc))
    return ConnectionTestResult(
        ok=True,
        tools=[
            ToolDescriptorOut(
                name=tool.spec.name,
                description=tool.spec.description,
                side_effecting=tool.spec.side_effecting,
            )
            for tool in tools
        ],
    )


async def test_connection(
    settings: Settings,
    tenant: TenantContext,
    req: ConnectionTestRequest,
    *,
    source: McpToolSource | None = None,
) -> ConnectionTestResult:
    """Probe a not-yet-saved server from a validated request body.

    The URL and one-shot credential come straight from the (schema-validated)
    request. ``source`` is injectable for hermetic tests; production builds a real
    one.
    """
    headers = _auth_header(settings, base_url=req.base_url, credential=req.credential)
    return await _probe(settings, tenant, base_url=req.base_url, headers=headers, source=source)


async def oauth_grant_metadata(
    session: AsyncSession, tenant: TenantContext, server: McpServer
) -> SecretMetadata | None:
    """The sealed grant's METADATA for an oauth server, or None if unconnected.

    A metadata read needs no cipher, so listings work on a keyless deployment
    too - the grant itself is never decrypted to answer "is this connected".
    """
    store = SecretStore(session, tenant, None)
    return await store.get(
        oauth_token_secret_name(server.id, server.base_url), scope=SecretScope.WORKSPACE
    )


async def _reveal_for_test(store: SecretStore, server: McpServer) -> str | None:
    """Reveal a SAVED server's stored credential for a test, or None if unavailable.

    Reveals under the server's own scope (its shared workspace credential, or the
    acting member's own for a user-scoped server - the store scopes a USER reveal to
    the caller from its bound context). Never raises: a not-connected server, an
    unset master key, or a decrypt failure all resolve to None, so the probe falls
    back to the env-for-chassis rule (or no header) rather than failing - a test is
    a read-only diagnostic, not a mutation.
    """
    scope = (
        SecretScope.USER
        if server.credential_scope == McpCredentialScope.USER.value
        else SecretScope.WORKSPACE
    )
    try:
        return await store.reveal(credential_secret_name(server.id, server.base_url), scope=scope)
    except Exception:
        logger.debug(
            "no revealable stored credential for a test of server %s; probing without it",
            server.id,
            exc_info=True,
        )
        return None


async def test_saved_connection(
    settings: Settings,
    tenant: TenantContext,
    server: McpServer,
    store: SecretStore,
    *,
    override_credential: str | None,
    oauth_bearer: str | None = None,
) -> ConnectionTestResult:
    """Probe an already-registered server, authenticating as the chat path would.

    For a ``header`` row, the credential is the one-shot ``override_credential``
    when given (test a new key before replacing the stored one), else the
    server's STORED credential revealed from the SecretStore, else the env
    bootstrap key for that URL if the deployment configures one, else none. So a
    test of a connected server sends what a real listing would, closing the gap
    where a credential-less probe reported ``auth_rejected`` on a healthy
    server. The revealed secret is used for one outbound handshake and never
    returned or logged.

    Probes ``server.base_url`` RAW - as the chat path does - rather than
    re-validating it through the request schema, so a seeded row whose
    ``MCP_CHASSIS_URL`` bypassed validation degrades to a classified error, never a
    500 that would break the always-200 contract.

    An ``oauth`` row probes with the ``oauth_bearer`` the caller resolved from
    its grant - the same token, renewed if it was stale, a chat turn would
    send - or bare when nothing usable is connected, which is also exactly
    what chat sends then. Its ``override_credential`` is DELIBERATELY ignored:
    a pasted token testing green while real calls use the grant would re-open
    the divergence. A ``none`` row probes with NO credential at all: the chat
    path attaches none either, and a probe that authenticated here could go
    green while real calls behave differently - the divergence this endpoint
    exists to close.
    """
    if server.auth_type == McpAuthType.OAUTH.value:
        headers = {"Authorization": f"Bearer {oauth_bearer}"} if oauth_bearer else {}
        return await _probe(settings, tenant, base_url=server.base_url, headers=headers)
    if server.auth_type != McpAuthType.HEADER.value:
        return await _probe(settings, tenant, base_url=server.base_url, headers={})
    credential = override_credential
    if credential is None:
        credential = await _reveal_for_test(store, server)
    headers = _auth_header(settings, base_url=server.base_url, credential=credential)
    return await _probe(settings, tenant, base_url=server.base_url, headers=headers)
