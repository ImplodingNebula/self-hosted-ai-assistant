"""HTTP schemas for the Integrations REST API - the frozen wire contract.

The write surface over the ``mcp_servers`` registry and the per-server
credentials. As everywhere in ``schemas/``, DTOs are separate from the DB models:
UUIDs go out as strings, ``tenant_id`` never leaves the backend, and a credential
VALUE never appears in any response - only its ``last4`` hint does.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING, Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

from app.models import McpAuthType, McpCredentialScope
from app.schemas.common import UtcDatetime

if TYPE_CHECKING:
    from datetime import datetime

    from app.models import McpServer
    from app.providers.secrets import SecretMetadata

_MAX_URL_LEN = 2048

# Names with no public DNS meaning, so plain http to them stays on the operator's
# own network: the special-use suffixes (RFC 6762 .local, RFC 8375 home.arpa, and
# the de-facto .internal / .localdomain) plus loopback.
_PRIVATE_HOST_SUFFIXES = (".local", ".internal", ".localdomain", ".home.arpa", ".localhost")


def _host_may_use_cleartext(host: str) -> bool:
    """True when plain http to ``host`` cannot cross the public internet.

    The credential a registered server carries rides in an ``Authorization``
    header on every listing and every tool call, so cleartext to a routable host
    publishes it. Cleartext is still legitimate INSIDE a deployment - the seeded
    chassis is reached as ``http://mcp-chassis:8765/mcp`` and local development
    lives on loopback - so the rule is "no public host", not "no http".

    Judged lexically, with no DNS lookup, for three reasons: a pydantic validator
    is synchronous and must not block the event loop on a resolver; validation
    that depends on DNS turns a resolver outage into a 422 on a previously valid
    URL; and a name that resolves privately at registration time can resolve
    publicly a second later, so a lookup here would buy confidence it cannot
    keep. The dial-time guard against a re-pointed host is the transport's
    (no redirects, see :mod:`app.providers.mcp.source`), not this validator's.

    Args:
        host: The URL's hostname, already lowercased and de-bracketed by
            :func:`urlsplit`.

    Returns:
        True when ``http://`` is acceptable for this host.
    """
    host = host.strip().rstrip(".")
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        # Same rule the web fetcher's SSRF guard applies to an IP literal:
        # anything not globally routable is internal (loopback, RFC1918,
        # link-local, CGN), and an IPv4-mapped IPv6 form is unwrapped first so a
        # mapped internal address cannot slip through.
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            ip = mapped
        return not ip.is_global
    if host == "localhost" or host.endswith(_PRIVATE_HOST_SUFFIXES):
        return True
    # A single-label name (a compose service, a kubernetes short name) resolves
    # only within the deployment's own network.
    return "." not in host


def _validate_base_url(value: str) -> str:
    """Require an http(s) URL with a host - the transport speaks Streamable HTTP.

    Rejects other schemes (``file:``, ``gopher:``) and hostless values up front,
    so a bad registration is a plain 422 rather than a confusing runtime failure.
    Plain ``http`` is allowed only to a host that is not publicly routable (see
    :func:`_host_may_use_cleartext`), mirroring the rule the OAuth endpoint
    validator already applies: an operator pasting a vendor's ``http://`` URL
    would otherwise put the server's bearer token on the wire in the clear on
    every listing and every call, with nothing warning them.

    Registration is still SSRF-adjacent (an admin may deliberately point at an
    internal host); that is gated by the admin role and recorded in the audit
    log, not by URL allow-listing here.
    """
    value = value.strip()
    if len(value) > _MAX_URL_LEN:
        raise ValueError(f"base_url must be at most {_MAX_URL_LEN} characters.")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https"):
        raise ValueError("base_url must be an http:// or https:// URL.")
    if not parts.netloc:
        raise ValueError("base_url must include a host.")
    if parts.username or parts.password:
        # A url like https://user:token@host embeds a credential that the audit
        # log records verbatim; reject it so the credential goes through the
        # write-only credential endpoint instead of into a log line.
        raise ValueError(
            "base_url must not embed credentials (user:pass@host); "
            "connect a credential through the credential endpoint instead."
        )
    if parts.scheme == "http" and not _host_may_use_cleartext(parts.hostname or ""):
        raise ValueError(
            "base_url must be https:// for a public host - plain http would send "
            "this server's credential in the clear. http is allowed only for "
            "loopback and internal hosts (a compose service name, .local, "
            ".internal)."
        )
    return value


def _validate_config(value: dict) -> dict:
    """Validate the operator-editable ``config`` blob's known keys.

    Only ``disabled_tools`` is typed today (a list of tool-name strings, the
    per-tool deny-list); any other key is passed through untouched so future
    transport knobs need no schema change here.
    """
    disabled = value.get("disabled_tools")
    if disabled is not None and (
        not isinstance(disabled, list) or not all(isinstance(item, str) for item in disabled)
    ):
        raise ValueError("config.disabled_tools must be a list of tool-name strings.")
    return value


class IntegrationServerCreate(BaseModel):
    """Register a new MCP server for the tenant."""

    name: str = Field(min_length=1, max_length=200)
    base_url: str
    credential_scope: McpCredentialScope = McpCredentialScope.WORKSPACE
    # HOW the server authenticates (scope above says WHOSE credential). Defaults
    # to ``header`` so unmodified clients keep today's behaviour exactly.
    auth_type: McpAuthType = McpAuthType.HEADER
    config: dict = Field(default_factory=dict)

    _check_url = field_validator("base_url")(_validate_base_url)
    _check_config = field_validator("config")(_validate_config)

    @model_validator(mode="after")
    def _no_per_user_oauth(self) -> IntegrationServerCreate:
        """Reject a per-user OAuth row - the combination is a lie today.

        The consent flow seals ONE workspace grant per server (per-user OAuth
        waits for real login), so a row declaring ``credential_scope=user``
        would pass the member gate while every reveal still hit the shared
        grant. This guard leaves with the Phase 8b per-user flow.
        """
        if self.auth_type == McpAuthType.OAUTH and self.credential_scope == McpCredentialScope.USER:
            raise ValueError(
                "per-user OAuth connections are not available yet: an OAuth "
                "server's grant is workspace-scoped until real login lands."
            )
        return self

    @model_validator(mode="after")
    def _no_per_user_none(self) -> IntegrationServerCreate:
        """Reject a per-user auth-none row - the combination promises nothing it keeps.

        An auth-none source attaches no credential for anyone (the row's
        declared auth is the authority at call time), so ``per-user`` would
        promise a fail-closed per-member gate that no call path enforces.
        Unlike the OAuth guard this one is permanent: with no credential there
        is nothing whose scope could ever differ per member.
        """
        if self.auth_type == McpAuthType.NONE and self.credential_scope == McpCredentialScope.USER:
            raise ValueError(
                "a per-user scope needs a per-user credential: a server that "
                "takes no credential is shared by nature."
            )
        return self


class IntegrationServerUpdate(BaseModel):
    """Patch a server. Every field optional; ``name`` is deliberately immutable.

    The name is the key the stored credential, the bootstrap seed, and the
    per-call reveal all hang on, so a rename would orphan the credential - a
    rename is delete + recreate + reconnect instead.
    """

    enabled: bool | None = None
    base_url: str | None = None
    credential_scope: McpCredentialScope | None = None
    auth_type: McpAuthType | None = None
    config: dict | None = None

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, value: str | None) -> str | None:
        return None if value is None else _validate_base_url(value)

    @field_validator("config")
    @classmethod
    def _check_config(cls, value: dict | None) -> dict | None:
        return None if value is None else _validate_config(value)


class IntegrationServerOut(BaseModel):
    """One registered server, as returned - metadata only, never a credential."""

    id: str
    name: str
    base_url: str
    enabled: bool
    credential_scope: McpCredentialScope
    auth_type: McpAuthType
    config: dict
    created_at: UtcDatetime
    # OAuth grant status, meaningful only on auth_type=oauth rows: whether a
    # sealed grant exists, and when it was last (re)connected. Metadata only -
    # the grant itself is never read to answer a listing.
    oauth_connected: bool = False
    oauth_connected_at: UtcDatetime | None = None
    # Whether this row can authenticate, in ONE answer across all three auth
    # types, so a listing can say so without a per-row round trip. The list
    # already answered this for oauth rows and refused to for header rows; that
    # asymmetry is what left "has a credential" invisible until a card was
    # expanded. Metadata only - existence is checked, nothing is decrypted.
    #
    # - ``connected``: a credential is sealed for the acting principal, or an
    #   OAuth grant exists.
    # - ``deployment_key``: nothing is stored, but an env bootstrap key backs
    #   this URL (any env-configured sidecar, not only the chassis). Working,
    #   and worth naming separately: it breaks if the deployment drops that
    #   variable, and no card action would explain why.
    # - ``missing``: this row needs a credential and has none - calls fail closed.
    # - ``not_required``: ``auth_type: none``. A fine state, never a warning.
    credential_state: Literal["connected", "deployment_key", "missing", "not_required"] = (
        "not_required"
    )
    # Whether the deployment's environment currently bootstraps a row of this
    # NAME. COMPUTED per response against the live settings and never stored:
    # the seeding path is get-or-create by (tenant, name), so a matching name is
    # exactly what comes back at the next restart. What it means for a client:
    # delete is refused with a 409 while this is true, and the durable off
    # switch is ``enabled`` (which seeding never overwrites). It goes false on
    # its own once the deployment stops naming the row, and delete opens up.
    seeded: bool = False

    @classmethod
    def from_entity(
        cls,
        server: McpServer,
        *,
        oauth_connected: bool = False,
        oauth_connected_at: datetime | None = None,
        credential_state: Literal[
            "connected", "deployment_key", "missing", "not_required"
        ] = "not_required",
        seeded: bool = False,
    ) -> IntegrationServerOut:
        """Map a DB row to the wire shape (UUID -> str; no tenant_id)."""
        return cls(
            id=str(server.id),
            name=server.name,
            base_url=server.base_url,
            enabled=server.enabled,
            credential_scope=McpCredentialScope(server.credential_scope),
            auth_type=McpAuthType(server.auth_type),
            config=server.config,
            created_at=server.created_at,
            oauth_connected=oauth_connected,
            oauth_connected_at=oauth_connected_at,
            credential_state=credential_state,
            seeded=seeded,
        )


class IntegrationOAuthEndpoints(BaseModel):
    """Operator-supplied authorization-server endpoints (the presets path).

    Naming these skips discovery entirely - the generic-provider path for
    services like GitHub or Google that are not MCP-native. Endpoint hygiene
    (https or loopback, no embedded credentials) is enforced where the values
    are consumed, by the service-layer endpoint model.
    """

    issuer: str = Field(min_length=1, max_length=_MAX_URL_LEN)
    authorization_endpoint: str = Field(min_length=1, max_length=_MAX_URL_LEN)
    token_endpoint: str = Field(min_length=1, max_length=_MAX_URL_LEN)
    revocation_endpoint: str | None = Field(default=None, max_length=_MAX_URL_LEN)


class IntegrationOAuthIdentity(BaseModel):
    """Where to ask a provider who the connected account is, and which field says.

    Read once at callback time with the fresh access token, best effort - the
    label ("Connected as octocat") is a display nicety, never a gate. Endpoint
    hygiene is enforced where the value is consumed, by the flow's validator.
    """

    userinfo_endpoint: str = Field(min_length=1, max_length=_MAX_URL_LEN)
    label_field: str = Field(min_length=1, max_length=64)


class IntegrationOAuthConnectRequest(BaseModel):
    """Start a workspace OAuth consent for a server.

    ``endpoints`` absent means the MCP-native path: the authorization server is
    discovered from the MCP server itself. ``scopes`` is required and explicit -
    the platform never widens a request to whatever the provider supports.
    """

    client_id: str = Field(min_length=1, max_length=500)
    # Write-only, like every credential: sealed on arrival, never echoed back.
    client_secret: str | None = Field(default=None, max_length=8192)
    scopes: list[str] = Field(min_length=1)
    endpoints: IntegrationOAuthEndpoints | None = None
    # Optional identity lookup for the "Connected as <account>" line; the
    # presets supply it for providers whose lookup is known.
    identity: IntegrationOAuthIdentity | None = None
    # Provider extensions outside RFC 6749 that some consent screens need
    # (Google's access_type=offline / prompt=consent). Operator-visible prefill
    # only; the authorize builder refuses keys that shadow the protocol's own.
    extra_authorize_params: dict[
        Annotated[str, StringConstraints(min_length=1, max_length=64)],
        Annotated[str, StringConstraints(min_length=1, max_length=256)],
    ] = Field(default_factory=dict, max_length=8)

    @field_validator("extra_authorize_params")
    @classmethod
    def _no_protocol_shadowing(cls, value: dict[str, str]) -> dict[str, str]:
        """Refuse extras that shadow the flow's own parameters (422, not 500).

        The authorize builder enforces the same rule as defense in depth; this
        validator makes the failure a clean request error at the boundary.
        """
        from app.services.oauth.authorize import RESERVED_AUTHORIZE_PARAMS

        shadowed = {key for key in value if key.strip().casefold() in RESERVED_AUTHORIZE_PARAMS}
        if shadowed:
            raise ValueError(f"these parameters belong to the flow itself: {sorted(shadowed)}")
        return value


class IntegrationOAuthConnectResponse(BaseModel):
    """Where to send the admin's browser to approve the connection."""

    authorization_url: str


class IntegrationOAuthRedirectUriOut(BaseModel):
    """The exact callback address this deployment sends to OAuth providers.

    Providers refuse a consent whose redirect does not match a URL registered
    on the app, so the operator needs this literal value at registration time;
    it is deployment config, not a secret.
    """

    redirect_uri: str


class IntegrationPresetServer(BaseModel):
    """The add-server form values a preset prefills."""

    name_suggestion: str
    base_url: str
    auth_type: McpAuthType
    credential_scope: McpCredentialScope


class IntegrationPresetOAuth(BaseModel):
    """The connect-form values a preset prefills.

    ``endpoints`` absent means the provider is MCP-native: discovery finds the
    authorization server from the MCP server itself, and the form needs no
    manual endpoints. ``extra_authorize_params`` are shown, editable, in the
    form - a preset makes them VISIBLE, never silent.
    """

    scopes: list[str]
    endpoints: IntegrationOAuthEndpoints | None = None
    identity: IntegrationOAuthIdentity | None = None
    extra_authorize_params: dict[str, str] = Field(default_factory=dict)


class IntegrationPresetOut(BaseModel):
    """One curated integration: prefill values plus honest setup notes."""

    id: str
    name: str
    description: str
    server: IntegrationPresetServer
    oauth: IntegrationPresetOAuth
    # Where the operator registers their own OAuth app, and the caveats that
    # decide whether this preset is viable for their deployment.
    registration_url: str
    setup_notes: list[str]


class IntegrationPresetListResponse(BaseModel):
    """The curated presets, in display order."""

    items: list[IntegrationPresetOut]


class IntegrationServerListResponse(BaseModel):
    """A tenant's servers (enabled and disabled), name-ordered.

    ``credential_storage_available`` describes the DEPLOYMENT, not any one
    server: false means ``SECRET_MASTER_KEY`` is unset, so no credential can be
    sealed anywhere. It rides on the list because otherwise a client can only
    learn it by attempting a write and failing - which is exactly the flow that
    strands a freshly registered server without its credential.
    """

    items: list[IntegrationServerOut]
    credential_storage_available: bool


class CredentialWrite(BaseModel):
    """The one-time plaintext credential to seal for a server."""

    value: str = Field(min_length=1, max_length=8192)


class CredentialStatus(BaseModel):
    """Whether a server has a credential, plus its metadata hint - never plaintext.

    ``connected`` is the single field a client can always rely on; the rest are
    present only when a credential is stored (``last4`` is the recognisable hint,
    empty for a value too short to hint safely).
    """

    connected: bool
    scope: McpCredentialScope | None = None
    last4: str | None = None
    created_at: UtcDatetime | None = None
    updated_at: UtcDatetime | None = None

    @classmethod
    def from_metadata(cls, metadata: SecretMetadata) -> CredentialStatus:
        """A connected status carrying the stored secret's metadata."""
        return cls(
            connected=True,
            scope=metadata.scope,
            last4=metadata.last4,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
        )

    @classmethod
    def not_connected(cls) -> CredentialStatus:
        """The status for a server with no stored credential."""
        return cls(connected=False)


class ToolDescriptorOut(BaseModel):
    """One tool a server advertises, as surfaced by a test-connection handshake."""

    name: str
    description: str
    side_effecting: bool


class ConnectionTestError(BaseModel):
    """Why a test-connection handshake did not succeed, classified for the UI."""

    code: Literal["unreachable", "timeout", "auth_rejected", "protocol_mismatch", "unknown"]
    detail: str


class ConnectionTestRequest(BaseModel):
    """Probe a server BEFORE saving it: an initialize + tools/list handshake only.

    ``credential`` is a one-shot plaintext used for THIS handshake and never
    stored; absent, the handshake uses the workspace header the chat path would
    (so a test faithfully mirrors a real listing).
    """

    base_url: str
    config: dict = Field(default_factory=dict)
    credential_scope: McpCredentialScope = McpCredentialScope.WORKSPACE
    credential: str | None = Field(default=None, max_length=8192)

    _check_url = field_validator("base_url")(_validate_base_url)
    _check_config = field_validator("config")(_validate_config)


class ServerConnectionTestRequest(BaseModel):
    """Re-test an ALREADY-REGISTERED server (by id).

    Carries no URL - the target is the stored server row. ``credential`` is an
    optional one-shot override to test a new key before replacing the stored one;
    absent, the handshake uses the server's stored credential, exactly as the chat
    path would. Never stored.
    """

    credential: str | None = Field(default=None, max_length=8192)


class ConnectionTestResult(BaseModel):
    """The handshake outcome: reachable + its tools, or a classified error.

    Always returned with HTTP 200 - a target being down is a normal diagnostic
    result, not a fault of this API - so a client reads ``ok`` and, on failure,
    ``error.code``.
    """

    ok: bool
    tools: list[ToolDescriptorOut] = Field(default_factory=list)
    error: ConnectionTestError | None = None
