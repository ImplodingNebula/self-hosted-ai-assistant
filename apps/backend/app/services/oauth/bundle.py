"""The sealed OAuth token bundle - what a connected server's grant looks like at rest.

An OAuth connection is more than one string: the short-lived access token, the
optional refresh token that renews it, and the endpoints and identity needed to
renew and revoke without re-running discovery. This module is the codec for that
bundle: a typed model, sealed to and unsealed from the single ``str`` that
:meth:`SecretStore.store` accepts - the crypto seam itself is deliberately not
widened (no structured-value API; the caller serializes).

Two properties of the sealed form are load-bearing:

- **The tail is never secret material.** ``SecretStore`` exposes the last 4
  characters of whatever it seals as the readable ``last4`` hint, so the JSON
  must END on a non-secret field. ``issuer`` is declared last for exactly this
  reason, and a test pins the serialized tail to it - reordering the fields is
  a leak, not a style choice.
- **Expiry is absolute, never relative.** Providers answer with ``expires_in``
  (seconds from now); persisting that across a restart is a silently-wrong
  expiry. The bundle stores a timezone-aware UTC ``expires_at`` computed once
  at grant time, and rejects naive datetimes outright.

``issuer`` is stored from day one even though nothing reads it yet: issuer
validation (RFC 9207) and issuer-bound credentials both need it later, and it
costs nothing to record at grant time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

from app.services.tools.naming import credential_secret_name

if TYPE_CHECKING:
    from uuid import UUID

# urlsplit().hostname strips brackets from IPv6 literals, so "::1" is the
# form that actually matches (never "[::1]").
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def oauth_token_secret_name(server_id: UUID, base_url: str) -> str:
    """The SecretStore name a server's sealed token bundle lives under.

    Derives from :func:`credential_secret_name`, so the bundle inherits the same
    stable-identity binding as a pasted credential: a recreated server (new id)
    or a re-pointed one (new url) starts fresh and fails closed rather than
    inheriting a grant that consented to a different endpoint.
    """
    return f"{credential_secret_name(server_id, base_url)}:oauth-token"


def oauth_client_secret_name(server_id: UUID, base_url: str) -> str:
    """The SecretStore name a server's OAuth client registration lives under.

    The operator-supplied client id/secret for the authorization server - a
    separate secret from the token bundle so revoking a grant (delete the
    token) never destroys the registration needed to grant again.
    """
    return f"{credential_secret_name(server_id, base_url)}:oauth-client"


def validate_endpoint(value: str) -> str:
    """Require an https URL - or http on loopback only - with a host.

    A token endpoint reached over plain http hands the access and refresh
    tokens to the network; the one legitimate http case is a local development
    authorization server on the loopback interface.
    """
    value = value.strip()
    parts = urlsplit(value)
    if parts.username or parts.password:
        # Matching the base_url rule in the integrations schema: a credential
        # embedded in a URL leaks wherever the URL is logged or displayed.
        raise ValueError("an OAuth endpoint must not embed credentials (user:pass@host).")
    if parts.scheme == "https":
        pass
    elif parts.scheme == "http":
        if parts.hostname not in _LOOPBACK_HOSTS:
            raise ValueError("an OAuth endpoint must be https (http is allowed on loopback only).")
    else:
        raise ValueError("an OAuth endpoint must be an http(s) URL.")
    if not parts.netloc:
        raise ValueError("an OAuth endpoint must include a host.")
    return value


class OAuthTokenBundle(BaseModel):
    """One server's OAuth grant, as sealed into the SecretStore.

    Field order is part of the contract: ``issuer`` is LAST so the sealed
    JSON's tail - which ``SecretStore`` exposes as the readable ``last4``
    hint - is never token material. See the module docstring.
    """

    access_token: str = Field(min_length=1)
    refresh_token: str | None = None
    # Absolute, timezone-aware UTC - computed once at grant time from the
    # provider's relative ``expires_in``. Never store the relative form.
    expires_at: datetime
    granted_scope: str = Field(min_length=1)
    token_endpoint: str
    revocation_endpoint: str | None = None
    client_id: str = Field(min_length=1)
    # The RFC 8707 resource identifier the grant is bound to (MCP-native path
    # only; None for generic providers and for bundles sealed before the field
    # existed). Refresh repeats it so the renewed token keeps its audience.
    resource: str | None = None
    # Set when a refresh fails terminally: the connection needs a human to
    # re-consent, and the UI reads this to say so instead of retrying forever.
    needs_reconnect: bool = False
    issuer: str = Field(min_length=1)

    @field_validator("expires_at")
    @classmethod
    def _require_aware_utc(cls, value: datetime) -> datetime:
        """Reject naive datetimes; normalize aware ones to UTC."""
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("expires_at must be timezone-aware (UTC) - never a naive datetime.")
        return value.astimezone(UTC)

    @field_validator("token_endpoint")
    @classmethod
    def _check_token_endpoint(cls, value: str) -> str:
        return validate_endpoint(value)

    @field_validator("revocation_endpoint")
    @classmethod
    def _check_revocation_endpoint(cls, value: str | None) -> str | None:
        return None if value is None else validate_endpoint(value)

    def seal(self) -> str:
        """Serialize to the compact JSON string the SecretStore seals.

        Emits fields in declaration order (the ``last4`` contract above) with
        no whitespace; :meth:`unseal` is the exact inverse.
        """
        return self.model_dump_json()

    @classmethod
    def unseal(cls, value: str) -> OAuthTokenBundle:
        """Parse a revealed SecretStore value back into a validated bundle.

        Validation runs in full, so a tampered or hand-edited value fails
        loudly here rather than surfacing as a malformed request downstream.
        """
        return cls.model_validate_json(value)
