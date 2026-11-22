"""Building the authorization request - PKCE, state, and the consent URL.

The pieces of an OAuth 2.1 authorization-code request that must be exactly
right, as pure functions: the PKCE verifier/challenge pair (RFC 7636, S256
only - the plain method is banned in OAuth 2.1), the CSRF ``state`` value, the
RFC 8707 ``resource`` canonicalization, and the consent-URL assembly itself.
No routes, no storage - the connect flow wires these to HTTP later.

Two rules are load-bearing enough to restate here:

- **Scopes are operator-supplied, never auto-expanded.** The metadata's
  ``scopes_supported`` is advisory; requesting all of it turns a credential
  anyone reaching the port can spend into the widest one the provider offers.
  :func:`build_authorization_url` therefore REQUIRES a non-empty scope list
  and never reads scopes from metadata.
- **``resource`` is sent only on the MCP-native path.** It binds the issued
  token to the MCP server it was requested for (so a token for one server
  cannot be replayed against another); a generic provider like GitHub does not
  take the parameter, which is what :class:`OAuthEndpoints.send_resource`
  records.
"""

from __future__ import annotations

import hashlib
import secrets
import string
from base64 import urlsafe_b64encode
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, field_validator

from app.services.oauth.bundle import validate_endpoint

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mcp.shared.auth import OAuthMetadata

# RFC 7636's unreserved alphabet for the code verifier.
_VERIFIER_ALPHABET = string.ascii_letters + string.digits + "-._~"
_VERIFIER_LENGTH = 128  # the RFC maximum; entropy is free here

_DEFAULT_PORTS = {"http": 80, "https": 443}


class OAuthEndpoints(BaseModel):
    """Where a server's OAuth conversation happens - discovered or operator-supplied.

    One shape for both paths: :meth:`from_metadata` builds it from a discovery
    walk (``send_resource=True`` - an MCP-native authorization server takes the
    RFC 8707 ``resource`` parameter), while an operator-configured row for a
    generic provider (GitHub, Google) constructs it directly with
    ``send_resource=False``. Downstream code never needs to know which.
    """

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    revocation_endpoint: str | None = None
    # Whether authorize/token requests carry the RFC 8707 ``resource``
    # parameter. True on the MCP-native (discovered) path; False for generic
    # providers, which do not take the parameter.
    send_resource: bool

    @field_validator("authorization_endpoint", "token_endpoint")
    @classmethod
    def _check_endpoint(cls, value: str) -> str:
        return validate_endpoint(value)

    @field_validator("revocation_endpoint")
    @classmethod
    def _check_revocation(cls, value: str | None) -> str | None:
        return None if value is None else validate_endpoint(value)

    @classmethod
    def from_metadata(cls, metadata: OAuthMetadata) -> OAuthEndpoints:
        """Adapt a discovery result; discovered servers take ``resource``."""
        revocation = metadata.revocation_endpoint
        return cls(
            issuer=str(metadata.issuer).rstrip("/"),
            authorization_endpoint=str(metadata.authorization_endpoint),
            token_endpoint=str(metadata.token_endpoint),
            revocation_endpoint=None if revocation is None else str(revocation),
            send_resource=True,
        )


def generate_pkce_verifier() -> str:
    """A fresh 128-character PKCE code verifier (RFC 7636 unreserved alphabet)."""
    return "".join(secrets.choice(_VERIFIER_ALPHABET) for _ in range(_VERIFIER_LENGTH))


def pkce_challenge(verifier: str) -> str:
    """The S256 code challenge for a verifier: base64url(sha256), unpadded.

    S256 is the only method OAuth 2.1 permits; there is deliberately no code
    path that could fall back to the ``plain`` method.
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def generate_state() -> str:
    """A fresh CSRF ``state`` value (32 random bytes, URL-safe).

    Single-use by contract: the callback consumes it or it expires - reuse is
    the authorization-code-injection door, and the pending-authorization store
    (a later PR) enforces the single use.
    """
    return secrets.token_urlsafe(32)


def canonical_resource(base_url: str) -> str:
    """The RFC 8707 canonical resource URI for an MCP server's base URL.

    Lowercases the scheme and host, drops a default port, keeps the path
    exactly (no trailing-slash normalization), and drops any fragment or
    query - the identifier names the server, not a request to it.
    """
    parts = urlsplit(base_url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if ":" in host:
        # urlsplit strips the brackets off an IPv6 literal; a rebuilt netloc
        # must restore them or the port separator becomes ambiguous.
        host = f"[{host}]"
    if parts.port is not None and parts.port != _DEFAULT_PORTS.get(scheme):
        host = f"{host}:{parts.port}"
    return urlunsplit((scheme, host, parts.path, "", ""))


# Parameters the protocol itself owns. An extra authorize parameter may never
# shadow one of these: silently overriding, say, redirect_uri or the PKCE
# challenge would dismantle the flow's guarantees from inside its own request.
RESERVED_AUTHORIZE_PARAMS = frozenset(
    {
        "response_type",
        "client_id",
        "redirect_uri",
        "scope",
        "state",
        "code_challenge",
        "code_challenge_method",
        "resource",
    }
)


def build_authorization_url(
    endpoints: OAuthEndpoints,
    *,
    client_id: str,
    redirect_uri: str,
    scopes: Sequence[str],
    state: str,
    code_challenge: str,
    resource: str | None = None,
    extra_params: Mapping[str, str] | None = None,
) -> str:
    """Assemble the consent URL the admin's browser is sent to.

    Args:
        endpoints: Where the authorization server lives (either path).
        client_id: The operator-registered OAuth client id.
        redirect_uri: The callback URL registered for that client.
        scopes: The OPERATOR-SUPPLIED scope list. Must be non-empty; this
            function never widens or substitutes scopes from metadata.
        state: The CSRF token from :func:`generate_state`.
        code_challenge: The S256 challenge from :func:`pkce_challenge`.
        resource: The canonical MCP server URI; attached only when the
            endpoints declare ``send_resource`` (the MCP-native path).
        extra_params: Operator-supplied provider extensions outside RFC 6749
            (Google's ``access_type=offline``/``prompt=consent``). Visible
            prefill only, never platform-defaulted; may not shadow a
            protocol-owned parameter.

    Raises:
        ValueError: ``scopes`` is empty - an unscoped request would leave the
            grant's breadth to the provider's defaults, unrecorded. Or the
            endpoints declare ``send_resource`` and no ``resource`` was given -
            the MCP-native path's token binding is mandatory, not optional.
            Or an extra parameter shadows a protocol-owned one.
    """
    if not scopes:
        raise ValueError("scopes must be operator-supplied and non-empty - never defaulted.")
    if endpoints.send_resource and resource is None:
        raise ValueError(
            "the MCP-native path requires the canonical resource - "
            "an unbound token request drops its replay protection."
        )
    # Normalized comparison: "Redirect_Uri" or "state " is still an attempt
    # to shadow a protocol parameter, even though the key is sent verbatim.
    shadowed = {
        key for key in (extra_params or ()) if key.strip().casefold() in RESERVED_AUTHORIZE_PARAMS
    }
    if shadowed:
        raise ValueError(
            f"extra authorize parameters may not shadow the protocol's own: {sorted(shadowed)}"
        )
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if endpoints.send_resource:
        params["resource"] = resource
    if extra_params:
        params.update(extra_params)
    separator = "&" if urlsplit(endpoints.authorization_endpoint).query else "?"
    return f"{endpoints.authorization_endpoint}{separator}{urlencode(params)}"
