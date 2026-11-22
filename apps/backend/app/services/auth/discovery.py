"""Where the configured issuer's login endpoints live.

One question: given ``OIDC_ISSUER``, what are its authorize, token and JWKS
URLs? The walk itself is the same well-known lookup the MCP OAuth package
already implements, so the candidate order
(:func:`~app.services.oauth.discovery.authorization_server_metadata_urls`), the
byte-capped redirect-refusing fetch
(:func:`~app.services.oauth.discovery.fetch_first_metadata`), the endpoint
validation (:func:`~app.services.oauth.bundle.validate_endpoint`) and the
public-host anchor (:func:`~app.services.oauth.discovery.require_fetchable`)
are all imported rather than rewritten.

What differs is the TRUST ANCHOR, and it differs in the deployment's favour.
There the base URL was typed into an admin form; here it is an environment
variable, which is the most trusted input the process has. So a PRIVATE issuer
- a Keycloak on the compose network, a dev provider on loopback - is free to
name private endpoints, exactly as a privately-registered MCP server is, while
a PUBLIC issuer (Google, Okta) may not steer the backend at an internal
address: a metadata document is still fetched over the network, and a
compromised or hijacked one naming ``http://169.254.169.254/...`` as the token
endpoint would have this process POST the client secret and the authorization
code straight at a metadata service.

RFC 8414's mix-up defense is enforced here too: the document must declare the
issuer it was resolved from. Without it, a look-alike host could answer for
somebody else's issuer and supply its own signing keys.

The result is CACHED in-process with a short TTL. Provider metadata changes on
the order of years, and a login that pays a metadata fetch plus a JWKS fetch
before it can verify anything is a login that feels broken.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator

from app.services.auth.errors import LoginConfigurationError
from app.services.oauth.bundle import validate_endpoint
from app.services.oauth.discovery import (
    DiscoveryError,
    authorization_server_metadata_urls,
    fetch_first_metadata,
    registered_host_is_public,
    require_fetchable,
)

if TYPE_CHECKING:
    import httpx

# Long enough that no login pays for it twice in a session, short enough that a
# provider rotating an endpoint is picked up the same day without a restart.
CACHE_TTL_SECONDS = 3600.0

# Discovery is one small document behind a well-known URL. A provider that
# cannot serve it quickly is a provider nobody can log in through, and the
# person is waiting on a browser redirect while this runs.
DISCOVERY_TIMEOUT_SECONDS = 10.0


class OidcProviderMetadata(BaseModel):
    """The four facts a login needs from an OpenID Connect provider.

    Deliberately narrower than the OAuth metadata model the MCP package parses:
    nothing here needs a revocation endpoint or a scope inventory, and every
    field that IS here is one a login cannot proceed without - so a provider
    missing any of them fails at discovery with a clear reason rather than at
    the callback with a confusing one.
    """

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str

    @field_validator("authorization_endpoint", "token_endpoint", "jwks_uri")
    @classmethod
    def _check_endpoint(cls, value: str) -> str:
        """Require https (or http on loopback) with a host and no credentials."""
        return validate_endpoint(value)


_cache: dict[str, tuple[float, OidcProviderMetadata]] = {}


def clear_discovery_cache() -> None:
    """Forget every cached provider document (tests, and a config reload)."""
    _cache.clear()


async def discover_provider(
    issuer: str, *, client: httpx.AsyncClient, now: float | None = None
) -> OidcProviderMetadata:
    """Resolve an issuer to its authorize, token and JWKS endpoints.

    Args:
        issuer: The configured ``OIDC_ISSUER``.
        client: The httpx client to fetch with; injected so tests use a
            MockTransport and callers own the timeout.
        now: Monotonic-ish clock for the cache, injectable for tests.

    Returns:
        The validated provider metadata, from cache when it is fresh.

    Raises:
        LoginConfigurationError: No metadata was found, it did not parse, it
            declared a different issuer, or an endpoint failed validation.
            Every one of these is the deployment's configuration or its
            provider - never the person signing in.
    """
    key = issuer.rstrip("/")
    clock = time.monotonic() if now is None else now
    cached = _cache.get(key)
    if cached is not None and cached[0] > clock:
        return cached[1]

    # The issuer gets the SAME rule as the endpoints it goes on to name, and it
    # needs it most: this document is the trust anchor - it says where the
    # signing keys live - so fetching it over plain http lets anyone on the path
    # substitute a jwks_uri they control and mint accepted tokens at will. Every
    # discovered endpoint is validated below; the thing that names them cannot be
    # the one exception. http on loopback stays legal, which is the local
    # development provider the rule was written for.
    try:
        validate_endpoint(key)
    except ValueError as exc:
        raise LoginConfigurationError(f"OIDC_ISSUER is not usable: {exc}") from exc

    # The env var is the trust anchor. A private issuer may name private
    # endpoints (the operator pointed the deployment into that network); a
    # public one may not (see the module docstring).
    try:
        issuer_public = await registered_host_is_public(key)
    except Exception as exc:  # noqa: BLE001 - a name that will not resolve is a config fault
        raise LoginConfigurationError(f"OIDC_ISSUER {key!r} could not be resolved") from exc

    try:
        _url, raw = await fetch_first_metadata(
            client, authorization_server_metadata_urls(key), what="OIDC provider metadata"
        )
    except DiscoveryError as exc:
        raise LoginConfigurationError(f"no OpenID provider metadata was served by {key!r}") from exc

    try:
        metadata = OidcProviderMetadata.model_validate(raw)
    except ValueError as exc:
        raise LoginConfigurationError(
            f"the provider metadata at {key!r} is missing an endpoint a login needs"
        ) from exc

    # RFC 8414's mix-up defense: the document must declare the issuer it was
    # resolved from, or a look-alike host can answer for this one and hand over
    # its own signing keys.
    if metadata.issuer.rstrip("/") != key:
        raise LoginConfigurationError(
            f"the provider metadata declares issuer {metadata.issuer!r}, not the configured {key!r}"
        )

    for endpoint, what in (
        (metadata.authorization_endpoint, "authorization endpoint"),
        (metadata.token_endpoint, "token endpoint"),
        (metadata.jwks_uri, "JWKS endpoint"),
    ):
        try:
            await require_fetchable(endpoint, registered_public=issuer_public, what=what)
        except DiscoveryError as exc:
            raise LoginConfigurationError(str(exc)) from exc

    _cache[key] = (clock + CACHE_TTL_SECONDS, metadata)
    return metadata
