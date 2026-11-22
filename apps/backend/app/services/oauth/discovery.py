"""Discovering a protected MCP server's authorization server - the metadata walk.

An OAuth-protected MCP server does not name its authorization server directly;
the client walks there: the server's *protected resource metadata* (RFC 9728)
names one or more authorization servers, and the authorization server's own
metadata (RFC 8414 / OIDC discovery) names the authorize and token endpoints.
This module is that walk as pure(ish) functions over an injected httpx client -
no routes, no storage, no SDK client flow (the SDK's interactive
``OAuthClientProvider`` is built for desktop apps that can block on a browser;
only its metadata MODELS are reused here, so the parsing stays spec-shaped).

Candidate order is part of the spec contract, not a style choice: the
resource-metadata URL from a 401 ``WWW-Authenticate`` challenge is tried first
when present, then the path-aware well-known location, then the root one; an
issuer with a path component tries the RFC 8414 path-insertion forms before the
OIDC path-suffix form. Tests pin both orders.

The walk's FETCH TARGETS are hostile input too, not just its results - the 401
challenge's hint and the metadata's authorization server are both named by the
remote server, and each steers an outbound request from the backend. Three
guards close that surface:

- the challenge hint must be SAME-ORIGIN with the registered base URL (a
  resource's metadata lives on the resource; a cross-origin hint is ignored);
- when the registered server is on a public address, every derived fetch
  target must resolve public too (:func:`~app.providers.search.fetcher.host_is_public`,
  the same SSRF guard the web fetcher uses) - a public resource naming a
  private authorization server is how a metadata endpoint gets probed. That
  applies to the DISCOVERED ENDPOINTS as well as to the issuer: the token and
  revocation endpoints are server-side POST targets carrying the client
  secret, the code and the verifier, so a metadata document naming
  ``http://127.0.0.1:9200/_bulk`` there is blind SSRF with credential
  exfiltration. A PRIVATE registered server (a sidecar, a loopback dev stack)
  may name private targets: the admin pointed the deployment into that network
  when they registered it, and registration is already admin-gated and audited;
- redirects are never followed (enforced per request, whatever the injected
  client's default) - a 3xx to an internal host would bypass the pre-fetch
  check.

The walk also verifies the two RFC identity bindings: the resource metadata
must describe the server it was fetched for (RFC 9728), and the authorization
server metadata must declare the issuer it was resolved from (RFC 8414's
mix-up defense). Every discovered endpoint then passes
:func:`~app.services.oauth.bundle.validate_endpoint` (https - or http on
loopback only - and no embedded credentials) AND :func:`require_fetchable`
(the public-host anchor above) before it is returned.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata

from app.providers.search.fetcher import host_is_public
from app.services.oauth.authorize import canonical_resource
from app.services.oauth.bundle import validate_endpoint
from app.services.oauth.fetch import read_json_capped

if TYPE_CHECKING:
    import httpx

# resource_metadata="<url>" inside a WWW-Authenticate challenge (RFC 9728
# section 5.1). The value is a quoted URL; the parameter name is
# case-insensitive per HTTP auth-param rules.
_RESOURCE_METADATA_PARAM = re.compile(r'resource_metadata\s*=\s*"([^"]+)"', re.IGNORECASE)

_WELL_KNOWN_PRM = "/.well-known/oauth-protected-resource"
_WELL_KNOWN_AS = "/.well-known/oauth-authorization-server"
_WELL_KNOWN_OIDC = "/.well-known/openid-configuration"


class DiscoveryError(Exception):
    """Discovery could not resolve a usable authorization server.

    Carries only a reason phrase and (already public) URLs - never a token or
    header value - so it is safe to log and to surface to an admin verbatim.
    """


def parse_resource_metadata_url(www_authenticate: str) -> str | None:
    """The ``resource_metadata`` URL from a 401 challenge, or None if absent."""
    match = _RESOURCE_METADATA_PARAM.search(www_authenticate)
    return match.group(1) if match else None


def protected_resource_metadata_urls(base_url: str) -> list[str]:
    """The well-known locations for a server's protected resource metadata, in order.

    Path-aware first (RFC 9728's path-insertion form, which lets one host serve
    metadata for several resources at distinct paths - the chassis pattern),
    then the host root.
    """
    parts = urlsplit(base_url)
    origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    path = parts.path.rstrip("/")
    candidates = []
    if path:
        candidates.append(f"{origin}{_WELL_KNOWN_PRM}{path}")
    candidates.append(f"{origin}{_WELL_KNOWN_PRM}")
    return candidates


def authorization_server_metadata_urls(issuer: str) -> list[str]:
    """The metadata locations for an issuer, in the spec's fallback order.

    An issuer WITH a path component (``https://as.example/tenant``) tries the
    RFC 8414 path-insertion forms first, then OIDC's path-suffix form; a
    rootless issuer tries the two well-known roots. First 200 that parses wins.
    """
    parts = urlsplit(issuer)
    origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    path = parts.path.rstrip("/")
    if path:
        return [
            f"{origin}{_WELL_KNOWN_AS}{path}",
            f"{origin}{_WELL_KNOWN_OIDC}{path}",
            f"{origin}{path}{_WELL_KNOWN_OIDC}",
        ]
    return [
        f"{origin}{_WELL_KNOWN_AS}",
        f"{origin}{_WELL_KNOWN_OIDC}",
    ]


def _same_origin(a: str, b: str) -> bool:
    """True when two URLs share scheme and netloc (host:port, verbatim)."""
    pa, pb = urlsplit(a), urlsplit(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)


def _covers_registered_url(declared_cmp: str, registered_cmp: str) -> bool:
    """Whether a declared resource identifier may stand for the registered URL.

    An exact match always may. A path PREFIX may too - Asana declares
    ``.../v2`` for its ``.../v2/mcp`` endpoint - but only a NEAR one: the
    declared path must be non-empty and may sit at most one segment above the
    registered URL.

    The reason is the shared host (the chassis pattern this project documents,
    several MCP servers mounted at distinct paths). A resource that declares an
    ancestor obtains a token whose audience covers every sibling beneath it,
    and the admin consented to exactly one of them. The bare origin is the
    degenerate case of that - an audience over the whole host - and a deep
    ancestor is the same claim wearing one path segment. RFC 8707 leaves
    granularity to the resource, so refusing the over-wide claim is a
    tightening, not a spec violation; one segment is what a real deployment
    needs (a version prefix over its endpoint) and nothing more.

    Both arguments are canonicalized and trailing-slash-stripped already.
    """
    if declared_cmp == registered_cmp:
        return True
    if not registered_cmp.startswith(declared_cmp + "/"):
        return False
    declared_path = urlsplit(declared_cmp).path.strip("/")
    if not declared_path:
        return False
    declared_depth = len(declared_path.split("/"))
    registered_depth = len(urlsplit(registered_cmp).path.strip("/").split("/"))
    return registered_depth - declared_depth <= 1


async def registered_host_is_public(base_url: str) -> bool:
    """Whether the ADMIN-REGISTERED server resolves to a globally-routable address.

    The trust anchor for every outbound target this module and the connect flow
    derive from remote input: a public registered server may not steer the
    backend at a private one, while a private registered server (a sidecar on
    the compose network, a loopback dev stack) may name private targets freely -
    the admin pointed the deployment into that network when they registered it.
    """
    return await host_is_public(urlsplit(base_url).hostname or "")


async def require_fetchable(url: str, *, registered_public: bool, what: str) -> None:
    """Refuse a request target the deployment must not be steered to.

    The target must be http(s) with a host and no embedded credentials, and -
    when the REGISTERED server resolves public - must itself resolve to a
    globally-routable address. A private registered server may name private
    targets: the admin granted that network at registration time.

    Public because the connect flow applies the SAME rule to the endpoints it
    later POSTs credentials to; one implementation, so the two cannot drift.

    Raises:
        DiscoveryError: The target is malformed, embeds credentials, or is
            private while the registered server is public.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise DiscoveryError(f"the {what} URL {url!r} is not a plain http(s) URL")
    if parts.username or parts.password:
        raise DiscoveryError(f"the {what} URL embeds credentials and was refused")
    if registered_public and not await host_is_public(parts.hostname):
        raise DiscoveryError(
            f"the {what} host {parts.hostname!r} does not resolve to a public "
            "address; a public server may not steer discovery to a private one"
        )


async def fetch_first_metadata(
    client: httpx.AsyncClient, urls: list[str], *, what: str
) -> tuple[str, dict]:
    """GET candidates in order; return (url, parsed json) for the first 200.

    A non-200 or a JSON parse failure moves to the next candidate rather than
    failing the walk - fallback order is the mechanism, not an error path.
    Redirects are never followed, whatever the injected client's default: a
    3xx to an internal host would bypass the pre-fetch trust check. The body is
    read under a byte cap (see :mod:`app.services.oauth.fetch`), because the
    host serving it was named by the remote server, not by an administrator.

    Public rather than private because the LOGIN discovery
    (:mod:`app.services.auth.discovery`) walks the same well-known candidate
    list against an operator-configured issuer. The two flows differ in what
    they trust and in what they do with the result, but "try these locations in
    order, take the first that parses, never follow a redirect, cap the body"
    is one behaviour and is kept as one implementation.
    """
    for url in urls:
        try:
            async with client.stream("GET", url, follow_redirects=False) as response:
                if response.status_code != 200:
                    continue
                return url, await read_json_capped(response, what=what)
        except Exception:  # noqa: BLE001 - a dead candidate is a fallthrough, not a crash
            # An oversized or unparseable document is a dead candidate like any
            # other: try the next well-known location rather than failing here.
            continue
    raise DiscoveryError(f"no {what} found at any well-known location: {', '.join(urls)}")


async def discover_authorization_server(
    base_url: str,
    *,
    client: httpx.AsyncClient,
    www_authenticate: str | None = None,
) -> tuple[OAuthMetadata, str]:
    """Walk from an MCP server's URL to its authorization server's metadata.

    Args:
        base_url: The MCP server's base URL (the protected resource).
        client: The httpx client to fetch with; injected so tests use a
            MockTransport and callers control timeouts.
        www_authenticate: The ``WWW-Authenticate`` header from a 401 challenge,
            when the caller has one - its ``resource_metadata`` URL is tried
            before the well-known candidates.

    Returns:
        The authorization server's parsed metadata, with the authorize and
        token endpoints validated (https or loopback, no embedded credentials),
        plus the resource identifier the metadata DECLARED (canonicalized).
        The declared identifier is what the RFC 8707 ``resource`` parameter
        must carry: it may name a path PREFIX covering the connection URL
        (Asana declares ``.../v2`` for its ``.../v2/mcp`` endpoint), and the
        authorization server binds tokens to that identifier, not to whatever
        URL the client happens to connect to.

    Raises:
        DiscoveryError: No metadata was found, the metadata named no
            authorization server, or a discovered endpoint failed validation.
    """
    # The trust anchor: what the ADMIN registered. If it resolves public, no
    # remote-named fetch target may be private; if it is itself private (a
    # sidecar, loopback dev), the admin granted that network at registration.
    registered_public = await registered_host_is_public(base_url)

    prm_candidates = protected_resource_metadata_urls(base_url)
    if www_authenticate:
        hinted = parse_resource_metadata_url(www_authenticate)
        # A resource's metadata lives on the resource: a cross-origin hint is
        # ignored (fall through to the well-known candidates), never fetched.
        if hinted and _same_origin(hinted, base_url):
            prm_candidates = [hinted, *prm_candidates]
    _, prm_raw = await fetch_first_metadata(
        client, prm_candidates, what="protected resource metadata"
    )
    try:
        # The SDK model requires at least one authorization server, so a
        # document naming none is rejected here - no separate emptiness check.
        prm = ProtectedResourceMetadata.model_validate(prm_raw)
    except ValueError as exc:
        raise DiscoveryError("the protected resource metadata did not parse") from exc
    # RFC 9728: the metadata must describe the server it was fetched FOR - a
    # document for some other resource must not steer this server's grant, and
    # must not claim an audience wider than the path the admin registered.
    declared = canonical_resource(str(prm.resource))
    declared_cmp = declared.rstrip("/")
    registered_cmp = canonical_resource(base_url).rstrip("/")
    if not _covers_registered_url(declared_cmp, registered_cmp):
        raise DiscoveryError(
            f"the protected resource metadata describes {str(prm.resource)!r}, "
            f"not the registered server"
        )

    issuer = str(prm.authorization_servers[0]).rstrip("/")
    await require_fetchable(
        issuer, registered_public=registered_public, what="authorization server"
    )
    as_url, as_raw = await fetch_first_metadata(
        client, authorization_server_metadata_urls(issuer), what="authorization server metadata"
    )
    try:
        metadata = OAuthMetadata.model_validate(as_raw)
    except ValueError as exc:
        raise DiscoveryError(
            f"the authorization server metadata at {as_url} did not parse"
        ) from exc

    # RFC 8414's mix-up defense: the metadata must declare the issuer it was
    # resolved from, or a look-alike server could answer for another issuer.
    if str(metadata.issuer).rstrip("/") != issuer:
        raise DiscoveryError(
            f"the authorization server metadata declares issuer "
            f"{str(metadata.issuer)!r}, not the discovered {issuer!r}"
        )

    # Hostile-input validation: the resource named its own authorization
    # server, so the endpoints that came back are not trusted configuration.
    try:
        validate_endpoint(str(metadata.authorization_endpoint))
        validate_endpoint(str(metadata.token_endpoint))
        if metadata.revocation_endpoint is not None:
            validate_endpoint(str(metadata.revocation_endpoint))
    except ValueError as exc:
        raise DiscoveryError(f"a discovered endpoint failed validation: {exc}") from exc
    # ...and the same public-host anchor the issuer got. validate_endpoint
    # alone permits ANY https host plus http on loopback, so without this a
    # metadata document could name https://10.0.0.5/collect as the token
    # endpoint and the exchange would POST the client secret, the code and the
    # verifier straight at an internal service. The token and revocation
    # endpoints are the real primitive (server-side POSTs carrying
    # credentials); the authorization endpoint is a BROWSER redirect target,
    # guarded on the same rule because steering the admin's browser onto an
    # internal origin is worth refusing too.
    await require_fetchable(
        str(metadata.authorization_endpoint),
        registered_public=registered_public,
        what="authorization endpoint",
    )
    await require_fetchable(
        str(metadata.token_endpoint),
        registered_public=registered_public,
        what="token endpoint",
    )
    if metadata.revocation_endpoint is not None:
        await require_fetchable(
            str(metadata.revocation_endpoint),
            registered_public=registered_public,
            what="revocation endpoint",
        )
    return metadata, declared
