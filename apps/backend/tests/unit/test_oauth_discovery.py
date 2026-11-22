"""The discovery walk: candidate orders, the metadata fetch, and hostile-input guards."""

from __future__ import annotations

import json

import httpx
import pytest

from app.services.oauth.discovery import (
    DiscoveryError,
    authorization_server_metadata_urls,
    discover_authorization_server,
    parse_resource_metadata_url,
    protected_resource_metadata_urls,
)

pytestmark = pytest.mark.anyio


def _client(routes: dict[str, dict | str]) -> httpx.AsyncClient:
    """An httpx client over a MockTransport serving canned JSON by exact URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in routes:
            body = routes[url]
            if isinstance(body, str):
                return httpx.Response(200, text=body)
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


AS_METADATA = {
    "issuer": "https://as.example",
    "authorization_endpoint": "https://as.example/authorize",
    "token_endpoint": "https://as.example/token",
}


# --- pure helpers ----------------------------------------------------------


def test_resource_metadata_url_parses_from_a_challenge() -> None:
    header = 'Bearer resource_metadata="https://mcp.example/.well-known/oauth-protected-resource"'
    assert (
        parse_resource_metadata_url(header)
        == "https://mcp.example/.well-known/oauth-protected-resource"
    )
    # Case-insensitive parameter name; absent parameter is None, not an error.
    assert parse_resource_metadata_url('Bearer RESOURCE_METADATA="https://x/prm"') is not None
    assert parse_resource_metadata_url('Bearer realm="mcp"') is None


def test_prm_candidates_are_path_aware_then_root() -> None:
    # The path-insertion form first (one host can serve several resources at
    # distinct paths - the chassis pattern), then the host root.
    assert protected_resource_metadata_urls("https://host/example/mcp") == [
        "https://host/.well-known/oauth-protected-resource/example/mcp",
        "https://host/.well-known/oauth-protected-resource",
    ]
    assert protected_resource_metadata_urls("https://host/") == [
        "https://host/.well-known/oauth-protected-resource",
    ]


def test_as_candidates_follow_the_spec_fallback_order() -> None:
    # Path-bearing issuer: RFC 8414 path-insertion forms before OIDC's
    # path-suffix form. This ORDER is the contract, not a preference.
    assert authorization_server_metadata_urls("https://as.example/tenant") == [
        "https://as.example/.well-known/oauth-authorization-server/tenant",
        "https://as.example/.well-known/openid-configuration/tenant",
        "https://as.example/tenant/.well-known/openid-configuration",
    ]
    assert authorization_server_metadata_urls("https://as.example") == [
        "https://as.example/.well-known/oauth-authorization-server",
        "https://as.example/.well-known/openid-configuration",
    ]


# --- the walk --------------------------------------------------------------


async def test_discovery_walks_prm_to_as_metadata() -> None:
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource/tools/mcp": {
                "resource": "https://mcp.example/tools/mcp",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": AS_METADATA,
        }
    ) as client:
        metadata, _ = await discover_authorization_server(
            "https://mcp.example/tools/mcp", client=client
        )
    assert str(metadata.token_endpoint) == "https://as.example/token"


async def test_discovery_prefers_the_challenge_hint() -> None:
    # A 401's resource_metadata URL is tried BEFORE the well-known candidates;
    # here it is the only route that exists, so passing means it was used.
    async with _client(
        {
            "https://mcp.example/custom/prm": {
                "resource": "https://mcp.example/tools/mcp",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": AS_METADATA,
        }
    ) as client:
        metadata, _ = await discover_authorization_server(
            "https://mcp.example/tools/mcp",
            client=client,
            www_authenticate='Bearer resource_metadata="https://mcp.example/custom/prm"',
        )
    assert str(metadata.issuer).rstrip("/") == "https://as.example"


async def test_discovery_falls_back_to_the_root_prm_and_oidc_form() -> None:
    # Path candidate 404s -> root PRM; RFC 8414 form 404s -> OIDC form.
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource": {
                "resource": "https://mcp.example/tools/mcp",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/openid-configuration": AS_METADATA,
        }
    ) as client:
        metadata, _ = await discover_authorization_server(
            "https://mcp.example/tools/mcp", client=client
        )
    assert str(metadata.authorization_endpoint) == "https://as.example/authorize"


async def test_prm_without_authorization_servers_is_a_discovery_error() -> None:
    # The SDK's ProtectedResourceMetadata requires at least one authorization
    # server, so a document naming none fails at PARSE - the walk surfaces it
    # as a DiscoveryError rather than indexing into an empty list.
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource": {
                "resource": "https://mcp.example/mcp",
                "authorization_servers": [],
            },
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="did not parse"):
            await discover_authorization_server("https://mcp.example/mcp", client=client)


async def test_nothing_found_anywhere_is_a_discovery_error() -> None:
    async with _client({}) as client:
        with pytest.raises(DiscoveryError, match="no protected resource metadata"):
            await discover_authorization_server("https://mcp.example/mcp", client=client)


async def test_a_discovered_http_endpoint_fails_validation() -> None:
    # The resource names its own authorization server, so what comes back is
    # hostile input: an AS whose token endpoint is plain http off loopback
    # must fail the walk, never reach the consent flow. Delete the
    # validate_endpoint calls in discovery and this passes a token-stealing AS.
    bad = dict(AS_METADATA, token_endpoint="http://as.example/token")
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource": {
                "resource": "https://mcp.example/mcp",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": bad,
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="failed validation"):
            await discover_authorization_server("https://mcp.example/mcp", client=client)


async def test_the_hint_wins_over_the_well_known_candidates() -> None:
    # Both routes resolve, to DIFFERENT authorization servers - the 401 hint
    # must be tried FIRST, so hint-AS wins. Reorder the candidate list (hint
    # appended instead of prepended) and this fails with well-known-AS.
    hint_as = dict(
        AS_METADATA,
        issuer="https://hint-as.example",
        authorization_endpoint="https://hint-as.example/authorize",
        token_endpoint="https://hint-as.example/token",
    )
    async with _client(
        {
            "https://mcp.example/custom/prm": {
                "resource": "https://mcp.example/mcp",
                "authorization_servers": ["https://hint-as.example"],
            },
            "https://mcp.example/.well-known/oauth-protected-resource": {
                "resource": "https://mcp.example/mcp",
                "authorization_servers": ["https://as.example"],
            },
            "https://hint-as.example/.well-known/oauth-authorization-server": hint_as,
            "https://as.example/.well-known/oauth-authorization-server": AS_METADATA,
        }
    ) as client:
        metadata, _ = await discover_authorization_server(
            "https://mcp.example/mcp",
            client=client,
            www_authenticate='Bearer resource_metadata="https://mcp.example/custom/prm"',
        )
    assert str(metadata.token_endpoint) == "https://hint-as.example/token"


async def test_a_cross_origin_hint_is_never_fetched() -> None:
    # The hint names another host; the ONLY resolving route lives there. A
    # resource's metadata lives on the resource, so the walk must ignore the
    # hint entirely and fail on the well-known candidates - success here would
    # mean the remote header steered a fetch off-origin.
    async with _client(
        {
            "https://evil.example/prm": {
                "resource": "https://mcp.example/mcp",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": AS_METADATA,
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="no protected resource metadata"):
            await discover_authorization_server(
                "https://mcp.example/mcp",
                client=client,
                www_authenticate='Bearer resource_metadata="https://evil.example/prm"',
            )


async def test_a_prm_for_another_resource_is_rejected() -> None:
    # RFC 9728: the metadata must describe the server it was fetched FOR. A
    # document describing some other resource must not steer this grant.
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource": {
                "resource": "https://other.example/mcp",
                "authorization_servers": ["https://as.example"],
            },
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="not the registered server"):
            await discover_authorization_server("https://mcp.example/mcp", client=client)


async def test_a_prefix_resource_identifier_is_accepted_and_returned() -> None:
    # RFC 8707: a resource identifier may cover several endpoints under one
    # path (Asana declares .../v2 for its .../v2/mcp connection URL). The
    # DECLARED identifier - not a recomputation from the base URL - is what
    # the flow must bind tokens to, so the walk returns it.
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource/v2/mcp": {
                "resource": "https://mcp.example/v2",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": AS_METADATA,
        }
    ) as client:
        metadata, resource = await discover_authorization_server(
            "https://mcp.example/v2/mcp", client=client
        )
    assert resource == "https://mcp.example/v2"
    assert str(metadata.token_endpoint) == "https://as.example/token"


async def test_a_prefix_without_a_segment_boundary_is_rejected() -> None:
    # "/v2" is NOT a path prefix of "/v2beta/mcp" - accepting it would let a
    # document for one resource steer a look-alike sibling. The AS route is
    # served so a broken guard would complete the walk instead of erroring
    # differently - this test must fail LOUDLY if the boundary check goes.
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource/v2beta/mcp": {
                "resource": "https://mcp.example/v2",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": AS_METADATA,
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="not the registered server"):
            await discover_authorization_server("https://mcp.example/v2beta/mcp", client=client)


async def test_a_bare_origin_resource_claim_is_rejected() -> None:
    # The degenerate widest claim. On a shared host - the chassis pattern this
    # project documents, several MCP servers at distinct paths - accepting the
    # bare origin hands back a token whose audience covers every sibling path
    # the admin never consented to, which is exactly what the RFC 8707 binding
    # exists to prevent. The AS route is served, so a broken guard completes
    # the walk rather than failing for some unrelated reason.
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource/v2/mcp": {
                "resource": "https://mcp.example",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": AS_METADATA,
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="not the registered server"):
            await discover_authorization_server("https://mcp.example/v2/mcp", client=client)


async def test_a_distant_ancestor_resource_claim_is_rejected() -> None:
    # One segment up is the real deployment need (a version prefix over its
    # endpoint) and is allowed; two is the same origin-wide claim wearing a
    # path segment, so "/api" may not stand for "/api/tenant-b/mcp" - that
    # audience would cover every co-tenant under /api.
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource/api/tenant-b/mcp": {
                "resource": "https://mcp.example/api",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": AS_METADATA,
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="not the registered server"):
            await discover_authorization_server(
                "https://mcp.example/api/tenant-b/mcp", client=client
            )


async def test_an_oversized_metadata_document_is_not_buffered() -> None:
    # The metadata host is named by the remote server, so its response size is
    # the remote side's choice. An unbounded read is an out-of-memory kill any
    # registered server can trigger, in the process that also serves every live
    # chat stream. An oversized document is a DEAD CANDIDATE, so the walk falls
    # through to the next well-known location and ultimately reports no
    # metadata found - never a body it accumulated anyway.
    from app.services.oauth.fetch import MAX_DOCUMENT_BYTES

    flood = json.dumps({"resource": "https://mcp.example/mcp", "pad": "x" * MAX_DOCUMENT_BYTES})
    async with _client(
        {"https://mcp.example/.well-known/oauth-protected-resource/mcp": flood}
    ) as client:
        with pytest.raises(DiscoveryError, match="no protected resource metadata"):
            await discover_authorization_server("https://mcp.example/mcp", client=client)


async def test_an_issuer_mismatch_is_rejected() -> None:
    # RFC 8414's mix-up defense: the metadata must declare the issuer it was
    # resolved from, or a look-alike answers for another issuer.
    imposter = dict(AS_METADATA, issuer="https://other-as.example")
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource": {
                "resource": "https://mcp.example/mcp",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": imposter,
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="declares issuer"):
            await discover_authorization_server("https://mcp.example/mcp", client=client)


async def test_a_public_server_may_not_name_a_private_authorization_server(
    monkeypatch,
) -> None:
    # The SSRF anchor: the ADMIN registered a public server, so a PRM naming a
    # private-resolving authorization server (a metadata endpoint, an internal
    # admin panel) must refuse to fetch. Delete the _require_fetchable call
    # and this walks straight into the internal host.
    import app.services.oauth.discovery as discovery_mod

    async def fake_public(host: str) -> bool:
        return host in {"mcp.example", "as.example"}

    monkeypatch.setattr(discovery_mod, "host_is_public", fake_public)
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource": {
                "resource": "https://mcp.example/mcp",
                "authorization_servers": ["https://internal-panel"],
            },
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="public"):
            await discover_authorization_server("https://mcp.example/mcp", client=client)


async def test_a_redirecting_candidate_is_skipped_not_followed() -> None:
    # A 3xx must count as a dead candidate. If redirects were followed, this
    # walk would land on the evil PRM (describing another resource) and FAIL;
    # with follow_redirects pinned off it falls through to the root candidate
    # and succeeds - so a client whose default enables redirects still cannot
    # be steered.
    def handler(request):
        url = str(request.url)
        if url == "https://mcp.example/.well-known/oauth-protected-resource/tools/mcp":
            return httpx.Response(301, headers={"Location": "https://mcp.example/evil-prm"})
        if url == "https://mcp.example/evil-prm":
            return httpx.Response(
                200,
                json={
                    "resource": "https://other.example/mcp",
                    "authorization_servers": ["https://as.example"],
                },
            )
        if url == "https://mcp.example/.well-known/oauth-protected-resource":
            return httpx.Response(
                200,
                json={
                    "resource": "https://mcp.example/tools/mcp",
                    "authorization_servers": ["https://as.example"],
                },
            )
        if url == "https://as.example/.well-known/oauth-authorization-server":
            return httpx.Response(200, json=AS_METADATA)
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        metadata, _ = await discover_authorization_server(
            "https://mcp.example/tools/mcp", client=client
        )
    assert str(metadata.token_endpoint) == "https://as.example/token"


async def test_a_discovered_bad_revocation_endpoint_fails_the_walk() -> None:
    bad = dict(AS_METADATA, revocation_endpoint="http://as.example/revoke")
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource": {
                "resource": "https://mcp.example/mcp",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": bad,
        }
    ) as client:
        with pytest.raises(DiscoveryError, match="failed validation"):
            await discover_authorization_server("https://mcp.example/mcp", client=client)


async def test_unparseable_json_falls_through_to_the_next_candidate() -> None:
    # A candidate serving garbage moves the walk along instead of crashing it.
    async with _client(
        {
            "https://mcp.example/.well-known/oauth-protected-resource/tools/mcp": "not json{",
            "https://mcp.example/.well-known/oauth-protected-resource": {
                "resource": "https://mcp.example/tools/mcp",
                "authorization_servers": ["https://as.example"],
            },
            "https://as.example/.well-known/oauth-authorization-server": AS_METADATA,
        }
    ) as client:
        metadata, _ = await discover_authorization_server(
            "https://mcp.example/tools/mcp", client=client
        )
    assert json.loads(metadata.model_dump_json())["issuer"].rstrip("/") == "https://as.example"
