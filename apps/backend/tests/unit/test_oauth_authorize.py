"""PKCE, state, resource canonicalization, and the consent-URL assembly."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from app.services.oauth.authorize import (
    OAuthEndpoints,
    build_authorization_url,
    canonical_resource,
    generate_pkce_verifier,
    generate_state,
    pkce_challenge,
)

_VERIFIER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


def _endpoints(**overrides) -> OAuthEndpoints:
    fields = {
        "issuer": "https://as.example",
        "authorization_endpoint": "https://as.example/authorize",
        "token_endpoint": "https://as.example/token",
        "send_resource": True,
        **overrides,
    }
    return OAuthEndpoints(**fields)


# --- PKCE and state --------------------------------------------------------


def test_verifier_is_128_chars_of_the_unreserved_alphabet_and_fresh() -> None:
    a, b = generate_pkce_verifier(), generate_pkce_verifier()
    assert len(a) == 128
    assert set(a) <= _VERIFIER_CHARS
    assert a != b  # a repeated verifier would let one interception replay forever


def test_challenge_matches_the_rfc_7636_test_vector() -> None:
    # The S256 vector from RFC 7636 appendix B. This pins the exact transform
    # (sha256 -> base64url, unpadded): any drift - plain method, wrong
    # encoding, stray padding - fails against the published constant.
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert pkce_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_state_is_fresh_and_long() -> None:
    a, b = generate_state(), generate_state()
    assert a != b
    assert len(a) >= 43  # 32 random bytes, URL-safe encoded


# --- canonical resource ----------------------------------------------------


def test_canonical_resource_normalizes_case_port_and_fragment() -> None:
    # Scheme/host lowercase, default port dropped, path kept exactly,
    # query/fragment dropped - the identifier names the server, not a request.
    assert canonical_resource("HTTPS://MCP.Example:443/Tools/mcp") == (
        "https://mcp.example/Tools/mcp"
    )
    assert canonical_resource("http://host:8811/mcp?x=1#frag") == "http://host:8811/mcp"
    assert canonical_resource("https://host/mcp/") == "https://host/mcp/"


# --- the consent URL -------------------------------------------------------


def test_authorization_url_carries_the_exact_parameter_set() -> None:
    url = build_authorization_url(
        _endpoints(),
        client_id="cid",
        redirect_uri="http://localhost:8000/api/v1/integrations/oauth/callback",
        scopes=["repo", "read:org"],
        state="st4te",
        code_challenge="ch4llenge",
        resource="https://mcp.example/mcp",
    )
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "https://as.example/authorize"
    params = parse_qs(parts.query, keep_blank_values=True)
    assert params == {
        "response_type": ["code"],
        "client_id": ["cid"],
        "redirect_uri": ["http://localhost:8000/api/v1/integrations/oauth/callback"],
        "scope": ["repo read:org"],
        "state": ["st4te"],
        "code_challenge": ["ch4llenge"],
        "code_challenge_method": ["S256"],
        "resource": ["https://mcp.example/mcp"],
    }


def test_resource_is_omitted_on_the_operator_supplied_path() -> None:
    # A generic provider (GitHub, Google) does not take RFC 8707's resource
    # parameter; the bypass endpoints record that and the URL must not carry
    # it even when a caller passes one.
    url = build_authorization_url(
        _endpoints(send_resource=False),
        client_id="cid",
        redirect_uri="http://localhost/cb",
        scopes=["repo"],
        state="s",
        code_challenge="c",
        resource="https://mcp.example/mcp",
    )
    assert "resource=" not in url


def test_empty_scopes_are_refused_never_defaulted() -> None:
    # The guard behind "scopes are operator-supplied, never auto-expanded":
    # remove it and an unscoped request silently takes the provider's default
    # grant, unrecorded.
    with pytest.raises(ValueError, match="operator-supplied"):
        build_authorization_url(
            _endpoints(),
            client_id="cid",
            redirect_uri="http://localhost/cb",
            scopes=[],
            state="s",
            code_challenge="c",
        )


def test_an_authorize_endpoint_with_a_query_is_extended_not_clobbered() -> None:
    url = build_authorization_url(
        _endpoints(
            authorization_endpoint="https://as.example/authorize?audience=mcp",
            send_resource=False,
        ),
        client_id="cid",
        redirect_uri="http://localhost/cb",
        scopes=["s1"],
        state="s",
        code_challenge="c",
    )
    params = parse_qs(urlsplit(url).query)
    assert params["audience"] == ["mcp"]
    assert params["client_id"] == ["cid"]


def test_the_native_path_requires_the_resource() -> None:
    # send_resource without a resource is an UNBOUND token request - the
    # replay protection the parameter exists for, silently dropped. Refused.
    with pytest.raises(ValueError, match="canonical resource"):
        build_authorization_url(
            _endpoints(),
            client_id="cid",
            redirect_uri="http://localhost/cb",
            scopes=["s1"],
            state="s",
            code_challenge="c",
        )


def test_canonical_resource_rebrackets_ipv6_literals() -> None:
    # urlsplit strips the brackets; a rebuilt netloc must restore them or the
    # port separator becomes ambiguous ("::1:8811" is not an address).
    assert canonical_resource("http://[::1]:8811/mcp") == "http://[::1]:8811/mcp"
    assert canonical_resource("https://[2001:DB8::1]:8443/mcp") == "https://[2001:db8::1]:8443/mcp"
    assert canonical_resource("https://[::1]:443/mcp") == "https://[::1]/mcp"


def test_endpoints_reject_http_off_loopback() -> None:
    # The operator-supplied path skips discovery, so this model's validators
    # are the only guard those endpoints pass through.
    with pytest.raises(ValidationError, match="https"):
        _endpoints(token_endpoint="http://as.example/token")
    assert (
        _endpoints(authorization_endpoint="http://127.0.0.1:9000/authorize").authorization_endpoint
    ).startswith("http://127.0.0.1")


def test_extra_params_ride_the_consent_url() -> None:
    # Google's non-standard refresh-token ask (access_type=offline) must reach
    # the consent URL exactly as the operator supplied it.
    url = build_authorization_url(
        _endpoints(send_resource=False),
        client_id="cid",
        redirect_uri="https://app.example/cb",
        scopes=["repo"],
        state="st",
        code_challenge="ch",
        extra_params={"access_type": "offline", "prompt": "consent"},
    )
    query = parse_qs(urlsplit(url).query)
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]


def test_extra_params_may_not_shadow_the_protocols_own() -> None:
    # An extra named redirect_uri (or state, or the PKCE pair) would rewrite
    # the flow's guarantees from inside its own request: refuse loudly.
    # Case and whitespace variants count too: the key is sent verbatim, but a
    # sloppy provider might honor "Redirect_Uri" - refuse the attempt itself.
    for key in ("redirect_uri", "Redirect_Uri", "state "):
        with pytest.raises(ValueError, match="shadow"):
            build_authorization_url(
                _endpoints(send_resource=False),
                client_id="cid",
                redirect_uri="https://app.example/cb",
                scopes=["repo"],
                state="st",
                code_challenge="ch",
                extra_params={key: "https://evil.example/cb"},
            )
