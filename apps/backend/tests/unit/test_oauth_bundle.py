"""The OAuth token bundle codec: round-trip, and the guards its storage rests on."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.services.oauth import (
    OAuthTokenBundle,
    oauth_client_secret_name,
    oauth_token_secret_name,
)
from app.services.tools.mcp_sources import credential_secret_name

EXPIRES = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)


def _bundle(**overrides) -> OAuthTokenBundle:
    fields = {
        "access_token": "gho_access-secret-value",
        "refresh_token": "ghr_refresh-secret-value",
        "expires_at": EXPIRES,
        "granted_scope": "repo read:org",
        "token_endpoint": "https://github.com/login/oauth/access_token",
        "revocation_endpoint": "https://github.com/settings/connections/revoke",
        "client_id": "Iv1.example",
        "issuer": "https://github.com",
        **overrides,
    }
    return OAuthTokenBundle(**fields)


def test_seal_unseal_round_trips() -> None:
    bundle = _bundle()
    assert OAuthTokenBundle.unseal(bundle.seal()) == bundle


def test_the_sealed_tail_is_the_issuer_never_token_material() -> None:
    # SecretStore exposes the LAST 4 CHARACTERS of whatever it seals as the
    # readable ``last4`` hint, so the serialized tail must be non-secret.
    # ``issuer`` is declared last for exactly this reason: move any token field
    # after it and this fails - that reorder is a leak, not a style choice.
    bundle = _bundle()
    sealed = bundle.seal()
    assert sealed.endswith('"issuer":"https://github.com"}')
    tail = sealed[-4:]
    assert tail not in bundle.access_token
    assert bundle.refresh_token is not None and tail not in bundle.refresh_token


def test_the_resource_binding_round_trips_and_defaults_off() -> None:
    # The RFC 8707 binding survives seal/unseal; a sealed grant from BEFORE
    # the field existed (its JSON has no ``resource`` key) still unseals, so
    # the migration cost of the new field is zero.
    bound = _bundle(resource="https://mcp.example/v2")
    assert OAuthTokenBundle.unseal(bound.seal()).resource == "https://mcp.example/v2"
    legacy = _bundle().seal().replace('"resource":null,', "")
    assert '"resource"' not in legacy
    assert OAuthTokenBundle.unseal(legacy).resource is None


def test_a_naive_expiry_is_rejected() -> None:
    # A naive datetime is how a relative ``expires_in`` sneaks in half-converted;
    # persisted across a restart it is a silently-wrong expiry.
    with pytest.raises(ValidationError, match="timezone-aware"):
        _bundle(expires_at=datetime(2026, 7, 21, 12, 0, 0))  # noqa: DTZ001


def test_a_non_utc_offset_normalizes_to_the_same_instant() -> None:
    offset = datetime(2026, 7, 21, 14, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    bundle = _bundle(expires_at=offset)
    assert bundle.expires_at == EXPIRES
    assert bundle.expires_at.tzinfo == UTC


def test_a_plain_http_token_endpoint_is_rejected_off_loopback() -> None:
    # A token endpoint reached over plain http hands both tokens to the
    # network. Loopback is the one legitimate http case (a local dev AS).
    with pytest.raises(ValidationError, match="https"):
        _bundle(token_endpoint="http://auth.example.com/token")
    assert (
        _bundle(token_endpoint="http://127.0.0.1:9000/token").token_endpoint
        == "http://127.0.0.1:9000/token"
    )
    assert (
        _bundle(token_endpoint="http://localhost:9000/token").token_endpoint
        == "http://localhost:9000/token"
    )


def test_the_revocation_endpoint_gets_the_same_scheme_guard() -> None:
    with pytest.raises(ValidationError, match="https"):
        _bundle(revocation_endpoint="http://auth.example.com/revoke")
    assert _bundle(revocation_endpoint=None).revocation_endpoint is None


def test_a_hostless_endpoint_is_rejected() -> None:
    # Delete the netloc guard and "https:///token" sails through as an https
    # URL with nowhere to send the tokens.
    with pytest.raises(ValidationError, match="host"):
        _bundle(token_endpoint="https:///token")


def test_ipv6_loopback_is_accepted_over_http() -> None:
    # urlsplit strips the brackets, so the set must hold "::1" - this pins the
    # whole IPv6 loopback branch.
    assert (
        _bundle(token_endpoint="http://[::1]:9000/token").token_endpoint
        == "http://[::1]:9000/token"
    )


def test_an_endpoint_with_embedded_credentials_is_rejected() -> None:
    # Same rule as the integrations base_url: a credential inside a URL leaks
    # wherever the URL is logged or displayed.
    with pytest.raises(ValidationError, match="embed"):
        _bundle(token_endpoint="https://user:pass@auth.example.com/token")


def test_empty_required_fields_are_rejected() -> None:
    # An empty scope means the grant's breadth was never recorded; an empty
    # access token is not a grant at all.
    with pytest.raises(ValidationError):
        _bundle(granted_scope="")
    with pytest.raises(ValidationError):
        _bundle(access_token="")
    with pytest.raises(ValidationError):
        _bundle(client_id="")
    # issuer emptiness is mildly load-bearing: it is the field the sealed
    # tail's non-secret guarantee stands on.
    with pytest.raises(ValidationError):
        _bundle(issuer="")


def test_unseal_rejects_a_tampered_value() -> None:
    sealed = _bundle().seal().replace('"granted_scope":"repo read:org"', '"granted_scope":""')
    with pytest.raises(ValidationError):
        OAuthTokenBundle.unseal(sealed)


def test_secret_names_derive_from_the_stable_identity_key() -> None:
    # The bundle inherits the pasted credential's binding: keyed on the
    # server's stable id + url digest, so a recreated or re-pointed server
    # starts fresh instead of inheriting a grant that consented to a different
    # endpoint. Token and client registration are SEPARATE names, so revoking
    # a grant never destroys the registration needed to grant again.
    server_id, url = uuid4(), "http://crm/mcp"
    base = credential_secret_name(server_id, url)
    token = oauth_token_secret_name(server_id, url)
    client = oauth_client_secret_name(server_id, url)
    assert token == f"{base}:oauth-token"
    assert client == f"{base}:oauth-client"
    assert token != client
    assert oauth_token_secret_name(server_id, "http://other/mcp") != token
