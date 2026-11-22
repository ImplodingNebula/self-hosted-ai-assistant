"""The base_url rule on the Integrations write surface: no cleartext to a public host.

A registered server's credential rides in an ``Authorization`` header on every
listing and every tool call, so ``http://`` to a routable host publishes it -
and an admin pasting a vendor's http URL from its docs gets no warning anywhere
else. These tests pin both halves of the rule: the public host is refused, and
every legitimate cleartext target of this deployment (the compose service name
the seeded chassis lives at, loopback, a private address) keeps working.

Driven through the real schemas rather than the validator function, because the
schemas are what the API actually validates with, and all three of them share
this rule.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.integrations import (
    ConnectionTestRequest,
    IntegrationServerCreate,
    IntegrationServerUpdate,
)

# What this deployment legitimately reaches in the clear: its own network.
CLEARTEXT_OK = [
    "http://mcp-chassis:8765/mcp",  # the seeded chassis, a compose service name
    "http://localhost:8765/mcp",
    "http://127.0.0.1:9/mcp",
    "http://[::1]:8765/mcp",
    "http://10.0.0.5/mcp",  # RFC1918
    "http://192.168.1.20:3000/mcp",
    "http://[::ffff:127.0.0.1]/mcp",  # an IPv4-mapped loopback is still loopback
    "http://mcp.internal/mcp",
    "http://sidecar.local:8000/mcp",
]

# What would put a bearer token on the public internet in the clear.
CLEARTEXT_REJECTED = [
    "http://mcp.vendor.example/mcp",  # the finding's exact case: pasted from docs
    "http://8.8.8.8/mcp",
    "http://[2001:4860:4860::8888]/mcp",
    "http://sub.domain.co.uk:8080/mcp",
]


@pytest.mark.parametrize("url", CLEARTEXT_OK)
def test_cleartext_is_allowed_to_a_host_that_is_not_publicly_routable(url: str) -> None:
    assert IntegrationServerCreate(name="x", base_url=url).base_url == url


@pytest.mark.parametrize("url", CLEARTEXT_REJECTED)
def test_cleartext_to_a_public_host_is_refused(url: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        IntegrationServerCreate(name="x", base_url=url)

    # The message has to say WHY, or an admin just retries the same paste.
    assert "https" in str(excinfo.value)


@pytest.mark.parametrize("url", CLEARTEXT_REJECTED)
def test_https_to_the_same_public_host_is_fine(url: str) -> None:
    secure = url.replace("http://", "https://", 1)
    assert IntegrationServerCreate(name="x", base_url=secure).base_url == secure


def test_the_rule_applies_to_patching_and_to_test_connection_too() -> None:
    # Re-pointing an existing row, or probing before saving, reaches the same
    # upstream with the same credential - so the same rule has to hold there.
    with pytest.raises(ValidationError):
        IntegrationServerUpdate(base_url="http://mcp.vendor.example/mcp")
    with pytest.raises(ValidationError):
        ConnectionTestRequest(base_url="http://mcp.vendor.example/mcp")

    assert IntegrationServerUpdate(base_url="http://mcp-chassis:8765/mcp") is not None
    assert ConnectionTestRequest(base_url="http://mcp-chassis:8765/mcp") is not None


def test_the_older_url_rules_still_apply() -> None:
    # The scheme, host and userinfo guards this rule was added alongside.
    for bad in ("ftp://nope/mcp", "http://", "https://user:token@host/mcp"):
        with pytest.raises(ValidationError):
            IntegrationServerCreate(name="x", base_url=bad)
