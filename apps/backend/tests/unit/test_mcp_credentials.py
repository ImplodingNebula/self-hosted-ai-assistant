"""PerUserCredentialResolver: the acting user's bearer, resolved per tool call.

Drives the resolver against a real :class:`SecretStore` over the test sqlite
session, so the scope isolation it relies on is the genuine store's, not a fake.
The identity always comes from the ``ToolContext``, never an argument - so the
central assertion is that one user's reveal can never reach another's secret.
"""

from __future__ import annotations

import base64
from uuid import uuid4

import pytest

from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.providers.secrets import SecretScope, build_cipher
from app.providers.tools.base import ToolContext
from app.services.secret_store import SecretStore
from app.services.tools.mcp_credentials import (
    MissingUserCredentialError,
    PerUserCredentialResolver,
)
from app.services.tools.mcp_sources import credential_secret_name

pytestmark = pytest.mark.anyio

CIPHER = build_cipher(base64.b64encode(b"m" * 32).decode())
# Two users of ONE tenant, so per-user isolation is a real boundary, not a
# tenant one - exactly the case OIDC will make live.
ALICE = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
BOB = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")

NAME = "mcp-chassis"


def _context(tenant: TenantContext) -> ToolContext:
    """The tool-call context a turn carries for this principal (ids as strings)."""
    return ToolContext(tenant_id=str(tenant.tenant_id), user_id=str(tenant.user_id))


async def _store_user_secret(db_sessionmaker, tenant: TenantContext, value: str) -> None:
    async with db_sessionmaker() as session:
        await SecretStore(session, tenant, CIPHER).store(NAME, value, scope=SecretScope.USER)
        await session.commit()


def _resolver(db_sessionmaker, *, cipher=CIPHER) -> PerUserCredentialResolver:
    return PerUserCredentialResolver(sessionmaker=db_sessionmaker, cipher=cipher, secret_name=NAME)


def _required_resolver(db_sessionmaker, *, cipher=CIPHER) -> PerUserCredentialResolver:
    """A fail-closed resolver, as a ``credential_scope=user`` server wires."""
    return PerUserCredentialResolver(
        sessionmaker=db_sessionmaker, cipher=cipher, secret_name=NAME, required=True
    )


async def test_the_resolver_returns_the_acting_users_bearer(db_sessionmaker) -> None:
    await _store_user_secret(db_sessionmaker, ALICE, "alice-token")
    headers = await _resolver(db_sessionmaker)(_context(ALICE))
    assert headers == {"Authorization": "Bearer alice-token"}


async def test_a_user_without_a_stored_secret_falls_back(db_sessionmaker) -> None:
    # No secret for BOB: empty headers, so the CALLER uses the static workspace
    # header rather than the call failing.
    headers = await _resolver(db_sessionmaker)(_context(BOB))
    assert headers == {}


async def test_one_users_reveal_cannot_reach_anothers_secret(db_sessionmaker) -> None:
    # The whole point of per-user credentials: Alice stores, Bob asks, and Bob
    # gets NOTHING of Alice's - the resolver reveals only the caller's own scope.
    await _store_user_secret(db_sessionmaker, ALICE, "alice-token")
    resolver = _resolver(db_sessionmaker)
    assert await resolver(_context(BOB)) == {}
    assert await resolver(_context(ALICE)) == {"Authorization": "Bearer alice-token"}


async def test_without_a_master_key_the_resolver_falls_back(db_sessionmaker) -> None:
    # Per-user mode on but SECRET_MASTER_KEY unset (cipher=None): the reveal
    # cannot run, so the resolver degrades to the static header instead of
    # blowing up the turn.
    await _store_user_secret(db_sessionmaker, ALICE, "alice-token")
    headers = await _resolver(db_sessionmaker, cipher=None)(_context(ALICE))
    assert headers == {}


async def test_required_resolver_still_returns_the_bearer_when_connected(db_sessionmaker) -> None:
    # Fail-closed does not mean "no credential": a user who HAS connected still
    # gets their bearer; fail-closed only changes the missing-credential branch.
    await _store_user_secret(db_sessionmaker, ALICE, "alice-token")
    headers = await _required_resolver(db_sessionmaker)(_context(ALICE))
    assert headers == {"Authorization": "Bearer alice-token"}


async def test_required_resolver_fails_closed_when_no_secret(db_sessionmaker) -> None:
    # A user-scoped server whose user has connected nothing must FAIL CLOSED -
    # raise, so the call cannot silently fall back to the shared workspace header.
    with pytest.raises(MissingUserCredentialError):
        await _required_resolver(db_sessionmaker)(_context(BOB))


async def test_required_resolver_fails_closed_without_a_master_key(db_sessionmaker) -> None:
    # No cipher (SECRET_MASTER_KEY unset): a user-scoped server fails closed
    # rather than degrading to the shared header, even for a user who stored one.
    await _store_user_secret(db_sessionmaker, ALICE, "alice-token")
    with pytest.raises(MissingUserCredentialError):
        await _required_resolver(db_sessionmaker, cipher=None)(_context(ALICE))


async def test_required_resolver_one_users_reveal_cannot_reach_anothers(db_sessionmaker) -> None:
    # The isolation guarantee holds under fail-closed too: Alice's connected
    # secret raises for Bob (nothing of hers reaches him) and returns for Alice.
    await _store_user_secret(db_sessionmaker, ALICE, "alice-token")
    resolver = _required_resolver(db_sessionmaker)
    with pytest.raises(MissingUserCredentialError):
        await resolver(_context(BOB))
    assert await resolver(_context(ALICE)) == {"Authorization": "Bearer alice-token"}


async def test_a_credential_is_not_revealed_under_a_different_server_key(db_sessionmaker) -> None:
    # The delete-recreate / re-point leak, closed: a credential stored under one
    # server's key (id + url) is NOT revealed for a differently-keyed server, so a
    # recreated same-name server (new id) cannot inherit a member's private token.
    name_a = credential_secret_name(uuid4(), "http://a/mcp")
    name_b = credential_secret_name(uuid4(), "http://b/mcp")
    async with db_sessionmaker() as session:
        await SecretStore(session, ALICE, CIPHER).store(
            name_a, "alice-token", scope=SecretScope.USER
        )
        await session.commit()

    resolver_b = PerUserCredentialResolver(
        sessionmaker=db_sessionmaker, cipher=CIPHER, secret_name=name_b, required=True
    )
    with pytest.raises(MissingUserCredentialError):
        await resolver_b(_context(ALICE))

    # Sanity: server A's own key still reveals it, so this is scope-of-key, not a
    # blanket failure.
    resolver_a = PerUserCredentialResolver(
        sessionmaker=db_sessionmaker, cipher=CIPHER, secret_name=name_a, required=True
    )
    assert await resolver_a(_context(ALICE)) == {"Authorization": "Bearer alice-token"}
