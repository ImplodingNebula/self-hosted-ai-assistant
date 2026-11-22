"""SecretStore: scope isolation, write-only reads, upsert, and the AAD tripwire."""

from __future__ import annotations

import base64
from uuid import uuid4

import pytest
from cryptography.exceptions import InvalidTag
from sqlalchemy import select

from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.models import Secret
from app.providers.secrets import (
    SecretNotFoundError,
    SecretScope,
    SecretStoreUnavailableError,
    build_cipher,
)
from app.services.secret_store import SecretStore

CIPHER = build_cipher(base64.b64encode(b"m" * 32).decode())
# Two users of ONE tenant, so scope isolation is a real boundary, not a tenant one.
ALICE = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
BOB = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")

WS = SecretScope.WORKSPACE
USER = SecretScope.USER


def _store(session, tenant, *, cipher=CIPHER) -> SecretStore:
    return SecretStore(session, tenant, cipher)


async def test_store_then_reveal_round_trips_and_get_never_exposes_plaintext(
    db_sessionmaker,
) -> None:
    async with db_sessionmaker() as session:
        store = _store(session, ALICE)
        meta = await store.store("github", "ghp_supersecret", scope=USER)
        await session.commit()

        # The write returns metadata only - last4, never the value.
        assert meta.name == "github"
        assert meta.scope is USER
        assert meta.last4 == "cret"
        assert "supersecret" not in meta.model_dump_json()

        # get() is metadata too; reveal() is the one call that returns the secret.
        got = await store.get("github", scope=USER)
        assert got is not None and got.last4 == "cret"
        assert await store.reveal("github", scope=USER) == "ghp_supersecret"


async def test_a_short_secret_hides_its_last4_so_the_hint_is_not_the_whole_value(
    db_sessionmaker,
) -> None:
    # last4 is "the only readable hint" - for a secret short enough that four
    # trailing characters would be most or all of it, the hint is suppressed so a
    # metadata read cannot spill the value it is meant to protect.
    async with db_sessionmaker() as session:
        store = _store(session, ALICE)
        short = await store.store("pin", "1234", scope=USER)
        assert short.last4 == ""
        # ...but the value itself still round-trips through reveal().
        assert await store.reveal("pin", scope=USER) == "1234"
        # A long secret still exposes its tail as a recognisable hint.
        long = await store.store("token", "abcdefghXYZ9", scope=USER)
        assert long.last4 == "XYZ9"


async def test_workspace_and_user_secrets_of_the_same_name_are_separate(
    db_sessionmaker,
) -> None:
    async with db_sessionmaker() as session:
        store = _store(session, ALICE)
        await store.store("token", "workspace-value", scope=WS)
        await store.store("token", "alice-value", scope=USER)
        await session.commit()

        assert await store.reveal("token", scope=WS) == "workspace-value"
        assert await store.reveal("token", scope=USER) == "alice-value"
        # Two distinct rows, one workspace (NULL owner) and one user-owned.
        rows = (await session.execute(select(Secret))).scalars().all()
        assert sorted((r.user_id is None) for r in rows) == [False, True]


async def test_a_user_secret_is_private_to_its_owner(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        await _store(session, ALICE).store("github", "alice-token", scope=USER)
        await _store(session, BOB).store("github", "bob-token", scope=USER)
        await session.commit()

        # Same name, same tenant, same scope - but each reveals only their own,
        # because the owner comes from the bound context, never an argument.
        assert await _store(session, ALICE).reveal("github", scope=USER) == "alice-token"
        assert await _store(session, BOB).reveal("github", scope=USER) == "bob-token"
        # Bob has no workspace secret of that name either - scopes do not bleed.
        assert await _store(session, BOB).get("github", scope=WS) is None


async def test_list_is_scoped_and_metadata_only(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        await _store(session, ALICE).store("a", "aaaa1111", scope=USER)
        await _store(session, ALICE).store("b", "bbbb2222", scope=USER)
        await _store(session, ALICE).store("shared", "wwww3333", scope=WS)
        await _store(session, BOB).store("c", "cccc4444", scope=USER)
        await session.commit()

        alice_user = await _store(session, ALICE).list(scope=USER)
        assert [m.name for m in alice_user] == ["a", "b"]  # not Bob's, not the workspace one
        workspace = await _store(session, ALICE).list(scope=WS)
        assert [m.name for m in workspace] == ["shared"]


async def test_store_upserts_on_reconnect(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        store = _store(session, ALICE)
        await store.store("github", "old-0001", scope=USER)
        await store.store("github", "new-0002", scope=USER)
        await session.commit()

        assert await store.reveal("github", scope=USER) == "new-0002"
        rows = (await session.execute(select(Secret))).scalars().all()
        assert len(rows) == 1  # replaced, not duplicated
        assert rows[0].last4 == "0002"


async def test_delete_removes_and_reports_absence(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        store = _store(session, ALICE)
        await store.store("github", "value123", scope=USER)
        await session.commit()

        assert await store.delete("github", scope=USER) is True
        await session.commit()
        assert await store.get("github", scope=USER) is None
        assert await store.delete("github", scope=USER) is False


async def test_reveal_of_a_missing_secret_raises(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        with pytest.raises(SecretNotFoundError):
            await _store(session, ALICE).reveal("nope", scope=USER)


async def test_without_a_master_key_sealing_and_revealing_are_unavailable(
    db_sessionmaker,
) -> None:
    async with db_sessionmaker() as session:
        store = _store(session, ALICE, cipher=None)
        with pytest.raises(SecretStoreUnavailableError):
            await store.store("github", "v", scope=USER)
        with pytest.raises(SecretStoreUnavailableError):
            await store.reveal("github", scope=USER)
        # Metadata paths do not need the key, so they stay usable.
        assert await store.get("github", scope=USER) is None
        assert await store.list(scope=USER) == []


async def test_a_swapped_ciphertext_fails_to_decrypt(db_sessionmaker) -> None:
    # The AAD tripwire end to end: swap two owners' ciphertext bytes in the DB,
    # and a reveal - which builds its AAD from the EXPECTED owner - raises rather
    # than returning the other user's token.
    async with db_sessionmaker() as session:
        await _store(session, ALICE).store("github", "alice-token", scope=USER)
        await _store(session, BOB).store("github", "bob-token", scope=USER)
        await session.commit()

        rows = {r.user_id: r for r in (await session.execute(select(Secret))).scalars().all()}
        alice_row, bob_row = rows[ALICE.user_id], rows[BOB.user_id]
        alice_row.ciphertext, bob_row.ciphertext = bob_row.ciphertext, alice_row.ciphertext
        session.add(alice_row)
        session.add(bob_row)
        await session.commit()

        with pytest.raises(InvalidTag):
            await _store(session, ALICE).reveal("github", scope=USER)
