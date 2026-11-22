"""DB spine tests: seeding, boot behavior, tenant isolation, cascades, limiter."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError, ProgrammingError

from app import main as main_module
from app.core.concurrency import StreamLimiter
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.db.seed import seed_defaults
from app.models import Message, Tenant, User, WorkspaceSettings
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository

TENANT_A = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


async def _count(session, model) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def test_seed_defaults_is_idempotent(db_sessionmaker) -> None:
    # The fixture already seeded once; a second run must not duplicate rows.
    await seed_defaults(db_sessionmaker)
    async with db_sessionmaker() as session:
        assert await _count(session, Tenant) == 1
        assert await _count(session, User) == 1
        assert await _count(session, WorkspaceSettings) == 1
        tenant = await session.get(Tenant, DEFAULT_TENANT_ID)
        user = await session.get(User, DEFAULT_USER_ID)
        assert tenant is not None and tenant.name == "default"
        assert user is not None and user.role == "admin"


async def test_boot_fails_loudly_when_schema_is_missing(monkeypatch) -> None:
    """A reachable DB without tables must abort the boot, not serve half-broken.

    ProgrammingError is what postgres raises for a missing relation (sqlite
    reports it as OperationalError, which stays on the tolerant path below);
    the lifespan translates it into a RuntimeError naming the remedy.
    """

    async def _schema_missing(_sessionmaker) -> None:
        raise ProgrammingError("SELECT ...", None, Exception("UndefinedTableError"))

    monkeypatch.setattr(main_module, "seed_defaults", _schema_missing)
    app = main_module.create_app()
    with pytest.raises(RuntimeError, match="make rebuild"):
        async with main_module._lifespan(app):
            pass


async def test_boot_tolerates_database_being_down(monkeypatch, caplog) -> None:
    """A connection-level seed failure only warns; the API must still come up."""

    async def _db_down(_sessionmaker) -> None:
        raise OperationalError("SELECT 1", None, Exception("connection refused"))

    monkeypatch.setattr(main_module, "seed_defaults", _db_down)
    app = main_module.create_app()
    with caplog.at_level(logging.WARNING):
        async with main_module._lifespan(app):
            pass
    assert any("seeding failed" in rec.message for rec in caplog.records)


async def test_repository_never_sees_foreign_tenant_rows(db_sessionmaker) -> None:
    tenant_b = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")
    async with db_sessionmaker() as session:
        repo_a = ConversationRepository(session, TENANT_A)
        repo_b = ConversationRepository(session, tenant_b)
        conv_a = await repo_a.create(title="mine")
        conv_b = await repo_b.create(title="theirs")
        await session.commit()

        # get: foreign rows are indistinguishable from missing ones.
        assert await repo_a.get(conv_b.id) is None
        assert await repo_b.get(conv_a.id) is None
        assert await repo_a.get_with_messages(conv_b.id) is None

        rows_a, total_a = await repo_a.list_summaries(limit=50, offset=0)
        assert total_a == 1
        assert [c.id for c in rows_a] == [conv_a.id]

        # add() must stamp the bound tenant even if the entity lies about it.
        from app.models import Conversation

        forged = Conversation(tenant_id=tenant_b.tenant_id, user_id=TENANT_A.user_id)
        repo_a.add(forged)
        await repo_a.flush()
        assert forged.tenant_id == TENANT_A.tenant_id
        await session.rollback()


async def test_delete_with_messages_removes_children(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        conv_repo = ConversationRepository(session, TENANT_A)
        msg_repo = MessageRepository(session, TENANT_A)
        conv = await conv_repo.create()
        await msg_repo.add(conv.id, role="user", content="hello")
        await msg_repo.add(conv.id, role="assistant", content="hi", model="fake")
        await session.commit()

        assert len(await msg_repo.list_for_conversation(conv.id)) == 2
        assert await conv_repo.delete_with_messages(conv.id) is True
        await session.commit()

        assert await conv_repo.get(conv.id) is None
        assert await _count(session, Message) == 0
        # Deleting a missing conversation reports False instead of raising.
        assert await conv_repo.delete_with_messages(conv.id) is False


async def test_message_ordering_and_recent_window(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        conv_repo = ConversationRepository(session, TENANT_A)
        msg_repo = MessageRepository(session, TENANT_A)
        conv = await conv_repo.create()
        base = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
        for i in range(5):
            await msg_repo.add(
                conv.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"m{i}",
                created_at=base.replace(minute=i),
            )
        await session.commit()

        all_msgs = await msg_repo.list_for_conversation(conv.id)
        assert [m.content for m in all_msgs] == ["m0", "m1", "m2", "m3", "m4"]
        recent = await msg_repo.list_recent(conv.id, limit=2)
        assert [m.content for m in recent] == ["m3", "m4"]


async def test_stream_limiter_cap_and_release() -> None:
    limiter = StreamLimiter(max_per_user=2)
    tenant = uuid4()
    user = uuid4()
    other_tenant = uuid4()

    assert await limiter.acquire(tenant, user) is True
    assert await limiter.acquire(tenant, user) is True
    assert await limiter.acquire(tenant, user) is False
    # The same user id in another tenant is another principal entirely, so the
    # key cannot be the user alone.
    assert await limiter.acquire(other_tenant, user) is True

    await limiter.release(tenant, user)
    assert await limiter.acquire(tenant, user) is True


async def test_stream_limiter_budget_is_per_user_not_per_workspace() -> None:
    # MUTATION GUARD: key the limiter on the tenant alone and this fails at the
    # second colleague - which is the starvation the per-user budget exists to
    # stop (one member's long tool turns locking everyone else out).
    limiter = StreamLimiter(max_per_user=1)
    tenant = uuid4()
    alice, bob = uuid4(), uuid4()

    assert await limiter.acquire(tenant, alice) is True
    assert await limiter.acquire(tenant, bob) is True
    # Each still holds their OWN cap: a per-user budget is not an unlimited one.
    assert await limiter.acquire(tenant, alice) is False
    assert await limiter.acquire(tenant, bob) is False

    await limiter.release(tenant, alice)
    assert await limiter.acquire(tenant, alice) is True
    # Alice's release freed Alice's slot and nobody else's.
    assert await limiter.acquire(tenant, bob) is False


async def test_stream_limiter_tolerates_over_release() -> None:
    limiter = StreamLimiter(max_per_user=1)
    tenant = uuid4()
    user = uuid4()

    await limiter.release(tenant, user)  # never acquired: must not go negative
    assert await limiter.acquire(tenant, user) is True
    await limiter.release(tenant, user)
    await limiter.release(tenant, user)  # double release
    assert await limiter.acquire(tenant, user) is True
    assert await limiter.acquire(tenant, user) is False  # cap still enforced


async def test_stream_limiter_any_active_tracks_stream_lifecycle() -> None:
    # The GPU-priority signal ingestion consults between embed batches. It asks
    # about the whole process, not one principal - the GPU is shared.
    limiter = StreamLimiter(max_per_user=2)
    tenant = uuid4()
    alice, bob = uuid4(), uuid4()

    assert limiter.any_active() is False
    assert await limiter.acquire(tenant, alice) is True
    assert limiter.any_active() is True
    assert await limiter.acquire(tenant, bob) is True
    await limiter.release(tenant, alice)
    # Bob is still generating, so the GPU is still busy.
    assert limiter.any_active() is True
    await limiter.release(tenant, bob)
    assert limiter.any_active() is False
