"""CollectionRepository tests: naming, get-or-create, and the insert race."""

from uuid import uuid4

from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import Collection
from app.repositories.collections import CollectionRepository

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


async def test_get_or_create_creates_then_reuses(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        repo = CollectionRepository(session, TENANT)
        first = await repo.get_or_create("default", user_id=TENANT.user_id)
        await repo.commit()
    async with db_sessionmaker() as session:
        repo = CollectionRepository(session, TENANT)
        second = await repo.get_or_create("default", user_id=TENANT.user_id)
        assert second.id == first.id


async def test_get_or_create_survives_losing_the_insert_race(db_sessionmaker) -> None:
    winner_id = None
    async with db_sessionmaker() as session:
        repo = CollectionRepository(session, TENANT)

        # Simulate the race: the not-exists check passes, then another
        # request commits the same name before this one flushes.
        original = repo.get_by_name
        calls = {"n": 0}

        async def racing_get_by_name(name: str):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return await original(name)

        repo.get_by_name = racing_get_by_name  # type: ignore[method-assign]
        async with db_sessionmaker() as other:
            other_repo = CollectionRepository(other, TENANT)
            winner = await other_repo.get_or_create("raced", user_id=TENANT.user_id)
            await other_repo.commit()
            winner_id = winner.id

        loser_result = await repo.get_or_create("raced", user_id=TENANT.user_id)
        assert loser_result.id == winner_id


async def test_same_name_allowed_across_tenants(db_sessionmaker) -> None:
    foreign = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")
    async with db_sessionmaker() as session:
        # The unique constraint is per tenant, not global.
        session.add(Collection(tenant_id=foreign.tenant_id, name="default"))
        await session.commit()
    async with db_sessionmaker() as session:
        repo = CollectionRepository(session, TENANT)
        row = await repo.get_or_create("default", user_id=TENANT.user_id)
        await repo.commit()
        assert row.tenant_id == TENANT.tenant_id


async def test_get_by_name_is_tenant_scoped(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        session.add(Collection(tenant_id=uuid4(), name="foreign-only"))
        await session.commit()
    async with db_sessionmaker() as session:
        repo = CollectionRepository(session, TENANT)
        assert await repo.get_by_name("foreign-only") is None
