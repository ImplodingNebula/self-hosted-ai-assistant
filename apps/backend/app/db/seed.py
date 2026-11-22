"""Idempotent seeding of the default tenant, user, and workspace settings.

Runs at every boot (and in the test fixtures). Select-first-then-insert
instead of dialect-specific upserts so the exact same code works on postgres
and the sqlite test database.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID
from app.models import Tenant, User, WorkspaceSettings


async def seed_defaults(sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    """Ensure the default tenant, its admin user, and workspace settings exist."""
    async with sessionmaker() as session:
        if await session.get(Tenant, DEFAULT_TENANT_ID) is None:
            session.add(Tenant(id=DEFAULT_TENANT_ID, name="default"))
        if await session.get(User, DEFAULT_USER_ID) is None:
            session.add(
                User(
                    id=DEFAULT_USER_ID,
                    tenant_id=DEFAULT_TENANT_ID,
                    display_name="Default User",
                    role="admin",
                )
            )
        if await session.get(WorkspaceSettings, DEFAULT_TENANT_ID) is None:
            session.add(WorkspaceSettings(tenant_id=DEFAULT_TENANT_ID))
        await session.commit()
