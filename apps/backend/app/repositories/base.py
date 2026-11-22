"""Tenant-injecting base repository - the mandatory data-access chokepoint.

Every read filters by the bound tenant and every write stamps it, so a
service simply CANNOT touch another tenant's rows through a repository. The
tenant comes from :class:`~app.core.tenancy.TenantContext` (server-derived),
never from client input.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from app.core.tenancy import TenantContext


class BaseRepository[ModelT: SQLModel]:
    """Generic CRUD helpers scoped to one tenant.

    Subclasses set ``model`` to their table class (which must carry ``id`` and
    ``tenant_id`` columns) and add entity-specific queries - each of which must
    keep filtering by ``self.tenant.tenant_id``.
    """

    # ClassVar cannot reference the type parameter; subclasses narrow it in practice.
    model: ClassVar[type[Any]]

    def __init__(self, session: AsyncSession, tenant: TenantContext) -> None:
        self.session = session
        self.tenant = tenant

    async def get(self, entity_id: UUID) -> ModelT | None:
        """Return the entity by id, or None if absent OR owned by another tenant.

        Foreign-tenant rows are indistinguishable from missing ones by design -
        existence itself must not leak across the tenant boundary.
        """
        stmt = select(self.model).where(
            self.model.id == entity_id,
            self.model.tenant_id == self.tenant.tenant_id,
        )
        result = await self.session.execute(stmt)
        return cast("ModelT | None", result.scalar_one_or_none())

    def add(self, entity: ModelT) -> ModelT:
        """Stage a new entity, overwriting its ``tenant_id`` with the bound tenant."""
        entity.tenant_id = self.tenant.tenant_id  # type: ignore[attr-defined]
        self.session.add(entity)
        return entity

    async def delete(self, entity: ModelT) -> None:
        """Stage deletion of an entity previously loaded through this repository."""
        await self.session.delete(entity)

    async def flush(self) -> None:
        """Push staged changes to the database without committing.

        Commit stays with the session owner (the request dependency), so one
        request is one transaction even across multiple repositories.
        """
        await self.session.flush()

    async def commit(self) -> None:
        """Commit the current transaction - the boundary for mutating endpoints.

        Mutations must commit BEFORE the response leaves: the get_session
        teardown also commits, but teardown runs after the response is sent,
        so relying on it alone lets an immediate follow-up read observe stale
        data (seen live: DELETE 204 then GET still listing the row).
        """
        await self.session.commit()

    async def rollback(self) -> None:
        """Discard staged changes, leaving the session usable.

        For the case where the world moved on underneath a loaded entity - the
        row was deleted elsewhere while this session held modifications to it -
        and flushing them would raise rather than accomplish anything.
        """
        await self.session.rollback()
