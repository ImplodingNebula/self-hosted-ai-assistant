"""User-scoped base repository - the per-user data-access chokepoint.

A sibling of :class:`~app.repositories.base.BaseRepository` for rows that a
per-user credential (or a per-user tool result) can reach. Every read filters
by BOTH the bound tenant and the acting user, and every write stamps both, so
one user cannot touch another user's rows through a repository even inside the
same tenant.

The acting user is ``self.tenant.user_id`` - read from the bound
:class:`~app.core.tenancy.TenantContext` and nowhere else. **No method here
accepts a ``user_id`` argument**, so passing another user's id is not
expressible: the wrong call cannot be written, not merely discouraged.

Today a single default user is seeded, so this filter is a no-op - every row
is already owned by that one principal. It stops being a no-op the moment real
identity lands: without it, one authenticated user could read another's private
conversation simply by naming its id, because the tenant filter alone does not
separate two users inside one tenant.
"""

from __future__ import annotations

from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlmodel import SQLModel

from app.repositories.base import BaseRepository


class UserScopedRepository[ModelT: SQLModel](BaseRepository[ModelT]):
    """Generic CRUD helpers scoped to one tenant AND one user.

    Subclasses set ``model`` to a table carrying ``id``, ``tenant_id`` and
    ``user_id`` columns, and every bespoke query they add must keep filtering by
    ``self.tenant.user_id`` alongside the tenant, exactly as the base repository
    requires for the tenant.
    """

    async def get(self, entity_id: UUID) -> ModelT | None:
        """Return the entity by id, or None if absent OR owned by another principal.

        Foreign-tenant AND foreign-user rows are both indistinguishable from
        missing ones by design - neither existence nor ownership may leak across
        the boundary.
        """
        stmt = select(self.model).where(
            self.model.id == entity_id,
            self.model.tenant_id == self.tenant.tenant_id,
            self.model.user_id == self.tenant.user_id,
        )
        result = await self.session.execute(stmt)
        return cast("ModelT | None", result.scalar_one_or_none())

    def add(self, entity: ModelT) -> ModelT:
        """Stage a new entity, stamping BOTH the bound tenant and the acting user."""
        entity.user_id = self.tenant.user_id  # type: ignore[attr-defined]
        return super().add(entity)  # stamps tenant_id and session.add()s
