"""Tenant-scoped access to RAG collection rows."""

from __future__ import annotations

from typing import ClassVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import Collection
from app.repositories.base import BaseRepository


class CollectionRepository(BaseRepository[Collection]):
    """CRUD for the durable catalog of RAG corpora."""

    model: ClassVar[type[Collection]] = Collection

    async def get_by_name(self, name: str) -> Collection | None:
        """Return this tenant's collection named ``name``, or None."""
        stmt = select(Collection).where(
            Collection.tenant_id == self.tenant.tenant_id,
            Collection.name == name,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_or_create(self, name: str, *, user_id: UUID | None = None) -> Collection:
        """Return the named collection, creating it on first use.

        Race-safe against concurrent first-ingests: the unique constraint on
        (tenant_id, name) makes the loser's insert fail, after which the
        winner's row is re-selected. The rollback is safe here because
        callers invoke this before staging any other writes.
        """
        existing = await self.get_by_name(name)
        if existing is not None:
            return existing
        entity = Collection(tenant_id=self.tenant.tenant_id, user_id=user_id, name=name)
        self.add(entity)
        try:
            await self.flush()
        except IntegrityError:
            await self.session.rollback()
            winner = await self.get_by_name(name)
            if winner is None:  # pragma: no cover - constraint failed some other way
                raise
            return winner
        return entity
