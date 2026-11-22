"""Tenant-scoped access to the registered MCP servers."""

from __future__ import annotations

from typing import ClassVar

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.errors import ConflictError
from app.models import McpAuthType, McpServer
from app.repositories.base import BaseRepository


class McpServerRepository(BaseRepository[McpServer]):
    """CRUD for the per-tenant registry of MCP servers.

    A server is a WORKSPACE resource - one namespace of servers per tenant, keyed
    by name - so this scopes by ``tenant_id`` only (``BaseRepository``), not by
    user: which servers exist is a tenant-wide fact, even though a user-scoped
    server's per-call CREDENTIAL is private to each user.
    """

    model: ClassVar[type[McpServer]] = McpServer

    async def list_enabled(self) -> list[McpServer]:
        """Return this tenant's enabled servers, name-ordered.

        Name ordering is not cosmetic: the tool registry gives a contested tool
        name to the EARLIEST source, so a deterministic order across turns keeps
        which server wins a name collision stable.
        """
        stmt = (
            select(McpServer)
            .where(
                McpServer.tenant_id == self.tenant.tenant_id,
                McpServer.enabled.is_(True),
            )
            .order_by(McpServer.name)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_all(self) -> list[McpServer]:
        """Return every server for this tenant (enabled and disabled), name-ordered.

        The registry read path uses :meth:`list_enabled`; the Integrations API
        lists ALL rows so an admin can see and re-enable a disabled one.
        """
        stmt = (
            select(McpServer)
            .where(McpServer.tenant_id == self.tenant.tenant_id)
            .order_by(McpServer.name)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_name(self, name: str) -> McpServer | None:
        """Return this tenant's server named ``name``, or None."""
        stmt = select(McpServer).where(
            McpServer.tenant_id == self.tenant.tenant_id,
            McpServer.name == name,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        name: str,
        base_url: str,
        credential_scope: str,
        auth_type: str = McpAuthType.HEADER.value,
        config: dict | None = None,
    ) -> McpServer:
        """Insert a new server for this tenant.

        Raises:
            ConflictError: A server with this name already exists for the tenant
                (the ``(tenant_id, name)`` uniqueness). The savepoint-free
                rollback is safe because callers create before staging other
                writes.
        """
        entity = McpServer(
            tenant_id=self.tenant.tenant_id,
            name=name,
            base_url=base_url,
            credential_scope=credential_scope,
            auth_type=auth_type,
            config=config or {},
        )
        self.add(entity)
        try:
            await self.flush()
        except IntegrityError as exc:
            await self.session.rollback()
            raise ConflictError(f"A server named {name!r} already exists.") from exc
        return entity

    async def ensure_seed(
        self,
        *,
        name: str,
        base_url: str,
        credential_scope: str,
        config: dict | None = None,
    ) -> McpServer:
        """Get-or-create the named server; NEVER overwrite an existing row.

        The bootstrap seed for the env-configured servers - one call per sidecar
        a deployment names in its environment.
        Idempotent and NON-DESTRUCTIVE: if a row with this name already exists it
        is returned untouched, so an operator who disabled, re-pointed, or
        rescoped the seeded server keeps their edit - the env value has receded to
        a first-boot seed, and the DB row is authoritative thereafter.

        Race-safe against a concurrent first-seed exactly as
        :meth:`CollectionRepository.get_or_create` is: the unique constraint on
        (tenant_id, name) fails the loser's insert, after which the winner's row
        is re-selected. Callers seed before staging other writes, so the rollback
        is safe.
        """
        existing = await self.get_by_name(name)
        if existing is not None:
            return existing
        entity = McpServer(
            tenant_id=self.tenant.tenant_id,
            name=name,
            base_url=base_url,
            credential_scope=credential_scope,
            config=config or {},
        )
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
