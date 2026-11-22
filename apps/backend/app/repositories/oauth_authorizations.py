"""Tenant-scoped access to pending OAuth authorizations."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, ClassVar

from sqlalchemy import delete

from app.models import OAuthAuthorization, utcnow
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID


class OAuthAuthorizationRepository(BaseRepository[OAuthAuthorization]):
    """The pending-consent rows between "Connect" and the callback.

    Tenant-scoped like every repository; ``consume`` is the only read, and it
    DELETES what it returns - single use is the contract that blocks an
    authorization code being replayed or injected into a second flow. The
    delete-returning is ATOMIC, so of two truly concurrent callbacks carrying
    the same state exactly one receives the row; the other sees None and
    fails like any unknown state.
    """

    model: ClassVar[type[OAuthAuthorization]] = OAuthAuthorization

    async def create(
        self,
        *,
        mcp_server_id: UUID,
        state: str,
        code_verifier_sealed: bytes,
        client_id: str,
        redirect_uri: str,
        resource: str | None,
        endpoints: dict,
        scopes: str,
        created_by: UUID,
        expires_at: datetime,
    ) -> OAuthAuthorization:
        """Insert a pending consent, sweeping this tenant's expired rows first.

        The sweep rides the insert (no scheduled job for a near-empty table):
        every started connect clears the tenant's dead states, so abandonment
        never accumulates.
        """
        await self.session.execute(
            delete(OAuthAuthorization).where(
                OAuthAuthorization.tenant_id == self.tenant.tenant_id,
                OAuthAuthorization.expires_at < utcnow(),
            )
        )
        entity = OAuthAuthorization(
            tenant_id=self.tenant.tenant_id,
            mcp_server_id=mcp_server_id,
            state=state,
            code_verifier_sealed=code_verifier_sealed,
            client_id=client_id,
            redirect_uri=redirect_uri,
            resource=resource,
            endpoints=endpoints,
            scopes=scopes,
            created_by=created_by,
            expires_at=expires_at,
        )
        self.add(entity)
        await self.flush()
        return entity

    async def consume(self, state: str) -> OAuthAuthorization | None:
        """Return the tenant's pending row for ``state`` and DELETE it, or None.

        None for an unknown state AND for an expired one - the caller cannot
        tell the difference, deliberately: a callback probing states learns
        nothing about which ever existed. The delete happens before the caller
        acts, so a second callback with the same state finds nothing.
        """
        stmt = (
            delete(OAuthAuthorization)
            .where(
                OAuthAuthorization.tenant_id == self.tenant.tenant_id,
                OAuthAuthorization.state == state,
            )
            .returning(OAuthAuthorization)
        )
        row = (await self.session.execute(stmt)).scalars().one_or_none()
        if row is None:
            return None
        # sqlite (the test database) returns naive datetimes even from a
        # timezone-aware column; normalize before comparing, as the chat
        # service does for the same reason.
        expires = row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=UTC)
        if expires < utcnow():
            return None
        return row
