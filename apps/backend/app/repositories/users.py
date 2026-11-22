"""Tenant-scoped access to the workspace's people.

A :class:`~app.repositories.base.BaseRepository`, deliberately NOT a
:class:`~app.repositories.user_scoped.UserScopedRepository`: who belongs to a
workspace is a tenant-wide fact, the same shape as the MCP server registry. A
user-scoped repository here would let each member see only themselves, which is
the opposite of what a members list is for - and it would make the last-admin
count unanswerable.

This is the first thing in the app that reads the ``users`` table at request
time. Nothing did before, because there was one seeded principal and no surface
that named anybody else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sqlalchemy import select

from app.models import User
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    from sqlalchemy import Select

ADMIN_ROLE = "admin"
MEMBER_ROLE = "member"


class UserRepository(BaseRepository[User]):
    """Read and role-update access to one tenant's users."""

    model: ClassVar[type[User]] = User

    async def list_all(self) -> list[User]:
        """Return every user of this tenant, oldest first.

        Joining order, not alphabetical: it puts the bootstrap admin at the top
        and keeps the list stable as people are added, so a row does not move
        under a cursor when somebody edits their display name upstream. ``id``
        breaks ties, because rows seeded in one transaction share a timestamp.
        """
        stmt = (
            select(User)
            .where(User.tenant_id == self.tenant.tenant_id)
            .order_by(User.created_at, User.id)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    def locked_admins_statement(self) -> Select[tuple[User]]:
        """Build the locking admin-set query (see :meth:`lock_admins`).

        Separated from its execution so the lock itself is assertable: sqlite -
        the test database - accepts ``FOR UPDATE`` by ignoring it, so running
        the query proves nothing about the clause that makes the guard safe on
        postgres. Compiling it does.
        """
        return (
            select(User)
            .where(User.tenant_id == self.tenant.tenant_id, User.role == ADMIN_ROLE)
            .order_by(User.id)
            .with_for_update()
            # Rows already in the session's identity map keep their loaded
            # attribute values on a plain re-select, which would hand the guard
            # back the very staleness the lock was taken to shed.
            .execution_options(populate_existing=True)
        )

    async def lock_admins(self) -> list[User]:
        """Fetch this tenant's admins holding a row lock, for the last-admin guard.

        The guard is a read-then-write across two rows' worth of state, so the
        read has to be serialized or it decides on a world that has already
        moved. Without the lock, two demotions of DIFFERENT admins overlapping
        in a two-admin workspace each see a count of two, each pass, and the
        workspace ends with no administrator at all - the exact state the guard
        exists to prevent, and one only a hand-edit of the database repairs.

        Every admin row is locked, not just the target: locking the target alone
        leaves each request free to count the other, which is how both come to
        pass. ``id`` order gives every caller the same lock order, so two of
        these can queue but cannot deadlock. A caller that waited on the lock
        re-reads the rows as they are once it is granted, so the loser of the
        race sees the winner's demotion and refuses.

        No lock on sqlite, whose single-writer model cannot interleave the two
        writes anyway. The caller commits, releasing the lock.

        Returns:
            The tenant's admin rows, refreshed from the database.
        """
        result = await self.session.execute(self.locked_admins_statement())
        return list(result.scalars().all())
