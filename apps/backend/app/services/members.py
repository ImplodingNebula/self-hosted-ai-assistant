"""Member administration: who is in the workspace, and what role they hold.

The role column has existed since the schema was written and nothing has ever
written to it. This is the surface that does, so every refusal it can make lives
here rather than in the router - the two guards below are the whole of the
policy, and both of them are about not locking a workspace out of its own
console.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import ConflictError, NotFoundError
from app.core.tenancy import TenantContext
from app.models import User
from app.repositories.users import ADMIN_ROLE, UserRepository

logger = logging.getLogger(__name__)

_NOT_FOUND_DETAIL = "No such member in this workspace."


def env_admin_emails(settings: Settings) -> frozenset[str]:
    """Return the deployment's standing-admin emails, lowercased.

    These come from the deployment's own configuration, so they are a FLOOR: a
    listed address is an admin at every login, and no in-app change can take
    that away. The list is read here rather than passed around so exactly one
    piece of code decides what "an env admin" means.

    Read defensively - by name, tolerating a missing field and either shape the
    environment can produce - because the setting is owned by the sign-in work
    and this surface must behave sanely on a build that predates it: absent
    means nobody is floored, which is precisely the pre-login truth.

    TODO(rebase): tighten after feat/oidc-login merges. This defensive read
    fails OPEN and undetectably if lane A named or shaped the field differently -
    every test here attaches the attribute itself, so nothing goes red on a
    mismatch. At rebase: (1) read ``settings.oidc_admin_emails`` directly and add
    a test that asserts ``Settings`` actually carries the field, so this marker
    going stale turns a test red rather than passing silently; and (2) confirm
    lane A's login-time admin floor lowercases BOTH sides of its comparison, to
    match ``is_env_admin`` below - a floor that compared raw case there while
    this compares lowercased would let the two disagree about the same address.

    Args:
        settings: Application settings.

    Returns:
        Lowercased email addresses, empty when the deployment names none.
    """
    raw = getattr(settings, "oidc_admin_emails", None)
    if raw is None:
        return frozenset()
    entries = raw.split(",") if isinstance(raw, str) else raw
    return frozenset(
        entry.strip().lower() for entry in entries if isinstance(entry, str) and entry.strip()
    )


def is_env_admin(email: str | None, env_admins: frozenset[str]) -> bool:
    """Whether this address is floored to admin by the deployment's config.

    Case-insensitive: an address is one identity however it was typed, and the
    floor has to hold for the person who signs in as ``Ada@example.com`` when
    the deployment wrote ``ada@example.com``.
    """
    if not email:
        return False
    return email.strip().lower() in env_admins


async def set_member_role(
    session: AsyncSession,
    admin: TenantContext,
    settings: Settings,
    *,
    member_id: UUID,
    role: str,
) -> User:
    """Change one member's role, refusing the two changes that cannot be undone.

    Both guards protect the same thing from opposite directions - a workspace
    that can no longer administer itself, and a role the app claims to own but
    does not:

    * **The last admin keeps the role.** Demoting them leaves nobody who can
      promote anyone, and the only repair would be an operator editing the
      database by hand.
    * **A deployment-listed admin keeps the role.** The environment re-floors
      them at their next sign-in anyway, so an in-app demotion would appear to
      work and silently revert; the refusal says where the change actually
      belongs instead.

    Self-demotion is otherwise ALLOWED. An admin stepping down is a legitimate
    thing to do, and the caller is told what it costs before they confirm.

    Setting the role a member already holds returns early, before the admin
    lock is taken. That is safe precisely because the branch WRITES NOTHING: the
    lock exists only to serialize the count-then-write of a real demotion, and a
    no-op has no write to serialize. Do not add a mutation to that branch without
    moving it under the lock.

    Args:
        session: The request's database session.
        admin: The acting principal (already role-gated by the router).
        settings: Application settings, for the deployment's admin floor.
        member_id: Who to change.
        role: The new role, already validated as ``admin`` or ``member``.

    Returns:
        The updated user row.

    Raises:
        NotFoundError: No such user in this tenant.
        ConflictError: The change would remove the last admin, or would try to
            demote an address the deployment floors to admin.
    """
    repo = UserRepository(session, admin)
    member = await repo.get(member_id)
    if member is None:
        raise NotFoundError(_NOT_FOUND_DETAIL)

    previous = member.role
    if previous == role:
        return member

    if previous == ADMIN_ROLE:
        if is_env_admin(member.email, env_admin_emails(settings)):
            raise ConflictError(
                "This account is listed as an administrator in the deployment's own "
                "configuration, so it is an administrator at every sign-in and cannot "
                "be changed here. Edit OIDC_ADMIN_EMAILS in the deployment environment "
                "and restart the backend instead."
            )
        # Locked, because the count and the write that acts on it are two steps:
        # see UserRepository.lock_admins for how two overlapping demotions would
        # otherwise agree that the other admin will still be there.
        admins = await repo.lock_admins()
        if member.id not in {row.id for row in admins}:
            # Somebody else demoted this account while this request waited for
            # the lock. The end state asked for is the end state already in the
            # database, so there is nothing to do and nothing to refuse - but
            # the row still has to be re-read before it is described back: the
            # locking query refreshes only the rows it RETURNED, and this one is
            # missing from them precisely because it is no longer an admin, so
            # without this the response would answer "admin" to a request that
            # asked for "member" and got it.
            await session.refresh(member)
            return member
        if len(admins) <= 1:
            raise ConflictError(
                "This is the workspace's only administrator. Promote somebody else "
                "first, then change this account."
            )

    member.role = role
    await repo.commit()
    # Actor, target and the change itself, in the tool-call log's shape: no new
    # table, and a line that only becomes meaningful once the principal is real
    # - which is exactly when somebody will want to read it back.
    logger.info(
        "member role change by tenant=%s user=%s: target=%s role %s -> %s",
        admin.tenant_id,
        admin.user_id,
        member.id,
        previous,
        role,
    )
    return member
