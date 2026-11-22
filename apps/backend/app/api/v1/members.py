"""Members endpoints: who is in the workspace, and what role each one holds.

Both routes are admin-gated. Reading the list is as admin-only as changing it:
the roster names every colleague's email address, which is not a fact a member
needs in order to use the assistant.

Routers stay HTTP-only: the demotion guards and the audit line live in
``app.services.members``, and both routes reach the database only through the
tenant-scoped ``UserRepository`` - so a foreign-tenant id is indistinguishable
from a missing one (404), and a malformed id is a 404 too. The one piece of
policy the list route reads for itself is the deployment's admin floor, to stamp
``is_env_admin`` per row - it is a read, not a guard, and the same helper the
service uses answers it.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, get_settings, require_role
from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.tenancy import TenantContext
from app.repositories.users import UserRepository
from app.schemas.members import MemberListResponse, MemberOut, MemberRoleUpdate
from app.services.members import env_admin_emails, is_env_admin, set_member_role

router = APIRouter(tags=["members"])

_NOT_FOUND_DETAIL = "No such member in this workspace."

SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[TenantContext, Depends(require_role("admin"))]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _parse_id(raw: str) -> UUID:
    """Parse a path id, treating a malformed value as an absent resource (404)."""
    try:
        return UUID(raw)
    except ValueError as exc:
        raise NotFoundError(_NOT_FOUND_DETAIL) from exc


@router.get("/members")
async def list_members(
    session: SessionDep,
    admin: AdminDep,
    settings: SettingsDep,
) -> MemberListResponse:
    """List the workspace's members, oldest first.

    Each row carries whether the deployment's configuration floors it to admin,
    so the console can render that account's role as fixed and say why rather
    than offering a control whose effect the next sign-in would undo.
    """
    rows = await UserRepository(session, admin).list_all()
    env_admins = env_admin_emails(settings)
    return MemberListResponse(
        items=[
            MemberOut.from_row(row, is_env_admin=is_env_admin(row.email, env_admins))
            for row in rows
        ]
    )


@router.patch("/members/{member_id}")
async def update_member_role(
    member_id: str,
    payload: MemberRoleUpdate,
    session: SessionDep,
    admin: AdminDep,
    settings: SettingsDep,
) -> MemberOut:
    """Set one member's role to ``admin`` or ``member``.

    Refused with a 409 when the change would leave the workspace without an
    administrator, or when the account's role is set by the deployment's own
    configuration - both messages name what to do instead. An admin may demote
    themselves as long as somebody else can still administer the workspace.

    No delete or deactivate verb here: an account that must lose access loses it
    at the identity provider, which is the only place that decision holds.
    """
    member = await set_member_role(
        session, admin, settings, member_id=_parse_id(member_id), role=payload.role
    )
    return MemberOut.from_row(
        member, is_env_admin=is_env_admin(member.email, env_admin_emails(settings))
    )
