"""HTTP schemas for the members API - the frozen wire contract.

Metadata about the people in one workspace. As everywhere in ``schemas/``, DTOs
are separate from the DB models: ids go out as strings and ``tenant_id`` never
leaves the backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from app.schemas.common import UtcDatetime

if TYPE_CHECKING:
    from app.models import User

MemberRole = Literal["admin", "member"]
"""The two roles this deployment has. Named permissions arrive later."""


class MemberOut(BaseModel):
    """One person in the workspace, as the console lists them."""

    id: str
    email: str | None
    display_name: str
    role: MemberRole
    created_at: UtcDatetime
    # Whether the deployment's own configuration floors this account to admin.
    # The console needs it to explain a control it cannot offer: the role is
    # re-applied at every sign-in, so a page that let somebody change it would
    # be promising something the next login undoes.
    is_env_admin: bool

    @classmethod
    def from_row(cls, user: User, *, is_env_admin: bool) -> MemberOut:
        """Build the DTO from a users row plus the deployment's admin floor."""
        return cls(
            id=str(user.id),
            email=user.email,
            # A role outside the pair cannot reach here: nothing writes one, and
            # the schema would reject it - which is the point of the Literal.
            role=user.role,  # type: ignore[arg-type]
            display_name=user.display_name,
            created_at=user.created_at,
            is_env_admin=is_env_admin,
        )


class MemberListResponse(BaseModel):
    """The workspace's members, oldest first."""

    items: list[MemberOut]


class MemberRoleUpdate(BaseModel):
    """Set one member's role.

    Role only. Nothing else about a person is editable here: display name and
    email come from the identity provider on every sign-in, so a field to edit
    them would be overwritten the next time they signed in.
    """

    role: MemberRole
