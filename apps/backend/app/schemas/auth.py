"""Wire shapes for the login endpoints."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class AuthMeResponse(BaseModel):
    """Who the caller is - the frontend's signed-in probe.

    ``auth`` is the field the UI branches on, and it answers a question no other
    field can: ``"none"`` means this deployment has no login at all, so there is
    no sign-in page to offer and no user chip to render, while ``"oidc"`` means
    there is. Without it the frontend would have to infer "login exists" from
    the absence of an error, which is exactly the guess that renders a Sign out
    button on a deployment nobody can sign in to.

    The identity fields are populated in both modes: with no login they describe
    the seeded principal, so a caller reading ``role`` to decide whether to show
    an admin surface needs no special case.
    """

    auth: Literal["none", "oidc"]
    user_id: UUID
    email: str | None = None
    display_name: str
    role: str = Field(description="Coarse role: 'admin' or 'member'.")
