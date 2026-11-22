"""Turning a verified identity into the ``users`` row it owns.

Keyed on **(issuer, sub)**, never on the email. A provider's ``sub`` is the one
claim the spec promises is stable for the life of an account; an address can be
renamed, released, and handed to somebody else, so keying on it would eventually
hand one person another person's history. The email is still stored and
refreshed at every login - it is what admission and the members list read - but
it is a LABEL here, not the key.

The seeded default user is deliberately never resolved by this. Its identity
columns stay NULL, so no login can ever land on it; it remains the owner of
whatever was created before login existed, which those rows keep pointing at.
That is a decision with a visible consequence, recorded in docs/DATA-MODEL.md:
turning login on does not hand a first signer-in the pre-auth conversations.

This is the one query in the codebase that reasonably runs OUTSIDE a
tenant-scoped repository, and the reason is structural rather than a shortcut:
the repositories inject the acting tenant, and this function's whole job is to
find out what that tenant IS. The identity key is unique across the table (a
partial unique index), so the row that comes back names its own tenant - the
tenant is the ANSWER here, never an input.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.tenancy import DEFAULT_TENANT_ID
from app.models import User
from app.services.auth.admission import role_for

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.config import Settings
    from app.services.auth.tokens import VerifiedIdentity

logger = logging.getLogger(__name__)


async def resolve_or_create(
    session: AsyncSession, identity: VerifiedIdentity, *, settings: Settings
) -> User:
    """Return the row this identity owns, creating it on a first login.

    Every login refreshes the label claims (email, display name) and re-applies
    the environment's admin floor, so a change at the provider or in ``.env``
    takes effect at the next sign-in rather than at the next migration.

    Args:
        session: An open session; the caller owns the commit.
        identity: The verified claims from the id_token.
        settings: The deployment's configuration (the admin floor).

    Returns:
        The persisted user row, flushed so its id is available.
    """
    user = await _find(session, identity)

    if user is None:
        # New person. The workspace is the seeded one: this deployment is
        # single-workspace, and inventing a tenant per signer-in would give each
        # of them a private, empty workspace instead of the shared one they were
        # admitted to.
        user = User(
            tenant_id=DEFAULT_TENANT_ID,
            email=identity.email,
            display_name=identity.display_name,
            role=role_for(identity.email, stored_role=None, settings=settings),
            oidc_issuer=identity.issuer,
            oidc_sub=identity.subject,
        )
        session.add(user)
        try:
            await session.flush()
        except IntegrityError:
            # Two first logins for the same account at once: both selected
            # nothing, both inserted, and the partial unique index refused one.
            # The loser re-reads instead of failing - the row it wanted now
            # exists, which is the outcome it was asking for. Without this the
            # refusal escapes as a 500 rather than as a sign-in, because an
            # IntegrityError is not one of the login errors the route catches.
            await session.rollback()
            user = await _find(session, identity)
            if user is None:
                raise
            return await _refresh(session, user, identity, settings=settings)
        logger.info(
            "auth.user_created user_id=%s tenant_id=%s role=%s",
            user.id,
            user.tenant_id,
            user.role,
        )
        return user

    return await _refresh(session, user, identity, settings=settings)


async def _find(session: AsyncSession, identity: VerifiedIdentity) -> User | None:
    """Look a row up by its identity key, or None on a first login."""
    result = await session.execute(
        select(User).where(User.oidc_issuer == identity.issuer, User.oidc_sub == identity.subject)
    )
    return result.scalar_one_or_none()


async def _refresh(
    session: AsyncSession, user: User, identity: VerifiedIdentity, *, settings: Settings
) -> User:
    """Re-apply the label claims and the environment's admin floor to a row."""
    user.email = identity.email
    user.display_name = identity.display_name
    promoted_from = user.role
    user.role = role_for(identity.email, stored_role=user.role, settings=settings)
    if user.role != promoted_from:
        # Write-through, not a per-request overlay: the row is what the members
        # surface reads and what every later request resolves, so a floor that
        # only applied in memory would show the wrong role everywhere else.
        logger.info(
            "auth.role_floor_applied user_id=%s from=%s to=%s",
            user.id,
            promoted_from,
            user.role,
        )
    await session.flush()
    return user
