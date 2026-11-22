"""messages carry a user_id.

Revision ID: c3d4e5f6a7b8
Revises: 1aab50fd762d
Create Date: 2026-07-16 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import sqlalchemy as sa

from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "1aab50fd762d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The seeded default user (app/core/tenancy.py). Used only as a backfill
# fallback for the impossible case of a message with no live parent
# conversation - messages CASCADE on conversation delete, so there is none.
_DEFAULT_USER_ID = UUID("00000000-0000-0000-0000-000000000002")


def upgrade() -> None:
    """Apply this revision.

    Phase 5 gives messages an owner, mirroring their conversation's owner, so
    the message repository is user-scoped in its own right. Every domain row now
    carries both ``tenant_id`` and ``user_id`` - the multi-tenant-ready schema
    the platform has always intended, finished before real identity needs it.

    The column is added nullable, backfilled from each message's conversation,
    and only then made NOT NULL. Postgres cannot add a NOT NULL column to a
    populated table without a server default, and the correct value here is the
    conversation's owner rather than any single constant - so the value is filled
    in by hand rather than by a default that would then have to be dropped.
    """
    op.add_column("messages", sa.Column("user_id", sa.Uuid(), nullable=True))
    # Every message belongs to a live conversation (the FK cascades), so the
    # subquery always resolves; the COALESCE is a belt-and-braces fallback for a
    # row that somehow outlived its parent.
    op.execute(
        sa.text(
            """
            UPDATE messages
            SET user_id = COALESCE(
                (
                    SELECT c.user_id
                    FROM conversations AS c
                    WHERE c.id = messages.conversation_id
                ),
                :fallback
            )
            WHERE user_id IS NULL
            """
        ).bindparams(
            # Typed as a UUID so it matches the column under COALESCE; a bare
            # string binds as varchar, which Postgres refuses to coalesce with a
            # uuid ("types uuid and character varying cannot be matched").
            sa.bindparam("fallback", value=_DEFAULT_USER_ID, type_=sa.Uuid())
        )
    )
    op.alter_column("messages", "user_id", nullable=False)


def downgrade() -> None:
    """Revert this revision."""
    op.drop_column("messages", "user_id")
