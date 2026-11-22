"""conversation pinned_at column.

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
Create Date: 2026-07-28 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f7a8b9c0d1e2"
down_revision: str | None = "e6f7a8b9c0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    Records when a conversation was pinned to the top of the sidebar, or NULL
    when it is not pinned. A timestamp rather than a boolean because the pinned
    group is ordered newest pin first, and a flag cannot express that order.

    Nullable with no server default and no backfill: an absent value already
    means "not pinned", so every existing row is correct the moment the column
    exists. There is no cap on how many threads a person may pin - the group's
    own visual weight is the limit, not a quota.
    """
    op.add_column(
        "conversations",
        sa.Column("pinned_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Revert this revision."""
    op.drop_column("conversations", "pinned_at")
