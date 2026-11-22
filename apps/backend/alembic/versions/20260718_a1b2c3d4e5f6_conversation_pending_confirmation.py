"""conversation pending_confirmation column.

Revision ID: a1b2c3d4e5f6
Revises: f6a7b8c9d0e1
Create Date: 2026-07-18 10:30:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    Phase 5's side-effecting-tool confirmation suspends a turn until the user
    approves the action. The suspended turn's checkpoint - its assembled prompt,
    the loop's counters and citations, and the calls awaiting approval - is kept
    on the conversation so the resume request can continue the exact turn without
    re-running its retrieval.

    Nullable with no server default: NULL means "no confirmation pending", which
    is the state of every existing conversation and the common case for every new
    one, so no backfill is needed. Client-side default (None) keeps the model
    driving both sqlite tests and postgres identically.
    """
    op.add_column(
        "conversations",
        sa.Column("pending_confirmation", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    """Revert this revision."""
    op.drop_column("conversations", "pending_confirmation")
