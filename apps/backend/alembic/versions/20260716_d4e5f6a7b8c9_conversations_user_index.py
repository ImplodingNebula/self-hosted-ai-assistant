"""conversations sidebar index carries user_id.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-07-16 10:30:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    The conversation repository is now user-scoped, so the sidebar query filters
    ``tenant_id`` AND ``user_id`` before ordering by ``updated_at``. The old
    index led with ``(tenant_id, updated_at)`` only, so with more than one user
    per tenant it would walk another user's rows to fill one page and discard
    them. Leading with both equality columns and ending on the ordering column
    serves the whole query directly. A tenant-only lookup is still served by the
    index's ``tenant_id`` prefix, so nothing regresses for that path.
    """
    op.drop_index("ix_conversations_tenant_updated", table_name="conversations")
    op.create_index(
        "ix_conversations_tenant_user_updated",
        "conversations",
        ["tenant_id", "user_id", "updated_at"],
        unique=False,
    )


def downgrade() -> None:
    """Revert this revision."""
    op.drop_index("ix_conversations_tenant_user_updated", table_name="conversations")
    op.create_index(
        "ix_conversations_tenant_updated",
        "conversations",
        ["tenant_id", "updated_at"],
        unique=False,
    )
