"""attachment pipeline columns.

Revision ID: 1aab50fd762d
Revises: 8f2a41c6d0b3
Create Date: 2026-07-10 12:52:09.135617
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel  # noqa: F401  - autogenerate renders sqlmodel.sql.sqltypes.* columns

from alembic import op

revision: str = "1aab50fd762d"
down_revision: str | None = "8f2a41c6d0b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    Phase 4 gives attachments a real ingestion lifecycle: a worker parses the
    stored bytes, so a row needs somewhere to record why parsing failed, when it
    last moved, whether the same bytes were already handled, and whether the file
    was chosen from a disk or pasted into the composer.

    The two NOT NULL columns are added WITH a server default and then have it
    dropped. Postgres cannot add a NOT NULL column to a populated table without
    one, and while `attachments` is empty on every deployment today, a migration
    that only works on an empty table is a trap for the first one that is not.
    Dropping the default afterwards restores this schema's convention: defaults
    are client-side (see app/models/entities.py) so the same models drive the
    sqlite test database and postgres identically.
    """
    op.add_column("attachments", sa.Column("error", sa.Text(), nullable=True))
    op.add_column(
        "attachments",
        sa.Column("content_hash", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.add_column(
        "attachments",
        sa.Column(
            "origin",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="upload",
        ),
    )
    op.add_column(
        "attachments",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.alter_column("attachments", "origin", server_default=None)
    op.alter_column("attachments", "updated_at", server_default=None)

    # Every attachment read is scoped to one conversation within one tenant: the
    # composer lists them, a chat turn resolves the ids it references, and
    # deleting a conversation collects its blobs.
    op.create_index(
        "ix_attachments_tenant_conversation",
        "attachments",
        ["tenant_id", "conversation_id"],
        unique=False,
    )
    # The composite index above leads with tenant_id, so it already serves every
    # tenant-only lookup. Keeping the standalone index would cost a write on
    # every insert and buy nothing - the same reasoning `conversations` applies.
    op.drop_index("ix_attachments_tenant_id", table_name="attachments")
    # Not unique: the same bytes may legitimately be attached to two threads.
    op.create_index(
        op.f("ix_attachments_content_hash"), "attachments", ["content_hash"], unique=False
    )


def downgrade() -> None:
    """Revert this revision."""
    op.drop_index(op.f("ix_attachments_content_hash"), table_name="attachments")
    op.create_index("ix_attachments_tenant_id", "attachments", ["tenant_id"], unique=False)
    op.drop_index("ix_attachments_tenant_conversation", table_name="attachments")
    op.drop_column("attachments", "updated_at")
    op.drop_column("attachments", "origin")
    op.drop_column("attachments", "content_hash")
    op.drop_column("attachments", "error")
