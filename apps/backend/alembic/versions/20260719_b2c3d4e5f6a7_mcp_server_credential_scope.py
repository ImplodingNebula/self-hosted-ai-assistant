"""mcp_servers credential_scope column.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-19 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    Slice 6b makes ``mcp_servers`` the live, DB-backed source registry: a server
    row can now declare whether its tool CALLS carry the shared workspace header
    or the acting user's own credential (``McpCredentialScope`` - "workspace" or
    "user"). A user-scoped server FAILS CLOSED when the user has connected no
    credential.

    Added NOT NULL with a server default of "workspace" so the column populates
    on any existing row and every existing server keeps today's shared-header
    behaviour; the server default is then dropped so the app-side default drives
    new rows (the model default is client-side, matching every other column and
    keeping sqlite tests and postgres identical). A re-pointed or rescoped row is
    an operator edit and is never touched by this migration.
    """
    op.add_column(
        "mcp_servers",
        sa.Column(
            "credential_scope",
            sa.String(),
            nullable=False,
            server_default="workspace",
        ),
    )
    op.alter_column("mcp_servers", "credential_scope", server_default=None)


def downgrade() -> None:
    """Revert this revision."""
    op.drop_column("mcp_servers", "credential_scope")
