"""mcp_servers auth_type column.

Revision ID: c4d5e6f7a8b9
Revises: b2c3d4e5f6a7
Create Date: 2026-07-21 16:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c4d5e6f7a8b9"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    Slice 9 (OAuth) opener: a server row now records HOW it authenticates
    (``McpAuthType`` - "none" or "header"; the consent flow adds "oauth" when it
    lands), where ``credential_scope`` records WHOSE credential a call carries.

    Added NOT NULL with a server default of "header" so the column populates on
    any existing row with the pre-column behaviour - every registered server
    keeps sending its stored or env bearer; the server default is then dropped
    so the app-side (client) default drives new rows, matching every other
    column here and keeping sqlite tests and postgres identical.
    """
    op.add_column(
        "mcp_servers",
        sa.Column(
            "auth_type",
            sa.String(),
            nullable=False,
            server_default="header",
        ),
    )
    op.alter_column("mcp_servers", "auth_type", server_default=None)


def downgrade() -> None:
    """Revert this revision."""
    op.drop_column("mcp_servers", "auth_type")
