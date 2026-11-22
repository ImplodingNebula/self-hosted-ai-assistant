"""oauth_authorizations table.

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-07-21 20:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d5e6f7a8b9c0"
down_revision: str | None = "c4d5e6f7a8b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    Slice 9's one new table: the pending OAuth consent between "Connect" and
    the callback. A row is written when an admin starts a connect and consumed
    (deleted) exactly once by the callback carrying its ``state``; rows expire
    and are swept opportunistically, so the table stays near-empty. The PKCE
    verifier is stored SEALED (AES-GCM bytes), never plaintext.
    """
    op.create_table(
        "oauth_authorizations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("mcp_server_id", sa.Uuid(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("code_verifier_sealed", sa.LargeBinary(), nullable=False),
        sa.Column("client_id", sa.String(), nullable=False),
        sa.Column("redirect_uri", sa.String(), nullable=False),
        sa.Column("resource", sa.String(), nullable=True),
        sa.Column("endpoints", sa.JSON(), nullable=False),
        sa.Column("scopes", sa.String(), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("state", name="uq_oauth_authorizations_state"),
    )


def downgrade() -> None:
    """Revert this revision."""
    op.drop_table("oauth_authorizations")
