"""model_probes: the tool-check measurement log.

Machine-scoped by design (owner-approved 2026-07-15): a probe row describes
the shared model server, so the triggering tenant/user are audit-only columns
with no scoping or foreign keys. Append-only history - invalidation is a new
row under a new digest/runtime/harness version, never an update.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-07-16
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the model_probes measurement log."""
    op.create_table(
        "model_probes",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("digest", sa.String(), nullable=True),
        sa.Column("runtime_version", sa.String(), nullable=True),
        sa.Column("harness_version", sa.String(), nullable=False),
        sa.Column("depth", sa.String(), nullable=False),
        sa.Column("earned_agency", sa.String(), nullable=False),
        sa.Column("thinking_with_tools", sa.Boolean(), nullable=False),
        sa.Column("tokens_per_second", sa.Float(), nullable=True),
        sa.Column("vram_bytes", sa.BigInteger(), nullable=True),
        sa.Column("cases", sa.JSON(), nullable=False),
        sa.Column("triggered_by_tenant_id", sa.Uuid(), nullable=True),
        sa.Column("triggered_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_model_probes_lookup",
        "model_probes",
        ["provider", "model", "digest", "created_at"],
    )


def downgrade() -> None:
    """Drop the measurement log (history is lost - it is measurements)."""
    op.drop_index("ix_model_probes_lookup", table_name="model_probes")
    op.drop_table("model_probes")
