"""model_policy: the operator declaration store.

Machine-scoped by design, on the pattern model_probes established: a
declaration describes the shared model server, so the declaring tenant/user are
audit-only columns with no scoping or foreign keys. Append-only - each row
carries only the axes one action set, and the effective policy is folded per
axis at read time, so invalidation is a digest that no longer matches rather
than an update.

Schema only. Nothing is materialized from infra/models.yaml here: a migration
that read the catalog would produce different rows on every machine and
different rows again after an edit, which is not something a revision id can
describe. The catalog stays in the resolution chain until a later slice sweeps
its declarations into rows deliberately.

Revision ID: b9c0d1e2f3a4
Revises: a8b9c0d1e2f3
Create Date: 2026-08-06
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b9c0d1e2f3a4"
down_revision: str | None = "a8b9c0d1e2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the model_policy declaration store.

    Every axis column is nullable, and that is the schema encoding the design:
    a row states what one action declared and stays silent about the rest, so
    NULL means "this action said nothing about that axis" rather than "off".
    """
    op.create_table(
        "model_policy",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("digest", sa.String(), nullable=True),
        sa.Column("ceiling", sa.String(), nullable=True),
        sa.Column("thinking_with_tools", sa.Boolean(), nullable=True),
        sa.Column("think_mode", sa.String(), nullable=True),
        sa.Column("num_ctx", sa.Integer(), nullable=True),
        sa.Column("thinking_model", sa.String(), nullable=True),
        sa.Column("declared_by_tenant_id", sa.Uuid(), nullable=True),
        sa.Column("declared_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("probe_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_model_policy_lookup",
        "model_policy",
        ["provider", "model", "created_at"],
    )


def downgrade() -> None:
    """Drop the declaration store.

    Real, and lossy in the way a downgrade of a declaration store has to be:
    the rows are operator decisions with no other copy, so a deployment that
    reverses this revision falls back to the catalog for every answer they
    supplied. Nothing else breaks - the resolution chain reads the catalog
    beneath the store precisely so its absence is a supported state.
    """
    op.drop_index("ix_model_policy_lookup", table_name="model_policy")
    op.drop_table("model_policy")
