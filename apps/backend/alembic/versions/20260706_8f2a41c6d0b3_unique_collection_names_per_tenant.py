"""unique collection names per tenant.

Revision ID: 8f2a41c6d0b3
Revises: 5022b7cb3271
Create Date: 2026-07-06 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "8f2a41c6d0b3"
down_revision: str | None = "5022b7cb3271"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    The ingest API addresses collections by name (get-or-create), so names
    must be unique per tenant. The table has been schema-ready-but-idle since
    Phase 2, so no dedupe pass is needed before adding the constraint.
    """
    op.create_unique_constraint("uq_collections_tenant_name", "collections", ["tenant_id", "name"])


def downgrade() -> None:
    """Revert this revision."""
    op.drop_constraint("uq_collections_tenant_name", "collections", type_="unique")
