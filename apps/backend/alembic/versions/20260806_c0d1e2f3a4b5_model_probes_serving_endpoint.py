"""model_probes: bind a verdict to the server it was measured on.

A probe row already says which weights and which runtime it measured. It did
not say WHICH MACHINE, and that is the axis remote serving added: decode
throughput is the host's hardware, and tool behaviour is a property of the
(model, runtime) pair as that host runs it. Without this column a verdict
earned against a rented cloud pod goes on certifying the same tag served by a
laptop, and vice versa - a certificate migrating between hosts that never ran
a check.

Nullable, and no backfill. Existing rows were measured somewhere nobody
recorded, and inventing the current endpoint for them would manufacture
exactly the false certificate this column exists to prevent. A NULL row is
read as certifying nothing, while its FAILURE verdicts still clamp - the same
safe-direction asymmetry the digest and harness axes already use.

No index either: the endpoint is compared in memory against rows the existing
``(provider, model, digest, created_at)`` lookup already returned, so it
narrows no query.

Revision ID: c0d1e2f3a4b5
Revises: b9c0d1e2f3a4
Create Date: 2026-08-06
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c0d1e2f3a4b5"
down_revision: str | None = "b9c0d1e2f3a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable serving-endpoint column to model_probes."""
    op.add_column("model_probes", sa.Column("serving_endpoint", sa.String(), nullable=True))


def downgrade() -> None:
    """Drop the column.

    Lossy in one direction only, and harmlessly: what is lost is the record of
    where each verdict was earned, so a downgraded deployment goes back to
    treating every row as applying to whatever it serves from now. That is the
    behaviour before this revision, not a new failure - and re-upgrading leaves
    the pre-existing rows NULL, which certify nothing until a check re-runs.
    """
    op.drop_column("model_probes", "serving_endpoint")
