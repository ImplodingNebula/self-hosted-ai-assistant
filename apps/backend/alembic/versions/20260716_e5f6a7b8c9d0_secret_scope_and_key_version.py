"""secrets gain a user scope and a key version.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-07-16 11:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    Credentials become per-user as well as workspace-shared: ``user_id`` NULL is
    a workspace secret, a set ``user_id`` is one member's own. ``key_version``
    records which master key sealed a row, so a future key rotation is a version
    bump rather than a schema change - every row today is version 1.

    The uniqueness key widens from (tenant, name) to (tenant, user_id, name) so a
    workspace secret and each member's secret can share a name. It is declared
    ``NULLS NOT DISTINCT`` so two workspace rows (NULL owner) still cannot
    duplicate a name - Postgres would otherwise treat the NULLs as distinct.

    ``key_version`` is added with a server default so the statement is valid on a
    populated table, then the default is dropped to keep this schema's
    convention that defaults live in the app, not the database. The table is
    empty on every deployment today (nothing writes it yet), but a migration
    that only works on an empty table is a trap for the first one that is not.
    """
    op.add_column("secrets", sa.Column("user_id", sa.Uuid(), nullable=True))
    op.add_column(
        "secrets",
        sa.Column("key_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.alter_column("secrets", "key_version", server_default=None)

    op.drop_constraint("uq_secrets_tenant_name", "secrets", type_="unique")
    op.create_unique_constraint(
        "uq_secrets_tenant_user_name",
        "secrets",
        ["tenant_id", "user_id", "name"],
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    """Revert this revision.

    Narrowing the uniqueness key back to (tenant, name) can only succeed while no
    two rows share a name across owners - which is exactly what the upgrade
    enabled. On the empty ``secrets`` table this schema ships with it is clean; a
    downgrade after per-user secrets exist would need those reconciled first.
    """
    op.drop_constraint("uq_secrets_tenant_user_name", "secrets", type_="unique")
    op.create_unique_constraint("uq_secrets_tenant_name", "secrets", ["tenant_id", "name"])
    op.drop_column("secrets", "key_version")
    op.drop_column("secrets", "user_id")
