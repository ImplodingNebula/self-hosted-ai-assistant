"""users: the identity a login resolves by.

The pre-login schema promised that real login would need no migration, and the
claim-shaped part of that held exactly: ``tenant_id`` and ``user_id`` were on
every domain row from day one, so nothing had to be backfilled and no ownership
had to be rewritten. What it did NOT cover was the users table's own IDENTITY
KEY - there was no column saying which provider vouched for a person or what
that provider calls them - so login needs these two columns and nothing else.

Keyed on (issuer, sub) rather than on the email, because a provider's ``sub`` is
the claim the spec promises stays stable for the life of an account, while an
address can be renamed, released and reassigned. Both nullable, and the unique
index is PARTIAL over the rows that have both: the seeded pre-auth user carries
NULLs and must stay insertable beside every real account. That is also what
keeps it unreachable by any sign-in - it owns the rows created before login
existed, and no login ever resolves to it.

No backfill, deliberately. There is no honest value to write: nothing in an
existing row says who its owner would be at an identity provider, and guessing
would hand the first person who signs in somebody else's history.

Revision ID: d1e2f3a4b5c6
Revises: c0d1e2f3a4b5
Create Date: 2026-08-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d1e2f3a4b5c6"
down_revision: str | None = "c0d1e2f3a4b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "uq_users_oidc_identity"
_PARTIAL_WHERE = "oidc_issuer IS NOT NULL AND oidc_sub IS NOT NULL"


def upgrade() -> None:
    """Add the identity columns and their partial unique index."""
    op.add_column("users", sa.Column("oidc_issuer", sa.String(), nullable=True))
    op.add_column("users", sa.Column("oidc_sub", sa.String(), nullable=True))
    op.create_index(
        _INDEX_NAME,
        "users",
        ["oidc_issuer", "oidc_sub"],
        unique=True,
        postgresql_where=sa.text(_PARTIAL_WHERE),
        sqlite_where=sa.text(_PARTIAL_WHERE),
    )


def downgrade() -> None:
    """Drop the index and the two columns."""
    op.drop_index(_INDEX_NAME, table_name="users")
    op.drop_column("users", "oidc_sub")
    op.drop_column("users", "oidc_issuer")
