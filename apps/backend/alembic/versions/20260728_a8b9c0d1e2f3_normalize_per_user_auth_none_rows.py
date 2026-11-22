"""normalize per-user auth-none mcp_servers rows to workspace scope.

Revision ID: a8b9c0d1e2f3
Revises: f7a8b9c0d1e2
Create Date: 2026-07-28 20:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "a8b9c0d1e2f3"
down_revision: str | None = "f7a8b9c0d1e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    Data-only. ``auth_type=none`` + ``credential_scope=user`` promised a
    fail-closed per-member credential gate that no call path enforces: an
    auth-none source attaches no credential for anyone, so the scope was inert
    on the wire. The combination is now refused at create and patch, and this
    sweep retires the rows minted before the guards existed - without it, a
    guard that checks the RESULTING patch state would reject even a plain
    enable/disable on such a row, blaming credential scope for a click that
    never touched it.

    Rescoping to ``workspace`` changes no behaviour: neither value sends
    anything for an auth-none server. It only makes the stored state one the
    write paths could still produce.
    """
    op.execute(
        "UPDATE mcp_servers SET credential_scope = 'workspace' "
        "WHERE auth_type = 'none' AND credential_scope = 'user'"
    )


def downgrade() -> None:
    """Revert this revision.

    A no-op: the swept rows carried no behaviour to restore, and which rows
    once held the dead combination is not information worth resurrecting.
    """
