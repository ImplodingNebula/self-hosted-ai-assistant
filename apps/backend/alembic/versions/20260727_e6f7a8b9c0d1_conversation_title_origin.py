"""conversation title_origin column.

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-07-27 09:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e6f7a8b9c0d1"
down_revision: str | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision.

    Records where a conversation's title came from - ``fallback`` (the
    truncation the opening turn wrote), ``generated`` (the background job's
    wording), or ``manual`` (a name a person typed). Only the last is protected:
    nothing may overwrite a title its owner chose.

    Until now the title job protected a rename by comparing the stored title
    against the seed it was allowed to replace, which works exactly once. Editing
    a conversation's first question re-runs the naming, and a string comparison
    then cannot tell "still the fallback" from "renamed to those same words" - so
    the provenance has to be recorded rather than inferred.

    Added NOT NULL with a server default of ``fallback`` so every existing row
    populates without a backfill pass; the server default is then dropped so the
    app-side default drives new rows, matching every other column here (a
    client-side default keeps the sqlite tests and postgres identical). Existing
    rows that were in fact renamed by hand therefore read as ``fallback``, and a
    later edit of such a thread's first question may re-title it once. That is
    accepted: it is one-shot, cosmetic, and the alternative is inventing
    provenance for rows that never recorded any.
    """
    op.add_column(
        "conversations",
        sa.Column("title_origin", sa.String(), nullable=False, server_default="fallback"),
    )
    op.alter_column("conversations", "title_origin", server_default=None)


def downgrade() -> None:
    """Revert this revision."""
    op.drop_column("conversations", "title_origin")
