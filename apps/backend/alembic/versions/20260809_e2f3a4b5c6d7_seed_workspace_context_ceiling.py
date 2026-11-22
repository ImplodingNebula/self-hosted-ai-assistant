"""seed the workspace context ceiling from the tier this deployment was serving.

The window a turn asks for on a tag NOTHING sizes used to come from the active
model profile's ``defaults.num_ctx`` - so which tier a deployment ran silently
decided how much context every unsized model got. That is deployment
configuration wearing a catalog's clothes, and it is moving out: the window is
now the workspace's context ceiling when one is set, and a deployment-wide
``DEFAULT_CONTEXT_WINDOW`` otherwise.

This migration is what makes that move invisible to a deployment already
running. ``DEFAULT_CONTEXT_WINDOW`` ships at 8192 - the value the dev profile
served - so a deployment on a larger profile would otherwise wake up with a
smaller window than it went to sleep with. Writing the profile's old number
into the workspace ceiling preserves the behaviour exactly, and does it in a
row an operator can see and change rather than in a constant.

The tier is read from the ``MODEL_TIER`` environment variable AT MIGRATION
TIME, and the numbers below are FROZEN LITERALS rather than a catalog lookup.
That is deliberate twice over. A migration describes what a deployment WAS
serving, so it must not move when the catalog is edited afterwards - and the
tier concept is being retired, so a migration that imported the catalog would
stop reading cleanly the moment it goes. The raw environment is read rather
than the settings object for the second reason as well: that object's tier
field is going away, and this file has to keep parsing after it does. An unset
or unrecognised tier takes the same 8192 the setting defaults to.

The one deployment this describes wrongly is one that EDITED the profile's
``defaults.num_ctx`` in its own ``infra/models.yaml``. The frozen numbers are
what the shipped profiles declare, so such a deployment is seeded the shipped
number and its windows move to match. That is called out in the migration note
in ``infra/.env.example``; re-entering the customised number as a workspace
context ceiling restores it.

A ceiling that is already set is NEVER overwritten: it is a number a person
chose, and this migration has nothing better to say than they did.

Rows, not row: ``workspace_settings`` is keyed by tenant, so every tenant with
no stored ceiling is seeded. A deployment with NO rows at all - a fresh
database, migrated before the boot seed has ever run - gets nothing written
here, and needs nothing: an absent row reads as an unset ceiling, and the
runtime's ``DEFAULT_CONTEXT_WINDOW`` arm is exactly the answer such a
deployment wants.

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-08-09
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from alembic import op

revision: str = "e2f3a4b5c6d7"
down_revision: str | None = "d1e2f3a4b5c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The key inside the settings blob. Named here rather than imported, for the
# same reason the windows are frozen: this file must keep meaning what it means
# even after the code that reads the key moves on.
_CEILING_KEY = "context_ceiling"

# What each tier's profile declared as ``defaults.num_ctx`` on the day this was
# written. Frozen on purpose - see the module docstring.
_TIER_WINDOWS = {
    "dev": 8192,
    "dev-qwen35": 8192,
    "mid": 16384,
    "server-t4": 16384,
    "server-a10": 32768,
}

# What an unset or unrecognised MODEL_TIER seeds, matching the setting's own
# default so the two arms of the new resolution agree.
_FALLBACK_WINDOW = 8192

# A minimal projection of the table, typed so SQLAlchemy serializes the JSON
# column for whichever dialect is connected. Writing the blob as a bound string
# instead would need a per-dialect cast, and the two dialects disagree about
# what casting to JSON even means.
_workspace_settings = sa.table(
    "workspace_settings",
    sa.column("tenant_id", sa.Uuid()),
    sa.column("settings", sa.JSON()),
)


def seed_window(tier: str | None) -> int:
    """Return the window a deployment on ``tier`` was already serving.

    Args:
        tier: The ``MODEL_TIER`` value, or None when it is unset.

    Returns:
        The frozen per-tier window, or the 8192 fallback for an unset or
        unrecognised tier.
    """
    return _TIER_WINDOWS.get((tier or "").strip(), _FALLBACK_WINDOW)


def read_blob(stored: object) -> dict | None:
    """Return a stored settings blob as a dict, or None when it is unreadable.

    The whole row is rewritten to add one key, so a blob this cannot read must
    be LEFT ALONE rather than treated as empty - reading a value as ``{}`` and
    writing back would delete every other setting the workspace holds. A row
    skipped here simply keeps no ceiling, which the runtime already has an
    answer for.

    Args:
        stored: Whatever the driver returned for the JSON column.

    Returns:
        A mutable copy of the blob, or None when it is not a JSON object.
    """
    if stored is None:
        return {}
    if isinstance(stored, dict):
        return dict(stored)
    if isinstance(stored, str):
        # Belt and braces for a driver that hands JSON back as text.
        try:
            parsed = json.loads(stored)
        except ValueError:
            return None
        return dict(parsed) if isinstance(parsed, dict) else None
    return None


def seed_ceilings(bind: Connection, window: int) -> int:
    """Write ``window`` into every workspace row that has no ceiling of its own.

    Read-decide-write in Python rather than a dialect-specific JSON update: the
    blob is a plain JSON column, and expressing "add this key unless it is
    already there" in SQL means one statement per dialect. There are as many
    rows as tenants, so the loop costs nothing. It also costs the migration
    OFFLINE mode (``alembic upgrade --sql``), which cannot run a read; nothing
    in this repo generates SQL that way, and a dialect-specific JSON update is
    the price of getting it back.

    Locked for the duration, like every other read-merge-write over this blob.
    Migrations run inside the backend container against a database the previous
    release may still be serving, so a workspace-defaults save landing in that
    window would otherwise be overwritten wholesale. Ignored by sqlite, which
    has one writer anyway.

    Args:
        bind: The connection to run against.
        window: The ceiling to store where none is set.

    Returns:
        How many rows were seeded.
    """
    rows = bind.execute(
        sa.select(_workspace_settings.c.tenant_id, _workspace_settings.c.settings).with_for_update()
    ).all()
    seeded = 0
    for tenant_id, stored in rows:
        settings = read_blob(stored)
        if settings is None or settings.get(_CEILING_KEY) is not None:
            continue  # unreadable, or somebody chose this number already
        settings[_CEILING_KEY] = window
        bind.execute(
            _workspace_settings.update()
            .where(_workspace_settings.c.tenant_id == tenant_id)
            .values(settings=settings)
        )
        seeded += 1
    return seeded


def upgrade() -> None:
    """Seed each unset workspace ceiling from the tier this deployment serves."""
    seed_ceilings(op.get_bind(), seed_window(os.environ.get("MODEL_TIER")))


def downgrade() -> None:
    """Revert this revision.

    A no-op, deliberately. The upgrade cannot tell a value it wrote from one an
    operator has since edited, so un-seeding would mean deleting numbers people
    chose in order to undo one the deployment never noticed.

    Leaving the ceiling in place costs older code nothing either: every seeded
    value is the number that profile's own entries declared, so the clamp the
    old resolution applies to it lands on the window it was already asking for.
    """
