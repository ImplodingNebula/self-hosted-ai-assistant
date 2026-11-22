"""The migration that seeds each workspace's context ceiling from its old tier.

The window an unsized tag runs at moved out of the model profile and into
deployment configuration, and ``DEFAULT_CONTEXT_WINDOW`` ships at the smallest
profile's number. A deployment on a larger profile would therefore shrink on
upgrade, so revision ``e2f3a4b5c6d7`` writes the number it was already serving
into the workspace's own ceiling.

The migration is exercised directly rather than through ``alembic upgrade``:
the test database is built from the SQLModel metadata, so the revision chain
never runs here. Its two decisions - which window a tier seeds, and which rows
may be written - are therefore imported and driven against a real table.

The guard that matters most is the one that does NOT write: a ceiling somebody
chose outranks anything a migration has to say, and the whole settings blob is
rewritten to add one key, so a row this reads wrongly is a row whose other
settings are gone.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa

from app.core.config import Settings

_REVISION = "20260809_e2f3a4b5c6d7_seed_workspace_context_ceiling.py"


def _load_migration() -> ModuleType:
    """Import the revision module by path (its filename is not an identifier)."""
    path = Path(__file__).resolve().parents[2] / "alembic" / "versions" / _REVISION
    spec = importlib.util.spec_from_file_location("_ceiling_seed_revision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _load_migration()

# A stand-in for the real table, carrying only what the seed touches. Declared
# here rather than imported from the app models so the test pins the migration's
# own view of the column - which is the thing that has to keep working after the
# app's model moves on.
_metadata = sa.MetaData()
_workspace_settings = sa.Table(
    "workspace_settings",
    _metadata,
    sa.Column("tenant_id", sa.Uuid(), primary_key=True),
    sa.Column("settings", sa.JSON(), nullable=False),
)


@pytest.fixture
def connection():
    """An in-memory database holding just the table the migration writes."""
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as conn:
        _metadata.create_all(conn)
        yield conn
    engine.dispose()


def _insert(conn, settings: object, tenant_id: UUID | None = None) -> UUID:
    # Typed `object`, not `dict`, so a test can store a blob that is valid JSON
    # but not a settings object - the shape the seed has to refuse to touch.
    tenant_id = tenant_id or uuid4()
    conn.execute(_workspace_settings.insert().values(tenant_id=tenant_id, settings=settings))
    return tenant_id


def _stored(conn, tenant_id: UUID) -> dict:
    blob = conn.execute(
        sa.select(_workspace_settings.c.settings).where(
            _workspace_settings.c.tenant_id == tenant_id
        )
    ).scalar_one()
    return json.loads(blob) if isinstance(blob, str) else blob


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("dev", 8192),
        ("dev-qwen35", 8192),
        ("mid", 16384),
        ("server-t4", 16384),
        ("server-a10", 32768),
    ],
)
def test_each_tier_seeds_the_window_it_was_already_serving(tier: str, expected: int) -> None:
    # HISTORY, not configuration. These literals describe what each shipped
    # hardware profile was already serving on the day this revision was
    # written, so they stay frozen even now that the profiles themselves are
    # gone: a database upgrading from that release still has to land on the
    # window it went to sleep with, and nothing else in the tree can still say
    # what that was.
    assert migration.seed_window(tier) == expected


@pytest.mark.parametrize("tier", [None, "", "   ", "laptop", "server-h100"])
def test_an_unset_or_unknown_tier_seeds_the_settings_default(tier: str | None) -> None:
    # A deployment the migration cannot place is left exactly where the code
    # would have left it anyway - and after this slice that is EVERY fresh box,
    # because nothing sets MODEL_TIER any more. Read off the SETTING rather
    # than written as a literal: unlike the frozen numbers above, this one is a
    # live-agreement claim - the two arms of the new resolution have to answer
    # the same, and the setting's default is free to move.
    assert migration.seed_window(tier) == Settings(_env_file=None).default_context_window


def test_upgrade_seeds_the_deployment_default_when_no_tier_is_set(
    connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real entry point, on the box this slice creates: MODEL_TIER unset.

    ``seed_window(None)`` above tests the RULE; this drives the environment
    READ that feeds it, which is the line that has to keep working now that no
    settings field, template key or make target names a tier. A revision that
    reached for ``settings.model_tier`` instead would raise here rather than
    seed, and an ``upgrade`` that skipped the seed on an unset variable would
    leave the row without a ceiling - both of which look identical from
    ``seed_window``'s side.
    """
    monkeypatch.delenv("MODEL_TIER", raising=False)
    monkeypatch.setattr(migration.op, "get_bind", lambda: connection)
    tenant = _insert(connection, {"model": "qwen3:4b-instruct"})

    migration.upgrade()

    stored = _stored(connection, tenant)
    assert stored["context_ceiling"] == Settings(_env_file=None).default_context_window
    assert stored["model"] == "qwen3:4b-instruct"


def test_a_row_with_no_ceiling_is_seeded(connection) -> None:
    tenant = _insert(connection, {"model": "qwen3:4b-instruct"})

    assert migration.seed_ceilings(connection, 16384) == 1

    stored = _stored(connection, tenant)
    assert stored["context_ceiling"] == 16384
    # The other settings ride along: the whole blob is rewritten to add one key.
    assert stored["model"] == "qwen3:4b-instruct"


def test_an_empty_blob_is_seeded(connection) -> None:
    # The shape of a freshly seeded deployment: the boot seed inserts a row with
    # an empty settings blob, and it is exactly the row this migration is for.
    tenant = _insert(connection, {})

    assert migration.seed_ceilings(connection, 32768) == 1
    assert _stored(connection, tenant) == {"context_ceiling": 32768}


def test_a_ceiling_somebody_chose_is_never_overwritten(connection) -> None:
    # The guard. A stored ceiling is a decision an operator made on the settings
    # page, and a migration that "restores" the tier default over it would
    # silently undo that decision on the next deploy.
    tenant = _insert(connection, {"context_ceiling": 4096, "reranker": False})

    assert migration.seed_ceilings(connection, 32768) == 0

    stored = _stored(connection, tenant)
    assert stored == {"context_ceiling": 4096, "reranker": False}


def test_a_null_ceiling_counts_as_unset(connection) -> None:
    # An explicit null is how the PATCH endpoint CLEARS the field, and a cleared
    # ceiling is not a chosen one - it is the absence the seed exists to fill.
    tenant = _insert(connection, {"context_ceiling": None})

    assert migration.seed_ceilings(connection, 16384) == 1
    assert _stored(connection, tenant)["context_ceiling"] == 16384


def test_every_tenant_with_no_ceiling_is_seeded(connection) -> None:
    # workspace_settings is per-tenant, so the seed is per-row. Mixed input, so
    # a version that stopped at the first row - or swept them all - is visible.
    unset_a = _insert(connection, {})
    chosen = _insert(connection, {"context_ceiling": 4096})
    unset_b = _insert(connection, {"thinking": "auto"})

    assert migration.seed_ceilings(connection, 16384) == 2

    assert _stored(connection, unset_a)["context_ceiling"] == 16384
    assert _stored(connection, unset_b)["context_ceiling"] == 16384
    assert _stored(connection, chosen)["context_ceiling"] == 4096


def test_no_rows_is_not_an_error(connection) -> None:
    # A database migrated before the boot seed has ever run. There is nothing to
    # seed and nothing to fix: an absent row reads as an unset ceiling, and the
    # runtime's DEFAULT_CONTEXT_WINDOW arm is the answer such a deployment wants.
    assert migration.seed_ceilings(connection, 16384) == 0


def test_an_unreadable_row_is_left_byte_for_byte_alone(connection) -> None:
    # The guard where it ACTS, not one layer below it. The seed rewrites the
    # WHOLE blob, so a value it cannot parse must be skipped rather than read
    # as empty: treating it as {} and writing back would delete every setting
    # the workspace holds. Driven through seed_ceilings so that the natural
    # bug - `read_blob(stored) or {}` - is what fails this, since read_blob
    # itself keeps answering correctly under it.
    unreadable = _insert(connection, [1, 2, 3])
    readable = _insert(connection, {})

    assert migration.seed_ceilings(connection, 16384) == 1

    assert _stored(connection, unreadable) == [1, 2, 3]
    assert _stored(connection, readable)["context_ceiling"] == 16384


def test_the_blob_reader_accepts_only_json_objects() -> None:
    # The unit beneath the row test above, covering the shapes a driver can
    # hand back that no fixture can conveniently store.
    assert migration.read_blob("not json at all") is None
    assert migration.read_blob("[1, 2, 3]") is None
    assert migration.read_blob(12345) is None
    # And the readable shapes, including JSON handed back as text by a driver.
    assert migration.read_blob(None) == {}
    assert migration.read_blob({"context_ceiling": 4096}) == {"context_ceiling": 4096}
    assert migration.read_blob('{"context_ceiling": 4096}') == {"context_ceiling": 4096}
