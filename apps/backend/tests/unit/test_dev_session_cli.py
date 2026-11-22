"""The session-minting terminal tool (`scripts/dev_session.py`).

It writes a CREDENTIAL, which makes it a different kind of script from its
siblings: the failure that matters is not a wrong number in a report, it is a
usable session existing somewhere nobody meant one to. So the tests here are
mostly about refusals, and each of them asserts the same two things - a
non-zero exit AND an empty store, because a message printed beside a session
that got written anyway is not a refusal.

The layout guard is the one with teeth. `DEPLOY_ENV` reaches the backend only
because the make target passes it, so the guard is the whole distance between
"a development affordance" and "a way to mint sessions on a server". Two tests
pin it: the production layout, and a layout that was never stated.

Each test loads the script FRESH, because it reads its arguments from the
environment at import time.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID
from app.db.seed import seed_defaults
from app.models import Tenant, User
from app.services.auth.principal import SESSION_COOKIE_NAME
from app.services.auth.store import SESSION_KEY_PREFIX, SessionRecord
from app.workers.queue import FakeJobQueue, JobQueueUnavailableError

MEMBER_EMAIL = "member@example.com"


def _repo_root() -> Path | None:
    """Walk up for the repo root, identified by the Makefile + script pair.

    Mirrors test_tool_check_cli.py: counting parents breaks inside the backend
    image, where the backend directory is the filesystem root.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "Makefile").is_file() and (parent / "scripts" / "dev_session.py").is_file():
            return parent
    return None


_ROOT = _repo_root()

pytestmark = pytest.mark.skipif(_ROOT is None, reason="no Makefile/scripts above this checkout")


class RecordingStore(FakeJobQueue):
    """The fake key-value store, plus the TTL it was handed and its close.

    `FakeJobQueue` deliberately ignores expiry - a test instance is as
    short-lived as the test holding it - but the requested window is exactly
    what one of these tests is about, so it is remembered here rather than
    simulated. `aclose` exists on the real Redis-backed queue only, and the
    script calls it, so the stand-in has to answer it too.
    """

    def __init__(self, *, unavailable: bool = False) -> None:
        super().__init__(unavailable=unavailable)
        self.ttls: dict[str, int] = {}
        self.closed = False

    async def kv_set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Record the window alongside the value."""
        await super().kv_set(key, value, ttl_seconds)
        self.ttls[key] = ttl_seconds

    async def aclose(self) -> None:
        """Match the real queue's release, so the script's `finally` has something to call."""
        self.closed = True

    def sessions(self) -> list[tuple[str, SessionRecord]]:
        """Every session written, parsed back through the app's own reader."""
        found = []
        for key, raw in self.values.items():
            if key.startswith(SESSION_KEY_PREFIX):
                record = SessionRecord.loads(raw)
                assert record is not None, f"unparseable session at {key}"
                found.append((key.removeprefix(SESSION_KEY_PREFIX), record))
        return found


class UnavailableStore(RecordingStore):
    """A store whose Redis is down, which is what a stopped stack looks like."""

    async def kv_set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Refuse the write the way the real queue refuses it."""
        raise JobQueueUnavailableError("The job queue is unavailable.")


def _engine(url: str) -> AsyncEngine:
    """A sqlite engine with foreign keys enforced, as the app's postgres does."""
    engine = create_async_engine(url)

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fks(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def _prepare(url: str, *, members: list[tuple[str, str]] | None = None) -> dict[str, UUID]:
    """Create the schema, seed the workspace, and add the named members.

    Args:
        url: The sqlite URL to build.
        members: (email, role) pairs to insert as signed-in people.

    Returns:
        Each added member's id, keyed by the email it was inserted with.
    """
    ids: dict[str, UUID] = {}

    async def work() -> None:
        engine = _engine(url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            maker = async_sessionmaker(engine, expire_on_commit=False)
            await seed_defaults(maker)
            async with maker() as session:
                for email, role in members or []:
                    user = User(
                        tenant_id=DEFAULT_TENANT_ID,
                        email=email,
                        display_name=email,
                        role=role,
                        oidc_issuer="https://issuer.example",
                        oidc_sub=f"sub-{email}",
                    )
                    session.add(user)
                    await session.flush()
                    ids[email] = user.id
                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(work())
    return ids


def _load_cli(monkeypatch: pytest.MonkeyPatch, **env: str) -> ModuleType:
    """Import the script fresh, with its arguments set in the environment.

    Fresh every time because DEPLOY_ENV, EMAIL and TTL are module-level
    constants read at import: a cached module would answer with whichever
    values the first test happened to set.
    """
    assert _ROOT is not None
    for key in ("DEPLOY_ENV", "DEV_SESSION_EMAIL", "DEV_SESSION_TTL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "cli_dev_session_under_test", _ROOT / "scripts" / "dev_session.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    members: list[tuple[str, str]] | None = None,
    store: RecordingStore | None = None,
    oidc: bool = True,
    **env: str,
) -> tuple[int, RecordingStore, dict[str, UUID]]:
    """Drive the REAL main() over a tmp sqlite database and an in-memory store."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'dev-session.db'}"
    ids = _prepare(url, members=members)
    settings = Settings(
        _env_file=None,
        model_provider="fake",
        **(
            {
                "oidc_issuer": "https://issuer.example",
                "oidc_client_id": "client",
                "oidc_client_secret": "secret",
                "oidc_admin_emails": "admin@example.com",
                "oauth_redirect_base_url": "http://localhost:3000",
            }
            if oidc
            else {}
        ),
    )

    cli = _load_cli(monkeypatch, **env)
    recorder = store if store is not None else RecordingStore()
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "build_store", lambda _settings: recorder)
    # The one seam standing in for the deployment's real database; everything
    # above it is the script's own logic, which is what these tests measure.
    monkeypatch.setattr(cli, "build_engine", lambda _settings: _engine(url))

    return cli.main(), recorder, ids


def test_mints_for_the_seeded_user_when_no_address_is_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, store, _ = _run(monkeypatch, tmp_path, DEPLOY_ENV="dev")

    assert code == 0
    sessions = store.sessions()
    assert len(sessions) == 1
    session_id, record = sessions[0]
    assert record.user_id == DEFAULT_USER_ID
    assert record.tenant_id == DEFAULT_TENANT_ID
    # The default window, and the store was released rather than leaked.
    assert store.ttls[f"{SESSION_KEY_PREFIX}{session_id}"] == 120 * 60
    assert store.closed is True

    out = capsys.readouterr().out
    # The printed forms must carry the real cookie name and the real value, or
    # what is pasted into a browser is not the session that was written.
    assert f'document.cookie = "{SESSION_COOKIE_NAME}={session_id};' in out
    assert f'curl -H "Cookie: {SESSION_COOKIE_NAME}={session_id}"' in out


def test_mints_for_a_named_member_normalizing_the_address(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # As it might be typed: padded and capitalized. Login lowercases before it
    # writes the row, so a lookup that did not normalize would miss this member.
    code, store, ids = _run(
        monkeypatch,
        tmp_path,
        members=[(MEMBER_EMAIL, "member")],
        DEPLOY_ENV="dev",
        DEV_SESSION_EMAIL="  Member@Example.COM  ",
    )

    assert code == 0
    sessions = store.sessions()
    assert len(sessions) == 1
    _, record = sessions[0]
    assert record.user_id == ids[MEMBER_EMAIL]
    assert record.user_id != DEFAULT_USER_ID
    assert record.tenant_id == DEFAULT_TENANT_ID


def test_ttl_is_honoured_in_minutes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, store, _ = _run(monkeypatch, tmp_path, DEPLOY_ENV="dev", DEV_SESSION_TTL="30")

    assert code == 0
    assert list(store.ttls.values()) == [30 * 60]


def test_a_nonsense_ttl_mints_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, store, _ = _run(monkeypatch, tmp_path, DEPLOY_ENV="dev", DEV_SESSION_TTL="soon")

    assert code != 0
    assert store.sessions() == []


def test_an_unknown_address_refuses_and_names_the_members_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, store, _ = _run(
        monkeypatch,
        tmp_path,
        members=[(MEMBER_EMAIL, "member")],
        DEPLOY_ENV="dev",
        DEV_SESSION_EMAIL="stranger@example.com",
    )

    assert code != 0
    assert store.sessions() == []
    err = capsys.readouterr().err
    assert "stranger@example.com" in err
    # The person has to be told WHERE the answer is, or the error is a dead end.
    assert "Members" in err and "/api/v1/members" in err


def test_refuses_under_the_production_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The guard this tool exists behind: a production stack mints nothing."""
    code, store, _ = _run(monkeypatch, tmp_path, DEPLOY_ENV="prod")

    assert code != 0
    assert store.sessions() == []
    err = capsys.readouterr().err
    assert "DEPLOY_ENV=prod" in err


def test_refuses_when_the_layout_is_not_stated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fails CLOSED: an unstated layout is refused, not assumed to be development."""
    code, store, _ = _run(monkeypatch, tmp_path)

    assert code != 0
    assert store.sessions() == []
    assert "DEPLOY_ENV is unset" in capsys.readouterr().err


def test_refuses_a_layout_it_does_not_recognise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, store, _ = _run(monkeypatch, tmp_path, DEPLOY_ENV="staging")

    assert code != 0
    assert store.sessions() == []


def test_an_unreachable_store_is_a_message_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, store, _ = _run(monkeypatch, tmp_path, store=UnavailableStore(), DEPLOY_ENV="dev")

    assert code != 0
    assert store.sessions() == []
    assert "unreachable" in capsys.readouterr().err


def test_says_so_when_sign_in_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A session minted where nothing reads sessions is the one that wastes an hour."""
    code, _, _ = _run(monkeypatch, tmp_path, oidc=False, DEPLOY_ENV="dev")

    assert code == 0
    out = capsys.readouterr().out
    assert "sign-in is OFF" in out


def test_sign_in_note_is_absent_when_sign_in_is_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, _ = _run(monkeypatch, tmp_path, DEPLOY_ENV="dev")

    assert code == 0
    assert "sign-in is OFF" not in capsys.readouterr().out


def test_a_member_of_another_workspace_is_not_reachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The roster read is tenant-scoped, so a foreign row is simply not there."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'dev-session.db'}"
    _prepare(url)

    async def add_foreign() -> None:
        engine = _engine(url)
        try:
            maker = async_sessionmaker(engine, expire_on_commit=False)
            async with maker() as session:
                other_tenant = Tenant(id=uuid4(), name="somebody else")
                session.add(other_tenant)
                session.add(
                    User(
                        tenant_id=other_tenant.id,
                        email="outsider@example.com",
                        display_name="outsider",
                        role="admin",
                        oidc_issuer="https://issuer.example",
                        oidc_sub="sub-outsider",
                    )
                )
                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(add_foreign())

    settings = Settings(_env_file=None, model_provider="fake")
    cli = _load_cli(monkeypatch, DEPLOY_ENV="dev", DEV_SESSION_EMAIL="outsider@example.com")
    store = RecordingStore()
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "build_store", lambda _settings: store)
    monkeypatch.setattr(cli, "build_engine", lambda _settings: _engine(url))

    assert cli.main() != 0
    assert store.sessions() == []
    # Not-found, specifically - a read that blew up would also exit non-zero and
    # would prove nothing about the scoping.
    assert "no member of this workspace" in capsys.readouterr().err
