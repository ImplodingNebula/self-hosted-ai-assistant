"""The terminal diagnostic (`scripts/tool_check.py`) as a WRITER of verdicts.

`make tool-check` and the console's verify endpoint append to the same
`model_probes` log, and every row carries the running `harness_version` - which
is a promise that any two rows under it measured the same thing. These tests
exist because that promise is only kept if both writers resolve a sample's
serve-time options the same way, and the terminal one is the writer nothing
else covers: it is a script rather than a service, so no route test reaches it.

The failure they pin is not hypothetical and not cosmetic. Skip the stored
declaration here and a tag whose operator confirmed it always reasons gets
driven with the flag cleared - it reasons anyway, the raw `<think>` block lands
in the visible answer, `must-not-call` and `continuation` fail on it, and the
run writes `deterministic`. A failure verdict clamps regardless of harness
version and regardless of which host produced it, so that row demotes a model
the console certified minutes earlier, under the same version, with no way for
a reader to tell which of the two is describing the deployment.

Each test loads the script FRESH, because its flags are read from the
environment at import time.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.core.config import Settings
from app.db.seed import seed_defaults
from app.models.entities import ModelPolicy, ModelProbe
from app.providers.base import ChatChunk, ModelCapabilities, ModelInfo, ToolCall, Usage
from app.providers.fake_provider import FakeProvider
from app.repositories.model_probes import ModelProbeRepository
from app.services.tool_check import THINK_CASE_NAMES

CHAT = "chat-tag"
CHAT_DIGEST = "sha256:chatweights"
# The spelling a runtime may list the SAME weights under. Configuration and
# listing disagreeing about a version suffix is the repo's own worked example,
# and it is the one thing that decides whether a stored declaration is found.
CHAT_LISTED = "chat-tag:latest"
# A partner an operator paired from the console. Nothing else names a pair, so
# this tag can only ever be reached through the stored declaration.
PARTNER = "partner-tag"
PARTNER_DIGEST = "sha256:partnerweights"

# The runtime reports a template with no think plumbing, which resolves the tag
# to NEVER-thinking - and that is what makes the stored declaration below
# visible: without it the flag is omitted, with it the flag is forced on, and
# the two are trivially distinguishable in a call log.
_NO_THINK_TEMPLATE = ModelCapabilities(completion=True, tools=True, template_thinking=False)


def _repo_root() -> Path | None:
    """Walk up for the repo root, identified by the Makefile + script pair.

    Mirrors test_remote_guard.py: counting parents breaks inside the backend
    image, where the backend directory is the filesystem root.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "Makefile").is_file() and (parent / "scripts" / "tool_check.py").is_file():
            return parent
    return None


_ROOT = _repo_root()

pytestmark = pytest.mark.skipif(_ROOT is None, reason="no Makefile/scripts above this checkout")


class RecordingProvider(FakeProvider):
    """A model that passes the fast cases, recording the flag each call carried.

    Behaves like a well-mannered small model: answer arithmetic directly, answer
    once a tool result comes back, otherwise call the first offered tag. It
    reasons when asked to, so the reasoning-with-tools case measures what its
    name says.

    Every tag it holds reports a template with no think plumbing, so the
    resolution that runs when nothing is declared is a definite ``never``
    rather than the ``unknown`` an unreported template would leave - which is
    what keeps the declared and undeclared arms distinguishable.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("models", [ModelInfo(name=CHAT, digest=CHAT_DIGEST)])
        kwargs.setdefault(
            "capabilities_by_model",
            dict.fromkeys((CHAT, CHAT_LISTED, PARTNER), _NO_THINK_TEMPLATE),
        )
        super().__init__(**kwargs)
        self.thinks: list[bool | None] = []
        # Which TAG each generation was driven against, kept apart from the
        # flags because it answers the other half of "what did this run
        # measure" - the pair's reasoning case is the only sample that leaves
        # the tag under check, and nothing else here would show that it did.
        self.drove: list[str] = []

    async def chat_stream(self, messages, *, model=None, tools=None, options=None):
        self.thinks.append(options.think if options is not None else None)
        self.drove.append(model or "")
        if options is not None and options.think:
            yield ChatChunk(type="thinking", delta="thinking it through")
        offered = [tool["function"]["name"] for tool in tools or []]
        user = next(m.content for m in reversed(messages) if m.role == "user")
        if messages[-1].role == "tool":
            yield ChatChunk(type="token", delta="Answered from ")
            yield ChatChunk(type="token", delta="the tool result.")
        elif "2 + 2" in user or not offered:
            yield ChatChunk(type="token", delta="4")
            yield ChatChunk(type="token", delta=" exactly.")
        else:
            yield ChatChunk(
                type="tool_call",
                tool_call=ToolCall(id="t1", name=offered[0], arguments={"q": user}),
            )
        yield ChatChunk(type="done", finish_reason="stop", usage=Usage())


def _engine(url: str) -> AsyncEngine:
    """An engine over a FILE-backed sqlite, safe to build once per event loop.

    File-backed rather than in-memory because the script runs three separate
    ``asyncio.run`` calls and an async engine binds to the loop that created
    it: each call gets its own engine over the same file, exactly as the real
    script gets its own engine over the same postgres.
    """
    engine = create_async_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fks(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def _prepare(url: str, *, declare_always: bool, pairing: tuple[str, str] | None = None) -> None:
    """Create the schema, seed the tenant, and optionally store declarations.

    Args:
        url: The sqlite URL the run will write to.
        declare_always: Store a confirmed ``always`` think mode for the chat tag.
        pairing: ``(model key, partner tag)``. The key is a parameter because
            declarations are stored under the name the PROVIDER lists, which is
            not always the name the deployment configures - and reading them
            under the wrong one is a failure worth being able to reproduce.
    """

    async def work() -> None:
        engine = _engine(url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            maker = async_sessionmaker(engine, expire_on_commit=False)
            await seed_defaults(maker)
            rows: list[ModelPolicy] = []
            if declare_always:
                rows.append(
                    ModelPolicy(
                        provider="fake", model=CHAT, digest=CHAT_DIGEST, think_mode="always"
                    )
                )
            if pairing is not None:
                key, partner = pairing
                rows.append(
                    ModelPolicy(
                        provider="fake", model=key, digest=CHAT_DIGEST, thinking_model=partner
                    )
                )
            if rows:
                async with maker() as session:
                    for row in rows:
                        session.add(row)
                    await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(work())


def _rows(url: str) -> list[ModelProbe]:
    """Read back what the run recorded."""

    async def work() -> list[ModelProbe]:
        engine = _engine(url)
        try:
            maker = async_sessionmaker(engine, expire_on_commit=False)
            async with maker() as session:
                return await ModelProbeRepository(session).history("fake", CHAT)
        finally:
            await engine.dispose()

    return asyncio.run(work())


def _load_cli(monkeypatch: pytest.MonkeyPatch, **env: str) -> ModuleType:
    """Import the script fresh, with its environment flags set.

    Fresh every time because DEEP / AS_JSON / NO_DB are module-level constants
    read at import: a cached module would answer with whichever flags the first
    test happened to set.
    """
    assert _ROOT is not None
    for flag in ("TOOL_CHECK_DEEP", "TOOL_CHECK_JSON", "TOOL_CHECK_NO_DB"):
        monkeypatch.delenv(flag, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "cli_tool_check_under_test", _ROOT / "scripts" / "tool_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    declare_always: bool = True,
    no_db: bool = False,
    deep: bool = False,
    pairing: tuple[str, str] | None = None,
    installed: list[ModelInfo] | None = None,
) -> tuple[RecordingProvider, str, int]:
    """Drive the REAL main() over a tmp sqlite database.

    Returns the provider it drove (for the flags and tags every call carried),
    the database URL (to read back what was written), and main()'s exit code.
    """
    url = f"sqlite+aiosqlite:///{(tmp_path / 'probe.db').as_posix()}"
    _prepare(url, declare_always=declare_always, pairing=pairing)

    settings = Settings(
        model_provider="fake",
        default_chat_model=CHAT,
        thinking_chat_model=None,
        _env_file=None,
    )
    provider = RecordingProvider(**({"models": installed} if installed else {}))
    flags = {}
    if no_db:
        flags["TOOL_CHECK_NO_DB"] = "1"
    if deep:
        flags["TOOL_CHECK_DEEP"] = "1"
    cli = _load_cli(monkeypatch, **flags)
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "build_model_provider", lambda _settings: provider)
    # The one seam standing in for the deployment's real database. Everything
    # above it - which engine gets built, when, and whether the fetchers are
    # wired off it - is the script's own logic and is what these tests measure.
    monkeypatch.setattr(cli, "build_engine", lambda _settings: _engine(url))

    return provider, url, cli.main()


def _tool_flags(provider: RecordingProvider) -> set[bool | None]:
    """The flags the TOOL generations carried.

    The thinking group sends its flag literally in BOTH directions on the same
    tag, so its two samples always contribute True and False whatever the
    resolution does. Reading the raw call log without excluding them would
    therefore pass under every mutation - the exact hollow assertion this helper
    exists to prevent. A fast run is the thinking pair (one generation each)
    followed by the tool cases, so dropping the first two leaves the tool calls.
    """
    return set(provider.thinks[len(THINK_CASE_NAMES) :])


def test_a_terminal_run_drives_the_flag_the_stored_declaration_gives_it(
    monkeypatch, tmp_path
) -> None:
    # The wiring, end to end through main(). The runtime's template says this
    # tag cannot reason; an operator has confirmed that it always does. A turn
    # sends think:true, so a probe must too - unwire either fetcher in the
    # script and the tool cases come back with the flag OMITTED, which is a
    # measurement of a configuration nothing serves and a clamp written under
    # the harness version the console certifies under.
    provider, url, code = _run_cli(monkeypatch, tmp_path)

    assert code == 0
    assert _tool_flags(provider) == {True}
    # And it recorded what it ran, so the row says which resolution wrote it.
    rows = _rows(url)
    assert len(rows) == 1
    assert {sample["think"] for sample in rows[0].cases["samples"]} == {True, False}


def test_no_db_keeps_the_sessionless_half_and_writes_nothing(monkeypatch, tmp_path) -> None:
    # The documented arm, and the reason the wiring above is conditional at all:
    # TOOL_CHECK_NO_DB exists for a stack whose database is down, so a
    # resolution that required one would not survive the flag that says it may
    # not have one. The SAME stored declaration is present and deliberately not
    # read - the runtime's own template report stands, so the flag is omitted -
    # and nothing is appended to the log.
    provider, url, code = _run_cli(monkeypatch, tmp_path, no_db=True)

    assert code == 0
    assert _tool_flags(provider) == {None}
    assert _rows(url) == []


def test_with_nothing_declared_the_two_arms_agree(monkeypatch, tmp_path) -> None:
    # The control that keeps the two tests above honest: with no declaration
    # stored, reading the database changes nothing, so a difference between the
    # arms could only come from the sessionless rungs they share. Both omit the
    # flag, because that is what the runtime's template report resolves to.
    wired, _url, wired_code = _run_cli(monkeypatch, tmp_path, declare_always=False)
    assert wired_code == 0
    assert _tool_flags(wired) == {None}


# --- the PARTNER a terminal run drives ----------------------------------------
#
# The same promise on the other axis. A reasoning-with-tools certificate binds
# to the partner's weights, so the tag this script drives for that case has to
# be the tag the console's job drives and the tag a turn routes to - the one
# shared ranking, stored pairing included. Drive a different one and the row is
# unreadable: it names weights nothing serves, so the binding never matches and
# the capability is withheld with nothing anywhere saying why.

_PAIRED = [ModelInfo(name=CHAT, digest=CHAT_DIGEST), ModelInfo(name=PARTNER, digest=PARTNER_DIGEST)]


def test_a_terminal_run_drives_the_partner_an_operator_declared(monkeypatch, tmp_path) -> None:
    # A stored pairing is the ONLY thing that names a partner here - the
    # environment override is unset - so before the stored rung was read this
    # script drove the tag under check for the reasoning case and recorded no
    # partner digest, while the console's job, reading the declaration, drove
    # and recorded the partner. Two rows under one harness version, describing
    # two different pairs.
    provider, url, code = _run_cli(
        monkeypatch, tmp_path, deep=True, pairing=(CHAT, PARTNER), installed=_PAIRED
    )

    assert code == 0
    assert PARTNER in provider.drove
    rows = _rows(url)
    assert len(rows) == 1
    assert rows[0].cases["thinking_digest"] == PARTNER_DIGEST


def test_the_stored_pairing_is_found_under_the_spelling_the_writer_used(
    monkeypatch, tmp_path
) -> None:
    # Declarations are keyed on the name the PROVIDER lists, and this script
    # holds the CONFIGURED one - so the read has to canonicalize between them or
    # it silently finds nothing. Here the deployment configures "chat-tag" and
    # the runtime lists "chat-tag:latest", which is one model written two ways.
    listed = [
        ModelInfo(name=CHAT_LISTED, digest=CHAT_DIGEST),
        ModelInfo(name=PARTNER, digest=PARTNER_DIGEST),
    ]
    provider, _url, code = _run_cli(
        monkeypatch, tmp_path, deep=True, pairing=(CHAT_LISTED, PARTNER), installed=listed
    )

    assert code == 0
    assert PARTNER in provider.drove


def test_no_db_leaves_the_partner_unpaired_too(monkeypatch, tmp_path) -> None:
    # The documented arm, applied to this rung as well: TOOL_CHECK_NO_DB exists
    # for a stack whose database is down, so the SAME stored pairing is present
    # and deliberately not read. Nothing else names a partner, so the run stays
    # on the tag under check.
    provider, _url, code = _run_cli(
        monkeypatch,
        tmp_path,
        deep=True,
        no_db=True,
        pairing=(CHAT, PARTNER),
        installed=_PAIRED,
    )

    assert code == 0
    assert PARTNER not in provider.drove
