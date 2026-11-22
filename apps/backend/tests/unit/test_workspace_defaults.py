"""The workspace-defaults writer, and proof each field it writes is read.

Two halves, and the second is the point. The first covers the write surface -
the per-field merge, the tri-state, the validation, the admin gate. The second
drives a REAL chat turn (or the real warm / meter endpoint) for each of the
six fields and watches the stored value move behavior, because a settings page
whose value nothing consults is a control that lies about having done
something.

Each read test is written to fail if its wiring is removed:

- ``model`` - asserts the pair SIBLING (``qwen3:4b``), never the deployment's
  own default, so a stored pick that stopped reaching ``resolve_chat_models``
  cannot pass by accident.
- ``thinking`` - asserts the think flag on the wire, which only the stored
  default can raise on a message the heuristic reads as ordinary.
- ``web_search`` - asserts the search provider was never consulted AND that no
  web tool was offered, the two halves of "off means fully local".
- ``system_prompt`` - asserts the built-in persona is ABSENT as well as the
  stored one present, so a composition that appended rather than replaced
  cannot pass; the meter case asserts the reading FALLS, which an estimate
  still composing the default could not do.
- ``context_ceiling`` - asserts a window strictly smaller than the 8192 the
  runtime reports, so an unclamped resolution fails the assertion outright.
- ``reranker`` - asserts a wired reranker was never called, against a control
  in the same file proving it is called when the workspace says nothing.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_tenant, get_model_provider, get_settings
from app.core.concurrency import StreamLimiter
from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.providers.base import ModelCapabilities, ModelInfo, ToolCall
from app.providers.fake_provider import FakeProvider, FakeTurn
from app.providers.rag.base import Reranker
from app.providers.rag.embedder import FakeEmbedder
from app.providers.search.base import SearchResult
from app.providers.search.searxng import FakeSearchProvider
from app.repositories.workspace_settings import WorkspaceSettingsRepository
from app.schemas.settings import (
    MAX_CONTEXT_CEILING,
    MAX_SYSTEM_PROMPT_CHARS,
    MIN_CONTEXT_CEILING,
    WorkspaceDefaults,
    WorkspaceDefaultsPatch,
)
from app.services.context_budget import (
    ANSWER_RESERVE_TOKENS,
    THINKING_ANSWER_RESERVE_TOKENS,
    approx_tokens,
    history_budget_tokens,
)
from app.services.model_facts import clamp_num_ctx, clear_facts_cache
from app.services.model_manager import build_manager_view
from app.services.prompts import DEFAULT_SYSTEM_PROMPT
from app.services.workspace_defaults import (
    apply_reranker_default,
    load_defaults,
    reranker_availability,
    set_default_model,
)
from app.storage.local import FakeStorageBackend
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store
from tests.unit.test_chat_agency import (
    CHAT_MODEL,
    KB_TEXT,
    THINKING_MODEL,
    collect_events,
    make_settings,
    offered_names,
    provisioned,
    seed_agentic_verdict,
    seed_kb,
    status_codes,
)
from tests.unit.test_chat_mcp_tools import FakeMcpSource, RemoteTool

DEFAULTS_URL = "/api/v1/workspace/defaults"
CHAT_URL = "/api/v1/chat/stream"
TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")

# The window the model server REPORTS for these weights, which is the only
# thing a window may never exceed now that no committed file sizes a tag. Every
# ceiling assertion below is strictly under it, so an unclamped resolution
# cannot pass - and the deployment default is set to match, so a ceiling is the
# only thing that can move the number either way.
NATIVE_WINDOW = 8192

# A tag nothing pairs and nothing describes, for the manager-card cases: it has
# to take its window from the runtime under this deployment's affordance rather
# than from anything written down.
STRAY_CHAT = "stray:7b"
STRAY_EMBED = "stray-embed:1b"

# A question the thinking heuristic reads as ordinary (short, no hint verb, one
# question mark), so only a stored default can turn reasoning on.
PLAIN_QUESTION = "hello there"
# A question the freshness heuristic WOULD search on, so a stored off is the
# only thing that can keep the turn local.
FRESH_QUESTION = "What is the latest python version today?"

# A stored persona, deliberately far SHORTER than the built-in one so the meter
# assertion below can only pass if the stored text really replaced it.
PERSONA = "Answer as the workspace's own support desk."


@pytest.fixture(autouse=True)
def _fresh_facts_cache():
    # Every window on this page is resolved from a runtime report, and the
    # reports are cached per (provider, tag) across tests. A leaked one would
    # size a turn from another test's fake.
    clear_facts_cache()
    yield
    clear_facts_cache()


def sized(**kwargs) -> FakeProvider:
    """The configured pair, reporting a native window of :data:`NATIVE_WINDOW`.

    Windows resolve runtime-first, so a fake that reports no window at all
    leaves the deployment's affordance as the only input - and then a ceiling
    ABOVE the default would raise the window instead of being ignored, which is
    the opposite of what the ceiling tests are about.

    Args:
        **kwargs: Passed through to the shared ``provisioned`` fake.

    Returns:
        The fake, reporting both pair members at the native window.
    """
    report = ModelCapabilities(tools=True, completion=True, context_length=NATIVE_WINDOW)
    kwargs.setdefault("capabilities_by_model", {CHAT_MODEL: report, THINKING_MODEL: report})
    return provisioned(**kwargs)


class RecordingReranker(Reranker):
    """A reranker that records every call and changes nothing about the order."""

    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query, chunks, top_k):  # type: ignore[override]
        self.calls += 1
        return list(chunks)[:top_k]


@asynccontextmanager
async def defaults_client(
    provider,
    db_sessionmaker,
    settings,
    *,
    store=None,
    search=None,
    reranker=None,
    mcp_sources=None,
    tenant=None,
):
    """App-wired client, optionally carrying the RAG, web and MCP collaborators.

    A local twin of ``test_chat_agency.agency_client`` because these tests need
    two things it does not expose: an injectable reranker (the subject of three
    of them) and an overridable principal (the subject of the admin gate).
    """
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    if tenant is not None:
        app.dependency_overrides[get_current_tenant] = lambda: tenant
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    app.state.job_queue = FakeJobQueue()
    app.state.vector_store = store if store is not None else make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    # Always a fake: the real backend writes to disk, and a unit test must never
    # reach the developer's filesystem.
    app.state.storage = FakeStorageBackend()
    if search is not None:
        app.state.search_provider = search
    if reranker is not None:
        app.state.reranker = reranker
    if mcp_sources is not None:
        app.state.mcp_tool_sources = list(mcp_sources)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def stored(db_sessionmaker) -> dict:
    """The raw stored blob, so a test can see keys the typed view would drop."""
    async with db_sessionmaker() as session:
        return await WorkspaceSettingsRepository(session, TENANT).load()


async def store_defaults(db_sessionmaker, values: dict) -> None:
    """Seed the workspace defaults directly, bypassing the API under test."""
    async with db_sessionmaker() as session:
        await WorkspaceSettingsRepository(session, TENANT).merge_defaults(values)
        await session.commit()


# --------------------------------------------------------------------------
# The pure rules
# --------------------------------------------------------------------------


def test_the_floor_alone_does_not_make_the_smallest_window_safe() -> None:
    """The floor clears the answer reserve - and that was never the binding limit.

    An earlier version of this test asserted only ``MIN > reserve``, which
    passed while the real overflow was live: the reserve is 2048 and the
    default HISTORY_TOKEN_BUDGET is 4000, so history alone can outspend the
    smallest window a ceiling may set. The window arithmetic is what actually
    protects the prompt, so this states BOTH halves - the floor's own margin,
    and the fact that the configured history budget does not fit inside it,
    which is exactly why the budget has to be derived from the window.
    """
    assert MIN_CONTEXT_CEILING > THINKING_ANSWER_RESERVE_TOKENS
    # If this ever stops being true, the window-aware trim below stops being
    # load-bearing at the floor - and someone should notice deliberately.
    assert Settings(_env_file=None).history_token_budget > MIN_CONTEXT_CEILING - (
        THINKING_ANSWER_RESERVE_TOKENS
    )


def test_history_budget_is_derived_from_the_window_not_the_setting() -> None:
    """The configured budget is a ceiling; the window is the binding limit."""
    # Roomy window: the setting is what bites.
    assert history_budget_tokens(4000, num_ctx=32768, system_tokens=500, message_tokens=50) == 4000
    # The smallest window a ceiling may set: 4096 - 500 - 50 - 1024 = 2522.
    assert history_budget_tokens(4000, num_ctx=4096, system_tokens=500, message_tokens=50) == 2522
    # A thinking turn reserves more, so history gets less.
    assert (
        history_budget_tokens(
            4000,
            num_ctx=4096,
            system_tokens=500,
            message_tokens=50,
            reserve_tokens=THINKING_ANSWER_RESERVE_TOKENS,
        )
        == 1498
    )
    # Never negative: a window with no room for history yields no history,
    # rather than a budget the caller would read as unlimited.
    assert history_budget_tokens(4000, num_ctx=4096, system_tokens=4000, message_tokens=50) == 0


def test_clamp_num_ctx_only_ever_clamps_down() -> None:
    """A ceiling above a resolved window changes nothing; below it wins.

    Scoped to this function, not a claim about the whole ceiling. It is the
    final clamp applied to whatever the resolution granted - a number a setting
    may narrow and never widen. The resolution ITSELF treats the ceiling as the
    deployment's affordance, which can move an unsized tag's window in both
    directions bounded by the runtime's native figure; that half lives in
    ``model_facts._unsized_window`` and is pinned in ``test_model_facts.py``.
    """
    assert clamp_num_ctx(8192, None) == 8192
    assert clamp_num_ctx(8192, 4096) == 4096
    # The escalation this rule exists to refuse: a settings page must not be
    # able to hand a turn more window than the model was declared to have.
    assert clamp_num_ctx(8192, 131072) == 8192


def test_apply_reranker_default_withholds_only_on_an_explicit_false() -> None:
    """False takes a wired reranker away; nothing else does, and None stays None."""
    reranker = RecordingReranker()
    assert apply_reranker_default(reranker, WorkspaceDefaults(reranker=False)) is None
    assert apply_reranker_default(reranker, WorkspaceDefaults(reranker=True)) is reranker
    assert apply_reranker_default(reranker, WorkspaceDefaults()) is reranker
    # An enabling default never conjures a sidecar this deployment lacks.
    assert apply_reranker_default(None, WorkspaceDefaults(reranker=True)) is None


def test_reranker_availability_is_decided_by_the_url_alone() -> None:
    """One gate, two states, and each says which one it is.

    Two more gates used to have to AGREE with the URL, both lines in a
    committed per-deployment file: one naming the reranker and one flag saying
    whether it was on. The shipped configuration was a disagreement - a live
    URL beside a declaration that disabled reranking - so a sidecar somebody
    had deliberately started did nothing, and the page explained it by naming a
    file. Restore either as a gate and this test fails on a tuple.
    """
    unwired = reranker_availability(make_settings())
    assert (unwired.configured, unwired.model) == (False, None)
    # The sentence names the key to set, and nothing sends a reader to a file.
    assert "RERANKER_URL" in unwired.detail
    assert "profile" not in unwired.detail

    wired = reranker_availability(make_settings(reranker_url="http://reranker:7997"))
    assert wired.configured is True
    assert wired.model == "bge-reranker-v2-m3"
    assert "bge-reranker-v2-m3" in wired.detail


def test_the_reported_model_is_the_one_the_deployment_named() -> None:
    """RERANKER_MODEL travels with the URL, because only the operator knows it.

    A rerank call has to carry a model name and nothing can look up which
    weights a sidecar was started with, so the two keys move together. A page
    naming some other model would send an operator to change the wrong thing.
    """
    settings = make_settings(reranker_url="http://reranker:7997", reranker_model="bge-base")
    availability = reranker_availability(settings)
    assert (availability.configured, availability.model) == (True, "bge-base")
    assert "bge-base" in availability.detail


def test_stored_ceiling_out_of_range_reads_as_unset() -> None:
    """A hostile or legacy blob degrades to unclamped, never to crippled.

    The lenient reader drops a malformed field; for this one that matters more
    than for its siblings, because a stored ``1`` would otherwise clamp every
    turn to a single token.
    """
    assert WorkspaceDefaults.from_stored({"context_ceiling": 1}).context_ceiling is None
    assert WorkspaceDefaults.from_stored({"context_ceiling": 10**9}).context_ceiling is None
    assert WorkspaceDefaults.from_stored({"context_ceiling": 4096}).context_ceiling == 4096


def test_patch_merges_per_field_and_clears_on_an_explicit_null() -> None:
    """Absent leaves alone, null removes, a value overwrites - and unknown keys ride along."""
    base = {"model": "a:1b", "thinking": "on", "written_by_another_version": 7}
    absent = WorkspaceDefaultsPatch().merged_into(base)
    assert absent == base
    changed = WorkspaceDefaultsPatch(thinking="off").merged_into(base)
    assert changed["thinking"] == "off"
    assert changed["model"] == "a:1b"
    cleared = WorkspaceDefaultsPatch(model=None).merged_into(base)
    assert "model" not in cleared
    assert cleared["written_by_another_version"] == 7


# --------------------------------------------------------------------------
# The write surface
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_defaults_reports_unset_and_the_deployment_facts(
    db_sessionmaker,
) -> None:
    """A fresh workspace reads as every field unset, with the bounds it accepts."""
    # Hermetic settings rather than the shared client fixture: the reranker
    # status is a DEPLOYMENT fact, so reading it off the ambient environment
    # would make this assertion depend on whether the developer's shell exports
    # RERANKER_URL - green for a reason the test never states.
    settings = make_settings()
    async with defaults_client(provisioned(), db_sessionmaker, settings) as client:
        resp = await client.get(DEFAULTS_URL)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] is None
    assert body["thinking"] is None
    assert body["web_search"] is None
    assert body["system_prompt"] is None
    assert body["context_ceiling"] is None
    assert body["reranker"] is None
    assert body["min_context_ceiling"] == MIN_CONTEXT_CEILING
    assert body["max_context_ceiling"] == MAX_CONTEXT_CEILING
    assert body["max_system_prompt_chars"] == MAX_SYSTEM_PROMPT_CHARS
    # No RERANKER_URL in these settings, so the toggle governs nothing and the
    # page is told which reason applied rather than shown a dead switch.
    assert body["reranker_status"]["configured"] is False
    assert "RERANKER_URL" in body["reranker_status"]["detail"]


@pytest.mark.anyio
async def test_writes_are_admin_gated_and_reads_are_not(db_sessionmaker) -> None:
    """A member may see how their own workspace behaves; only an admin changes it.

    The role check passes for everybody until real login populates it, so this
    drives an explicitly non-admin principal through the same dependency OIDC
    will start filling - the seam has to be provably there before identity can
    make it mean anything.
    """
    member = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="member")
    async with defaults_client(
        provisioned(), db_sessionmaker, make_settings(), tenant=member
    ) as client:
        assert (await client.get(DEFAULTS_URL)).status_code == 200
        refused = await client.patch(DEFAULTS_URL, json={"thinking": "on"})
    assert refused.status_code == 403
    assert await stored(db_sessionmaker) == {}


@pytest.mark.anyio
async def test_patch_touches_only_the_fields_it_names(client, db_sessionmaker) -> None:
    """The bug this shape exists to prevent: a second save clobbering the first."""
    first = await client.patch(DEFAULTS_URL, json={"model": "qwen3:4b"})
    assert first.status_code == 200
    second = await client.patch(DEFAULTS_URL, json={"thinking": "on"})
    assert second.status_code == 200
    assert second.json()["model"] == "qwen3:4b"
    assert await stored(db_sessionmaker) == {"model": "qwen3:4b", "thinking": "on"}


@pytest.mark.anyio
async def test_patch_null_clears_one_field(client, db_sessionmaker) -> None:
    """An explicit null removes the key rather than storing a null nothing reads."""
    await client.patch(DEFAULTS_URL, json={"model": "qwen3:4b", "web_search": "off"})
    resp = await client.patch(DEFAULTS_URL, json={"model": None})
    assert resp.status_code == 200
    assert resp.json()["model"] is None
    assert await stored(db_sessionmaker) == {"web_search": "off"}


@pytest.mark.anyio
async def test_patch_stores_the_two_new_axes(client, db_sessionmaker) -> None:
    """The ceiling and the toggle round-trip through the API."""
    resp = await client.patch(DEFAULTS_URL, json={"context_ceiling": 4096, "reranker": False})
    assert resp.status_code == 200
    assert resp.json()["context_ceiling"] == 4096
    assert resp.json()["reranker"] is False
    assert await stored(db_sessionmaker) == {"context_ceiling": 4096, "reranker": False}


@pytest.mark.anyio
async def test_patch_stores_the_workspace_persona(client, db_sessionmaker) -> None:
    """The persona round-trips, trimmed, and an emptied field clears it.

    Blank is a CLEAR, never a kill switch: ``SYSTEM_PROMPT=""`` disables the
    system prompt outright and that power stays with the environment, so the
    stored blob has no way to spell it.
    """
    resp = await client.patch(DEFAULTS_URL, json={"system_prompt": "  Speak as the desk.\n"})
    assert resp.status_code == 200
    assert resp.json()["system_prompt"] == "Speak as the desk."
    assert await stored(db_sessionmaker) == {"system_prompt": "Speak as the desk."}

    for blank in ("", "   \n "):
        cleared = await client.patch(DEFAULTS_URL, json={"system_prompt": blank})
        assert cleared.status_code == 200
        assert cleared.json()["system_prompt"] is None
        assert await stored(db_sessionmaker) == {}
        await client.patch(DEFAULTS_URL, json={"system_prompt": "Speak as the desk."})
    # Explicit null clears it too, the same way every other axis clears.
    assert (await client.patch(DEFAULTS_URL, json={"system_prompt": None})).status_code == 200
    assert await stored(db_sessionmaker) == {}


@pytest.mark.anyio
async def test_patch_stores_the_embedding_model(client, db_sessionmaker) -> None:
    """The embedder round-trips through the API, and clears back to unset.

    Clearing is not "no embedder": it hands the question back to the
    deployment, which answers from the model its own stored passages were
    written by, and then from the only embedder the server holds.
    """
    resp = await client.patch(DEFAULTS_URL, json={"embedding_model": "  qwen3-embedding:4b  "})
    assert resp.status_code == 200
    assert resp.json()["embedding_model"] == "qwen3-embedding:4b"
    assert await stored(db_sessionmaker) == {"embedding_model": "qwen3-embedding:4b"}

    cleared = await client.patch(DEFAULTS_URL, json={"embedding_model": None})
    assert cleared.status_code == 200
    assert cleared.json()["embedding_model"] is None
    assert await stored(db_sessionmaker) == {}


@pytest.mark.anyio
async def test_patch_leaves_the_persona_alone_when_it_is_absent(client, db_sessionmaker) -> None:
    """Saving another control must not disturb a stored persona."""
    await client.patch(DEFAULTS_URL, json={"system_prompt": "Speak as the desk."})
    resp = await client.patch(DEFAULTS_URL, json={"thinking": "on"})
    assert resp.status_code == 200
    assert resp.json()["system_prompt"] == "Speak as the desk."
    assert await stored(db_sessionmaker) == {
        "system_prompt": "Speak as the desk.",
        "thinking": "on",
    }


def test_an_oversized_stored_persona_reads_as_unset() -> None:
    """A blob from a wider code version degrades to the deployment's own persona.

    Bounded on the way OUT as well as in, like the context ceiling: this value
    is prepended to every prompt, so an unbounded one is not a preference that
    falls through harmlessly - it eats the window. Falling back to the built-in
    persona is a working turn; carrying the wall of text is not.
    """
    huge = "x" * (MAX_SYSTEM_PROMPT_CHARS + 1)
    assert WorkspaceDefaults.from_stored({"system_prompt": huge}).system_prompt is None
    ok = "x" * MAX_SYSTEM_PROMPT_CHARS
    assert WorkspaceDefaults.from_stored({"system_prompt": ok}).system_prompt == ok
    # Whitespace-only stored text is unset, so no reader has to re-check it.
    assert WorkspaceDefaults.from_stored({"system_prompt": "   "}).system_prompt is None


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["http://evil/qwen3", "../../etc/passwd", "a b:1"])
async def test_patch_refuses_an_embedder_name_that_is_not_a_name(
    client, db_sessionmaker, name
) -> None:
    """Held to the same shape rules as the chat default, for the same reason.

    The stored tag becomes a collection name and a model-server argument, so a
    URL or a traversal is refused here rather than by whatever builds a request
    out of it later.
    """
    resp = await client.patch(DEFAULTS_URL, json={"embedding_model": name})
    assert resp.status_code == 422
    assert await stored(db_sessionmaker) == {}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"unknown_key": 1},
        {"thinking": "yes"},
        {"context_ceiling": MIN_CONTEXT_CEILING - 1},
        {"context_ceiling": MAX_CONTEXT_CEILING + 1},
        {"context_ceiling": "big"},
        {"model": ""},
        {"system_prompt": "x" * (MAX_SYSTEM_PROMPT_CHARS + 1)},
        {"system_prompt": 7},
        {"embedding_model": ""},
    ],
)
async def test_patch_refuses_a_malformed_body(client, db_sessionmaker, body) -> None:
    """Writes are strict where reads are lenient - and nothing is stored."""
    resp = await client.patch(DEFAULTS_URL, json=body)
    assert resp.status_code == 422
    assert await stored(db_sessionmaker) == {}


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["http://evil/qwen3", "../../etc/passwd", "a b:1"])
async def test_patch_refuses_a_model_name_that_is_not_a_name(client, db_sessionmaker, name) -> None:
    """The default tag goes on to be a path segment and a server argument.

    Same validator every management verb uses, so a URL or a traversal is
    refused here rather than at whatever builds a request out of it later.
    """
    resp = await client.patch(DEFAULTS_URL, json={"model": name})
    assert resp.status_code == 422
    assert await stored(db_sessionmaker) == {}


@pytest.mark.anyio
async def test_a_padded_tag_is_stored_trimmed(client, db_sessionmaker) -> None:
    """What gets STORED is what the parser returned, not what the field held.

    The parser trims, so validating the clean form and storing the dirty one
    would leave a default the read side can never honour: it compares stored
    tags against the provider's listing by equality, and " qwen3:4b" matches
    nothing on any server.
    """
    resp = await client.patch(DEFAULTS_URL, json={"model": "  qwen3:4b  "})
    assert resp.status_code == 200
    assert resp.json()["model"] == "qwen3:4b"
    assert await stored(db_sessionmaker) == {"model": "qwen3:4b"}


@pytest.mark.anyio
async def test_set_default_model_writes_and_clears(db_sessionmaker) -> None:
    """The narrow entry point a manager row action calls, both directions."""
    async with db_sessionmaker() as session:
        result = await set_default_model(session, TENANT, "qwen3:4b")
        await session.commit()
    assert result.model == "qwen3:4b"
    assert await stored(db_sessionmaker) == {"model": "qwen3:4b"}

    await store_defaults(db_sessionmaker, {"thinking": "on"})
    async with db_sessionmaker() as session:
        cleared = await set_default_model(session, TENANT, None)
        await session.commit()
    assert cleared.model is None
    # Its neighbour survived: this is a one-field write, not a blob write.
    assert await stored(db_sessionmaker) == {"thinking": "on"}


@pytest.mark.anyio
async def test_load_defaults_reads_back_what_was_written(db_sessionmaker) -> None:
    """The read helper every consumer uses sees the writer's result."""
    await store_defaults(db_sessionmaker, {"context_ceiling": 4096, "reranker": False})
    async with db_sessionmaker() as session:
        defaults = await load_defaults(session, TENANT)
    assert (defaults.context_ceiling, defaults.reranker) == (4096, False)


# --------------------------------------------------------------------------
# The read paths: each stored field actually moves a real turn
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_workspace_model_routes_the_turn(db_sessionmaker) -> None:
    """A stored default becomes the model the provider is actually asked for."""
    # Deliberately the pair SIBLING, never the deployment default: a stored
    # pick equal to it would pass with the workspace rung removed entirely.
    await store_defaults(db_sessionmaker, {"model": "qwen3:4b"})
    provider = sized(script=["Hi."])
    async with defaults_client(provider, db_sessionmaker, make_settings()) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    assert provider.last_model == "qwen3:4b"


@pytest.mark.anyio
async def test_workspace_thinking_default_turns_reasoning_on(db_sessionmaker) -> None:
    """A stored thinking=on reaches the wire on a message the heuristic would not."""
    provider = sized(script=["Hi."])
    settings = make_settings()
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    # Control: with nothing stored this plain question does not reason, and it
    # runs on the non-thinking member of the pair.
    assert provider.last_model == "qwen3:4b-instruct"

    await store_defaults(db_sessionmaker, {"thinking": "on"})
    provider = sized(script=["Hi."])
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    # The dev pair routes thinking turns to the sibling tag rather than sending
    # a flag, so the ROUTED model is the observable half of the decision.
    assert provider.last_model == "qwen3:4b"


@pytest.mark.anyio
async def test_workspace_web_off_keeps_the_turn_fully_local(db_sessionmaker) -> None:
    """A stored web_search=off stops both halves: no pre-fetch and no web tool."""
    await store_defaults(db_sessionmaker, {"web_search": "off"})
    provider = provisioned(script=["Answered", " locally."])
    search = FakeSearchProvider(results=[SearchResult(title="t", url="https://x", snippet="s")])
    async with defaults_client(provider, db_sessionmaker, make_settings(), search=search) as client:
        events = await collect_events(client, {"message": FRESH_QUESTION})
    assert "searching_web" not in status_codes(events)
    assert "web_search" not in offered_names(provider)
    assert search.last_query is None


def system_of(provider) -> str:
    """The system message the provider's last request opened with, or ""."""
    messages = provider.last_messages
    if not messages or messages[0].role != "system":
        return ""
    return messages[0].content


@pytest.mark.anyio
async def test_workspace_persona_replaces_the_built_in_one(db_sessionmaker) -> None:
    """A stored persona becomes the system message a real turn opens with.

    Both halves asserted, because only the second can fail if the wiring is
    removed: the stored text is THERE, and the built-in persona is GONE. A test
    that checked only for the stored text would pass against a composition that
    appended it to the default instead of replacing it.
    """
    provider = sized(script=["Hi."])
    settings = make_settings()
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    # Control: nothing stored, so the deployment's own persona runs.
    assert system_of(provider).startswith(DEFAULT_SYSTEM_PROMPT)

    await store_defaults(db_sessionmaker, {"system_prompt": PERSONA})
    provider = sized(script=["Hi."])
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    system = system_of(provider)
    assert system.startswith(PERSONA)
    assert DEFAULT_SYSTEM_PROMPT not in system


@pytest.mark.anyio
async def test_workspace_persona_wins_over_the_environments(db_sessionmaker) -> None:
    """The chain's order, on a real turn: workspace above env, never beside it."""
    await store_defaults(db_sessionmaker, {"system_prompt": PERSONA})
    provider = sized(script=["Hi."])
    settings = make_settings(system_prompt="Answer as the deployment configured.")
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    system = system_of(provider)
    assert system.startswith(PERSONA)
    assert "Answer as the deployment configured." not in system


@pytest.mark.anyio
async def test_a_workspace_persona_still_carries_the_tool_guidance(
    db_sessionmaker,
) -> None:
    """Rewriting the persona must not silently drop the never-invent-a-tool layer.

    Driven through a real turn rather than through ``append_tool_guidance``
    directly: the guarantee is about what the model is SENT, and the layer is
    appended by the turn, not by the composition.
    """
    await store_defaults(db_sessionmaker, {"system_prompt": PERSONA})
    provider = sized(script=["Hi."])
    async with defaults_client(provider, db_sessionmaker, make_settings()) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    # The turn genuinely offered something - otherwise there would be no
    # guidance to look for and this would pass while proving nothing.
    assert offered_names(provider)
    system = system_of(provider)
    assert system.startswith(PERSONA)
    assert "Never invent a tool" in system


@pytest.mark.anyio
async def test_workspace_persona_counts_toward_the_meter(db_sessionmaker) -> None:
    """The composer's fill reading measures the prompt a turn would actually send.

    The stored persona is deliberately much shorter than the built-in one, so
    the reading has to FALL - an estimate that kept composing the default would
    report the same number both times.
    """
    provider = sized(script=["Hi."])
    settings = make_settings()
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        created = await client.post("/api/v1/conversations", json={})
        conversation_id = created.json()["id"]
        url = f"/api/v1/conversations/{conversation_id}/context"
        before = (await client.get(url)).json()["prompt_tokens"]
        await store_defaults(db_sessionmaker, {"system_prompt": PERSONA})
        after = (await client.get(url)).json()["prompt_tokens"]
    assert after < before


@pytest.mark.anyio
async def test_workspace_context_ceiling_shrinks_the_turns_window(
    db_sessionmaker,
) -> None:
    """A stored ceiling below the reported window is what the turn actually requests."""
    provider = sized(script=["Hi."])
    settings = make_settings(default_context_window=NATIVE_WINDOW)
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    assert provider.last_options.num_ctx == NATIVE_WINDOW

    await store_defaults(db_sessionmaker, {"context_ceiling": MIN_CONTEXT_CEILING})
    provider = sized(script=["Hi."])
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    assert provider.last_options.num_ctx == MIN_CONTEXT_CEILING


@pytest.mark.anyio
async def test_workspace_context_ceiling_never_raises_the_window(db_sessionmaker) -> None:
    """A ceiling above what the WEIGHTS admit leaves the turn exactly where it was."""
    await store_defaults(db_sessionmaker, {"context_ceiling": 131072})
    provider = sized(script=["Hi."])
    settings = make_settings(default_context_window=NATIVE_WINDOW)
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        await collect_events(client, {"message": PLAIN_QUESTION})
    assert provider.last_options.num_ctx == NATIVE_WINDOW


@pytest.mark.anyio
async def test_context_ceiling_moves_the_warm_window_too(db_sessionmaker) -> None:
    """Warm must ask for the same window the turn will, or it primes a cache nothing reuses."""
    await store_defaults(db_sessionmaker, {"context_ceiling": MIN_CONTEXT_CEILING})
    provider = sized(script=["ok"])
    settings = make_settings(default_context_window=NATIVE_WINDOW)
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        resp = await client.post("/api/v1/models/warm", json={})
    assert resp.status_code == 200
    assert provider.last_options.num_ctx == MIN_CONTEXT_CEILING


@pytest.mark.anyio
async def test_context_ceiling_is_the_meters_denominator(db_sessionmaker) -> None:
    """The composer's fill reading must measure against the clamped window."""
    await store_defaults(db_sessionmaker, {"context_ceiling": MIN_CONTEXT_CEILING})
    provider = sized(script=["Hi."])
    settings = make_settings(default_context_window=NATIVE_WINDOW)
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        created = await client.post("/api/v1/conversations", json={})
        conversation_id = created.json()["id"]
        resp = await client.get(f"/api/v1/conversations/{conversation_id}/context")
    assert resp.status_code == 200
    assert resp.json()["num_ctx"] == MIN_CONTEXT_CEILING


@pytest.mark.anyio
async def test_workspace_reranker_off_withholds_the_wired_reranker(
    db_sessionmaker,
) -> None:
    """A stored reranker=false keeps a grounded turn on dense ranking alone."""
    store = make_memory_store()
    await seed_kb(store)
    reranker = RecordingReranker()
    provider = provisioned(script=["Refunds", " run", " 30", " days", " [1]."])
    # deterministic: the turn grounds on the ambient knowledge base, which is
    # the retrieval a reranker would re-score.
    settings = make_settings(agency_mode="deterministic")
    async with defaults_client(
        provider, db_sessionmaker, settings, store=store, reranker=reranker
    ) as client:
        await collect_events(client, {"message": KB_TEXT})
    # Control: nothing stored, so the deployment's reranker runs.
    assert reranker.calls > 0

    await store_defaults(db_sessionmaker, {"reranker": False})
    withheld = RecordingReranker()
    provider = provisioned(script=["Refunds", " run", " 30", " days", " [1]."])
    async with defaults_client(
        provider, db_sessionmaker, settings, store=store, reranker=withheld
    ) as client:
        await collect_events(client, {"message": KB_TEXT})
    assert withheld.calls == 0


@pytest.mark.anyio
async def test_workspace_reranker_off_reaches_the_search_endpoint(
    db_sessionmaker,
) -> None:
    """The same clamp on /rag/query: one workspace must not rank its corpus two ways."""
    store = make_memory_store()
    await seed_kb(store)
    reranker = RecordingReranker()
    provider = sized(script=["Hi."])
    settings = make_settings()
    async with defaults_client(
        provider, db_sessionmaker, settings, store=store, reranker=reranker
    ) as client:
        resp = await client.post("/api/v1/rag/query", json={"query": KB_TEXT})
        assert resp.status_code == 200
        assert reranker.calls > 0

        await store_defaults(db_sessionmaker, {"reranker": False})
        before = reranker.calls
        resp = await client.post("/api/v1/rag/query", json={"query": KB_TEXT})
        assert resp.status_code == 200
    assert reranker.calls == before


def _confirming_provider() -> object:
    """A model that asks for a side-effecting tool, then looks something up.

    Three turns, which is what makes this a RESUME test rather than another
    plain-turn one: the first suspends the turn on the confirmation, and the
    second - the one the approve resumes into - is the one that retrieves.
    """
    return provisioned(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c1", name="create_record", arguments={"q": "bug"})]),
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c2", name="search_knowledge", arguments={"query": KB_TEXT})
                ]
            ),
            FakeTurn(tokens=["Filed it."]),
        ]
    )


async def _pause_and_resume(client) -> None:
    """Drive the confirm-capable turn to its pause, then approve it."""
    events = await collect_events(client, {"message": KB_TEXT, "supports_confirmation": True})
    confirmations = [data for name, data in events if name == "tool_confirmation"]
    assert len(confirmations) == 1, "the turn should suspend on exactly one confirmation"
    await collect_events(
        client,
        {
            "conversation_id": confirmations[0]["conversation_id"],
            "confirmation": {"id": confirmations[0]["id"], "decision": "approve"},
        },
    )


@pytest.mark.anyio
async def test_workspace_reranker_off_survives_a_confirmation_resume(
    db_sessionmaker,
) -> None:
    """A resumed turn ranks the way the workspace asked, not the way the deployment would.

    The resume is a SEPARATE request that rebuilds its collaborators from the
    dependencies, so it reads the deployment's reranker unless the clamp is
    applied there too. Without it one turn would rank two ways: the halves
    before and after the user approved a side-effecting action.
    """
    await seed_agentic_verdict(db_sessionmaker)
    store = make_memory_store()
    await seed_kb(store)
    settings = make_settings()
    danger = RemoteTool("create_record", side_effecting=True, result="A RECORD WAS CREATED")

    reranker = RecordingReranker()
    async with defaults_client(
        _confirming_provider(),
        db_sessionmaker,
        settings,
        store=store,
        reranker=reranker,
        mcp_sources=[FakeMcpSource([danger])],
    ) as client:
        await _pause_and_resume(client)
    # Control: with nothing stored, the resumed turn's lookup is re-scored.
    assert reranker.calls > 0

    await store_defaults(db_sessionmaker, {"reranker": False})
    withheld = RecordingReranker()
    async with defaults_client(
        _confirming_provider(),
        db_sessionmaker,
        settings,
        store=store,
        reranker=withheld,
        mcp_sources=[FakeMcpSource([RemoteTool("create_record", side_effecting=True)])],
    ) as client:
        await _pause_and_resume(client)
    assert withheld.calls == 0


@pytest.mark.anyio
async def test_context_ceiling_reaches_the_title_job(db_sessionmaker) -> None:
    """The background titler must size its window exactly as the turn does.

    The two agreeing is what keeps the runtime from tearing down the loaded
    model to write six words. A ceiling that reached the turn and not this job
    would reintroduce that reload on the first title of every thread - and
    silently, since a title that costs a reload still produces a title.
    """
    from app.services.titles import derive_title
    from tests.unit.test_conversation_titles import _run, _seed
    from tests.unit.test_conversation_titles import provisioned as titling_provider

    await store_defaults(db_sessionmaker, {"context_ceiling": MIN_CONTEXT_CEILING})
    seed = derive_title("why does my login page hang after the redirect")
    conversation = await _seed(db_sessionmaker, title=seed)
    provider = titling_provider(["Login ", "redirect ", "loop"])
    await _run(db_sessionmaker, provider, conversation, seed)
    assert provider.last_options.num_ctx == MIN_CONTEXT_CEILING


def _inventory() -> FakeProvider:
    """A model server holding one chat tag and one embedder, both self-reported.

    Built here rather than borrowed from the manager suite so this file's
    ceiling assertions do not depend on another file's fixture: the cards under
    test are the ones a plain, undescribed inventory produces, which is what
    every deployment now has.

    Returns:
        The fake, reporting a native window on the chat tag and none on the
        embedder (nothing generates on an embedder, so nothing sizes it).
    """
    return FakeProvider(
        models=[
            ModelInfo(name=STRAY_CHAT, digest=f"sha256:{STRAY_CHAT}"),
            ModelInfo(name=STRAY_EMBED, digest=f"sha256:{STRAY_EMBED}"),
        ],
        capabilities_by_model={
            STRAY_CHAT: ModelCapabilities(
                tools=True, completion=True, context_length=NATIVE_WINDOW
            ),
            STRAY_EMBED: ModelCapabilities(embedding=True, completion=False),
        },
    )


async def _cards(db_sessionmaker, provider: FakeProvider) -> dict[str, int | None]:
    """The manager view's window per tag, as a card would render it."""
    view = await build_manager_view(
        provider=provider,
        provider_name="fake",
        settings=make_settings(default_context_window=NATIVE_WINDOW),
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        store=None,
    )
    return {row.model: row.num_ctx for group in view.groups for row in group.rows}


@pytest.mark.anyio
async def test_context_ceiling_reaches_the_manager_cards(db_sessionmaker) -> None:
    """A card's window is the window a turn requests, ceiling included.

    The card and the turn resolve through one rule precisely so they cannot
    disagree; a card reporting the model's full window under a ceiling that
    halves it would be off by exactly the amount the operator asked for.
    """
    provider = _inventory()
    unclamped = await _cards(db_sessionmaker, provider)
    # The deployment's affordance, well above the floor a ceiling may be set to.
    assert unclamped[STRAY_CHAT] == NATIVE_WINDOW

    await store_defaults(db_sessionmaker, {"context_ceiling": MIN_CONTEXT_CEILING})
    clamped = await _cards(db_sessionmaker, provider)
    assert clamped[STRAY_CHAT] == MIN_CONTEXT_CEILING


@pytest.mark.anyio
async def test_the_ceiling_leaves_the_non_chat_cards_alone(db_sessionmaker) -> None:
    """It is a chat-turn budget, so an embedder's card must not wear it.

    The embedder is driven by the ingest path and the reranker by a sidecar;
    neither reads this setting. A card showing them clamped would report a
    limit that does not apply - the same kind of lie as a card ignoring one
    that does, in the opposite direction.
    """
    provider = _inventory()
    before = (await _cards(db_sessionmaker, provider))[STRAY_EMBED]
    await store_defaults(db_sessionmaker, {"context_ceiling": MIN_CONTEXT_CEILING})
    after = (await _cards(db_sessionmaker, provider))[STRAY_EMBED]
    # Unmoved, and distinguishable from the ceiling - or the assertion could
    # not tell "left alone" from "happened to land on the same number".
    assert before == after
    assert after != MIN_CONTEXT_CEILING


# --------------------------------------------------------------------------
# The window arithmetic: the assembled prompt provably fits
# --------------------------------------------------------------------------


# Long enough that the history genuinely overruns BOTH budgets. Only the
# questions carry weight - the fake's replies are three characters - so ten
# surviving questions at 1800 chars (~450 tokens each) come to ~4500, above the
# configured 4000 ceiling and well above the ~2700 a 4096-token window can
# spare beside a system prompt and an answer reserve. A shorter fixture makes
# both window tests pass without the trim existing at all, which is how the
# first draft of this file was hollow.
_TURN_CHARS = 1800


def _long_turns(count: int) -> list[str]:
    """Questions long enough that replaying them overruns a small window."""
    body = "This is a stored exchange that exists only to take up room. "
    return [f"{index} {body * 40}"[:_TURN_CHARS] for index in range(count)]


async def _fill_thread(client, conversation_id: str, count: int) -> None:
    """Drive ``count`` complete exchanges onto one conversation."""
    for content in _long_turns(count):
        await collect_events(client, {"message": content, "conversation_id": conversation_id})


@pytest.mark.anyio
async def test_a_clamped_window_still_fits_its_whole_prompt(db_sessionmaker) -> None:
    """The prompt a turn actually sends fits the window it actually requests.

    The floor a ceiling may be set to is 4096, and HISTORY_TOKEN_BUDGET is
    4000 - so without a window-aware trim the history alone very nearly fills
    the window, and the system prompt and answer reserve do not fit beside it.
    Ollama then truncates from the HEAD, which is the system prompt, taking the
    persona, the tool guidance and the injection defense with it, on a turn
    that still streams a fluent answer. Nothing else in the suite would notice.

    Asserted the way the tool-result floor is: over the ASSEMBLED prompt the
    provider was actually handed, not over an intermediate the code could
    compute correctly and then not use.
    """
    await store_defaults(db_sessionmaker, {"context_ceiling": MIN_CONTEXT_CEILING})
    provider = sized(script=["Hi."])
    settings = make_settings(default_context_window=NATIVE_WINDOW)
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        created = await client.post("/api/v1/conversations", json={})
        conversation_id = created.json()["id"]
        # A long stored thread, so the history is the thing that overflows.
        await _fill_thread(client, conversation_id, 14)
        await collect_events(
            client, {"message": PLAIN_QUESTION, "conversation_id": conversation_id}
        )

    assert provider.last_options.num_ctx == MIN_CONTEXT_CEILING
    # The premise: without the trim this thread WOULD overrun. If the stored
    # history ever gets small enough to fit unaided, the assertion below stops
    # testing anything and this says so first.
    assert (
        sum(approx_tokens(content) for content in _long_turns(14)) * 2
        > MIN_CONTEXT_CEILING - ANSWER_RESERVE_TOKENS
    )
    sent = sum(approx_tokens(message.content) for message in provider.last_messages)
    assert sent + ANSWER_RESERVE_TOKENS <= MIN_CONTEXT_CEILING, (
        f"the assembled prompt ({sent} tokens) plus the answer reserve overruns the "
        f"{MIN_CONTEXT_CEILING}-token window it was sent with"
    )
    # And the trim took HISTORY, not the system prompt: the head survived.
    assert provider.last_messages[0].role == "system"


@pytest.mark.anyio
async def test_an_unclamped_window_keeps_the_configured_history_budget(
    db_sessionmaker,
) -> None:
    """With room to spare the setting is still the ceiling - no silent shrinking.

    The control for the trim above: a window that comfortably holds the
    configured budget must replay exactly what it replayed before, or the fix
    would have quietly cut every deployment's history in half.
    """
    provider = sized(script=["Hi."])
    settings = make_settings(num_ctx=32768)
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        created = await client.post("/api/v1/conversations", json={})
        conversation_id = created.json()["id"]
        await _fill_thread(client, conversation_id, 6)
        await collect_events(
            client, {"message": PLAIN_QUESTION, "conversation_id": conversation_id}
        )

    # Every stored message came back: 6 questions plus their 6 answers, the
    # system prompt, and the new question.
    assert len(provider.last_messages) == 1 + 12 + 1


@pytest.mark.anyio
async def test_the_meter_measures_the_prompt_a_turn_would_send(db_sessionmaker) -> None:
    """A ceiling shrinks the meter's numerator as well as its denominator.

    The fill reading is the fraction a turn would produce. Trimming the window
    without trimming the history it is measured against would make a clamped
    workspace read as permanently near-full, on threads the turn itself would
    have fitted comfortably.
    """
    provider = sized(script=["Hi."])
    settings = make_settings(default_context_window=NATIVE_WINDOW)
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        created = await client.post("/api/v1/conversations", json={})
        conversation_id = created.json()["id"]
        await _fill_thread(client, conversation_id, 14)
        wide = (await client.get(f"/api/v1/conversations/{conversation_id}/context")).json()

        await store_defaults(db_sessionmaker, {"context_ceiling": MIN_CONTEXT_CEILING})
        narrow = (await client.get(f"/api/v1/conversations/{conversation_id}/context")).json()

    assert wide["num_ctx"] == NATIVE_WINDOW
    assert narrow["num_ctx"] == MIN_CONTEXT_CEILING
    assert narrow["prompt_tokens"] < wide["prompt_tokens"]
    assert narrow["prompt_tokens"] + ANSWER_RESERVE_TOKENS <= MIN_CONTEXT_CEILING


# --------------------------------------------------------------------------
# Warm: the same model, and the same flag, a turn would ask for
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_warm_follows_the_stored_thinking_default(db_sessionmaker) -> None:
    """A workspace that reasons by default gets its REASONING model warmed.

    On the dev pair a thinking turn runs on a different TAG, so a warm that
    ignored the stored default would load the instruct member while every turn
    ran the thinking one - and on a card that fits one model at a time, the
    first real question evicts exactly what the warm just paid to load.
    """
    settings = make_settings()
    provider = sized(script=["ok"])
    async with defaults_client(provider, db_sessionmaker, settings) as client:
        assert (await client.post("/api/v1/models/warm", json={})).status_code == 200
    # Control: nothing stored, so the fast path - the pair's instruct member.
    assert provider.last_model == "qwen3:4b-instruct"

    await store_defaults(db_sessionmaker, {"thinking": "on"})
    reasoning = sized(script=["ok"])
    async with defaults_client(reasoning, db_sessionmaker, settings) as client:
        assert (await client.post("/api/v1/models/warm", json={})).status_code == 200
    assert reasoning.last_model == THINKING_MODEL
    # And the flag the turn would send goes with it. The quirk seed records
    # this sibling as always-thinking, so the warm asks for reasoning
    # explicitly rather than priming a prefix the turn will not reuse.
    assert reasoning.last_options.think is True


# A hybrid deployment: one tag that reasons on request, no pair sibling. The
# dev pair expresses the thinking decision by ROUTING, so on it
# `normalize_think` answers from the tag's own mode and the warm's think flag
# would look right whatever decision produced it. Only a hybrid model lets the
# FLAG itself be the observable - and a hybrid one is what a deployment with
# room for a larger model actually runs.
HYBRID_MODEL = "hybrid:8b"


def _hybrid_settings() -> Settings:
    """A deployment configured with one hybrid tag and no pair sibling."""
    return Settings(
        _env_file=None,
        model_provider="fake",
        default_chat_model=HYBRID_MODEL,
        # The literal that switches pair-routing OFF, which is what "one tag
        # does both" means as configuration.
        thinking_chat_model="none",
    )


def _hybrid_provider() -> FakeProvider:
    """A server holding the hybrid tag, whose template has real think plumbing.

    ``template_thinking`` is the stronger of the runtime's two thinking
    signals, and it is what makes this tag resolve as ``hybrid`` rather than as
    a tag nothing describes - the deployment says which model it runs, and the
    RUNTIME says what that model does with the flag.
    """
    return FakeProvider(
        script=["ok"],
        models=[ModelInfo(name=HYBRID_MODEL)],
        capabilities_by_model={
            HYBRID_MODEL: ModelCapabilities(
                tools=True,
                completion=True,
                thinking=True,
                template_thinking=True,
                context_length=NATIVE_WINDOW,
            )
        },
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("stored_mode", "expected_flag"),
    [("on", True), ("off", False), (None, False)],
)
async def test_warm_sends_the_think_flag_the_stored_default_implies(
    db_sessionmaker, stored_mode, expected_flag
) -> None:
    """On a hybrid model the decision rides the FLAG, and the warm must send it.

    A warm that primed the wrong flag primes a prefix the first real turn does
    not reuse, so the cold evaluation this exists to remove happens on that
    turn anyway - and on a reasoning model the two prefixes differ by the whole
    reasoning scaffold, not by a token.
    """
    if stored_mode is not None:
        await store_defaults(db_sessionmaker, {"thinking": stored_mode})
    provider = _hybrid_provider()
    async with defaults_client(provider, db_sessionmaker, _hybrid_settings()) as client:
        assert (await client.post("/api/v1/models/warm", json={})).status_code == 200
    assert provider.last_model == HYBRID_MODEL
    assert provider.last_options.think is expected_flag


@pytest.mark.anyio
async def test_warm_follows_a_stored_thinking_off(db_sessionmaker) -> None:
    """The other direction, so the pin is about the STORED value, not about "on".

    Without this, a warm hardcoded to the non-thinking member would pass the
    control above for the wrong reason.
    """
    await store_defaults(db_sessionmaker, {"thinking": "off"})
    provider = sized(script=["ok"])
    async with defaults_client(provider, db_sessionmaker, make_settings()) as client:
        assert (await client.post("/api/v1/models/warm", json={})).status_code == 200
    assert provider.last_model == "qwen3:4b-instruct"


# --------------------------------------------------------------------------
# The row lock, pinned where SQLite cannot show it
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_the_locking_read_asks_for_the_lock(db_sessionmaker) -> None:
    """``load_for_update`` asks for a row lock - pinned on the call it makes.

    The suite runs on SQLite, whose single-writer model renders FOR UPDATE as
    nothing, so no behavioural test here can tell a locking read from a plain
    one. Asserting the arguments the repository passes is what CAN fail when
    somebody deletes them; the compile below then says what that flag buys on
    the database this actually ships against.

    Deliberately a spy on the repository's own call rather than a statement
    this test builds. A test that constructs its own ``select(...)`` and
    compiles it proves that SQLAlchemy renders FOR UPDATE - which was never in
    doubt - while staying green if the repository stopped asking for it.
    """
    from sqlalchemy import select
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.models import WorkspaceSettings

    seen: list[dict] = []
    original = AsyncSession.get

    async def spy(self, entity, ident, **kwargs):  # noqa: ANN001, ANN202
        seen.append(kwargs)
        return await original(self, entity, ident, **kwargs)

    async with db_sessionmaker() as session:
        repo = WorkspaceSettingsRepository(session, TENANT)
        AsyncSession.get = spy  # type: ignore[method-assign]
        try:
            await repo.load_for_update()
        finally:
            AsyncSession.get = original  # type: ignore[method-assign]

    assert seen, "load_for_update should read the row through the session"
    assert seen[0].get("with_for_update") is True
    assert seen[0].get("populate_existing") is True

    # What that flag buys, on the dialect the deployment runs: the clause
    # SQLite cannot show. Documentation, not the pin - the assertions above
    # are the pin.
    statement = (
        select(WorkspaceSettings)
        .where(WorkspaceSettings.tenant_id == DEFAULT_TENANT_ID)
        .with_for_update()
    )
    assert "FOR UPDATE" in str(statement.compile(dialect=postgresql.dialect()))


@pytest.mark.anyio
async def test_the_locking_read_sees_a_write_from_another_session(db_sessionmaker) -> None:
    """``populate_existing`` is load-bearing, and this is where it shows.

    The lock re-reads the ROW; without ``populate_existing`` the ORM hands back
    the instance already in this session's identity map, with the values it was
    first loaded with. A caller that read the defaults earlier in the same
    session - the delete flow does - would then merge over a stale blob UNDER a
    lock, which is a lost update wearing a lock's clothing.
    """
    await store_defaults(db_sessionmaker, {"model": "first:1b"})
    async with db_sessionmaker() as first:
        # Load the row into this session's identity map...
        assert (await WorkspaceSettingsRepository(first, TENANT).load())["model"] == "first:1b"
        # ...then move it from somewhere else entirely.
        await store_defaults(db_sessionmaker, {"model": "second:1b"})
        fresh = await WorkspaceSettingsRepository(first, TENANT).load_for_update()
    assert fresh["model"] == "second:1b"
