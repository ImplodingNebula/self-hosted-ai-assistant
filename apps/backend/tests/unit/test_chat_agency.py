"""Chat-path agency tests: the offering decision in stream_chat (slice 3 PR B).

resolve_tool_offer is wired into stream_chat here, so these drive the real
endpoint with a FakeProvider and assert the whole decision end to end - which
tools reach the wire, whether the ambient pre-fetch still runs, and the standing
notices - across the modes:

- the shipped agentic default offers web_search and search_knowledge, suppresses
  the ambient knowledge base, and stands a tools_not_used notice when the model
  answers without a lookup;
- a model that DOES call a tool runs it and carries no notice;
- the web switch off drops the web tool (fully local stays literally true);
- an explicit thinking turn keeps its reasoning and is offered no tools;
- assisted keeps the ambient grounding AND offers tools;
- deterministic offers nothing but still grounds;
- a window too small for a tool result drops the tools.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from httpx import ASGITransport, AsyncClient

from app.api.deps import get_model_provider, get_settings
from app.core.concurrency import StreamLimiter
from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.models.entities import ModelPolicy, ModelProbe
from app.providers.base import ModelCapabilities, ModelInfo, ToolCall
from app.providers.fake_provider import FAKE_SERVING_ENDPOINT, FakeProvider, FakeTurn
from app.providers.rag.embedder import FakeEmbedder
from app.providers.search.base import SearchResult
from app.providers.search.searxng import FakeSearchProvider
from app.schemas.rag import RagIngestItem
from app.services import rag_service
from app.services.model_policy import clear_policy_cache
from app.services.prompts import DEFAULT_SYSTEM_PROMPT
from app.services.tool_check import harness_version
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store

CHAT_URL = "/api/v1/chat/stream"

# A question and a stored record whose text equals it, so the hash embedder
# retrieves the record at cosine 1.0 when the KB is actually searched.
KB_TEXT = "The refund window for orders is 30 days from delivery."

# The configured pair. The qwen3 tags carry their own meaning through the
# code-shipped quirk seed (app.services.known_models): qwen3:4b-instruct never
# reasons and qwen3:4b always does, which is the shipped qwen3-on-Ollama shape
# these tests arbitrate against - a pair that cannot reason and call tools in
# one turn.
CHAT_MODEL = "qwen3:4b-instruct"
THINKING_MODEL = "qwen3:4b"

# The pair as the provider reports holding it, so these turns take the
# "configured model is installed -> route to it" path: a fake listing neither
# tag would have the turn substitute onto some other installed one, and every
# offering decision below would then be judged against the wrong model.
#
# The digests are load-bearing rather than decorative: agency above `assisted`
# is EARNED from a deep tool-check verdict bound to the installed weights, and
# a digestless listing gives a verdict nothing to bind to. A suite about the
# agentic turn therefore needs both the digests and :func:`seed_probe`.
CHAT_DIGEST = "sha256:chat"
THINKING_DIGEST = "sha256:think"
DEV_PAIR = [
    ModelInfo(name=CHAT_MODEL, digest=CHAT_DIGEST),
    ModelInfo(name=THINKING_MODEL, digest=THINKING_DIGEST),
]


def provisioned(**kwargs) -> FakeProvider:
    """A FakeProvider that reports holding the configured chat pair.

    Args:
        **kwargs: Passed through to :class:`FakeProvider`.

    Returns:
        The fake, listing the pair unless the caller seeded its own models.
    """
    kwargs.setdefault("models", list(DEV_PAIR))
    return FakeProvider(**kwargs)


def windowed_provider(native_window: int, **kwargs) -> FakeProvider:
    """A provider whose runtime report gives the pair a native context window.

    A window is ``min(native, afforded)`` and the native half is a RUNTIME
    fact, so the only way to state a small one is to have the provider report
    it - ``DEFAULT_CONTEXT_WINDOW`` will not go under 4096, and the tool-result
    floor only bites well below that.

    Args:
        native_window: The context length the runtime reports for both tags.
        **kwargs: Passed through to :func:`provisioned`.

    Returns:
        The fake, reporting ``native_window`` for the configured pair.
    """
    report = ModelCapabilities(
        tools=True, embedding=True, completion=True, context_length=native_window
    )
    kwargs.setdefault("capabilities_by_model", {CHAT_MODEL: report, THINKING_MODEL: report})
    return provisioned(**kwargs)


async def seed_probe(
    db_sessionmaker,
    *,
    depth: str,
    earned: str,
    model: str = CHAT_MODEL,
    digest: str = CHAT_DIGEST,
) -> None:
    """Record one tool-check verdict for a set of installed weights.

    Args:
        db_sessionmaker: The test session factory.
        depth: The check depth the verdict was earned at (fast or deep).
        earned: The agency mode the verdict awards.
        model: The tag the verdict is recorded against.
        digest: The installed digest it binds to.
    """
    async with db_sessionmaker() as session:
        session.add(
            ModelProbe(
                provider="fake",
                model=model,
                digest=digest,
                # The RUNNING harness's version: only verdicts it measured
                # certify anything, in either regime.
                harness_version=harness_version(),
                # And the host the fake serves from, for the same reason: a
                # verdict earned on another server certifies nothing here.
                serving_endpoint=FAKE_SERVING_ENDPOINT,
                depth=depth,
                earned_agency=earned,
            )
        )
        await session.commit()


async def seed_agentic_verdict(db_sessionmaker) -> None:
    """Arm the pair to drive a turn: BOTH halves, on BOTH members.

    Neither half is agentic by default. The CEILING an undeclared model
    resolves to is ``assisted``, because the only thing describing it is the
    runtime's report and "this template accepts a tool list" is a mechanical
    fact rather than a decision to hand weights the turn; agentic is a human
    declaration, stored digest-bound. The EARNED half needs a deep verdict
    bound to the same weights. Seed one without the other and the test
    measures an assisted turn under an agentic name.

    Both go on EACH member of the pair, because a declaration is per tag and
    every tag is its own probe key. The planner asks its capability question of
    whichever member the raw thinking decision points at, so a reasoning-shaped
    message on a half-armed pair lands on an assisted turn.

    Args:
        db_sessionmaker: The test session factory.
    """
    for tag, digest in ((CHAT_MODEL, CHAT_DIGEST), (THINKING_MODEL, THINKING_DIGEST)):
        await store_policy(db_sessionmaker, model=tag, digest=digest, ceiling="agentic")
        await seed_probe(db_sessionmaker, model=tag, digest=digest, depth="deep", earned="agentic")


def make_settings(**overrides) -> Settings:
    """Settings for a deployment configured with the pair above.

    Args:
        **overrides: Fields to set on top of the pair configuration.

    Returns:
        Settings built without reading the developer's own ``.env``.
    """
    return Settings(
        _env_file=None,
        model_provider="fake",
        default_chat_model=CHAT_MODEL,
        thinking_chat_model=THINKING_MODEL,
        **overrides,
    )


def citing_provider() -> FakeProvider:
    """A provider whose answer references passage [1], so a citation can emit."""
    return provisioned(script=["Refunds", " run", " 30", " days", " [1]."])


async def seed_kb(store, text: str = KB_TEXT) -> None:
    """Ingest one record into the persistent knowledge base collection."""
    tenant = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")
    await rag_service.ingest_structured(
        source_type="records",
        tenant=tenant,
        items=[RagIngestItem(id="kb-1", title="Refund policy", text=text)],
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=Settings(_env_file=None, model_provider="fake"),
        collection_id="col-1",
    )


@asynccontextmanager
async def agency_client(provider, db_sessionmaker, settings, *, store, search=None):
    """App-wired client with the RAG collaborators and an optional web provider."""
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    # Faked like storage: the lazy fallback opens a real Redis pool, and a first
    # turn now queues the background title job.
    app.state.job_queue = FakeJobQueue()
    app.state.vector_store = store
    app.state.embedder = FakeEmbedder(dim=32)
    if search is not None:
        app.state.search_provider = search
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def collect_events(client: AsyncClient, payload: dict) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current = ""
    async with client.stream("POST", CHAT_URL, json=payload) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                current = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((current, json.loads(line.removeprefix("data: "))))
    return events


def status_codes(events: list[tuple[str, dict]]) -> list[str]:
    return [d.get("code") for n, d in events if n == "status"]


def offered_names(provider: FakeProvider) -> list[str]:
    return [t["function"]["name"] for t in (provider.last_tools or [])]


# A question with a freshness cue ("today"), so the notice's freshness gate is
# satisfied and the deterministic pre-fetch heuristic would have wanted the web.
FRESH_QUESTION = "What is the latest Python version today?"


async def test_agentic_default_offers_both_tools_and_suppresses_ambient_kb(db_sessionmaker) -> None:
    # An agentic turn: the ceiling declares it and a deep check earned it. The
    # KB has a matching record, but this turn does NOT pre-fetch it: it is
    # offered the tools instead, and because this fake never calls one, the
    # answer is ungrounded - and, since the query wanted current info, it says
    # so.
    await seed_agentic_verdict(db_sessionmaker)
    store = make_memory_store()
    await seed_kb(store)
    provider = citing_provider()
    search = FakeSearchProvider(results=[SearchResult(title="t", url="https://x/", snippet="s")])
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        events = await collect_events(c, {"message": FRESH_QUESTION, "web_search": "on"})

    # Both core tools reached the wire, in the OpenAI-function shape.
    assert set(offered_names(provider)) == {"web_search", "search_knowledge"}
    # No ambient grounding happened: the record was never pre-fetched, so nothing
    # was cited even though the answer wrote a [1] marker.
    assert not any(n == "citation" for n, _ in events)
    # ...and the freshness-cue turn admits it answered without a lookup.
    assert "tools_not_used" in status_codes(events)


async def test_agentic_ordinary_query_offers_tools_but_stays_silent(db_sessionmaker) -> None:
    # Silence by default: a creative/reasoning turn with no freshness cue is still
    # offered tools (agentic), but a model that answers from its own knowledge
    # carries NO tools_not_used notice - the notice would be pure noise there.
    await seed_agentic_verdict(db_sessionmaker)
    store = make_memory_store()
    provider = citing_provider()
    search = FakeSearchProvider(results=[SearchResult(title="t", url="https://x/", snippet="s")])
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        events = await collect_events(c, {"message": "Write me a short poem about the sea."})

    assert set(offered_names(provider)) == {"web_search", "search_knowledge"}
    assert "tools_not_used" not in status_codes(events)


async def test_agentic_tool_call_runs_the_tool_and_carries_no_notice(db_sessionmaker) -> None:
    # When the model DOES drive - calls search_knowledge, then answers - the tool
    # runs and the honesty notice is absent.
    await seed_agentic_verdict(db_sessionmaker)
    store = make_memory_store()
    await seed_kb(store)
    provider = provisioned(
        turns=[
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c1", name="search_knowledge", arguments={"query": KB_TEXT})
                ]
            ),
            FakeTurn(tokens=["Refunds run 30 days."]),
        ]
    )
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store) as c:
        events = await collect_events(c, {"message": KB_TEXT})

    assert "tool_running" in status_codes(events)
    assert "tools_not_used" not in status_codes(events)
    # The tool result was fed back: the second provider call carried a tool
    # message with the record's text.
    assert len(provider.calls) == 2
    assert any(m.role == "tool" and "refund window" in m.content for m in provider.calls[1])


async def test_web_switch_off_offers_no_web_tool(db_sessionmaker) -> None:
    # "off" = fully local stays literally true: the live-fetch tool is never even
    # offered, so the model cannot make a fetch the user refused. search_knowledge
    # (local) still is.
    store = make_memory_store()
    provider = provisioned(script=["ok"])
    search = FakeSearchProvider(results=[])
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        await collect_events(c, {"message": "hello there", "web_search": "off"})

    assert offered_names(provider) == ["search_knowledge"]


async def test_explicit_thinking_beats_tools_and_offers_none(db_sessionmaker) -> None:
    # thinking "on" on a model that cannot do both: keep the reasoning, offer no
    # tools. The thinking frames stream and nothing reached the tools wire.
    #
    # Seeded agentic on purpose: this is the arbitration's HARD case. On an
    # agentic turn with thinking left to the heuristic the tools win, so only
    # an EXPLICIT ask is allowed to take them away - and only an armed
    # deployment can show that, since an assisted turn yields to thinking
    # either way and would pass without exercising the rule.
    await seed_agentic_verdict(db_sessionmaker)
    store = make_memory_store()
    provider = provisioned(script=["answer"], thinking_script=["reason ", "more"])
    search = FakeSearchProvider(results=[])
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        events = await collect_events(c, {"message": "why is the sky blue?", "thinking": "on"})

    assert provider.last_tools is None
    assert any(n == "thinking" for n, _ in events)


async def test_assisted_mode_grounds_ambiently_and_still_offers_tools(db_sessionmaker) -> None:
    # assisted keeps today's pre-fetch (the KB record is retrieved and cited) AND
    # offers tools on top - the one-line clamp that adds tools without regressing
    # the grounded path.
    store = make_memory_store()
    await seed_kb(store)
    provider = citing_provider()
    search = FakeSearchProvider(results=[])
    settings = make_settings(agency_mode="assisted", rag_min_score=-1.0)
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        events = await collect_events(c, {"message": KB_TEXT, "web_search": "off"})

    # Ambient grounding still happened...
    assert any(n == "citation" for n, _ in events)
    # ...and tools were offered too (search_knowledge; web is off).
    assert offered_names(provider) == ["search_knowledge"]
    assert "tools_not_used" not in status_codes(events)


async def test_deterministic_mode_offers_no_tools_but_grounds(db_sessionmaker) -> None:
    # The kill switch: the pre-tool pipeline verbatim - grounding on, no tools.
    store = make_memory_store()
    await seed_kb(store)
    provider = citing_provider()
    settings = make_settings(agency_mode="deterministic", rag_min_score=-1.0)
    async with agency_client(provider, db_sessionmaker, settings, store=store) as c:
        events = await collect_events(c, {"message": KB_TEXT})

    assert provider.last_tools is None
    assert any(n == "citation" for n, _ in events)
    assert "tools_not_used" not in status_codes(events)


async def test_no_room_for_a_tool_result_reverts_to_the_deterministic_pipeline(
    db_sessionmaker,
) -> None:
    # Second hard floor: a window too small to hold a tool result offers no tools
    # even though the mode resolved agentic - and, crucially, falls back to the
    # FULL deterministic pipeline (web pre-fetch restored), NOT a silent agentic
    # turn stripped of both tools and grounding. So no tools reach the wire, the
    # web is pre-fetched (searching_web), and there is no tools_not_used notice.
    #
    # The verdict is seeded so the mode really does resolve agentic first: with
    # nothing declared the turn would land on assisted and the assertions below
    # could not tell the floor from the mode.
    await seed_agentic_verdict(db_sessionmaker)
    store = make_memory_store()
    # 256 tokens of native window - the runtime is the only thing that can say
    # a window is this small, so the fake reports it.
    provider = windowed_provider(256, script=["ok"])
    search = FakeSearchProvider(results=[])
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        events = await collect_events(c, {"message": "a short question", "web_search": "on"})

    codes = status_codes(events)
    assert provider.last_tools is None
    assert "tools_not_used" not in codes  # deterministic, not a stripped agentic turn
    assert "searching_web" in codes  # the pre-fetch was restored
    # A floored turn has no tools, so it must carry no tool guidance either.
    assert provider.calls[0][0].content == DEFAULT_SYSTEM_PROMPT


async def test_tools_offered_turn_layers_guidance_onto_the_persona(db_sessionmaker) -> None:
    # A tools-on turn carries the guidance layer: the persona alone actively
    # suppresses calls (its "outside available data, say so plainly" rule reads
    # a live lookup as that case - measured 0/6 must-call on qwen3:4b-instruct
    # without the layer, 6/6 with it). Both offered tools get their bullet.
    store = make_memory_store()
    provider = citing_provider()
    search = FakeSearchProvider(results=[])
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        await collect_events(c, {"message": "hello there"})

    system = provider.calls[0][0]
    assert system.role == "system"
    assert system.content.startswith(DEFAULT_SYSTEM_PROMPT)
    assert "- web_search:" in system.content
    assert "- search_knowledge:" in system.content


async def test_guidance_names_only_the_offered_tools(db_sessionmaker) -> None:
    # web off -> the web tool is not offered, and the guidance must not tell the
    # model to call it (the prompt would be inventing a tool).
    store = make_memory_store()
    provider = provisioned(script=["ok"])
    search = FakeSearchProvider(results=[])
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        await collect_events(c, {"message": "hello there", "web_search": "off"})

    system = provider.calls[0][0]
    assert "- search_knowledge:" in system.content
    assert "web_search" not in system.content


async def test_deterministic_turn_carries_no_guidance(db_sessionmaker) -> None:
    # No tools, no tool talk: the deterministic pipeline's prompt is byte-for-byte
    # the pre-tool persona.
    store = make_memory_store()
    provider = provisioned(script=["ok"])
    settings = make_settings(agency_mode="deterministic")
    async with agency_client(provider, db_sessionmaker, settings, store=store) as c:
        await collect_events(c, {"message": "hello there"})

    assert provider.calls[0][0].content == DEFAULT_SYSTEM_PROMPT


async def test_second_floor_strips_guidance_with_the_tools(db_sessionmaker) -> None:
    # The tool-result floor runs twice: on the base prompt in stream_chat, then
    # again in _generate at the ASSEMBLED prompt once grounding lands. When only
    # the second pass fails, the tools are dropped mid-assembly - and the
    # guidance layer appended when the offer still looked affordable must go
    # with them, or the model is told "you have real, working tools" while
    # holding none (an instruction to fabricate a lookup).
    #
    # A raised tool_result_min_tokens brackets the two floors deterministically
    # (measured with this exact persona + guidance + record): the base prompt is
    # ~503 tokens, so the first floor sees ~1673 tokens of room (>= 1450, tools
    # stay offered); the grounded ~250-token record plus scaffolding brings the
    # assembled room to ~1322 (< 1450), so only the second floor fires.
    store = make_memory_store()
    await seed_kb(store, text="The refund window for orders is 30 days from delivery. " * 18)
    provider = windowed_provider(3200, script=["ok"])
    search = FakeSearchProvider(results=[])
    settings = make_settings(
        agency_mode="assisted",
        rag_min_score=-1.0,
        tool_result_min_tokens=1450,
    )
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        events = await collect_events(c, {"message": "how do refunds work?", "web_search": "off"})

    assert "retrieving" in status_codes(events)
    # ...the second floor then stripped the tools from the wire...
    assert provider.last_tools is None
    # ...and the system message went back to the bare persona with them.
    assert provider.calls[0][0].content == DEFAULT_SYSTEM_PROMPT


async def test_assisted_tool_sources_share_the_ambient_numbering(db_sessionmaker) -> None:
    # Assisted grounds ambiently ([1] offered from the KB) AND offers tools. A
    # tool call that surfaces the SAME record must reuse [1], not mint [2]:
    # numbering is global to the turn, keyed on the document. The answer citing
    # [1] then yields exactly one citation event.
    store = make_memory_store()
    await seed_kb(store)
    provider = provisioned(
        turns=[
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c1", name="search_knowledge", arguments={"query": KB_TEXT})
                ]
            ),
            FakeTurn(tokens=["Refunds run 30 days", " [1]."]),
        ]
    )
    settings = make_settings(agency_mode="assisted", rag_min_score=-1.0)
    async with agency_client(provider, db_sessionmaker, settings, store=store) as c:
        events = await collect_events(c, {"message": KB_TEXT, "web_search": "off"})

    # The REAL SearchKnowledgeTool retrieved the seeded record; its digest
    # reused the ambient number instead of minting a new one.
    tool_msg = next(m for m in provider.calls[1] if m.role == "tool")
    assert "[1]" in tool_msg.content
    assert "[2]" not in tool_msg.content
    citation_indexes = [d.get("index") for n, d in events if n == "citation"]
    assert citation_indexes == [1]


# --- the earned-verdict wiring (slice 3b's consuming half) -----------------------

# Kept as an alias so the tests below read as being about the pair with its
# verdicts; the digests moved onto DEV_PAIR itself once EVERY agentic turn
# needed one.
PAIR_MODELS = DEV_PAIR


async def test_require_verification_with_no_probe_runs_deterministic(db_sessionmaker) -> None:
    # Hardened mode, never probed: an armed ceiling alone counts for nothing -
    # the grace that would otherwise carry an unmeasured model to `assisted` is
    # gone, so no tool reaches the wire at all.
    await store_policy(db_sessionmaker, ceiling="agentic")
    provider = provisioned(models=PAIR_MODELS)
    settings = make_settings(agency_require_verification=True)
    store = make_memory_store()
    async with agency_client(provider, db_sessionmaker, settings, store=store) as c:
        await collect_events(c, {"message": FRESH_QUESTION})

    assert provider.last_tools is None


async def test_require_verification_deep_verdict_enables_tools(db_sessionmaker) -> None:
    # The same hardened deployment, once a DEEP run earned agentic on exactly
    # these weights: the turn drives again.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")
    provider = provisioned(models=PAIR_MODELS)
    search = FakeSearchProvider(results=[SearchResult(title="t", url="https://x/", snippet="s")])
    settings = make_settings(agency_require_verification=True)
    store = make_memory_store()
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        await collect_events(c, {"message": FRESH_QUESTION, "web_search": "on"})

    assert set(offered_names(provider)) == {"web_search", "search_knowledge"}


async def _grounded_turn(db_sessionmaker) -> list[tuple[str, dict]]:
    """One turn whose ambient grounding is OBSERVABLE, holding everything else fixed.

    The question is the stored record verbatim (the hash embedder matches it at
    cosine 1.0) and the answer writes a ``[1]`` marker, so a ``citation`` event
    appears exactly when the pre-fetch ran. That single bit is what separates
    ``assisted`` from ``agentic`` on the wire: the tool OFFER is identical in
    both, and only the agentic turn withdraws the ambient knowledge base.
    """
    store = make_memory_store()
    await seed_kb(store)
    provider = citing_provider()
    settings = make_settings(rag_min_score=-1.0)
    async with agency_client(provider, db_sessionmaker, settings, store=store) as c:
        events = await collect_events(c, {"message": KB_TEXT, "web_search": "off"})
    assert offered_names(provider) == ["search_knowledge"]  # tools offered either way
    return events


async def test_grace_holds_a_fast_pass_at_assisted(db_sessionmaker) -> None:
    # The default regime with the real data shape a dev machine holds: a FAST
    # run that earned assisted is a pass at its depth CAP. That is not failure
    # evidence, so nothing CLAMPS - and it is not a deep certificate either, so
    # nothing LIFTS. The turn lands on the grace floor, where the ambient
    # knowledge base is still pre-fetched and cited.
    await seed_probe(db_sessionmaker, depth="fast", earned="assisted")
    events = await _grounded_turn(db_sessionmaker)
    assert any(n == "citation" for n, _ in events)


async def test_a_declared_ceiling_alone_does_not_withdraw_the_grounding(
    db_sessionmaker,
) -> None:
    # The state an operator lands in the moment they arm a model from the
    # console: the ceiling says agentic and NOTHING has been measured behind
    # it. The grounding stays. A declaration is a ceiling, not a grant - it can
    # only raise the bar a measurement is then allowed to clear.
    await store_policy(db_sessionmaker, ceiling="agentic")
    events = await _grounded_turn(db_sessionmaker)
    assert any(n == "citation" for n, _ in events)


async def test_grace_lifts_to_agentic_only_on_a_deep_pass(db_sessionmaker) -> None:
    # The other direction of the same rule, on the real path: the SAME
    # declaration and the SAME deployment, with a deep certificate bound to the
    # installed weights, withdraws the ambient pre-fetch and hands the turn
    # over. This is the only pairing of inputs that does.
    await seed_agentic_verdict(db_sessionmaker)
    events = await _grounded_turn(db_sessionmaker)
    assert not any(n == "citation" for n, _ in events)


# --- the stored declarations (the policy store's consuming half) -----------------


async def store_policy(
    db_sessionmaker, *, model: str = CHAT_MODEL, digest: str = CHAT_DIGEST, **fields
) -> None:
    """Append one operator declaration for a set of installed weights.

    Args:
        db_sessionmaker: The test session factory.
        model: The tag the declaration is about.
        digest: The installed digest the declaration is bound to.
        **fields: The axes this declaration sets; every other column stays NULL.
    """
    async with db_sessionmaker() as session:
        session.add(ModelPolicy(provider="fake", model=model, digest=digest, **fields))
        await session.commit()
    clear_policy_cache()


async def test_a_stored_clamp_takes_the_tools_off_a_real_turn(db_sessionmaker) -> None:
    # A deep check earned agentic on exactly these weights, so the EARNED axis
    # would let this turn drive - except that an operator clamped the model.
    # The clamp is the ceiling, and min(CEILING, EARNED) holds regardless of
    # what the log proves.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")
    await store_policy(db_sessionmaker, ceiling="deterministic")
    store = make_memory_store()
    provider = provisioned(script=["ok"])
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store) as c:
        await collect_events(c, {"message": FRESH_QUESTION})

    assert provider.last_tools is None


async def test_a_stored_clamp_declared_on_other_weights_does_not_apply(db_sessionmaker) -> None:
    # The auto-re-clamp read the other way round: the clamp was declared
    # against weights this server no longer holds, so it retires silently and
    # the turn falls back to what the runtime's report allows - which still
    # offers tools. Without the digest gate this turn would be clamped by a
    # decision about different weights.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")
    await store_policy(
        db_sessionmaker, digest="sha256:weights-we-replaced", ceiling="deterministic"
    )
    store = make_memory_store()
    provider = provisioned(script=["ok"])
    search = FakeSearchProvider(results=[SearchResult(title="t", url="https://x/", snippet="s")])
    settings = make_settings()
    async with agency_client(provider, db_sessionmaker, settings, store=store, search=search) as c:
        await collect_events(c, {"message": FRESH_QUESTION, "web_search": "on"})

    assert set(offered_names(provider)) == {"web_search", "search_knowledge"}


async def test_grace_ignores_a_deep_pass_the_weights_outran(db_sessionmaker) -> None:
    # And the certificate has to be bound to what is INSTALLED. The ceiling is
    # armed exactly as in the deep-pass test beside this one; only the verdict's
    # digest differs. A re-pull leaves that verdict describing weights this
    # machine no longer holds, so the turn falls back to the grace floor -
    # ambient grounding restored - rather than keeping a grant it no longer
    # earns.
    await store_policy(db_sessionmaker, ceiling="agentic")
    async with db_sessionmaker() as session:
        session.add(
            ModelProbe(
                provider="fake",
                model=CHAT_MODEL,
                digest="sha256:weights-we-replaced",
                harness_version=harness_version(),
                depth="deep",
                earned_agency="agentic",
            )
        )
        await session.commit()
    events = await _grounded_turn(db_sessionmaker)
    assert any(n == "citation" for n, _ in events)


async def test_grace_failed_probe_clamps_the_turn(db_sessionmaker) -> None:
    # A digest-bound guard failure (a FAST run earning deterministic) is
    # failure EVIDENCE and clamps even in the default regime: no tools.
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    provider = provisioned(models=PAIR_MODELS)
    settings = make_settings()
    store = make_memory_store()
    async with agency_client(provider, db_sessionmaker, settings, store=store) as c:
        await collect_events(c, {"message": FRESH_QUESTION})

    assert provider.last_tools is None
