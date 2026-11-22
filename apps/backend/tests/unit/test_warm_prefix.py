"""The warm prime and a real turn must open with the SAME prompt prefix.

A model server's prompt cache is prefix-MATCHED: it reuses evaluated tokens
only while the head of the new prompt is byte-identical to the head of one it
already holds. So a warm is worth exactly nothing unless what it sends starts
the way a real turn starts - and once a deployment advertises tools, most of
that head IS the tool block, which a chat template renders into its system
region alongside the system prompt.

These tests drive both sides through the real app and compare literally: the
warm endpoint's request against the chat endpoint's first request, on the same
settings and the same collaborators. They are deliberately end-to-end rather
than unit-level, because the failure mode being guarded is two code paths
drifting apart, and only a comparison that runs both can see it.

The suite also pins the two cases where the prefix is NOT a tool-carrying one -
an explicitly thinking turn on a model that cannot reason and call tools at
once, and a deployment with no collaborators wired - because a warm that
bolted a tool block onto those would mis-prime just as badly in the other
direction.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import get_model_provider, get_settings
from app.core.concurrency import StreamLimiter
from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.models.entities import ModelPolicy, ModelProbe
from app.providers.base import ChatMessage, ModelCapabilities, ModelInfo
from app.providers.fake_provider import FAKE_SERVING_ENDPOINT, FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.providers.search.base import SearchResult
from app.providers.search.searxng import FakeSearchProvider
from app.providers.tools.base import (
    Tool,
    ToolContext,
    ToolResult,
    ToolSource,
    ToolSpec,
)
from app.repositories.workspace_settings import WorkspaceSettingsRepository
from app.services import warmup
from app.services.context_budget import ANSWER_RESERVE_TOKENS, approx_tokens
from app.services.model_policy import clear_policy_cache
from app.services.prompts import DEFAULT_SYSTEM_PROMPT, tool_guidance
from app.services.tool_check import harness_version
from app.services.tools.local import SearchKnowledgeTool, WebSearchTool
from app.services.tools.offer import tools_to_wire
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store

CHAT_URL = "/api/v1/chat/stream"
WARM_URL = "/api/v1/models/warm"

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")

# The shipped dev shape: a tool-native pair that cannot reason and call tools
# in the same turn (qwen3-on-Ollama), with a separate thinking tag - so the
# thinking case genuinely routes to different weights. Those two facts come
# from the code-shipped quirk seed (app.services.known_models) for these exact
# tags, which is why the names are load-bearing.
CHAT_MODEL = "qwen3:4b-instruct"
THINKING_MODEL = "qwen3:4b"

CHAT_DIGEST = "sha256:chat"
THINKING_DIGEST = "sha256:think"
DEV_PAIR = [
    ModelInfo(name=CHAT_MODEL, digest=CHAT_DIGEST),
    ModelInfo(name=THINKING_MODEL, digest=THINKING_DIGEST),
]

# A plain question with no freshness cue and no reasoning hint, so the turn it
# provokes is the ordinary one a warm is priming for.
QUESTION = "What does this workspace sell?"

# A stored persona, nothing like the built-in one, so an assertion that it
# replaced rather than joined the default cannot pass by overlap.
PERSONA = "Answer as the workspace's own support desk."


class RemoteTool(Tool):
    """A tool shaped like one a registered MCP server offers.

    Read-only by default, because that is what the document-generation sidecar
    this deployment ships declares. The side-effecting spelling exists for the
    confirmation-gate case: which of those a client is offered decides how wide
    its tool block is, and therefore which prefix it opens with.
    """

    def __init__(self, name: str, *, side_effecting: bool = False) -> None:
        self._name = name
        self._side_effecting = side_effecting

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description="Generate a document from structured content.",
            parameters={
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
            side_effecting=self._side_effecting,
        )

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        return ToolResult(content="done")


class FakeMcpSource(ToolSource):
    """A stand-in for a registered server, landing where a real one's source does."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = list(tools)

    async def list_tools(self, context: ToolContext) -> list[Tool]:
        return list(self._tools)


class CountingEmbedder(FakeEmbedder):
    """A fake embedder that records how many QUERIES it was asked to embed.

    Tool selection embeds the query and nothing else on a warm cache, so the
    count is the honest test of whether the warm ranked at all - an order
    assertion alone could be satisfied by a ranking that happened to agree
    with registry order.
    """

    def __init__(self, *, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.queries = 0

    async def embed_query(self, text: str) -> list[float]:
        self.queries += 1
        return await super().embed_query(text)


async def seed_agentic_verdict(db_sessionmaker) -> None:
    """Arm BOTH pair members to drive a turn: each half of ``min(CEILING, EARNED)``.

    Two things this needs that a single row cannot give it.

    The ceiling an undeclared model resolves to is ``assisted`` - the runtime's
    report can vouch for a template, never for weights - so agentic takes a
    stored operator declaration, and the earned half takes a deep check bound
    to the installed digest.

    And both go on EACH member of the pair. A declaration is per tag and every
    tag is its own probe key, so arming only the direct-answer tag leaves the
    thinking sibling at the report's ``assisted`` - and the planner resolves
    the capability question against whichever member the raw thinking decision
    points at, which for a reasoning-shaped message is the sibling. That is a
    real deployment shape (an operator arms the pair they run), and it is what
    the tests below mean by "an agentic deployment".

    Args:
        db_sessionmaker: The test session factory.
    """
    async with db_sessionmaker() as session:
        for tag, digest in ((CHAT_MODEL, CHAT_DIGEST), (THINKING_MODEL, THINKING_DIGEST)):
            session.add(ModelPolicy(provider="fake", model=tag, digest=digest, ceiling="agentic"))
            session.add(
                ModelProbe(
                    provider="fake",
                    model=tag,
                    digest=digest,
                    harness_version=harness_version(),
                    serving_endpoint=FAKE_SERVING_ENDPOINT,
                    depth="deep",
                    earned_agency="agentic",
                )
            )
        await session.commit()
    clear_policy_cache()


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


def windowed_provider(native_window: int, **kwargs) -> FakeProvider:
    """A fake whose runtime report gives the pair a native context window.

    A window is ``min(native, afforded)`` and the native half is a RUNTIME
    fact, so a window under ``DEFAULT_CONTEXT_WINDOW``'s own 4096 floor can
    only be stated by the provider.

    Args:
        native_window: The context length the runtime reports for both tags.
        **kwargs: Passed through to :class:`FakeProvider`.

    Returns:
        The fake, listing the pair and reporting ``native_window`` for it.
    """
    report = ModelCapabilities(
        tools=True, embedding=True, completion=True, context_length=native_window
    )
    kwargs.setdefault("models", list(DEV_PAIR))
    kwargs.setdefault("capabilities_by_model", {CHAT_MODEL: report, THINKING_MODEL: report})
    return FakeProvider(**kwargs)


@asynccontextmanager
async def prefix_client(
    provider: FakeProvider,
    db_sessionmaker: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    rag: bool = True,
    search: bool = True,
    mcp_sources: list[ToolSource] | None = None,
    embedder: FakeEmbedder | None = None,
):
    """One app serving BOTH endpoints, so neither can be wired differently.

    That is the point of the harness: if the warm and the turn were given
    separate apps, a difference in their wiring could pass for a difference in
    their prompts, which is precisely the confusion these tests exist to end.
    """
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    app.state.job_queue = FakeJobQueue()
    app.state.vector_store = make_memory_store()
    if rag:
        app.state.embedder = embedder if embedder is not None else FakeEmbedder(dim=32)
    if search:
        app.state.search_provider = FakeSearchProvider(
            results=[SearchResult(title="t", url="https://x/", snippet="s")]
        )
    if mcp_sources is not None:
        app.state.mcp_tool_sources = mcp_sources
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def store_defaults(db_sessionmaker, **fields) -> None:
    """Write the workspace defaults both endpoints read."""
    async with db_sessionmaker() as session:
        await WorkspaceSettingsRepository(session, TENANT).merge_defaults(fields)
        await session.commit()


def system_of(messages: list[ChatMessage] | None) -> str:
    """The system message a request opened with, or "" when it carried none."""
    if not messages or messages[0].role != "system":
        return ""
    return messages[0].content


async def drain_chat(client: AsyncClient, payload: dict) -> None:
    """Run one chat turn to completion, discarding its events."""
    async with client.stream("POST", CHAT_URL, json=payload) as resp:
        assert resp.status_code == 200
        async for _line in resp.aiter_lines():
            pass


async def test_warm_primes_the_prefix_an_agentic_turn_opens_with(db_sessionmaker) -> None:
    # THE pin: the warm's system message and tool block, compared literally
    # against the ones the very next turn sends. Anything that diverts either
    # side by a single character fails here, which is the whole guarantee -
    # a prompt cache is prefix-matched, so "nearly the same" is worth nothing.
    await seed_agentic_verdict(db_sessionmaker)
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    async with prefix_client(provider, db_sessionmaker, settings) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_system = system_of(provider.last_messages)
        warm_tools = provider.last_tools
        warm_model = provider.last_model
        warm_num_ctx = provider.last_options.num_ctx
        warm_think = provider.last_options.think

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_system = system_of(provider.calls[-1])
        turn_tools = provider.last_tools
        turn_model = provider.last_model
        turn_num_ctx = provider.last_options.num_ctx
        turn_think = provider.last_options.think

    # The tool block really is present - a test that compared two empty offers
    # would pass while proving nothing.
    assert [tool["function"]["name"] for tool in warm_tools] == [
        "web_search",
        "search_knowledge",
    ]
    # ORDER included: the template renders this list verbatim, so a reordered
    # block is a different prefix even with identical members.
    assert json.dumps(warm_tools) == json.dumps(turn_tools)
    # The persona AND the tool-guidance layer that rides on it.
    assert warm_system == turn_system
    assert warm_system.startswith(DEFAULT_SYSTEM_PROMPT)
    assert warm_system != DEFAULT_SYSTEM_PROMPT
    # And the three request-level knobs that make the runtime reload or
    # re-render regardless of the text.
    assert warm_model == turn_model
    assert warm_num_ctx == turn_num_ctx
    assert warm_think == turn_think


async def test_a_wired_deployment_primes_more_than_an_unwired_one(db_sessionmaker) -> None:
    # What an operator sees: the warm response's prompt_tokens is how anyone
    # tells a persona-only prime from a tools-carrying one without reading the
    # code, and the tool block roughly quadruples it. The fake reports zero
    # usage, so the assertion is on the PROMPT this suite can measure - the
    # size the real number would count.
    await seed_agentic_verdict(db_sessionmaker)
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    async with prefix_client(provider, db_sessionmaker, settings) as client:
        with_tools = await client.post(WARM_URL, json={})
        assert with_tools.status_code == 200
        primed = system_of(provider.last_messages)
    async with prefix_client(
        provider, db_sessionmaker, settings, rag=False, search=False
    ) as client:
        bare = await client.post(WARM_URL, json={})
        assert bare.status_code == 200
        bare_primed = system_of(provider.last_messages)

    assert len(primed) > len(bare_primed)
    assert bare_primed == DEFAULT_SYSTEM_PROMPT


async def test_a_registered_servers_tools_ride_in_the_primed_prefix(db_sessionmaker) -> None:
    # The block this whole change is about: on the shipped deployment most of
    # the standing prefix is the REMOTE tools' schemas (a document-generation
    # sidecar's among them), so a warm that listed only the in-process pair
    # would still miss the turn's cache by most of its width.
    await seed_agentic_verdict(db_sessionmaker)
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    sources = [FakeMcpSource([RemoteTool("create_document")])]
    async with prefix_client(provider, db_sessionmaker, settings, mcp_sources=sources) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_system = system_of(provider.last_messages)
        warm_tools = provider.last_tools

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_system = system_of(provider.calls[-1])
        turn_tools = provider.last_tools

    assert [tool["function"]["name"] for tool in warm_tools] == [
        "web_search",
        "search_knowledge",
        "create_document",
    ]
    assert json.dumps(warm_tools) == json.dumps(turn_tools)
    # The guidance layer counts the remote tool too (its header and footer key
    # on "any tool at all"), so the two system messages match only if both
    # sides saw the same three.
    assert warm_system == turn_system


async def test_warm_primes_the_prefix_a_confirming_client_sends(db_sessionmaker) -> None:
    # A side-effecting tool is offered only to a client that can resume a
    # confirmation - so a deployment with one has TWO possible prefixes, and
    # the warm has to pick the browser's, the client whose first turn it is
    # there to speed up. Picking the other would prime a block narrower than
    # every UI turn's by exactly that schema.
    await seed_agentic_verdict(db_sessionmaker)
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    sources = [FakeMcpSource([RemoteTool("create_issue", side_effecting=True)])]
    async with prefix_client(provider, db_sessionmaker, settings, mcp_sources=sources) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_tools = provider.last_tools

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_tools = provider.last_tools

    assert "create_issue" in [tool["function"]["name"] for tool in warm_tools]
    assert json.dumps(warm_tools) == json.dumps(turn_tools)


async def test_above_the_selection_threshold_the_warm_takes_the_stable_head(
    db_sessionmaker,
) -> None:
    # Where prefix identity ENDS, pinned rather than left to the docstring.
    # Past the selection threshold the offered set becomes a function of the
    # QUESTION, so no single prefix can match every turn. The warm therefore
    # ranks nothing: it takes the core tools plus the registry-order head, and
    # spends no embedder call doing it.
    #
    # The zero-embed assertion is the load-bearing half. Ranking on the warm's
    # placeholder words would still produce SOME plausible order, so an order
    # check alone could pass by coincidence; an embed count cannot.
    await seed_agentic_verdict(db_sessionmaker)
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    remote = [RemoteTool(f"remote_{index}") for index in range(10)]
    assert len(remote) + 2 > settings.tool_selection_threshold, "fixture must clear the threshold"
    embedder = CountingEmbedder(dim=32)
    async with prefix_client(
        provider,
        db_sessionmaker,
        settings,
        embedder=embedder,
        mcp_sources=[FakeMcpSource(list(remote))],
    ) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_tools = provider.last_tools

    assert embedder.queries == 0, "the warm must not rank, and so must not embed"
    assert [tool["function"]["name"] for tool in warm_tools] == [
        "web_search",
        "search_knowledge",
        *[f"remote_{index}" for index in range(settings.tool_selection_top_k)],
    ]


async def test_the_warms_own_words_cannot_route_it_to_other_weights(
    db_sessionmaker, monkeypatch
) -> None:
    # The warm's message is a placeholder, not a question - so it must not be
    # able to decide which MODEL gets loaded. On a pair that cannot reason and
    # call tools at once, an agentic auto turn gives up its reasoning to keep
    # its tools and therefore runs on the direct-answer member; a warm that
    # picked its tag from the raw thinking heuristic instead would load the
    # thinking sibling, and on a card that fits one model the first real
    # question would evict exactly what the warm just paid to load.
    #
    # Driven by making the placeholder heuristically-thinking (three question
    # marks trip resolve_thinking), which is one edit away from being true by
    # accident - the invariant has to hold by construction rather than by the
    # current wording.
    monkeypatch.setattr(warmup, "_WARM_USER_MESSAGE", "Why? How? Which one???")
    await seed_agentic_verdict(db_sessionmaker)
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    async with prefix_client(provider, db_sessionmaker, settings) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_model = provider.last_model
        warm_tools = provider.last_tools

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_model = provider.last_model

    # The direct-answer member, on both sides - and the response names it.
    assert warm_model == turn_model == CHAT_MODEL
    assert resp.json()["model"] == CHAT_MODEL
    # And it still carries the tools that arbitration kept.
    assert [tool["function"]["name"] for tool in warm_tools] == [
        "web_search",
        "search_knowledge",
    ]


async def test_an_explicitly_thinking_warm_primes_a_toolless_prefix(db_sessionmaker) -> None:
    # A model that cannot reason and call tools at once runs an explicitly
    # thinking turn WITHOUT tools - so the warm for such a workspace must not
    # bolt a tool block onto a prefix no thinking turn ever carries. It keys on
    # the same arbitration the turn does, which is why it comes out right on a
    # deployment whose tools are otherwise fully wired.
    #
    # A registered server is wired on purpose: the local tools are withheld by
    # the offer's own booleans, so a remote source is the ONLY thing that can
    # tell a warm honouring the deterministic arbitration from one that merely
    # asked for the local pair and got nothing.
    await seed_agentic_verdict(db_sessionmaker)
    await store_defaults(db_sessionmaker, thinking="on")
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    sources = [FakeMcpSource([RemoteTool("create_document")])]
    async with prefix_client(provider, db_sessionmaker, settings, mcp_sources=sources) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_system = system_of(provider.last_messages)
        warm_tools = provider.last_tools
        warm_model = provider.last_model

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_system = system_of(provider.calls[-1])
        turn_tools = provider.last_tools
        turn_model = provider.last_model

    # The thinking sibling, on both sides - warming the direct-answer member
    # would prime weights this workspace's turns never load.
    assert warm_model == THINKING_MODEL
    assert turn_model == THINKING_MODEL
    # No tool block on either side, and the bare persona on both.
    assert not warm_tools
    assert not turn_tools
    assert warm_system == turn_system == DEFAULT_SYSTEM_PROMPT


async def test_a_workspace_persona_is_primed_by_the_warm(db_sessionmaker) -> None:
    # The persona is the HEAD of the prefix - the first bytes of the system
    # message - so a warm that composed without the workspace tier misses the
    # turn's cache from character one, and every first turn pays the whole
    # prefill while the warm still reports success.
    #
    # Three assertions, and the middle one is what makes this fail if the warm
    # loses the value: warm and turn must agree, the agreed text must be the
    # WORKSPACE persona, and the built-in one must be gone from it. Equality
    # alone would still pass if BOTH sides quietly composed the default.
    await seed_agentic_verdict(db_sessionmaker)
    await store_defaults(db_sessionmaker, system_prompt=PERSONA)
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    async with prefix_client(provider, db_sessionmaker, settings) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_system = system_of(provider.last_messages)
        warm_tools = provider.last_tools

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_system = system_of(provider.calls[-1])
        turn_tools = provider.last_tools

    assert warm_system == turn_system
    assert warm_system.startswith(PERSONA)
    assert DEFAULT_SYSTEM_PROMPT not in warm_system
    # The guidance layer still rides on it, on both sides - a persona rewrite
    # must not cost the model its tool instructions, and dropping the layer on
    # one side only would be a prefix mismatch too.
    assert warm_system != PERSONA
    assert json.dumps(warm_tools) == json.dumps(turn_tools)


async def test_a_workspace_persona_beats_the_environments_on_both_sides(db_sessionmaker) -> None:
    # The tier order has to be the same on both paths, not merely present on
    # both: a warm reading the env persona while the turn reads the workspace
    # one is a mismatch that no "is the workspace prompt wired" check catches.
    await seed_agentic_verdict(db_sessionmaker)
    await store_defaults(db_sessionmaker, system_prompt=PERSONA)
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings(system_prompt="Answer as the deployment configured.")
    async with prefix_client(provider, db_sessionmaker, settings) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_system = system_of(provider.last_messages)

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_system = system_of(provider.calls[-1])

    assert warm_system == turn_system
    assert warm_system.startswith(PERSONA)
    assert "Answer as the deployment configured." not in warm_system


async def test_a_web_off_workspace_warms_without_the_web_schema(db_sessionmaker) -> None:
    # "Web search off = fully local" is locked, so the web tool is withheld -
    # which changes the prefix, not just the behaviour. The warm has to follow
    # the stored switch or it primes a block with one extra schema in it.
    await seed_agentic_verdict(db_sessionmaker)
    await store_defaults(db_sessionmaker, web_search="off")
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    async with prefix_client(provider, db_sessionmaker, settings) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_system = system_of(provider.last_messages)
        warm_tools = provider.last_tools

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_system = system_of(provider.calls[-1])
        turn_tools = provider.last_tools

    assert [tool["function"]["name"] for tool in warm_tools] == ["search_knowledge"]
    assert json.dumps(warm_tools) == json.dumps(turn_tools)
    assert warm_system == turn_system


async def test_an_unwired_deployment_warms_the_bare_persona(db_sessionmaker) -> None:
    # No collaborators, no tools - for the warm AND for the turn. The prime
    # then has to be the persona alone: a tool block here would advertise
    # capabilities this deployment cannot build.
    await seed_agentic_verdict(db_sessionmaker)
    provider = FakeProvider(models=list(DEV_PAIR), script=["Sure."])
    settings = make_settings()
    async with prefix_client(
        provider, db_sessionmaker, settings, rag=False, search=False
    ) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_system = system_of(provider.last_messages)
        warm_tools = provider.last_tools

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_system = system_of(provider.calls[-1])
        turn_tools = provider.last_tools

    assert not warm_tools
    assert not turn_tools
    assert warm_system == turn_system == DEFAULT_SYSTEM_PROMPT


async def test_a_window_too_small_for_a_result_warms_without_tools(db_sessionmaker) -> None:
    # The turn's affordability floor drops the tools when a result could not
    # fit, and the warm applies the same floor on the same numbers - otherwise
    # it would prime a tool block for a deployment whose turns never send one.
    #
    # The window is sized so BOTH of the floor's easily-forgotten terms decide
    # the outcome: without the tool guidance the prompt would look affordable,
    # and without the tool SCHEMAS it would too. A window that was hopeless
    # before either term was counted (num_ctx at or under the answer reserve)
    # would pass this test while measuring nothing - the band assertions below
    # fail loudly if the fixture ever drifts back into that state.
    await seed_agentic_verdict(db_sessionmaker)
    num_ctx = 2048
    provider = windowed_provider(num_ctx, script=["Sure."])
    settings = make_settings()
    prompt = approx_tokens(DEFAULT_SYSTEM_PROMPT) + approx_tokens(warmup._WARM_USER_MESSAGE)
    guidance = approx_tokens(tool_guidance(web=True, knowledge=True, other=0))
    schemas = approx_tokens(
        json.dumps(
            tools_to_wire(
                [
                    WebSearchTool(provider=None, fetcher=None, settings=settings),
                    SearchKnowledgeTool(
                        embedder=None, store=None, reranker=None, settings=settings
                    ),
                ]
            )
        )
    )
    room = num_ctx - ANSWER_RESERVE_TOKENS - prompt
    floor = settings.tool_result_min_tokens
    assert room - guidance - schemas < floor, "the fixture must be unaffordable WITH both terms"
    assert room - schemas >= floor, "...and affordable if the guidance went uncounted"
    assert room - guidance >= floor, "...and affordable if the schemas went uncounted"

    async with prefix_client(provider, db_sessionmaker, settings) as client:
        resp = await client.post(WARM_URL, json={})
        assert resp.status_code == 200
        warm_system = system_of(provider.last_messages)
        warm_tools = provider.last_tools

        await drain_chat(client, {"message": QUESTION, "supports_confirmation": True})
        turn_tools = provider.last_tools

    assert not warm_tools
    assert not turn_tools
    assert warm_system == DEFAULT_SYSTEM_PROMPT
