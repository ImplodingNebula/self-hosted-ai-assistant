"""The verify job: the probe's order, its guards, and the endpoint that streams it.

Every test drives the REAL service - the same `run_verification` /
`stream_verification` the endpoint calls - with a scripted provider that
records each call it receives in one ordered list. That list is what makes the
order assertable: "the unload happened" and "the unload happened BEFORE the
probe" are different claims, and only the second one is worth having.
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import get_args

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_tenant, get_model_provider
from app.api.v1.models import _FRAME_NAMES, hold_verify_claim
from app.core.config import Settings, get_settings
from app.core.errors import BadRequestError, ConflictError, NotFoundError
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.models.entities import ModelPolicy, ModelProbe
from app.providers.base import (
    ChatChunk,
    ManagementCapabilities,
    ModelCapabilities,
    ModelInfo,
    ModelResidency,
    ProviderUnavailableError,
    ToolCall,
    Usage,
)
from app.providers.fake_provider import FakeProvider
from app.repositories.model_probes import ModelProbeRepository
from app.repositories.workspace_settings import WorkspaceSettingsRepository
from app.schemas.models import (
    ModelVerifyRequest,
    PullDoneEvent,
    VerifyDoneEvent,
    VerifyErrorEvent,
    VerifyProgressEvent,
)
from app.schemas.settings import MIN_CONTEXT_CEILING, WorkspaceDefaultsPatch
from app.services import model_pull, model_verify
from app.services.earned import clear_earned_cache, effective_verdicts
from app.services.model_policy import resolve_policy
from app.services.model_resolution import (
    resolve_chat_models,
    thinking_sibling_for,
)
from app.services.prompts import DEFAULT_SYSTEM_PROMPT
from app.services.tool_check import THINK_CASE_NAMES, expected_sample_count, harness_version
from app.services.workspace_defaults import apply_defaults

# One progress frame per sample, so the counts here are derived rather than
# written down: a case added to the harness must not need a number edited in
# five places, and a number that had to be edited would be a number nobody
# could tell from a real regression.
FAST_SAMPLES = expected_sample_count(deep=False)
DEEP_SAMPLES = expected_sample_count(deep=True)
FAST_FRAMES = ["verify_progress"] * FAST_SAMPLES + ["verify_done"]

CHAT = "chat-tag"
THINK = "think-tag"
EMBED = "embed-tag"
CHAT_DIGEST = "sha256:chatweights"
THINK_DIGEST = "sha256:thinkweights"
# A partner an operator paired from the console, and therefore the tag that
# tells a resolution reading only the environment apart from the one a turn
# actually routes on.
DECLARED_THINK = "declared-think"
DECLARED_THINK_DIGEST = "sha256:declaredweights"
# A tag nothing in this module configures - the free-text pull's shape.
MYSTERY = "mystery:1b"
MYSTERY_DIGEST = "sha256:mysteryweights"

# What the runtime reports about each tag, and it is load-bearing twice over:
# the ROLE the verify job refuses on is derived from it, and so is the think
# mode a tool case is driven at when nothing is declared.
#
# The direct-answer tags carry a template with NO think plumbing, which
# resolves to "never" and OMITS the flag - the answer a stored declaration then
# has something to change. The reasoning partners carry one, so a case routed
# to them is driven with the flag set, which is what makes the reasoning-with-
# tools case measure what its name says.
DIRECT_REPORT = ModelCapabilities(completion=True, tools=True, template_thinking=False)
REASONING_REPORT = ModelCapabilities(completion=True, tools=True, template_thinking=True)
EMBED_REPORT = ModelCapabilities(embedding=True)

REPORTS: dict[str, ModelCapabilities] = {
    CHAT: DIRECT_REPORT,
    "chat-tag:latest": DIRECT_REPORT,
    MYSTERY: DIRECT_REPORT,
    "ws-tag": DIRECT_REPORT,
    THINK: REASONING_REPORT,
    DECLARED_THINK: REASONING_REPORT,
    EMBED: EMBED_REPORT,
}

# A non-admin of the same tenant, for the role gate.
MEMBER = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="member")

INSTALLED = [
    ModelInfo(name=CHAT, digest=CHAT_DIGEST),
    ModelInfo(name=THINK, digest=THINK_DIGEST),
    ModelInfo(name=EMBED, digest="sha256:embedweights"),
]


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    model_verify.clear_verifications()
    model_pull.clear_pulls_in_flight()
    yield
    model_verify.clear_verifications()
    model_pull.clear_pulls_in_flight()


class ProbeProvider(FakeProvider):
    """A model that passes both fast cases, recording every call it receives.

    The recording is one ordered list across unloads, generations, warms and
    embeds - not a counter per verb - because the properties worth pinning are
    about ORDER: that the card was freed before anything was measured, and that
    the re-warm happened after the measuring stopped.

    A warm generation is told apart from a probe generation by its one-token
    cap, exactly as :mod:`app.services.warmup` sends it.
    """

    def __init__(self, *, fail: Exception | None = None, **kwargs) -> None:
        kwargs.setdefault("models", list(INSTALLED))
        # Merged rather than replaced, so a test that describes one extra tag
        # does not silently take the roles and think modes away from the rest.
        kwargs["capabilities_by_model"] = {
            **REPORTS,
            **(kwargs.get("capabilities_by_model") or {}),
        }
        super().__init__(**kwargs)
        self.events: list[tuple[str, str]] = []
        # The serve-time options every MEASURING generation requested, in
        # order, one list each. Kept apart from ``events`` because they answer
        # a different question - not what happened when, but what configuration
        # was measured - and apart from each other because they move
        # independently.
        self.windows: list[int | None] = []
        self.thinks: list[tuple[str, bool | None]] = []
        # The system message each WARM generation opened with. The re-warm's
        # PREFIX is a second thing a stored default moves, alongside the weights
        # and the window, and nothing else this fake records exposes it.
        self.warm_systems: list[str] = []
        self._fail = fail

    async def chat_stream(self, messages, *, model=None, tools=None, options=None):
        warming = options is not None and options.max_tokens == 1
        self.events.append(("warm" if warming else "chat", model or ""))
        if warming:
            head = messages[0] if messages else None
            self.warm_systems.append(
                head.content if head is not None and head.role == "system" else ""
            )
        if not warming:
            self.windows.append(options.num_ctx if options is not None else None)
            self.thinks.append((model or "", options.think if options is not None else None))
        if warming:
            yield ChatChunk(type="done", finish_reason="stop", usage=Usage(prompt_tokens=3))
            return
        if self._fail is not None:
            raise self._fail
        if options is not None and options.think:
            # A model asked to reason, reasons - the same fidelity the harness's
            # own fake keeps. Case (h) passes only on reasoning AND a call AND
            # an answer, so a fake that answered identically whatever flag it
            # was sent would let a deep run here certify reasoning-with-tools
            # from an exchange in which nothing reasoned.
            yield ChatChunk(type="thinking", delta="thinking it through")
        offered = [tool["function"]["name"] for tool in tools or []]
        user = next(m.content for m in reversed(messages) if m.role == "user")
        # Every answer streams at least TWO token chunks. The throughput
        # aggregate is measured over token INTERVALS (tokens minus one, per
        # round), so a one-chunk answer makes every run in this module report
        # "no measurable throughput" - and a test asserting that a spilled run
        # publishes none would then hold whether or not the code suppressed it.
        if messages[-1].role == "tool":
            yield ChatChunk(type="token", delta="Answered from ")
            yield ChatChunk(type="token", delta="the tool result.")
        elif "2 + 2" in user or not offered:
            yield ChatChunk(type="token", delta="4")
            yield ChatChunk(type="token", delta=" exactly.")
        else:
            wants_kb = "refund" in user or "returns" in user
            name = "search_knowledge" if wants_kb and "search_knowledge" in offered else offered[0]
            yield ChatChunk(
                type="tool_call", tool_call=ToolCall(id="t1", name=name, arguments={"query": user})
            )
        yield ChatChunk(type="done", finish_reason="stop", usage=Usage())

    async def embed(self, texts, *, model=None, options=None):
        self.events.append(("embed", model or ""))
        return await super().embed(texts, model=model, options=options)

    async def unload_model(self, model: str) -> bool:
        self.events.append(("unload", model))
        return await super().unload_model(model)

    def kinds(self, kind: str) -> list[str]:
        """Return the models this verb was called with, in call order."""
        return [name for seen, name in self.events if seen == kind]


class UnreachableProvider(ProbeProvider):
    """A model server that answers every generation with a provider error.

    The shape the real Ollama provider produces: a typed error CHUNK rather
    than a raised exception, which is exactly why a caller cannot simply catch
    its way to the right answer. The default text is a connect failure, but the
    same code carries an out-of-memory refusal from the model server's own
    response body - which is why the message is a parameter.
    """

    def __init__(self, *, error: str = "Cannot reach the model server.", **kwargs) -> None:
        super().__init__(**kwargs)
        self.error = error

    async def chat_stream(self, messages, *, model=None, tools=None, options=None):
        warming = options is not None and options.max_tokens == 1
        self.events.append(("warm" if warming else "chat", model or ""))
        yield ChatChunk(type="error", error=self.error, error_code="provider_unavailable")


class OneModelCardProvider(ProbeProvider):
    """A card that fits exactly one model: whatever generated last is resident.

    The dev tier's real shape, and the reason the residency reading is taken
    DURING the run: a deep check's last case drives the pair's thinking tag,
    which evicts the very tag the measurement row is about.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resident: str | None = None

    async def chat_stream(self, messages, *, model=None, tools=None, options=None):
        if options is None or options.max_tokens != 1:
            self.resident = model
        async for chunk in super().chat_stream(messages, model=model, tools=tools, options=options):
            yield chunk

    async def residency(self, model: str) -> ModelResidency | None:
        if self.resident != model:
            return ModelResidency(model=model, loaded=False)
        return ModelResidency(model=model, loaded=True, total_bytes=4_000, vram_bytes=4_000)

    async def unload_model(self, model: str) -> bool:
        if self.resident == model:
            self.resident = None
        return await super().unload_model(model)


class FlakyProvider(ProbeProvider):
    """One generation loses the connection; the rest are fine.

    The ordinary shape of a busy box, and the dangerous one: the harness scores
    an unreachable sample as a case FAILURE, so a single blip in a two-sample
    check earns "deterministic" - a verdict the log then reads as evidence
    against the model.

    ``think_only`` puts the blip in the THINKING group instead, which the
    harness offers no tools - the other side of the scoping rule, where losing
    the run would throw away an agency verdict that was measured perfectly well.
    """

    def __init__(self, *, think_only: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.generations = 0
        self._think_only = think_only
        self._blipped = False

    async def chat_stream(self, messages, *, model=None, tools=None, options=None):
        warming = options is not None and options.max_tokens == 1
        if not warming:
            self.generations += 1
            # The offered tool list is what tells the two groups apart on the
            # wire: the thinking cases are the only ones driven with none.
            target_group = (not tools) if self._think_only else bool(tools)
            if target_group and not self._blipped:
                self._blipped = True
                self.events.append(("chat", model or ""))
                yield ChatChunk(
                    type="error",
                    error="Cannot reach the model server.",
                    error_code="provider_unavailable",
                )
                return
        async for chunk in super().chat_stream(messages, model=model, tools=tools, options=options):
            yield chunk


class SlotWatchingProvider(ProbeProvider):
    """Records who held the probe slot at the moment the re-warm ran.

    The re-warm loads models onto the same card the probe was measuring, so the
    slot has to outlast it: releasing first opens a window where a second probe
    can start unloading what this one is busy putting back.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.slot_during_rewarm: list[str | None] = []

    async def chat_stream(self, messages, *, model=None, tools=None, options=None):
        if options is not None and options.max_tokens == 1:
            self.slot_during_rewarm.append(model_verify.verification_in_flight())
        async for chunk in super().chat_stream(messages, model=model, tools=tools, options=options):
            yield chunk


class LateCpuCardProvider(OneModelCardProvider):
    """A one-model card whose first residency reading misses, and that runs on the CPU.

    Both halves are real: a runtime can answer before it has registered a load,
    and a model can end up entirely in host memory. Together they are the case
    a single early reading gets wrong in the dangerous direction.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reads = 0

    async def residency(self, model: str) -> ModelResidency | None:
        self.reads += 1
        if self.reads == 1:
            return ModelResidency(model=model, loaded=False)
        view = await super().residency(model)
        if view is not None and view.loaded:
            return ModelResidency(model=model, loaded=True, total_bytes=4_000, vram_bytes=0)
        return view


def make_settings(
    *,
    heartbeat: float = 0,
    chat: str = CHAT,
    thinking: str | None = THINK,
) -> Settings:
    """The deployment's configuration: a pair-routed chat tag, by default.

    Heartbeats are off by default so a stream's frames are exactly its own; one
    test raises it to the deployment value to exercise the other branch. The
    two model names are parameters for the tests about which partner a run
    drives - the environment is the strongest rung of that ranking, and a
    stored declaration is the only other one.
    """
    return Settings(
        model_provider="fake",
        default_chat_model=chat,
        thinking_chat_model=thinking,
        sse_heartbeat_seconds=heartbeat,
        _env_file=None,
    )


# What this deployment affords one turn on a tag whose weights report no native
# window - which is every tag here. Read off the settings rather than written
# down, so a changed default moves the assertions with it.
DEPLOYMENT_WINDOW = make_settings().default_context_window


def tenant() -> TenantContext:
    return TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


async def begin(provider, *, name: str = CHAT) -> str:
    """Run the refusals-and-claim step exactly as the route's dependency does."""
    return await model_verify.begin_verification(provider, provider_name="fake", name=name)


async def run(provider, sessionmaker, *, deep: bool = False, model: str = CHAT):
    """Drive the service end to end (steps 2-5), as the stream does."""
    return await model_verify.run_verification(
        provider=provider,
        settings=make_settings(),
        sessionmaker=sessionmaker,
        tenant=tenant(),
        model=model,
        deep=deep,
    )


async def drain(provider, sessionmaker, *, deep: bool = False):
    """Collect every event the streaming wrapper yields (steps 1-6)."""
    return [
        event
        async for event in model_verify.stream_verification(
            provider=provider,
            settings=make_settings(),
            sessionmaker=sessionmaker,
            tenant=tenant(),
            model=CHAT,
            deep=deep,
        )
    ]


@asynccontextmanager
async def verify_client(
    provider,
    sessionmaker,
    *,
    principal: TenantContext | None = None,
    heartbeat: float = 0,
) -> AsyncIterator[AsyncClient]:
    """App-wired client over the sqlite measurement log."""
    settings = make_settings(heartbeat=heartbeat)
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    if principal is not None:
        app.dependency_overrides[get_current_tenant] = lambda: principal
    app.state.db_sessionmaker = sessionmaker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


def frames(body: str) -> list[tuple[str, str]]:
    """Parse an SSE body into (event, data) pairs, dropping keep-alive comments."""
    parsed: list[tuple[str, str]] = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if line and not line.startswith(":")]
        if not lines:
            continue
        event = next((line[len("event: ") :] for line in lines if line.startswith("event: ")), "")
        data = next((line[len("data: ") :] for line in lines if line.startswith("data: ")), "")
        parsed.append((event, data))
    return parsed


async def rows(sessionmaker, *, model: str = CHAT) -> list[ModelProbe]:
    async with sessionmaker() as session:
        return await ModelProbeRepository(session).history("fake", model)


async def store_ceiling(sessionmaker, ceiling: int) -> None:
    """Set the workspace's stored context ceiling, as the settings page does."""
    async with sessionmaker() as session:
        await WorkspaceSettingsRepository(session, tenant()).merge_defaults(
            {"context_ceiling": ceiling}
        )
        await session.commit()


# --- the order the job happens in ---------------------------------------------


async def test_the_card_is_freed_before_anything_is_measured(db_sessionmaker) -> None:
    # The claim worth pinning is the ORDER, not the presence: an unload that
    # lands after the first sample has measured a load that raced the runtime's
    # own eviction, which is the exact thing this step exists to avoid.
    provider = ProbeProvider()
    await run(provider, db_sessionmaker)

    kinds = [kind for kind, _ in provider.events]
    assert "unload" in kinds and "chat" in kinds
    assert max(i for i, k in enumerate(kinds) if k == "unload") < kinds.index("chat")


async def test_the_target_and_everything_it_would_evict_are_unloaded(
    db_sessionmaker,
) -> None:
    provider = ProbeProvider()
    await run(provider, db_sessionmaker)

    unloaded = provider.kinds("unload")
    # The target first - a stale runner for it is the one thing that would
    # silently change what the run measures - then the deployment's own pair,
    # which the probe would have evicted anyway. No workspace has chosen an
    # embedder here, but the inventory holds exactly one embedding-capable
    # tag, and _free_the_card resolves through that rung too - so it is freed
    # as well, rather than staying resident through a run that exists to
    # measure a fresh load. The test below covers the stored-choice rung.
    assert unloaded[0] == CHAT
    assert set(unloaded) == {CHAT, THINK, EMBED}


async def test_the_embedder_freed_is_the_one_the_workspace_chose(db_sessionmaker) -> None:
    # Freeing the card means naming what is ON it, and the embedder in use is a
    # workspace choice: unloading some other tag drops a model that is not
    # resident and leaves the one that IS competing with the target through
    # every sample, which corrupts the reading rather than annoying it. Read the
    # stored choice from anywhere but the workspace and this fails on the tuple.
    async with db_sessionmaker() as session:
        await apply_defaults(
            session, tenant(), WorkspaceDefaultsPatch(embedding_model="chosen-embed:4b")
        )
        await session.commit()
    provider = ProbeProvider()
    await run(provider, db_sessionmaker)

    unloaded = provider.kinds("unload")
    # The chosen embedder, beside the deployment's own pair - and nothing else,
    # so a resolution that fell back to some other tag shows up as an extra
    # name rather than passing on the one it got right.
    assert set(unloaded) == {CHAT, THINK, "chosen-embed:4b"}


async def test_a_backend_that_cannot_unload_is_probed_anyway(db_sessionmaker) -> None:
    # Never a refusal: a runtime with no unload verb can still be measured, and
    # the reading is simply less controlled. The alternative - refusing to
    # verify at all - would leave such a deployment with no measurements.
    provider = ProbeProvider(
        management=ManagementCapabilities(
            can_pull=True, can_delete=True, can_list_installed=True, can_unload=False
        )
    )
    outcome = await run(provider, db_sessionmaker)

    assert provider.kinds("unload") == []
    assert outcome.summary.earned == "assisted"


# --- one probe at a time ------------------------------------------------------


async def test_a_second_probe_is_refused_while_one_runs() -> None:
    model_verify.claim_verification("something-else")
    with pytest.raises(ConflictError) as refusal:
        await begin(ProbeProvider())
    # The message names what IS running: "a check is already running" without
    # saying which one reads like a bug in the page.
    assert "something-else" in str(refusal.value)


async def test_only_the_holder_can_release_the_slot() -> None:
    # A release keyed on nothing at all is a silent hazard in the case that
    # matters: a path releasing a claim it never took hands somebody else's
    # running probe to a third caller, and nothing anywhere says so.
    model_verify.claim_verification(CHAT)
    model_verify.release_verification("some-other-model")
    assert model_verify.verification_in_flight() == CHAT
    model_verify.release_verification(CHAT)
    assert model_verify.verification_in_flight() is None


async def test_the_guard_is_not_per_model() -> None:
    # Unlike a download, two DIFFERENT models are just as bad as the same one
    # twice: each probe's model load is what the other's residency reading sees.
    model_verify.claim_verification(CHAT)
    with pytest.raises(ConflictError):
        model_verify.claim_verification("a-completely-different:model")


async def test_a_probe_of_an_uninstalled_model_is_refused_before_it_claims() -> None:
    provider = ProbeProvider(models=[ModelInfo(name="other:1b")])
    with pytest.raises(NotFoundError):
        await begin(provider)
    assert model_verify.verification_in_flight() is None


async def test_a_probe_of_a_model_the_runtime_says_cannot_chat_is_refused() -> None:
    # The harness measures tool-calling. Running it against the embedder would
    # evict the card for minutes and then append a "deterministic" verdict for
    # a model that was never going to drive a turn.
    #
    # The refusal reads the RUNTIME's report now - nothing declares a role any
    # more - so the message says what the server reported rather than what a
    # file claimed.
    with pytest.raises(BadRequestError, match="embedding model"):
        await begin(ProbeProvider(), name=EMBED)
    assert model_verify.verification_in_flight() is None


async def test_a_probe_of_a_tag_the_runtime_will_not_describe_goes_ahead() -> None:
    # The other side of the same rule, and the one it would be easy to get
    # backwards: "the runtime would not say" is not evidence that a model
    # cannot chat. A free-text pull against a half-answering server is exactly
    # the tag somebody most needs a measurement about, so an unreadable report
    # is checked rather than refused.
    #
    # MUTATION: make `_wrong_role_reason` refuse on a None report and this
    # raises BadRequestError instead of claiming the slot.
    class Silent(ProbeProvider):
        async def capabilities(self, model: str):
            raise ProviderUnavailableError("The model server will not describe its models.")

    assert await begin(Silent()) == CHAT
    assert model_verify.verification_in_flight() == CHAT


# --- residency ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("view", "representative"),
    [
        # No reading at all: unknown is not evidence.
        (None, True),
        (ModelResidency(model=CHAT, loaded=False), True),
        (ModelResidency(model=CHAT, loaded=True), True),
        # A partial split is how a small card runs a model at all.
        (ModelResidency(model=CHAT, loaded=True, total_bytes=1000, vram_bytes=600), True),
        (ModelResidency(model=CHAT, loaded=True, total_bytes=1000, vram_bytes=1000), True),
        # No GPU share worth the name: the run measured a machine the
        # deployment does not serve from.
        (ModelResidency(model=CHAT, loaded=True, total_bytes=1000, vram_bytes=0), False),
    ],
)
def test_only_a_run_with_no_gpu_share_is_unrepresentative(view, representative: bool) -> None:
    assert model_verify.assess_residency(view)[0] is representative


async def test_a_cpu_bound_run_is_marked_rather_than_trusted(db_sessionmaker) -> None:
    provider = ProbeProvider(
        residency=ModelResidency(model=CHAT, loaded=True, total_bytes=4_000, vram_bytes=0)
    )
    outcome = await run(provider, db_sessionmaker)

    assert outcome.representative is False
    assert outcome.warning and "CPU" in outcome.warning
    # The verdicts are kept exactly as measured - a spill does not make a
    # model's tool calls better or worse...
    assert outcome.summary.earned == "assisted"
    # ...but a CPU-bound decode rate is a fact about the spill, so it is
    # dropped rather than published as this model's throughput.
    assert outcome.summary.tokens_per_second is None

    (row,) = await rows(db_sessionmaker)
    assert row.cases["representative"] is False
    assert row.cases["residency"]["gpu_fraction"] == 0.0
    assert row.tokens_per_second is None


async def test_residency_is_read_while_the_run_still_owns_the_card(
    db_sessionmaker,
) -> None:
    # On a card that fits one model, a deep run's last case drives the pair's
    # THINKING tag and evicts the chat tag this row is about. A reading taken
    # only after the run would report "not loaded" for a model that had been on
    # the GPU the whole time it was measured - and an unknown placement is one
    # the operator cannot act on.
    provider = OneModelCardProvider()
    outcome = await run(provider, db_sessionmaker, deep=True)

    assert provider.resident == THINK  # the chat tag really was evicted
    assert outcome.residency is not None and outcome.residency.loaded is True
    assert outcome.residency.vram_bytes == 4_000
    assert outcome.representative is True


async def test_an_early_missed_reading_does_not_settle_the_question(
    db_sessionmaker,
) -> None:
    # A reading can legitimately miss - the runtime answers before it has
    # registered the load - and keeping that first miss would let a run that
    # spent every sample on the CPU report an UNKNOWN placement, which is
    # exactly the run the marker exists for. The end-of-run fallback cannot
    # rescue it either: by then the pair's thinking tag owns the card.
    provider = LateCpuCardProvider()
    outcome = await run(provider, db_sessionmaker, deep=True)

    assert provider.reads > 1  # it asked again rather than trusting the miss
    assert outcome.representative is False
    assert outcome.warning is not None


async def test_a_gpu_resident_run_publishes_what_it_measured(db_sessionmaker) -> None:
    provider = ProbeProvider(
        residency=ModelResidency(model=CHAT, loaded=True, total_bytes=4_000, vram_bytes=3_000)
    )
    outcome = await run(provider, db_sessionmaker)

    assert outcome.representative is True
    assert outcome.warning is None
    # The other half of the suppression rule, and the half that makes the
    # spilled-run assertion mean something: a representative run DOES publish
    # the throughput it measured.
    assert outcome.summary.tokens_per_second is not None
    (row,) = await rows(db_sessionmaker)
    assert row.cases["representative"] is True
    assert row.vram_bytes == 3_000
    assert row.tokens_per_second == outcome.summary.tokens_per_second


# --- what gets recorded -------------------------------------------------------


async def test_one_run_records_exactly_one_bound_measurement(db_sessionmaker) -> None:
    provider = ProbeProvider(
        runtime_version="0.32.0",
        serving_endpoint="https://pod-abc.proxy.runpod.net",
        residency=ModelResidency(model=CHAT, loaded=True, total_bytes=4_000, vram_bytes=4_000),
    )
    outcome = await run(provider, db_sessionmaker)

    written = await rows(db_sessionmaker)
    assert len(written) == 1
    row = written[0]
    assert row.provider == "fake"
    assert row.model == CHAT
    assert row.depth == "fast"
    assert row.earned_agency == "assisted"
    # Bound to the WEIGHTS, the serving stack AND the machine, never to the
    # mutable tag alone: a re-pull, a runtime upgrade or a move to another
    # server each has to be able to invalidate this.
    assert row.digest == CHAT_DIGEST
    assert row.runtime_version == "0.32.0"
    assert row.serving_endpoint == "https://pod-abc.proxy.runpod.net"
    assert row.vram_bytes == 4_000
    assert row.cases["thinking_model"] == THINK
    assert row.cases["thinking_digest"] == "sha256:thinkweights"
    # Audit-only columns, populated because a UI click has an actor where a
    # terminal run does not.
    assert row.triggered_by_tenant_id == DEFAULT_TENANT_ID
    assert row.triggered_by_user_id == DEFAULT_USER_ID
    assert outcome.probe_id == str(row.id)


async def test_a_failed_write_still_reports_the_measurement(db_sessionmaker) -> None:
    # The expensive part - GPU minutes - already happened. Reporting a database
    # problem as a failed verification would send someone to re-run the probe
    # when the thing to fix is the database, so the verdict comes back with a
    # null probe id instead.
    class BrokenSessionmaker:
        def __call__(self):
            raise RuntimeError("the database is down")

    outcome = await run(ProbeProvider(), BrokenSessionmaker())

    assert outcome.probe_id is None
    assert outcome.summary.earned == "assisted"  # the measurement itself stands
    assert await rows(db_sessionmaker) == []


async def test_an_unreachable_server_is_not_recorded_as_a_verdict(
    db_sessionmaker,
) -> None:
    # The dangerous case: every sample "fails", so a naive run would record
    # "deterministic" and clamp a perfectly healthy model on the strength of a
    # network outage. The measurement did not happen, so nothing is written.
    provider = UnreachableProvider()
    with pytest.raises(ProviderUnavailableError):
        await run(provider, db_sessionmaker)
    assert await rows(db_sessionmaker) == []


async def test_what_the_model_server_actually_said_survives_the_discard(
    db_sessionmaker, caplog
) -> None:
    # The error code is broader than its name: the provider raises
    # "provider_unavailable" for a connect failure, a generation timeout, a
    # runner crash, AND any error line in the backend's own response body -
    # which is where a model too large for the card reports that it needs more
    # memory than the machine has. Discarding the run throws away every
    # sample's detail, so the one sentence naming the real problem has to be
    # logged before it goes.
    provider = UnreachableProvider(
        error="model requires more system memory (5.6 GiB) than is available (3.1 GiB)"
    )
    with caplog.at_level("WARNING"), pytest.raises(ProviderUnavailableError) as refused:
        await run(provider, db_sessionmaker)

    assert "more system memory" in caplog.text
    # ...and the message a user sees names BOTH possibilities rather than
    # sending them to check a stack that is plainly up.
    assert "could not run on this machine" in str(refused.value)


async def test_even_one_unreachable_sample_discards_the_run(db_sessionmaker) -> None:
    # ONE blip in the AGENCY group is enough. The harness scores an unreachable
    # sample as a case failure, and a fast run that did not pass everything
    # earns "deterministic" - which the log reads as evidence against the model
    # and clamps the running deployment with. Discarding a partly-measured run
    # is much cheaper than quietly demoting a healthy model, and this now
    # happens unattended after every download.
    provider = FlakyProvider()
    with pytest.raises(ProviderUnavailableError):
        await run(provider, db_sessionmaker)
    assert provider.generations > 1  # the rest of the run really did succeed
    assert await rows(db_sessionmaker) == []


async def test_a_blip_in_the_thinking_group_alone_keeps_the_run(db_sessionmaker) -> None:
    # The other direction, and the scoping this discard needs. The thinking
    # group decides NOTHING about agency - it cannot blame a case, and its own
    # classifier already refuses to read an errored sample - so losing the run
    # over a blip confined to it would throw away an agency verdict that was
    # measured perfectly well. On the automatic post-download check, that is
    # the only verdict a model gets.
    provider = FlakyProvider(think_only=True)
    outcome = await run(provider, db_sessionmaker)

    assert outcome.summary.earned == "assisted"  # the agency verdict survived
    assert outcome.summary.measured_think_mode is None  # and the think mode did not
    stored = await rows(db_sessionmaker)
    assert len(stored) == 1
    assert stored[0].cases["measured_think_mode"] is None


# --- the window a probe runs at -----------------------------------------------


async def test_the_probe_runs_at_the_window_this_deployment_serves(
    db_sessionmaker,
) -> None:
    # The defect: the harness sized every generation from a committed per-model
    # file, which answered nothing for a tag it did not name - so a check on a
    # free-text pull ran unsized (the runtime picking its own VRAM-scaled
    # default) while every real turn on that tag asked for a resolved window.
    # The verdict then certified a configuration nothing serves, and the two
    # different requests made the runtime reload the model between the check
    # and the re-warm behind it.
    #
    # The stored ceiling is the other half: a workspace that clamped its turns
    # had every one of them clamped while the probe certified the unclamped
    # window.
    provider = ProbeProvider(models=[*INSTALLED, ModelInfo(name=MYSTERY, digest=MYSTERY_DIGEST)])
    await store_ceiling(db_sessionmaker, MIN_CONTEXT_CEILING)

    await run(provider, db_sessionmaker, model=MYSTERY)

    # Sized at all - the failure being pinned is a None here - and sized to
    # what a turn on this workspace would request.
    assert provider.windows
    assert set(provider.windows) == {MIN_CONTEXT_CEILING}


async def test_the_recorded_verdict_names_the_window_it_measured(db_sessionmaker) -> None:
    # A verdict that does not say which window it was earned at cannot be read
    # against the turns it certifies: the window is the one serve-time option a
    # deployment moves without touching the weights, so the digest says nothing
    # about it.
    #
    # Measured on a tag nothing configures, deliberately: the stored ceiling is
    # the only thing that could have produced this number, so a run that had
    # resolved no window at all would show up rather than pass by coincidence.
    provider = ProbeProvider(models=[*INSTALLED, ModelInfo(name=MYSTERY, digest=MYSTERY_DIGEST)])
    await store_ceiling(db_sessionmaker, MIN_CONTEXT_CEILING)

    await run(provider, db_sessionmaker, model=MYSTERY)

    stored = await rows(db_sessionmaker, model=MYSTERY)
    assert len(stored) == 1
    assert {sample["num_ctx"] for sample in stored[0].cases["samples"]} == {MIN_CONTEXT_CEILING}


async def test_a_settings_read_failure_probes_unsized_by_nothing(db_sessionmaker) -> None:
    # The fallback is newly load-bearing, so it gets its own guard: a database
    # that will not answer must cost the CLAMP and nothing else. The run still
    # happens, and it still resolves a real window - reverting to unsized here
    # would reinstate exactly the defect this slice removed, behind a single
    # logged warning nobody reads.
    class BrokenSessionmaker:
        def __call__(self):
            raise RuntimeError("the database is down")

    provider = ProbeProvider()
    events = await drain(provider, BrokenSessionmaker())

    assert isinstance(events[-1], VerifyDoneEvent)
    # What this deployment affords one turn, unclamped by a ceiling nobody
    # could read - never None, which is what "unsized" would look like.
    assert provider.windows
    assert set(provider.windows) == {DEPLOYMENT_WINDOW}


# --- the think flag a probe runs at -------------------------------------------


async def test_the_probe_runs_at_the_think_flag_this_deployment_serves(
    db_sessionmaker,
) -> None:
    # The other half of "serve-time", and the half only this job can supply.
    # The runtime reports a template with no think plumbing for this tag, so a
    # turn would omit the flag - until an operator confirms a measurement that
    # says otherwise, which lives in the database the harness deliberately
    # cannot read. Drop either fetcher from the run_tool_check call and the
    # tool samples come back with the flag omitted, measuring a configuration
    # nobody serves.
    provider = ProbeProvider()
    async with db_sessionmaker() as session:
        session.add(
            ModelPolicy(provider="fake", model=CHAT, digest=CHAT_DIGEST, think_mode="always")
        )
        await session.commit()

    outcome = await run(provider, db_sessionmaker)

    samples = outcome.summary.samples
    # The TOOL cases carry the declared flag...
    assert {s.think for s in samples if s.case not in THINK_CASE_NAMES} == {True}
    # ...while the thinking group keeps sending its own flag literally, both
    # ways, on the same tag - it is measuring the very claim above.
    assert {s.case: s.think for s in samples if s.case in THINK_CASE_NAMES} == {
        "think-on": True,
        "think-off": False,
    }
    # And the row says so, so the verdict names the flag it was earned under.
    stored = await rows(db_sessionmaker)
    assert {sample["think"] for sample in stored[0].cases["samples"]} == {True, False}
    # The provider really was driven that way, not merely recorded as such.
    assert (CHAT, True) in provider.thinks


async def test_a_dead_database_leaves_the_probe_on_the_sessionless_half(
    db_sessionmaker,
) -> None:
    # The degradation, pinned. Both declaration reads fail closed to "nothing
    # declared" rather than raising, so a database that will not answer costs
    # the two top rungs of the think resolution and nothing else: the run still
    # happens, and it still resolves the sessionless half - the quirk seed and
    # the runtime's own template report.
    class BrokenSessionmaker:
        def __call__(self):
            raise RuntimeError("the database is down")

    provider = ProbeProvider()
    events = await drain(provider, BrokenSessionmaker())

    assert isinstance(events[-1], VerifyDoneEvent)
    # The template carries no think plumbing, so the tool cases OMIT the flag -
    # never send it as false, which is the request a 400 comes back from. The
    # thinking group's own literal pair supplies the True and the False.
    assert {think for model, think in provider.thinks if model == CHAT} == {None, True, False}


async def test_a_later_ceiling_change_leaves_an_earned_verdict_standing(
    db_sessionmaker,
) -> None:
    # The guard on part three, and the line it draws. The window a probe ran at
    # is DISPLAY: it tells a reader what was measured. It is deliberately not a
    # staleness axis - those are the digest, the harness version and the serving
    # host, all facts about the weights or the machine - so an operator editing
    # a setting must not silently void GPU-minutes of evidence, nor re-clamp a
    # model that passed. Wire the ceiling into the certificate check and this
    # goes red.
    settings = make_settings()
    provider = ProbeProvider()
    await store_ceiling(db_sessionmaker, MIN_CONTEXT_CEILING)

    outcome = await run(provider, db_sessionmaker, deep=True)
    assert outcome.summary.earned == "agentic"

    async def earned_now() -> str:
        # Cleared first, or the five-minute lookup cache would answer for the
        # second read and the test would pass whatever the resolution does.
        clear_earned_cache()
        verdicts = await effective_verdicts(
            provider=provider,
            provider_name="fake",
            model=CHAT,
            settings=settings,
            # The hardened regime, so the certificate is what is being read:
            # under the grace floor a discounted certificate would still leave
            # the model at assisted, and the difference this pins would be one
            # rung instead of two.
            require_verification=True,
            sessionmaker=db_sessionmaker,
            effective_ceiling="agentic",
        )
        return verdicts.earned

    assert await earned_now() == "agentic"
    measured = await rows(db_sessionmaker)

    await store_ceiling(db_sessionmaker, 4 * MIN_CONTEXT_CEILING)

    assert await earned_now() == "agentic"
    # And nothing rewrote or re-ran anything: a measurement is history, and a
    # settings edit does not append to it.
    assert [row.id for row in await rows(db_sessionmaker)] == [row.id for row in measured]


# --- the partner a run drives -------------------------------------------------
#
# A reasoning-with-tools certificate binds to the PARTNER's weights, so the tag
# a run drives for that case and the tag the earned resolution binds against
# have to be the same one - and that tag is whatever a TURN would route to
# (app.services.model_resolution.thinking_sibling_for: the environment
# override, then a stored operator pairing, and nothing else). The
# configurations below are exactly where a resolution reading only the
# environment used to differ, and the failure was silent every time: the
# binding simply never matched, and the capability was withheld with nothing
# anywhere saying why.


# The terminal check, which is a piped SCRIPT rather than an importable module,
# so the parity test loads it by path. Safe to import: everything below its own
# imports sits behind a __main__ guard.
_TOOL_CHECK_CLI = Path(__file__).resolve().parents[4] / "scripts" / "tool_check.py"


@lru_cache(maxsize=1)
def _terminal_check() -> ModuleType:
    """Load the terminal tool-check script, so its resolution can be asked directly."""
    spec = importlib.util.spec_from_file_location("terminal_tool_check", _TOOL_CHECK_CLI)
    assert spec is not None and spec.loader is not None, _TOOL_CHECK_CLI
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_provider(**kwargs) -> ProbeProvider:
    """A backend holding the environment's pair AND a separately declarable partner."""
    return ProbeProvider(
        models=[*INSTALLED, ModelInfo(name=DECLARED_THINK, digest=DECLARED_THINK_DIGEST)], **kwargs
    )


async def store_pairing(sessionmaker, partner: str, *, grant: bool | None = None) -> None:
    """Pair the chat tag with ``partner``, as the console's declare verb does."""
    async with sessionmaker() as session:
        session.add(
            ModelPolicy(
                provider="fake",
                model=CHAT,
                digest=CHAT_DIGEST,
                thinking_model=partner,
                thinking_with_tools=grant,
            )
        )
        await session.commit()


async def run_over(settings: Settings, provider, sessionmaker, *, deep: bool = True):
    """Drive the service against caller-supplied settings rather than the default."""
    return await model_verify.run_verification(
        provider=provider,
        settings=settings,
        sessionmaker=sessionmaker,
        tenant=tenant(),
        model=CHAT,
        deep=deep,
    )


async def test_a_declared_partner_is_the_one_the_check_drives(db_sessionmaker) -> None:
    # Divergence one, and the only rung a click can reach: an operator paired
    # this model with something the environment does not name, and a TURN
    # reasons on what the operator chose. A run that drove some other tag would
    # spend its GPU minutes measuring weights nothing serves, and record the
    # wrong digest beside the verdict.
    provider = paired_provider()
    await store_pairing(db_sessionmaker, DECLARED_THINK)

    outcome = await run_over(make_settings(thinking=None), provider, db_sessionmaker)

    assert outcome.summary.thinking_model == DECLARED_THINK
    driven = set(provider.kinds("chat"))
    assert DECLARED_THINK in driven
    assert THINK not in driven  # nothing else is ever touched
    # And the row binds to the weights that were actually driven.
    assert outcome.thinking_digest == DECLARED_THINK_DIGEST
    stored = await rows(db_sessionmaker)
    assert stored[0].cases["thinking_digest"] == DECLARED_THINK_DIGEST


async def test_a_declared_partner_pairs_a_deployment_nothing_else_pairs(
    db_sessionmaker,
) -> None:
    # Divergence two, and the one that made a stored grant unusable outright:
    # with no environment override a run that read only the environment
    # recorded no partner digest at all while the earned resolution looked one
    # up - two values that could never match, on a deployment where declaring a
    # pairing is the ONLY way to have one. The loop is asserted end to end
    # here, from the run that measures to the turn that acts on it.
    settings = make_settings(thinking=None)
    provider = paired_provider()
    await store_pairing(db_sessionmaker, DECLARED_THINK, grant=True)

    outcome = await run_over(settings, provider, db_sessionmaker)

    assert outcome.summary.thinking_model == DECLARED_THINK
    assert outcome.thinking_digest == DECLARED_THINK_DIGEST
    # The deep run passed the reasoning case on the partner it drove...
    assert outcome.summary.thinking_with_tools is True
    # ...and the grant an operator stored now reaches a turn, which is the whole
    # point: a stored grant is evidence-gated in BOTH regimes, so it takes
    # effect only on a deep certificate bound to the partner's installed
    # weights. Resolve the binding any other way and this stays False forever.
    policy = await resolve_policy(
        provider=provider, provider_name="fake", model=CHAT, sessionmaker=db_sessionmaker
    )
    verdicts = await effective_verdicts(
        provider=provider,
        provider_name="fake",
        model=CHAT,
        settings=settings,
        require_verification=False,
        sessionmaker=db_sessionmaker,
        effective_ceiling="agentic",
        policy=policy,
    )
    assert verdicts.thinking_with_tools is True


async def test_an_unpaired_tag_records_no_partner_at_all(db_sessionmaker) -> None:
    # The shape of the remote server's inventory: no stored pairing and no
    # environment override. There is no partner, so the run stays on the
    # single-model path where the think flag does the work and the row records
    # no thinking digest - which is what lets a certificate already standing on
    # such weights survive this change untouched.
    settings = make_settings(thinking=None)
    provider = paired_provider()

    outcome = await run_over(settings, provider, db_sessionmaker)

    assert outcome.summary.thinking_model is None
    assert outcome.thinking_digest is None
    stored = await rows(db_sessionmaker)
    assert stored[0].cases["thinking_digest"] is None
    assert DECLARED_THINK not in set(provider.kinds("chat"))


async def test_the_environment_override_outranks_a_declared_partner(
    db_sessionmaker,
) -> None:
    # The top rung, pinned on the writer that used to ignore both of the ones
    # below it: deployment configuration is not overruled by a click, and the
    # literal "none" switches pair-routing off outright - at which point the
    # honest recorded partner is no partner.
    provider = paired_provider()
    await store_pairing(db_sessionmaker, DECLARED_THINK)

    forced = await run_over(make_settings(thinking=THINK), provider, db_sessionmaker)
    assert forced.thinking_digest == THINK_DIGEST

    clear_earned_cache()
    disabled = await run_over(make_settings(thinking="none"), paired_provider(), db_sessionmaker)
    assert disabled.summary.thinking_model is None
    assert disabled.thinking_digest is None


@pytest.mark.parametrize(
    ("env", "declared", "expected"),
    [
        pytest.param(None, None, None, id="nothing pairs the tag"),
        pytest.param(None, DECLARED_THINK, DECLARED_THINK, id="a stored pairing"),
        pytest.param(THINK, None, THINK, id="an environment override"),
        pytest.param(THINK, DECLARED_THINK, THINK, id="the env override wins"),
        pytest.param("none", DECLARED_THINK, None, id="pair-routing switched off"),
    ],
)
def test_every_writer_and_the_turn_name_one_partner(
    env: str | None, declared: str | None, expected: str | None
) -> None:
    """The parity this slice exists for, across the configurations that differ.

    Three surfaces have to agree, and each is asked in its own words: the
    console's verify job through the shared ranking it drives on, the terminal
    check through the wrapper it resolves its partner with (loaded from the
    script itself, so this is its real code and not a restatement), and the
    TURN through ``resolve_chat_models`` - a different code path that composes
    the same two rungs from the chat pair rather than from a tag.

    MUTATION: reorder the two rungs inside ``thinking_sibling_for`` and the
    "env override wins" and "switched off" rows go red; drop the terminal
    check's stored-pairing read and the two stored-pairing rows do. The verify
    JOB's own use of the ranking is pinned by the behavioural tests above,
    which is the half this one cannot see.
    """
    settings = make_settings(thinking=env)

    job = thinking_sibling_for(settings, CHAT, declared=declared)
    turn = resolve_chat_models(settings, declared_thinking_model=declared).thinking
    terminal = _terminal_check()._partner_for(settings, CHAT, declared)

    assert job == expected
    assert turn == expected
    assert terminal == (expected or "")


@pytest.mark.parametrize(
    ("env", "declared", "expected"),
    [
        pytest.param(None, None, None, id="nothing declares a partner"),
        pytest.param(THINK, None, THINK, id="an environment override"),
        pytest.param(None, DECLARED_THINK, DECLARED_THINK, id="a stored declaration"),
        pytest.param(THINK, DECLARED_THINK, THINK, id="the override outranks the declaration"),
    ],
)
def test_the_pair_and_the_tag_keyed_resolution_agree_about_an_undescribed_tag(
    env: str | None, declared: str | None, expected: str | None
) -> None:
    """The disagreement this fold-in removed, asserted as agreement.

    The two paths used to answer differently about a tag nothing described.
    ``resolve_chat_models`` resolved a PAIR and fell back to the active
    profile's chat binding for the partner, so an undescribed default silently
    inherited that binding's sibling; the tag-keyed
    ``thinking_sibling_for`` read the entry governing the tag ASKED about,
    found none, and answered no partner. A run measured one pair while a turn
    reasoned on another, and nothing anywhere said so.

    The binding is gone, so both now compose the SAME two rungs - the
    environment override, then a stored declaration - and there is nothing left
    for a tag to inherit. This is the whole point of the fold-in, so it is
    asserted on the tag that used to expose it: ``MYSTERY``, which is the
    deployment's chat model here and which nothing else mentions.

    The one difference that remains is deliberate and lives one test below:
    ``thinking_sibling_for`` excludes a tag from being its own partner, which a
    pair resolution has no tag to exclude against.

    MUTATION: give either function a rung the other lacks - a per-tag fallback
    inside ``resolve_chat_models``, or a second source inside
    ``thinking_sibling_for`` - and the two sides of the final assertion part
    company on at least one row.
    """
    settings = make_settings(chat=MYSTERY, thinking=env)

    pair = resolve_chat_models(settings, declared_thinking_model=declared)
    tag_keyed = thinking_sibling_for(settings, MYSTERY, declared=declared)

    assert pair.default == MYSTERY
    # Asserted against the expected value AND against each other: agreeing on
    # None for every row would otherwise pass a resolution that had stopped
    # answering at all.
    assert pair.thinking == expected
    assert tag_keyed == expected
    assert pair.thinking == tag_keyed


@pytest.mark.parametrize(
    ("requested", "override"),
    [
        pytest.param(CHAT, CHAT, id="named exactly"),
        pytest.param("chat-tag:latest", CHAT, id="named through an alias"),
    ],
)
def test_a_model_is_never_its_own_partner(requested: str, override: str) -> None:
    # The self-exclusion, which is what keeps a deployment that points the pair
    # override at its own chat tag on the single-model path where the think flag
    # does the work. Both spellings, because both sides here are written-down
    # configuration: an override naming `chat-tag:latest` against a request for
    # `chat-tag` is one model under two names, and an exact-match guard would
    # pass the tag back as its own partner under the second.
    settings = make_settings(thinking=override)

    assert thinking_sibling_for(settings, requested, declared=None) is None


async def test_an_override_naming_the_tag_itself_records_no_partner(
    db_sessionmaker,
) -> None:
    # The same guard where it costs GPU minutes: without it the run would drive
    # the tag under check a second time for the reasoning case and record its
    # own digest as its partner's - a certificate that reads as a PAIR having
    # been measured when no pair exists.
    provider = paired_provider()

    outcome = await run_over(make_settings(thinking=CHAT), provider, db_sessionmaker)

    assert outcome.summary.thinking_model is None
    assert outcome.thinking_digest is None


# --- the re-warm --------------------------------------------------------------


async def test_the_probe_rewarms_chat_and_the_embedder(db_sessionmaker) -> None:
    provider = ProbeProvider()
    events = await drain(provider, db_sessionmaker)

    assert isinstance(events[-1], VerifyDoneEvent)
    kinds = [kind for kind, _ in provider.events]
    # Both, and only after the measuring stopped: the probe evicted them.
    assert provider.kinds("warm") == [CHAT]
    assert provider.kinds("embed") == [EMBED]
    assert kinds.index("warm") > max(i for i, k in enumerate(kinds) if k == "chat")


async def test_a_failed_probe_still_rewarms(db_sessionmaker) -> None:
    # The failure paths are the ones that leave the most damage behind: the
    # models are already evicted by the time anything can go wrong.
    provider = ProbeProvider(fail=RuntimeError("the runner crashed"))
    events = await drain(provider, db_sessionmaker)

    assert isinstance(events[-1], VerifyErrorEvent)
    assert provider.kinds("warm") == [CHAT]
    assert provider.kinds("embed") == [EMBED]


async def test_the_rewarm_restores_the_workspace_default(db_sessionmaker) -> None:
    # The re-warm puts the deployment's steady state back, and the steady
    # state is what a DEFAULT TURN runs - which the workspace default decides,
    # exactly as POST /models/warm resolves it. Resolving without it heated
    # the profile's binding instead: a model no turn on that deployment runs,
    # cold-loaded onto the card, while the actual daily driver stayed as cold
    # as the probe left it (observed live, 2026-08-06).
    ws = ModelInfo(name="ws-tag", digest="sha256:wsweights")
    provider = ProbeProvider(models=[*INSTALLED, ws])
    async with db_sessionmaker() as session:
        await WorkspaceSettingsRepository(session, tenant()).merge_defaults({"model": "ws-tag"})
        await session.commit()

    events = await drain(provider, db_sessionmaker)

    assert isinstance(events[-1], VerifyDoneEvent)
    assert provider.kinds("warm") == ["ws-tag"]
    assert provider.kinds("embed") == [EMBED]


async def test_the_rewarm_restores_the_workspace_persona(db_sessionmaker) -> None:
    # The same argument as the default model one line up, one layer in: the
    # steady state a re-warm restores is a PREFIX, not just a set of weights.
    # Re-warming with the deployment's built-in persona leaves the card holding
    # a cache keyed on a prompt no turn on this workspace ever sends, so the
    # first real question after a check pays the whole prefill anyway - the
    # exact cost the re-warm exists to avoid.
    persona = "Answer as the workspace's own support desk."
    provider = ProbeProvider()
    async with db_sessionmaker() as session:
        await WorkspaceSettingsRepository(session, tenant()).merge_defaults(
            {"system_prompt": persona}
        )
        await session.commit()

    events = await drain(provider, db_sessionmaker)

    assert isinstance(events[-1], VerifyDoneEvent)
    assert provider.kinds("warm") == [CHAT]
    # Both halves, so a re-warm that appended rather than replaced still fails.
    assert provider.warm_systems[-1].startswith(persona)
    assert DEFAULT_SYSTEM_PROMPT not in provider.warm_systems[-1]


async def test_a_dead_database_does_not_stop_the_rewarm(db_sessionmaker) -> None:
    # The workspace read is a best-effort refinement on a cleanup path. With
    # the database gone the re-warm still restores the profile's resolution -
    # a cleanup that could raise would replace the outcome the caller is
    # waiting for, and a card left empty costs more than a suboptimal warm.
    class BrokenSessionmaker:
        def __call__(self):
            raise RuntimeError("the database is down")

    provider = ProbeProvider()
    events = await drain(provider, BrokenSessionmaker())

    assert isinstance(events[-1], VerifyDoneEvent)
    assert provider.kinds("warm") == [CHAT]
    assert provider.kinds("embed") == [EMBED]


async def test_an_unexpected_failure_never_leaks_its_exception(db_sessionmaker) -> None:
    provider = ProbeProvider(fail=RuntimeError("secret internal detail"))
    events = await drain(provider, db_sessionmaker)

    error = events[-1]
    assert isinstance(error, VerifyErrorEvent)
    assert error.code == "internal_error"
    assert "secret internal detail" not in error.message


# --- the streamed endpoint ----------------------------------------------------


def test_every_event_type_has_a_frame_name() -> None:
    # A missing entry is a KeyError raised mid-stream, which is loud only in
    # the server log: what the client gets is a truncated body with no terminal
    # frame - exactly the "a stream that simply stops" failure the whole design
    # is written to prevent. Checked against the unions themselves so adding an
    # event type without a name fails here rather than in production.
    declared = set(get_args(model_pull.PullEvent)) | set(get_args(model_verify.VerifyEvent))
    assert declared <= set(_FRAME_NAMES)


async def test_verify_streams_a_frame_per_sample_then_the_verdict(
    db_sessionmaker,
) -> None:
    provider = ProbeProvider(
        runtime_version="0.32.0",
        residency=ModelResidency(model=CHAT, loaded=True, total_bytes=4_000, vram_bytes=4_000),
    )
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/verify", json={"name": CHAT})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = frames(resp.text)
    assert [name for name, _ in events] == FAST_FRAMES
    first = VerifyProgressEvent.model_validate_json(events[0][1])
    assert (first.completed, first.total) == (1, FAST_SAMPLES)
    assert first.case and first.model == CHAT
    done = VerifyDoneEvent.model_validate_json(events[-1][1])
    assert (done.depth, done.earned) == ("fast", "assisted")
    # The think-mode measurement rides the same terminal frame: a fast check
    # runs the thinking group too, so the verdict a reader sees carries it.
    assert done.measured_think_mode is not None
    assert done.digest == CHAT_DIGEST
    assert done.runtime_version == "0.32.0"
    assert done.residency.gpu_resident is True
    assert done.probe_id is not None


async def test_a_deep_check_reports_its_own_sample_count(db_sessionmaker) -> None:
    provider = ProbeProvider()
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/verify", json={"name": CHAT, "depth": "deep"})

    events = frames(resp.text)
    done = VerifyDoneEvent.model_validate_json(events[-1][1])
    assert done.depth == "deep"
    # Every sample the depth declares, and the reasoning case drives the pair's
    # OTHER tag - which is the whole reason a deep verdict says nothing about
    # one tag alone.
    assert len([name for name, _ in events if name == "verify_progress"]) == DEEP_SAMPLES
    assert THINK in {
        VerifyProgressEvent.model_validate_json(data).model
        for name, data in events
        if name == "verify_progress"
    }


async def test_a_second_verification_is_refused_with_409(db_sessionmaker) -> None:
    model_verify.claim_verification("other-model")
    provider = ProbeProvider()
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/verify", json={"name": CHAT})

    assert resp.status_code == 409
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "conflict"
    # Refused before any work: the card is untouched.
    assert provider.events == []


async def test_a_refusal_leaves_the_running_probe_alone(db_sessionmaker) -> None:
    # The refused request must not release the claim it failed to take, or the
    # second click would cancel the first check's guard.
    model_verify.claim_verification("other-model")
    async with verify_client(ProbeProvider(), db_sessionmaker) as client:
        await client.post("/api/v1/models/verify", json={"name": CHAT})
    assert model_verify.verification_in_flight() == "other-model"


async def test_a_finished_check_releases_the_slot(db_sessionmaker) -> None:
    async with verify_client(ProbeProvider(), db_sessionmaker) as client:
        await client.post("/api/v1/models/verify", json={"name": CHAT})
        assert model_verify.verification_in_flight() is None
        again = await client.post("/api/v1/models/verify", json={"name": CHAT})
    assert again.status_code == 200


async def test_a_failed_check_releases_the_slot(db_sessionmaker) -> None:
    # A wedged slot is unrecoverable without a restart, so the path that raises
    # has to drop it too.
    provider = ProbeProvider(fail=RuntimeError("boom"))
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/verify", json={"name": CHAT})
    assert frames(resp.text)[-1][0] == "verify_error"
    assert model_verify.verification_in_flight() is None


async def test_the_slot_is_held_until_the_probe_itself_ends(db_sessionmaker) -> None:
    # Driven directly, because the release that matters here is the one the
    # SERVICE owns. On a mid-stream disconnect the server unwinds the request
    # stack at once - releasing the request scope's claim - while the heartbeat
    # wrapper's producer task keeps draining this generator and the probe keeps
    # generating. A release owned only by the request scope would open the slot
    # in that window and let the next click start a second probe underneath the
    # first.
    provider = SlotWatchingProvider()
    model_verify.claim_verification(CHAT)
    held: list[str | None] = []
    async for event in model_verify.stream_verification(
        provider=provider,
        settings=make_settings(),
        sessionmaker=db_sessionmaker,
        tenant=tenant(),
        model=CHAT,
        deep=False,
    ):
        held.append(model_verify.verification_in_flight())
        assert isinstance(event, VerifyProgressEvent | VerifyDoneEvent)
    # Held for every frame, including the terminal one - the re-warm is still
    # to come after that - and released once the generator is done.
    assert held == [CHAT] * (FAST_SAMPLES + 1)
    # Still held while the re-warm loads models back onto the same card:
    # releasing first would let a second probe start unloading what this one is
    # busy putting back.
    assert provider.slot_during_rewarm == [CHAT]
    assert model_verify.verification_in_flight() is None


async def test_a_hang_up_mid_verify_stream_still_rewarms(db_sessionmaker) -> None:
    # The console offers Stop on a running check, and Stop is an abort of this
    # stream. The hint beside it promises the chat model comes back, which is
    # only true if a hang-up re-warms: the probe has already evicted the chat
    # model and the embedder by the time the first frame arrives.
    provider = ProbeProvider()
    model_verify.claim_verification(CHAT)
    events = model_verify.stream_verification(
        provider=provider,
        settings=make_settings(),
        sessionmaker=db_sessionmaker,
        tenant=tenant(),
        model=CHAT,
        deep=False,
    )
    async for event in events:
        if isinstance(event, VerifyProgressEvent):
            break
    await events.aclose()
    assert model_verify.verification_in_flight() is None
    assert provider.kinds("warm") == [CHAT]
    assert provider.kinds("embed") == [EMBED]


async def test_the_stream_holds_up_at_the_deployment_heartbeat(db_sessionmaker) -> None:
    # Every other route test runs with heartbeats OFF, which is the plain
    # passthrough branch. The shipped default is 15 seconds, which takes the
    # producer-task branch instead: the frames cross a queue, and cancellation
    # has a second place to land. Worth one test at the value that actually
    # ships.
    provider = ProbeProvider()
    async with verify_client(provider, db_sessionmaker, heartbeat=15.0) as client:
        resp = await client.post("/api/v1/models/verify", json={"name": CHAT})

    assert [name for name, _ in frames(resp.text)] == FAST_FRAMES
    assert model_verify.verification_in_flight() is None
    assert provider.kinds("warm") == [CHAT]


async def test_the_probe_slot_is_released_when_the_request_dies() -> None:
    # The case an in-process ASGI client cannot stage, and the reason the
    # release lives in a yield dependency at all: the request's exit stack
    # unwinding with an exception - a client that vanished, or a failure above
    # the route. Driven directly for that reason; through the transport the app
    # always runs to completion, so a release that only happened on the happy
    # path would look perfectly fine.
    claim = hold_verify_claim(
        ModelVerifyRequest(name=CHAT), ProbeProvider(), make_settings(), tenant()
    )
    assert (await anext(claim)).tag == CHAT
    assert model_verify.verification_in_flight() == CHAT
    with pytest.raises(RuntimeError):
        await claim.athrow(RuntimeError("the client vanished"))
    assert model_verify.verification_in_flight() is None


def handed_off(provider, sessionmaker, *, verify: bool = True):
    return model_verify.stream_pull_with_check(
        provider=provider,
        settings=make_settings(),
        sessionmaker=sessionmaker,
        tenant=tenant(),
        tag=CHAT,
        verify=verify,
    )


async def test_a_hang_up_mid_check_releases_the_slot(db_sessionmaker) -> None:
    # The hand-off's claim is taken inside the stream rather than by the
    # request scope, so its release has to survive the consumer walking away
    # part-way through - otherwise one abandoned download wedges every later
    # check until a restart.
    provider = ProbeProvider()
    events = handed_off(provider, db_sessionmaker)
    async for event in events:
        if isinstance(event, VerifyProgressEvent):
            break
    assert model_verify.verification_in_flight() == CHAT
    await events.aclose()
    assert model_verify.verification_in_flight() is None
    # ...and the abandoned check still put back what it evicted, rather than
    # leaving the deployment cold because somebody closed a tab.
    assert provider.kinds("warm") == [CHAT]
    assert provider.kinds("embed") == [EMBED]


async def test_a_hang_up_at_the_download_s_last_frame_releases_the_slot(
    db_sessionmaker,
) -> None:
    # The narrowest window there is, and the one a guard scoped to the
    # verification loop jumps straight past: the claim is taken while this
    # generator is suspended at the ``done`` frame, and the next thing it does
    # is close the download's own stream - an await, and a place a client can
    # vanish. Landing there with the slot taken and no release wedges every
    # later check in the process.
    events = handed_off(ProbeProvider(), db_sessionmaker)
    async for event in events:
        if isinstance(event, PullDoneEvent):
            break
    assert model_verify.verification_in_flight() == CHAT
    await events.aclose()
    assert model_verify.verification_in_flight() is None


async def test_verifying_a_model_the_backend_does_not_hold_is_404(
    db_sessionmaker,
) -> None:
    provider = ProbeProvider(models=[ModelInfo(name="other:1b")])
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/verify", json={"name": CHAT})

    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    assert provider.events == []


async def test_verify_refuses_an_unsafe_name(db_sessionmaker) -> None:
    provider = ProbeProvider()
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/verify", json={"name": "../../etc/passwd"})

    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"
    assert provider.events == []


async def test_verify_is_admin_gated(db_sessionmaker) -> None:
    # The check passes for everyone today (the default principal is an admin),
    # so this pins that the SEAM is wired: the day OIDC populates a real role,
    # a member stops being able to evict the deployment's warm model.
    provider = ProbeProvider()
    async with verify_client(provider, db_sessionmaker, principal=MEMBER) as client:
        resp = await client.post("/api/v1/models/verify", json={"name": CHAT})

    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    assert provider.events == []


async def test_an_unreachable_server_ends_the_stream_with_a_typed_error(
    db_sessionmaker,
) -> None:
    provider = UnreachableProvider()
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/verify", json={"name": CHAT})

    events = frames(resp.text)
    assert events[-1][0] == "verify_error"
    error = VerifyErrorEvent.model_validate_json(events[-1][1])
    assert error.code == "provider_unavailable"


# --- the pull hand-off --------------------------------------------------------


async def test_a_successful_download_continues_into_a_fast_check(db_sessionmaker) -> None:
    provider = ProbeProvider(pull_script=[{"status": "pulling manifest"}, {"status": "success"}])
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": CHAT})

    events = frames(resp.text)
    assert [name for name, _ in events] == ["progress", "progress", "done", *FAST_FRAMES]
    done = PullDoneEvent.model_validate_json(events[2][1])
    assert done.verifying is True and done.verify_skipped_reason is None
    assert VerifyDoneEvent.model_validate_json(events[-1][1]).depth == "fast"
    assert model_verify.verification_in_flight() is None


async def test_a_caller_can_opt_out_of_the_automatic_check(db_sessionmaker) -> None:
    provider = ProbeProvider()
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": CHAT, "verify": False})

    events = frames(resp.text)
    assert [name for name, _ in events] == ["progress", "progress", "done"]
    done = PullDoneEvent.model_validate_json(events[-1][1])
    assert done.verifying is False and done.verify_skipped_reason is None
    # Nothing was measured, so nothing was evicted either.
    assert provider.events == []
    assert await rows(db_sessionmaker) == []


async def test_a_failing_check_still_reports_the_download_as_successful(
    db_sessionmaker,
) -> None:
    # The download DID succeed - the weights are on disk. Folding the check's
    # failure into the download's outcome would send someone to re-download a
    # model that is already here.
    provider = ProbeProvider(fail=RuntimeError("the runner crashed"))
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": CHAT})

    events = frames(resp.text)
    assert [name for name, _ in events][-1] == "verify_error"
    done = PullDoneEvent.model_validate_json(next(d for n, d in events if n == "done"))
    assert done.status == "success"
    assert done.verifying is True
    assert provider.pulled == [CHAT]


async def test_the_automatic_check_respects_the_same_single_flight_guard(
    db_sessionmaker,
) -> None:
    # An automatic check must never be the thing that evicts a model out from
    # under a deliberate one.
    model_verify.claim_verification("other-model")
    provider = ProbeProvider()
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": CHAT})

    events = frames(resp.text)
    assert [name for name, _ in events] == ["progress", "progress", "done"]
    done = PullDoneEvent.model_validate_json(events[-1][1])
    assert done.verifying is False
    assert "already running" in (done.verify_skipped_reason or "")
    # ...and the claim the other check holds is still theirs.
    assert model_verify.verification_in_flight() == "other-model"


async def test_a_download_drops_what_the_chat_path_cached_about_the_old_weights(
    db_sessionmaker,
) -> None:
    # The chat path caches earned verdicts for five minutes. A re-pull changes
    # the digest, which is what makes an old verdict stop applying - but only
    # once the cache lets go of it. Without this, a download whose check was
    # skipped (opted out, refused, or the wrong role) leaves the old row
    # certifying the NEW weights for the rest of the window, which is exactly
    # the auto-revert this measurement log exists to enforce.
    settings = make_settings()
    old = ProbeProvider(models=[ModelInfo(name=CHAT, digest="sha256:before")])
    new = ProbeProvider(models=[ModelInfo(name=CHAT, digest="sha256:after")])
    async with db_sessionmaker() as session:
        session.add(
            ModelProbe(
                provider="fake",
                model=CHAT,
                digest="sha256:before",
                harness_version=harness_version(),
                depth="deep",
                # Failure evidence, and deliberately below the grace FLOOR: an
                # unclamped model already resolves to assisted, so a clamp to
                # assisted would be invisible and this test would pass whether
                # or not the cache was dropped.
                earned_agency="deterministic",
                cases={},
            )
        )
        await session.commit()

    async def earned_now(provider) -> str:
        verdicts = await effective_verdicts(
            provider=provider,
            provider_name="fake",
            model=CHAT,
            settings=settings,
            require_verification=False,
            sessionmaker=db_sessionmaker,
            effective_ceiling="agentic",
        )
        return verdicts.earned

    assert await earned_now(old) == "deterministic"  # the clamp is now cached

    downloaded = [event async for event in handed_off(new, db_sessionmaker, verify=False)]
    assert isinstance(downloaded[-1], PullDoneEvent)

    # The weights moved on, so the clamp measured against the old ones is gone
    # immediately rather than in five minutes: the model returns to the
    # unmeasured floor, which is what an un-probed set of weights resolves to.
    assert await earned_now(new) == "assisted"


async def test_a_download_drops_the_stored_declarations_too(db_sessionmaker) -> None:
    # The third digest-bound cache, and the subtlest. A policy row's TRUST axes
    # are honoured only while the declaring row's digest is installed, so a
    # fold cached across a pull would keep an operator's ceiling armed against
    # weights that no longer exist - the auto-re-clamp working everywhere
    # except in the minutes right after the pull that triggered it.
    old = ProbeProvider(models=[ModelInfo(name=CHAT, digest="sha256:before")])
    new = ProbeProvider(models=[ModelInfo(name=CHAT, digest="sha256:after")])
    async with db_sessionmaker() as session:
        session.add(
            ModelPolicy(provider="fake", model=CHAT, digest="sha256:before", ceiling="agentic")
        )
        await session.commit()

    async def ceiling_now(provider) -> str | None:
        view = await resolve_policy(
            provider=provider,
            provider_name="fake",
            model=CHAT,
            sessionmaker=db_sessionmaker,
        )
        return view.ceiling if view is not None else None

    assert await ceiling_now(old) == "agentic"  # bound to the installed weights, and cached

    downloaded = [event async for event in handed_off(new, db_sessionmaker, verify=False)]
    assert isinstance(downloaded[-1], PullDoneEvent)

    # Retired immediately: the fold re-reads, finds the declaring row bound to
    # weights this server no longer holds, and drops the axis.
    assert await ceiling_now(new) is None


async def test_a_downloaded_embedder_is_not_probed_for_tool_calling(
    db_sessionmaker,
) -> None:
    provider = ProbeProvider()
    async with verify_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": EMBED})

    done = PullDoneEvent.model_validate_json(frames(resp.text)[-1][1])
    assert done.verifying is False
    assert "embedding" in (done.verify_skipped_reason or "")
    assert provider.events == []
