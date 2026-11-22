"""The embedder-check job: its guards, its placement, and the row it writes.

Two halves. The first drives the REAL runner - the same functions the endpoint
calls - with a provider that records every embed it receives, so "the device
pin reached the seam" and "the active embedder was put back" are assertions
about calls rather than about intentions.

The second is the one that matters most and is the easiest to lose: an
embedder-check row lands in the SAME ``model_probes`` log the agency resolver
reads, under a depth that resolver has never heard of. Nothing enforces that
separation with a check - it falls out of every agency read naming its depth -
so the tests here seed the most dangerous row that could exist (an ``agentic``
verdict against the chat tag's own installed weights) and pin that every reader
on the agency path ignores it, against a control proving the same test WOULD
see it at a depth the resolver reads.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_tenant, get_model_provider
from app.core.config import Settings, get_settings
from app.core.errors import BadRequestError, ConflictError, NotFoundError, UnprocessableEntityError
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.models.entities import ModelProbe
from app.providers.base import (
    EmbedOptions,
    ModelCapabilities,
    ModelInfo,
    ProviderUnavailableError,
)
from app.providers.fake_provider import FAKE_SERVING_ENDPOINT, FakeProvider
from app.repositories.model_probes import ModelProbeRepository
from app.schemas.settings import WorkspaceDefaultsPatch
from app.services import embedder_verify, model_verify
from app.services.earned import effective_verdicts, explain_earned
from app.services.embedder_check import PROBE_DEPTH
from app.services.embedding_resolution import resolve_embedding
from app.services.model_facts import clear_facts_cache
from app.services.model_manager import build_manager_view
from app.services.tool_check import harness_version
from app.services.workspace_defaults import apply_defaults, load_defaults
from tests.conftest import make_memory_store

CHAT = "chat-tag"
EMBED = "embed-tag"
OTHER_EMBED = "qwen3-embedding:0.6b"
CHAT_DIGEST = "sha256:chatweights"
EMBED_DIGEST = "sha256:embedweights"

# The runtime's reports, per role. The chat tag needs an explicit one: the
# fake's fallback declares BOTH capabilities, and the refusal under test keys
# on a report that positively says these weights cannot embed.
CHAT_REPORT = ModelCapabilities(tools=True, completion=True, embedding=False)
EMBED_REPORT = ModelCapabilities(completion=False, embedding=True)

INSTALLED = [
    ModelInfo(name=CHAT, digest=CHAT_DIGEST),
    ModelInfo(name=EMBED, digest=EMBED_DIGEST),
    ModelInfo(name=OTHER_EMBED, digest="sha256:otherweights"),
]
REPORTS = {CHAT: CHAT_REPORT, EMBED: EMBED_REPORT, OTHER_EMBED: EMBED_REPORT}

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")
MEMBER = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="member")


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    clear_facts_cache()
    embedder_verify.clear_embedder_checks()
    model_verify.clear_verifications()
    yield
    clear_facts_cache()
    embedder_verify.clear_embedder_checks()
    model_verify.clear_verifications()


class RecordingProvider(FakeProvider):
    """A model server that records which tag every embed call was for.

    One ordered list rather than a counter, because the properties worth
    pinning are about ORDER and IDENTITY: that the probe embedded the tag under
    test, and that the deployment's own embedder was loaded again AFTER it.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("models", list(INSTALLED))
        kwargs.setdefault("capabilities_by_model", dict(REPORTS))
        super().__init__(**kwargs)
        self.embedded: list[tuple[str | None, EmbedOptions | None]] = []

    async def embed(self, texts, *, model=None, options=None):
        self.embedded.append((model, options))
        return await super().embed(texts, model=model, options=options)

    def tags(self) -> list[str | None]:
        return [model for model, _options in self.embedded]


def make_settings(**overrides) -> Settings:
    """Hermetic settings on the fake provider, matching the shipped defaults."""
    return Settings(
        model_provider="fake",
        default_chat_model="",
        thinking_chat_model=None,
        sse_heartbeat_seconds=0,
        _env_file=None,
        **overrides,
    )


async def store_active_embedder(sessionmaker, tag: str | None = EMBED) -> None:
    """Make ``tag`` the workspace's embedder, the way a settings page would.

    The check resolves "which embedder does this deployment actually query
    with" through the ordinary chain, so a test that wants an ACTIVE embedder
    has to give the deployment one rather than declare it in a fixture.
    """
    async with sessionmaker() as session:
        await apply_defaults(session, TENANT, WorkspaceDefaultsPatch(embedding_model=tag))
        await session.commit()


async def active_embedder(sessionmaker) -> str | None:
    """Resolve the active embedder as the route's claim dependency does."""
    async with sessionmaker() as session:
        stored = await load_defaults(session, TENANT)
    return (await resolve_embedding(settings=make_settings(), stored=stored)).model


async def begin(provider, sessionmaker, *, name: str | None = None, active: str | None = EMBED):
    """Run the refusals-and-claim step exactly as the route's dependency does."""
    return await embedder_verify.begin_embedder_check(
        provider, provider_name="fake", active=active, name=name
    )


async def run(provider, sessionmaker, *, model: str = EMBED, settings: Settings | None = None):
    """Drive the runner end to end, as the endpoint does after its claim."""
    return await embedder_verify.run_embedder_check_job(
        provider=provider,
        settings=settings or make_settings(),
        sessionmaker=sessionmaker,
        tenant=TENANT,
        model=model,
    )


@asynccontextmanager
async def check_client(
    provider, sessionmaker, *, principal: TenantContext | None = None
) -> AsyncIterator[AsyncClient]:
    """App-wired client over the sqlite measurement log and an in-memory store.

    The store is injected rather than left to the lazy fallback because the
    route's claim resolves the active embedder through the full chain, which
    consults it - and the fallback would point a unit test at a Qdrant URL.
    """
    settings = make_settings()
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    if principal is not None:
        app.dependency_overrides[get_current_tenant] = lambda: principal
    app.state.db_sessionmaker = sessionmaker
    app.state.vector_store = make_memory_store()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def seed_probe(
    sessionmaker,
    *,
    model: str = CHAT,
    digest: str | None = CHAT_DIGEST,
    depth: str,
    earned: str = "agentic",
) -> ModelProbe:
    """Append one probe row at an arbitrary depth, as a harness would."""
    async with sessionmaker() as session:
        row = ModelProbe(
            provider="fake",
            model=model,
            digest=digest,
            runtime_version="0.32.0",
            serving_endpoint=FAKE_SERVING_ENDPOINT,
            harness_version=harness_version(),
            depth=depth,
            earned_agency=earned,
            thinking_with_tools=False,
            cases={"tallies": {}, "representative": True},
        )
        await ModelProbeRepository(session).add(row)
        await session.commit()
        return row


# --------------------------------------------------------------------------
# Resolution and placement
# --------------------------------------------------------------------------


async def test_the_active_embedder_is_the_default_target(db_sessionmaker) -> None:
    # A bare request measures what this deployment actually queries with, and
    # the caller resolves that through the ordinary chain rather than the
    # service looking it up - the chain needs a workspace and a store, and this
    # service is meant to hold neither.
    await store_active_embedder(db_sessionmaker)
    provider = RecordingProvider()
    assert await begin(
        provider, db_sessionmaker, active=await active_embedder(db_sessionmaker)
    ) == (EMBED)
    assert embedder_verify.check_in_flight() == EMBED


async def test_the_run_embeds_the_target_at_the_deployments_device_pin(db_sessionmaker) -> None:
    # EMBEDDING_DEVICE is this machine's policy for where embedding work runs,
    # and a tight card pins it to the CPU precisely so a retrieval cannot evict
    # the chat model. Probing it onto the card instead would measure a
    # placement the deployment never serves from - and evict the chat model to
    # do it.
    await store_active_embedder(db_sessionmaker)
    provider = RecordingProvider()
    outcome = await run(provider, db_sessionmaker)

    assert provider.tags() == [EMBED] * len(provider.embedded)
    assert all(options is not None and options.device == "cpu" for _t, options in provider.embedded)
    assert outcome.digest == EMBED_DIGEST
    # Nothing describes this tag, so its queries embed RAW and the run says so
    # rather than borrowing a wrapper from a neighbouring model.
    assert outcome.summary.prefix.source == "none"
    assert outcome.summary.prefix.prefix == ""


async def test_the_placement_follows_the_setting_in_both_directions(db_sessionmaker) -> None:
    # The other half of the pin, so the assertion above is about the SETTING
    # rather than about a constant that happens to read "cpu".
    await store_active_embedder(db_sessionmaker)
    provider = RecordingProvider()
    await run(provider, db_sessionmaker, settings=make_settings(embedding_device="auto"))
    assert all(
        options is not None and options.device == "auto" for _t, options in provider.embedded
    )


async def test_a_candidate_embedder_uses_the_seeded_prefix_and_is_put_back(
    db_sessionmaker,
) -> None:
    # A tag this deployment does not query with is measured under the wrapper
    # its own model card prescribes, read from the quirk seed - and the
    # deployment's own embedder is reloaded afterwards, because the check just
    # put something else in its place on a card that fits one model.
    await store_active_embedder(db_sessionmaker)
    provider = RecordingProvider()
    outcome = await run(provider, db_sessionmaker, model=OTHER_EMBED)

    assert outcome.active is False
    assert outcome.summary.prefix.source == "known_models"
    assert outcome.summary.prefix.prefix.startswith("Instruct:")
    assert provider.tags()[-1] == EMBED
    assert OTHER_EMBED in provider.tags()


async def test_checking_the_active_embedder_leaves_it_where_it_already_was(
    db_sessionmaker,
) -> None:
    # Nothing was displaced, so there is nothing to put back: a re-warm here
    # would be a second load of the model that is already resident.
    await store_active_embedder(db_sessionmaker)
    provider = RecordingProvider()
    outcome = await run(provider, db_sessionmaker)
    assert outcome.active is True
    assert set(provider.tags()) == {EMBED}


async def test_a_deployment_with_no_active_embedder_has_nothing_to_put_back(
    db_sessionmaker,
) -> None:
    # Nothing stored, so the chain resolves no embedder for this run to work
    # around. The check still measures the named tag - a hand-pulled embedder
    # on an unconfigured box is exactly who needs bands - and simply skips the
    # restore rather than warming a model nobody chose.
    provider = RecordingProvider()
    outcome = await run(provider, db_sessionmaker, model=OTHER_EMBED)
    assert outcome.active is False
    assert set(provider.tags()) == {OTHER_EMBED}


async def test_a_run_that_could_not_embed_raises_rather_than_recording_bands(
    db_sessionmaker,
) -> None:
    # Bands from a failed run describe the outage, and a floors verdict taken
    # on them could invalidate perfectly correct configuration.
    class DeadProvider(RecordingProvider):
        async def embed(self, texts, *, model=None, options=None):
            raise ProviderUnavailableError("Cannot reach the model server.")

    with pytest.raises(ProviderUnavailableError):
        await run(DeadProvider(), db_sessionmaker)
    async with db_sessionmaker() as session:
        assert await ModelProbeRepository(session).latest("fake", EMBED) is None


# --------------------------------------------------------------------------
# Refusals, decided before anything is embedded
# --------------------------------------------------------------------------


async def test_a_tag_the_runtime_calls_a_chat_model_is_refused(db_sessionmaker) -> None:
    # The RUNTIME's report decides, because whether weights can embed is
    # mechanical and the server holding them is the only party that can say.
    with pytest.raises(BadRequestError, match="only measures embedding models"):
        await begin(RecordingProvider(), db_sessionmaker, name=CHAT)
    assert embedder_verify.check_in_flight() is None


async def test_a_tag_whose_probe_failed_is_checked_anyway(db_sessionmaker) -> None:
    # "The runtime would not say" is not evidence that a model cannot embed,
    # and a hand-pulled embedder on a half-answering server is exactly the case
    # somebody most needs bands for. Refusing on silence would make it
    # unmeasurable.
    class Silent(RecordingProvider):
        async def capabilities(self, model: str):
            raise RuntimeError("the runtime would not say")

    assert await begin(Silent(), db_sessionmaker, name=OTHER_EMBED) == OTHER_EMBED


async def test_weights_the_backend_does_not_hold_are_refused(db_sessionmaker) -> None:
    with pytest.raises(NotFoundError, match="not installed"):
        await begin(RecordingProvider(), db_sessionmaker, name="never-pulled:1b")


async def test_an_unusable_name_is_refused_before_the_listing(db_sessionmaker) -> None:
    with pytest.raises(UnprocessableEntityError):
        await begin(RecordingProvider(), db_sessionmaker, name="../../etc/passwd")


async def test_no_resolved_embedder_and_no_name_is_a_refusal_not_a_404(db_sessionmaker) -> None:
    # The CONFIGURATION is what is missing, not a model - nothing is stored,
    # nothing was observed, and either nothing installed can embed or several
    # can and nobody chose. So it reads as a bad request, and the message says
    # a name would work.
    with pytest.raises(BadRequestError, match="Name a model explicitly"):
        await begin(RecordingProvider(), db_sessionmaker, active=None)


async def test_a_second_check_is_refused_while_one_runs(db_sessionmaker) -> None:
    provider = RecordingProvider()
    await begin(provider, db_sessionmaker)
    with pytest.raises(ConflictError, match="already running"):
        await begin(provider, db_sessionmaker)


async def test_a_check_waits_for_a_running_tool_check(db_sessionmaker) -> None:
    # The tool check is mid-way through deliberately unloading models to
    # measure a fresh load - and the embedder is one of the things it unloads.
    model_verify.claim_verification(CHAT)
    with pytest.raises(ConflictError, match="tool check is running"):
        await begin(RecordingProvider(), db_sessionmaker)
    assert embedder_verify.check_in_flight() is None


async def test_a_running_check_also_blocks_a_tool_check(db_sessionmaker) -> None:
    # The other direction, and the one with teeth: a tool check started
    # underneath a running embedder check gets a model loaded back onto the
    # card mid-measurement, its disturbed sample scores as a case FAILURE, and
    # failure evidence clamps a model's earned agency whether or not the run
    # was representative. A persisted clamp caused by an unrelated diagnostic
    # is exactly what the mutual exclusion prevents.
    provider = RecordingProvider()
    tag = await begin(provider, db_sessionmaker)
    assert model_verify.verification_in_flight() == tag
    with pytest.raises(ConflictError):
        await model_verify.begin_verification(provider, provider_name="fake", name=CHAT)
    embedder_verify.release_probe_slots(tag)
    # Both slots come back, so the next probe of either kind can start.
    assert model_verify.verification_in_flight() is None
    assert embedder_verify.check_in_flight() is None


async def test_the_shared_slot_does_not_blame_a_tool_check_that_is_not_running(
    db_sessionmaker,
) -> None:
    """The refusal must not name a harness that is not the one holding the slot.

    ``model_verify`` owns the single probe slot and the embedder check takes it
    too, so its wording is read by BOTH. Saying "a tool check is already
    running" while an embedder check holds it sends an operator looking through
    the tool-check UI for a run that does not exist.

    MUTATION: put "tool check" back in ``claim_verification``'s ConflictError
    and this fails on the substring assertion.
    """
    provider = RecordingProvider()
    tag = await begin(provider, db_sessionmaker)
    with pytest.raises(ConflictError) as raised:
        await model_verify.begin_verification(provider, provider_name="fake", name=CHAT)
    message = str(raised.value)
    assert "tool check" not in message
    # The MODEL is what locates the running probe either way, so it stays.
    assert tag in message
    embedder_verify.release_probe_slots(tag)


def test_the_slot_is_released_only_by_its_holder() -> None:
    # An unconditional release would hand a running check's slot to a third
    # caller with nothing anywhere saying so.
    embedder_verify.claim_check(EMBED)
    embedder_verify.release_check("someone-else")
    assert embedder_verify.check_in_flight() == EMBED
    embedder_verify.release_check(EMBED)
    assert embedder_verify.check_in_flight() is None


# --------------------------------------------------------------------------
# What gets recorded
# --------------------------------------------------------------------------


async def test_the_reading_is_recorded_at_the_embedder_depth(
    db_sessionmaker,
) -> None:
    provider = RecordingProvider()
    outcome = await run(provider, db_sessionmaker)

    assert outcome.probe_id is not None
    async with db_sessionmaker() as session:
        row = await ModelProbeRepository(session).latest("fake", EMBED)
    assert row is not None
    assert row.depth == PROBE_DEPTH
    # An embedder never drives a turn, so the honest verdict for "did these
    # weights demonstrate they can drive" is the floor - and it can only clamp
    # a model nothing routes a turn to.
    assert row.earned_agency == "deterministic"
    assert row.thinking_with_tools is False
    # No tokens were generated, so the column the grid reads as decode
    # throughput stays empty rather than carrying an embedding latency.
    assert row.tokens_per_second is None
    assert row.digest == EMBED_DIGEST
    assert row.triggered_by_tenant_id == DEFAULT_TENANT_ID
    cases = row.cases
    assert cases["embedder_harness_version"] == outcome.summary.harness_version
    # The seed is the ONLY source of a query wrapper now, so the row records
    # which of its two answers governed - never a per-deployment file, whose
    # copy of the string could differ from the one live retrieval uses.
    assert cases["prefix"]["source"] in {"known_models", "none"}
    assert cases["bands"]["wrapped"]["related"]["count"] > 0
    assert cases["bands"]["raw"]["separation"] == outcome.summary.raw.separation
    assert [t["name"] for t in cases["bands"]["wrapped"]["triplets"]]
    assert cases["floors"]["declared"]["source"] in {"known_models", "none"}


async def test_the_floors_in_force_are_judged_beside_the_declared_ones(
    db_sessionmaker,
) -> None:
    # The finding this exists for: an operator who edits RAG_MIN_SCORE gets a
    # report validating the number the SEED declares, which is not the number
    # deciding what their answers may cite. Both are measured, and the row says
    # which is which.
    settings = make_settings()
    outcome = await embedder_verify.run_embedder_check_job(
        provider=RecordingProvider(),
        settings=settings,
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=EMBED,
    )
    configured = outcome.configured
    assert (configured.min_score, configured.rel_margin) == (
        settings.rag_min_score,
        settings.rag_rel_margin,
    )
    # EMBED has no seed entry, so nothing is declared and the two cannot match.
    assert outcome.summary.floors.declared.source == "none"
    assert outcome.summary.floors.composed is None  # nothing declared to compose
    assert configured.matches_declared is False
    # The configured pair is COMPOSED even though nothing was declared - which
    # is the whole point: an undeclared embedder still has floors in force, and
    # they are what decides what its answers may cite.
    assert configured.hold == configured.composed.holds
    assert len(configured.composed.queries) == 8
    assert {check.name for check in configured.checks} == {
        "min_score_admits_related",
        "min_score_excludes_unrelated",
    }
    async with db_sessionmaker() as session:
        row = await ModelProbeRepository(session).latest("fake", EMBED)
    assert row is not None
    stored = row.cases["floors"]["configured"]
    assert stored["min_score"] == settings.rag_min_score
    assert stored["matches_declared"] is False
    assert stored["hold"] == configured.hold
    # The per-query simulation is kept, not just the summary: a row that only
    # said "it leaked" would send the next reader back to re-run the probe.
    assert stored["composed"]["holds"] == configured.hold
    assert len(stored["composed"]["queries"]) == 8


async def test_edited_floors_are_the_ones_the_report_measures(
    db_sessionmaker,
) -> None:
    # Drive the divergence directly: a deployment that moved its floor gets a
    # verdict about the moved floor, not about the template default.
    settings = make_settings()
    moved = settings.model_copy(update={"rag_min_score": 0.99, "rag_rel_margin": 0.01})
    outcome = await embedder_verify.run_embedder_check_job(
        provider=RecordingProvider(),
        settings=moved,
        sessionmaker=None,
        tenant=None,
        model=EMBED,
    )
    assert outcome.configured.min_score == 0.99
    # A floor at 0.99 admits nothing, so every real answer is lost - which the
    # COMPOSED gate reports, because that is the failure a user would notice.
    assert outcome.configured.hold is False
    assert outcome.configured.composed.related_lost
    assert "excluded the real answer" in outcome.configured.composed.detail
    admits = next(
        check for check in outcome.configured.checks if check.name == "min_score_admits_related"
    )
    assert admits.passed is False
    assert "0.99" in admits.detail


async def test_a_proposal_records_that_nothing_applied_it(db_sessionmaker) -> None:
    # Floors are configuration. A stored number that looks like configuration
    # invites being treated as configuration, so the row says out loud that
    # nothing read it back.
    provider = RecordingProvider()
    await run(provider, db_sessionmaker)
    async with db_sessionmaker() as session:
        row = await ModelProbeRepository(session).latest("fake", EMBED)
    assert row is not None
    proposal = row.cases["floors"]["proposal"]
    # The fake's hash vectors describe an embedder nobody declared floors for,
    # so a proposal is exactly what this run should carry.
    assert proposal is not None
    assert proposal["applied"] is False
    assert proposal["derivation"]


async def test_no_sessionmaker_measures_without_writing(db_sessionmaker) -> None:
    settings = make_settings()
    outcome = await embedder_verify.run_embedder_check_job(
        provider=RecordingProvider(),
        settings=settings,
        sessionmaker=None,
        tenant=None,
        model=EMBED,
    )
    assert outcome.probe_id is None
    assert outcome.summary.wrapped.related.count > 0
    async with db_sessionmaker() as session:
        assert await ModelProbeRepository(session).latest("fake", EMBED) is None


async def test_a_failed_write_does_not_lose_the_measurement(
    monkeypatch: pytest.MonkeyPatch, db_sessionmaker
) -> None:
    # The expensive part already happened; reporting it as a failed check would
    # send someone to re-run the probe when the thing to fix is the database.
    async def boom(self, probe):
        raise RuntimeError("the log is down")

    monkeypatch.setattr(ModelProbeRepository, "add", boom)
    outcome = await run(RecordingProvider(), db_sessionmaker)
    assert outcome.probe_id is None
    assert outcome.summary.wrapped.separation is not None


# --------------------------------------------------------------------------
# The embedder depth is invisible to every agency reader
# --------------------------------------------------------------------------


async def verdicts(provider, db_sessionmaker, *, require: bool):
    """Resolve the earned half of the agency inputs for the chat tag.

    The ceiling is passed as ``agentic`` deliberately: a lower one would clamp
    the answer on its own, and every assertion below about a row being ignored
    would then hold for the wrong reason.
    """
    return await effective_verdicts(
        provider=provider,
        provider_name="fake",
        model=CHAT,
        settings=make_settings(),
        require_verification=require,
        sessionmaker=db_sessionmaker,
        effective_ceiling="agentic",
    )


async def test_an_embedder_row_never_reaches_the_agency_resolver(
    db_sessionmaker,
) -> None:
    # The most dangerous row that could exist: an "agentic" verdict recorded
    # against the CHAT tag's own installed weights, at the embedder depth. If a
    # depth filter were ever dropped from an agency read, this row would hand a
    # model the drive on the strength of a measurement that never generated a
    # token.
    provider = RecordingProvider()
    await seed_probe(db_sessionmaker, depth=PROBE_DEPTH, earned="agentic")

    hardened = await verdicts(provider, db_sessionmaker, require=True)
    assert hardened.earned == "deterministic"


async def test_the_same_row_at_a_read_depth_would_have_been_seen(
    db_sessionmaker,
) -> None:
    # The control the test above needs to mean anything: identical row, one
    # field different, and the resolver acts on it. Without this, a test
    # asserting "deterministic" would pass just as happily against a query that
    # returned nothing for unrelated reasons.
    provider = RecordingProvider()
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")

    hardened = await verdicts(provider, db_sessionmaker, require=True)
    assert hardened.earned == "agentic"


async def test_an_embedder_row_is_invisible_to_the_explanation_too(
    db_sessionmaker,
) -> None:
    # The reader's read spans every set of weights, so a row a re-pull left
    # behind still shows up there - which is exactly why the embedder depth has
    # to be excluded from it as well, or a chat model's card would explain
    # itself with a measurement of an embedder.
    settings = make_settings()
    await seed_probe(db_sessionmaker, depth=PROBE_DEPTH, earned="agentic")

    explanation = await explain_earned(
        provider_name="fake",
        model=CHAT,
        settings=settings,
        require_verification=True,
        sessionmaker=db_sessionmaker,
        digests={CHAT: CHAT_DIGEST},
        serving_endpoint=FAKE_SERVING_ENDPOINT,
        effective_ceiling="agentic",
    )
    assert explanation.fast is None
    assert explanation.deep is None
    assert explanation.latest_any_weights == ()
    assert explanation.earned == "deterministic"


async def test_the_depth_filtered_reads_skip_it_and_history_keeps_it(
    db_sessionmaker,
) -> None:
    # The repository's contract, stated as a test: the depth-filtered reads are
    # what make the separation structural, while the unfiltered ones are
    # reporting queries that legitimately span every depth.
    row = await seed_probe(db_sessionmaker, depth=PROBE_DEPTH)
    async with db_sessionmaker() as session:
        repo = ModelProbeRepository(session)
        assert await repo.latest("fake", CHAT, depth="fast") is None
        assert await repo.latest("fake", CHAT, depth="deep") is None
        unfiltered = await repo.latest("fake", CHAT)
        assert unfiltered is not None and unfiltered.id == row.id
        assert [probe.id for probe in await repo.history("fake", CHAT)] == [row.id]


async def test_the_manager_view_reports_the_chat_model_as_unmeasured(
    db_sessionmaker,
) -> None:
    # The grid joins the same log. An embedder row against the chat tag must
    # leave the chat row honestly unmeasured rather than filling its measured
    # facts with a reading that generated no tokens.
    settings = make_settings()
    await seed_probe(db_sessionmaker, depth=PROBE_DEPTH, earned="agentic")

    view = await build_manager_view(
        provider=RecordingProvider(),
        provider_name="fake",
        settings=settings,
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        store=None,
    )
    chat_row = next(row for group in view.groups for row in group.rows if row.model == CHAT)
    assert chat_row.measured is None
    # Nor does it surface as a verdict a re-pull left behind, which is the
    # other place a row measured on other terms could show up.
    assert chat_row.weights_change is None


# --------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------


async def test_the_endpoint_answers_with_the_reading(db_sessionmaker) -> None:
    await store_active_embedder(db_sessionmaker)
    provider = RecordingProvider()
    async with check_client(provider, db_sessionmaker) as client:
        # No body at all: the common case is "measure what this deployment
        # actually queries with", and it should not need one.
        response = await client.post("/api/v1/models/embedder-check")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == EMBED
    assert body["active"] is True
    assert body["digest"] == EMBED_DIGEST
    assert body["prefix"]["source"] == "none"
    assert body["wrapped"]["related"]["count"] > 0
    assert len(body["wrapped"]["triplets"]) == len(body["raw"]["triplets"])
    assert body["floors"]["proposal"]["applied"] is False
    # The floors actually in force travel beside the declared verdict.
    assert body["configured"]["min_score"] == make_settings().rag_min_score
    assert {check["name"] for check in body["configured"]["checks"]}
    assert body["probe_id"]
    # The slot is released with the request, or the next check would 409
    # forever.
    assert embedder_verify.check_in_flight() is None


async def test_the_endpoint_measures_a_named_candidate(db_sessionmaker) -> None:
    provider = RecordingProvider()
    async with check_client(provider, db_sessionmaker) as client:
        response = await client.post("/api/v1/models/embedder-check", json={"name": OTHER_EMBED})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == OTHER_EMBED
    assert body["active"] is False
    assert body["prefix"]["source"] == "known_models"


async def test_the_endpoint_refuses_before_it_measures(db_sessionmaker) -> None:
    provider = RecordingProvider()
    async with check_client(provider, db_sessionmaker) as client:
        missing = await client.post(
            "/api/v1/models/embedder-check", json={"name": "never-pulled:1b"}
        )
        chat = await client.post("/api/v1/models/embedder-check", json={"name": CHAT})
    assert missing.status_code == 404
    assert chat.status_code == 400
    assert provider.embedded == []


async def test_the_endpoint_reports_a_ragged_answer_as_a_backend_fault(
    db_sessionmaker,
) -> None:
    # The harness raises ValueError when the vectors cannot be matched to the
    # passages they came from. That is the model server misbehaving, and it
    # must not reach a client as a bare 500 the endpoint never documented.
    class RaggedProvider(RecordingProvider):
        async def embed(self, texts, *, model=None, options=None):
            vectors = await super().embed(texts, model=model, options=options)
            return vectors[:-1]

    await store_active_embedder(db_sessionmaker)
    async with check_client(RaggedProvider(), db_sessionmaker) as client:
        response = await client.post("/api/v1/models/embedder-check")
    assert response.status_code == 503
    assert "could not be matched" in response.json()["detail"]
    assert embedder_verify.check_in_flight() is None
    assert model_verify.verification_in_flight() is None


async def test_the_endpoint_is_admin_gated(db_sessionmaker) -> None:
    # It loads a model and writes to the shared measurement log, so it declares
    # the gate now even though the pre-identity role check always passes for
    # the seeded user.
    provider = RecordingProvider()
    async with check_client(provider, db_sessionmaker, principal=MEMBER) as client:
        response = await client.post("/api/v1/models/embedder-check")
    assert response.status_code == 403
