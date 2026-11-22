"""The Model Manager read view: a pure INVENTORY, joined to what is known of it.

One row per tag the model server actually holds, grouped by the role the
RUNTIME's report derives for it. There is no second list of models a deployment
might have downloaded but did not: such a row has no digest, no measurement, no
verb worth offering, and a page full of them buries the ones somebody came for.

So a fixture here is a SERVER - a listing plus a capability report per tag -
and the declarations, probe rows and workspace settings are what get layered
over it. Nothing is read off a committed file, which is why every test that
used to describe a model now describes what the runtime says about it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models.entities import ModelPolicy, ModelProbe, WorkspaceSettings
from app.providers.base import (
    ManagementCapabilities,
    ModelCapabilities,
    ModelInfo,
    ProviderUnavailableError,
    tag_aliases,
)
from app.providers.fake_provider import FAKE_SERVING_ENDPOINT, FakeProvider
from app.providers.rag.base import ChunkPayload, ChunkPoint
from app.repositories.conversations import ConversationRepository
from app.schemas.chat import ChatRequest
from app.schemas.models import ManagerRow, MeasuredFacts, ModelManagerResponse
from app.schemas.settings import WorkspaceDefaultsPatch
from app.services.agency import parse_agency_mode, resolve_agency
from app.services.earned import ProbeView, effective_verdicts
from app.services.model_facts import resolve_num_ctx, runtime_capabilities
from app.services.model_manager import (
    _failed_case_detail,
    active_chat_role,
    build_manager_view,
    derive_role,
    management_of,
)
from app.services.model_policy import clear_policy_cache
from app.services.model_resolution import ChatModelPair, resolve_chat_models
from app.services.tool_check import harness_version
from app.services.turn_plan import plan_turn
from app.services.workspace_defaults import apply_defaults
from tests.conftest import make_memory_store

CHAT = "chat:4b"
THINK = "chat:4b-think"
EMBED = "embed:0.6b"
OTHER_CHAT = "other-chat:8b"
# A tag whose template takes no tool list - the mechanical floor's own shape.
PROSE = "prose:7b"
# A tag nothing describes at all, and which the runtime will not describe
# either: the empty role, which is a different claim from "no capabilities".
UNKNOWN = "mystery:1b"

# What the runtime says about the server these tests join against. This is the
# ONLY source of a role now, so it is what decides which group a row lands in,
# whether the tool-check button is offered, and whether the embedding lock
# applies - which is why every tag a test touches is described here.
CHAT_REPORT = ModelCapabilities(completion=True, tools=True, template_thinking=False)
THINKING_REPORT = ModelCapabilities(completion=True, tools=True, template_thinking=True)
EMBED_REPORT = ModelCapabilities(embedding=True)
TOOLLESS_REPORT = ModelCapabilities(completion=True, tools=False, template_thinking=False)

REPORTS: dict[str, ModelCapabilities] = {
    CHAT: CHAT_REPORT,
    THINK: THINKING_REPORT,
    OTHER_CHAT: THINKING_REPORT,
    EMBED: EMBED_REPORT,
    PROSE: TOOLLESS_REPORT,
}


def installed(*names: str) -> list[ModelInfo]:
    """A provider listing where every named tag is present with a digest."""
    return [ModelInfo(name=name, size=1_000, digest=f"sha256:{name}") for name in names]


class PartlySilentProvider(FakeProvider):
    """A server that refuses to describe some of the tags it holds.

    "The runtime would not say" is a different claim from a report of no
    capabilities, and the view has to read it that way: an empty role rather
    than a refusal, and every capability column at its floor.
    """

    def __init__(self, *, silent: set[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._silent = silent

    async def capabilities(self, model: str) -> ModelCapabilities:
        if model in self._silent:
            raise ProviderUnavailableError("the model server dropped the probe")
        return await super().capabilities(model)


def server(*names: str, silent: set[str] | None = None, **kwargs) -> FakeProvider:
    """A fake model server holding ``names`` and reporting :data:`REPORTS`.

    A caller's own ``capabilities_by_model`` is merged OVER the shared map, so
    describing one tag differently does not silently take the reports away from
    every other tag in the same listing.

    Args:
        *names: The tags the server holds.
        silent: Tags whose capability probe raises, staging the "the runtime
            would not say" answer that a report of no capabilities must never
            be confused with.
        **kwargs: Passed to :class:`FakeProvider`.

    Returns:
        The configured provider.
    """
    kwargs.setdefault("models", installed(*names))
    kwargs["capabilities_by_model"] = {**REPORTS, **(kwargs.get("capabilities_by_model") or {})}
    if silent:
        return PartlySilentProvider(silent=silent, **kwargs)
    return FakeProvider(**kwargs)


def SilentProvider(**kwargs) -> FakeProvider:  # noqa: N802 - reads as a type at the call site
    """A server that will not describe ANY of the tags it holds."""
    listing = kwargs.get("models") or []
    return PartlySilentProvider(silent={info.name for info in listing}, **kwargs)


def make_settings(**overrides: object) -> Settings:
    """The deployment's configuration: the chat pair, and whatever a test moves."""
    fields: dict[str, object] = {
        "model_provider": "fake",
        "default_chat_model": CHAT,
        "thinking_chat_model": THINK,
        "_env_file": None,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


async def build(
    db_sessionmaker,
    *,
    provider: FakeProvider | None = None,
    settings: Settings | None = None,
    store=None,
) -> ModelManagerResponse:
    return await build_manager_view(
        provider=provider if provider is not None else server(),
        provider_name="fake",
        settings=settings if settings is not None else make_settings(),
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        store=store,
    )


async def choose_embedder(db_sessionmaker, model: str) -> None:
    """Store ``model`` as the workspace's embedder, as a settings page would.

    The top rung of the embedding chain. Tests that are about a corpus written
    by SOMEBODY ELSE have to set it, because with nothing stored the chain
    falls back to what the corpus itself shows - and then no tag can be
    stranded, since the tag in use is the one that wrote everything.
    """
    async with db_sessionmaker() as session:
        await apply_defaults(session, TENANT, WorkspaceDefaultsPatch(embedding_model=model))
        await session.commit()


async def pin_conversations(db_sessionmaker, model: str, count: int = 1) -> None:
    """Store ``count`` conversations pinned to ``model``, as the composer does."""
    async with db_sessionmaker() as session:
        repo = ConversationRepository(session, TENANT)
        for index in range(count):
            await repo.create(title=f"thread {index}", settings={"model": model})
        await session.commit()


async def store_policy(db_sessionmaker, *, model: str = CHAT, digest: str | None = None, **fields):
    """Append one operator declaration, keyed on the row's installed digest."""
    async with db_sessionmaker() as session:
        session.add(
            ModelPolicy(
                provider="fake",
                model=model,
                digest=digest if digest is not None else f"sha256:{model}",
                **fields,
            )
        )
        await session.commit()
    clear_policy_cache()


async def arm(db_sessionmaker, *, model: str = CHAT) -> None:
    """Declare ``model`` agentic - the ONE thing that can lift a ceiling that far.

    A runtime report stops at ``assisted``: "this template accepts a tool list"
    is a mechanical fact, and handing a turn to the model is a decision about
    weights that only a human makes. So every test about a clamp BELOW agentic
    has to put the declaration there first, or the ceiling it is measuring
    against does not exist.
    """
    await store_policy(db_sessionmaker, model=model, ceiling="agentic")


async def store_pair(db_sessionmaker, *, model: str, thinking_model: str) -> None:
    """Append a thinking-pair declaration, exactly as the console writer does."""
    from app.repositories.model_policy import ModelPolicyRepository

    async with db_sessionmaker() as session:
        await ModelPolicyRepository(session).add(
            ModelPolicy(provider="fake", model=model, thinking_model=thinking_model)
        )
        await session.commit()
    clear_policy_cache()


async def set_workspace_default(db_sessionmaker, model: str) -> None:
    """Store a workspace default chat model, the tier the Settings page writes."""
    async with db_sessionmaker() as session:
        row = await session.get(WorkspaceSettings, DEFAULT_TENANT_ID)
        assert row is not None, "the default tenant's workspace row should be seeded"
        row.settings = {**dict(row.settings or {}), "model": model}
        session.add(row)
        await session.commit()


async def store_ceiling(db_sessionmaker, ceiling: int) -> None:
    """Store a workspace context ceiling, as the workspace-defaults page does."""
    async with db_sessionmaker() as session:
        row = await session.get(WorkspaceSettings, DEFAULT_TENANT_ID)
        assert row is not None, "the default tenant's workspace row should be seeded"
        row.settings = {**dict(row.settings or {}), "context_ceiling": ceiling}
        session.add(row)
        await session.commit()


def rows_of(view: ModelManagerResponse, role: str) -> list[ManagerRow]:
    return next(group.rows for group in view.groups if group.role == role)


def every_row(view: ModelManagerResponse) -> list[ManagerRow]:
    """Every row the page renders. One list now: every row is inside a group."""
    return [row for group in view.groups for row in group.rows]


def row_for(view: ModelManagerResponse, model: str) -> ManagerRow:
    for group in view.groups:
        for row in group.rows:
            if row.model == model:
                return row
    raise AssertionError(f"no row for {model!r}")


async def seed_probe(
    db_sessionmaker,
    *,
    model: str = CHAT,
    digest: str | None = f"sha256:{CHAT}",
    depth: str = "fast",
    earned: str = "assisted",
    harness: str | None = None,
    runtime_version: str | None = None,
    serving_endpoint: str | None = FAKE_SERVING_ENDPOINT,
    tokens_per_second: float | None = None,
    vram_bytes: int | None = None,
    representative: bool = True,
    gpu_fraction: float | None = None,
    created_at: datetime | None = None,
    tallies: dict[str, list[int]] | None = None,
    think_mode: object = None,
    ttw: bool = False,
    thinking_digest: str | None = None,
) -> None:
    cases: dict = {
        "representative": representative,
        "residency": {"gpu_fraction": gpu_fraction},
        # The pair's thinking weights the run drove, which is what binds a
        # reasoning-case verdict. None is the honest default: it is what a row
        # carries when nothing pairs the tag, and it matches the None the
        # resolver computes there.
        "thinking_digest": thinking_digest,
    }
    # Omitted by default for the same reason as the tallies below: every other
    # row keeps the shape of a probe written before the thinking group existed.
    if think_mode is not None:
        cases["measured_think_mode"] = think_mode
    # Omitted rather than empty by default, so the rows every other test seeds
    # keep the shape of a probe written before tallies were recorded.
    if tallies is not None:
        cases["tallies"] = tallies
    row = ModelProbe(
        provider="fake",
        model=model,
        digest=digest,
        harness_version=harness if harness is not None else harness_version(),
        depth=depth,
        earned_agency=earned,
        thinking_with_tools=ttw,
        runtime_version=runtime_version,
        serving_endpoint=serving_endpoint,
        tokens_per_second=tokens_per_second,
        vram_bytes=vram_bytes,
        cases=cases,
    )
    if created_at is not None:
        # Pinned where the ORDER between two rows is what a test is about -
        # insertion order alone leaves that to clock resolution.
        row.created_at = created_at
    async with db_sessionmaker() as session:
        session.add(row)
        await session.commit()


# --- grouping ---------------------------------------------------------------------


async def test_rows_are_grouped_by_the_role_the_runtime_derives(db_sessionmaker) -> None:
    """The one derivation, applied to a whole inventory.

    Nothing declares a role any more, so the group a row lands in is entirely
    what the runtime said about those weights: it completes, it only embeds, or
    it established neither. The empty role sorts LAST, which is where a tag the
    runtime would not describe belongs - it is listed because the weights are
    here and occupy disk, not because anything is known about what they do.

    MUTATION: key the grouping on anything but ``derive_role`` - the tag name,
    a stored declaration, a hard-coded "chat" - and the embedder joins the chat
    group, picking up its verbs and losing the vector-space lock.
    """
    provider = server(CHAT, EMBED, UNKNOWN, silent={UNKNOWN})
    view = await build(db_sessionmaker, provider=provider)
    assert [group.role for group in view.groups] == ["chat", "embedding", ""]
    assert [row.model for row in rows_of(view, "chat")] == [CHAT]
    assert [row.model for row in rows_of(view, "embedding")] == [EMBED]
    assert [row.model for row in rows_of(view, "")] == [UNKNOWN]


def test_the_role_derivation_is_asymmetric_and_says_so() -> None:
    # A model that can ALSO complete is a chat model everywhere, even when it
    # embeds too: counting such a tag as an embedder would make one model a
    # candidate for two roles and put a chat model inside the embedding lock.
    # A report that could not be taken answers "" - "the runtime would not
    # say" - which is a different claim from a report of no capabilities.
    assert derive_role(ModelCapabilities(completion=True, embedding=True)) == "chat"
    assert derive_role(ModelCapabilities(embedding=True)) == "embedding"
    assert derive_role(ModelCapabilities()) == ""
    assert derive_role(None) == ""


def test_each_role_carries_its_own_management_policy() -> None:
    # The embedding lock is the load-bearing one, and the empty role is
    # deliberately MANAGEABLE: nothing establishes what such a tag does, and
    # refusing to manage it would be a guess in the more surprising direction
    # than simply offering the provider's own verbs.
    assert management_of("chat") == ("manageable", None)
    assert management_of("embedding")[0] == "locked"
    assert "vector space" in management_of("embedding")[1]
    # Not a model this provider serves at all - it runs in a sidecar - so it
    # gets no row from this view, and the policy stays for whatever reports it.
    assert management_of("reranker")[0] == "status_only"
    assert management_of("") == ("manageable", None)


async def test_the_thinking_sibling_gets_a_row_and_is_named_on_its_partner(
    db_sessionmaker,
) -> None:
    # The inventory lists what is HELD, so a pair's thinking half is a row like
    # any other - it used to be hidden so a conversation could not pin it, and
    # hiding installed weights is what made a remote server's inventory
    # unreadable. The pairing is still reported on the partner's row, because
    # that is the tag a thinking turn is routed FROM.
    view = await build(db_sessionmaker, provider=server(CHAT, THINK))
    assert row_for(view, CHAT).thinking_model == THINK
    assert row_for(view, THINK).model == THINK


async def test_embedding_is_locked_against_change(db_sessionmaker) -> None:
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED))
    group = next(g for g in view.groups if g.role == "embedding")
    assert group.management == "locked"
    assert group.lock_reason and "vector space" in group.lock_reason


async def test_the_locked_role_offers_neither_deletion_nor_updates(db_sessionmaker) -> None:
    # Both verbs withdrawn, for the same reason: an UPDATE to an installed
    # embedder is the silent form of the swap this lock prevents - changed
    # weights under an unchanged tag shift the vector space while the
    # name-keyed provenance keeps the stranded-vectors alarm quiet. (No store
    # is wired here, so the delete side is also the fail-closed answer - the
    # per-row policy below is what hands the verb back.)
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED))
    row = row_for(view, EMBED)
    assert (row.installed, row.can_pull, row.can_delete) == (True, False, False)


EMBED_ALT = "embed-alt:4b"


def embed_server(*names: str) -> FakeProvider:
    """A server holding two embedding-capable tags beside the chat model."""
    return server(*names, capabilities_by_model={EMBED_ALT: EMBED_REPORT})


async def test_an_installed_embedder_nothing_references_gets_the_delete_verb(
    db_sessionmaker,
) -> None:
    # The lock is on what STRANDS vectors. An installed alternate no stored
    # passage was embedded by strands nothing, so its row gets Delete back -
    # while the ACTIVE embedder keeps the refusal, with the reason on the row.
    await choose_embedder(db_sessionmaker, EMBED)
    store = make_memory_store()
    view = await build(db_sessionmaker, provider=embed_server(CHAT, EMBED, EMBED_ALT), store=store)
    await store.aclose()
    alternate = row_for(view, EMBED_ALT)
    assert (alternate.can_delete, alternate.delete_blocked_reason) == (True, None)
    active = row_for(view, EMBED)
    assert active.can_delete is False
    assert "active embedding model" in active.delete_blocked_reason


async def test_an_embedder_with_stored_passages_keeps_the_refusal(db_sessionmaker) -> None:
    # Stored passages can only be searched through the weights that embedded
    # them, so a referenced alternate stays refused - and the reason carries
    # the count, because "how much would this strand" is the first question.
    #
    # The workspace names its embedder so the alternate is a BYSTANDER holding
    # passages rather than the active model itself, which is a different (and
    # separately tested) refusal.
    await choose_embedder(db_sessionmaker, EMBED)
    store = make_memory_store()
    await store.ensure_ready("embed-alt-4b", 32)
    await store.upsert(
        "kb",
        "embed-alt-4b",
        str(DEFAULT_TENANT_ID),
        [
            ChunkPoint(
                id=str(uuid.uuid4()),
                vector=[0.1] * 32,
                payload=ChunkPayload(
                    tenant_id=str(DEFAULT_TENANT_ID),
                    doc_id="doc-1",
                    text="a passage in the alternate's space",
                    created_at=1000.0,
                    embed_model=EMBED_ALT,
                ),
            )
        ],
    )
    view = await build(db_sessionmaker, provider=embed_server(CHAT, EMBED, EMBED_ALT), store=store)
    await store.aclose()
    row = row_for(view, EMBED_ALT)
    assert row.can_delete is False
    assert "1 stored passage" in row.delete_blocked_reason


async def test_embedding_deletion_fails_closed_without_a_store_answer(db_sessionmaker) -> None:
    # Every other unknown on this page renders as an honest "unknown"; this
    # one gates a destructive verb whose damage is exactly what the missing
    # report would have warned about, so no answer means no verb.
    await choose_embedder(db_sessionmaker, EMBED)
    view = await build(db_sessionmaker, provider=embed_server(CHAT, EMBED, EMBED_ALT))
    row = row_for(view, EMBED_ALT)
    assert row.can_delete is False
    assert "did not answer" in row.delete_blocked_reason


async def test_chat_and_undescribed_rows_are_manageable(db_sessionmaker) -> None:
    provider = server(CHAT, UNKNOWN, silent={UNKNOWN})
    view = await build(db_sessionmaker, provider=provider)
    modes = {group.role: group.management for group in view.groups}
    assert modes["chat"] == "manageable"
    assert modes[""] == "manageable"


# --- live state -------------------------------------------------------------------


async def test_installed_state_comes_from_the_provider(db_sessionmaker) -> None:
    view = await build(db_sessionmaker, provider=server(CHAT))
    here = row_for(view, CHAT)
    # Always true, because the view IS the inventory - and kept as a field so a
    # client need not infer from a row's presence what the row can state.
    assert here.installed is True
    assert here.installed_size_bytes == 1_000
    assert here.digest == f"sha256:{CHAT}"


async def test_the_declared_reasoning_claim_rides_along(db_sessionmaker) -> None:
    # A stored operator grant is the ONLY source of this claim - nothing infers
    # it - and it is a DECLARATION: what a probe found is in `measured`, and
    # what a turn will do is in `thinking_with_tools_effective`.
    await store_policy(db_sessionmaker, thinking_with_tools=True)
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT))
    assert row_for(view, CHAT).thinking_with_tools is True
    assert row_for(view, CHAT).thinking_with_tools_declared is True
    # Nothing declares the other one, so it reads as withheld rather than as an
    # absence a reader has to interpret.
    assert row_for(view, OTHER_CHAT).thinking_with_tools is False
    assert row_for(view, OTHER_CHAT).thinking_with_tools_declared is False


async def test_a_passing_reasoning_case_reaches_the_row_before_anyone_declares(
    db_sessionmaker,
) -> None:
    """The field a console needs to say what DECLARING would do.

    The state is the ordinary one: a deep check passed the reasoning case and
    nobody has declared the capability yet. The row must report both halves
    honestly - not in force, and established - because a client with only the
    first would tell the operator their passing check failed.

    MUTATION: source `thinking_with_tools_passed` from
    `explanation.thinking_with_tools` (the effective value) and the second
    assertion flips to False while the check plainly passed.
    """
    await seed_probe(
        db_sessionmaker,
        depth="deep",
        earned="agentic",
        ttw=True,
        thinking_digest=f"sha256:{THINK}",
    )
    view = await build(db_sessionmaker, provider=server(CHAT, THINK))
    row = row_for(view, CHAT)
    assert row.thinking_with_tools_effective is False  # nothing declares it
    assert row.measured is not None
    assert row.measured.thinking_with_tools_passed is True  # ...and yet it is established


async def test_a_reasoning_case_measured_absent_is_reported_as_absent(db_sessionmaker) -> None:
    # The other definite answer, and the one the old wording assumed was the
    # only possibility.
    await seed_probe(
        db_sessionmaker,
        depth="deep",
        earned="agentic",
        ttw=False,
        thinking_digest=f"sha256:{THINK}",
    )
    view = await build(db_sessionmaker, provider=server(CHAT, THINK))
    assert row_for(view, CHAT).measured.thinking_with_tools_passed is False


async def test_a_fast_run_leaves_the_reasoning_case_unknown_on_the_row(db_sessionmaker) -> None:
    # A measured row exists, so `measured` is present - but FAST never runs the
    # reasoning case, and reporting its column would invent a verdict.
    await seed_probe(db_sessionmaker, depth="fast", earned="assisted")
    view = await build(db_sessionmaker, provider=server(CHAT, THINK))
    row = row_for(view, CHAT)
    assert row.measured is not None
    assert row.measured.thinking_with_tools_passed is None


# --- the runtime report: the facts nothing written down can hold --------------------


async def test_the_runtime_describes_the_weights_that_are_here(db_sessionmaker) -> None:
    # Quantization, parameter count and the native window are properties of the
    # BLOB, not of anything a human wrote down, so the runtime is their only
    # source - and the native window is reported BESIDE the configured one,
    # because "8k because that is all it has" and "8k because this card cannot
    # afford more" are different answers to a reader deciding what to run.
    provider = server(
        CHAT,
        capabilities_by_model={
            CHAT: ModelCapabilities(
                completion=True,
                tools=True,
                context_length=262144,
                quantization="Q4_K_M",
                parameter_size="4.0B",
                family="qwen3",
            )
        },
    )
    settings = make_settings(default_context_window=8192)
    view = await build(db_sessionmaker, provider=provider, settings=settings)
    here = row_for(view, CHAT)
    assert (here.quantization, here.parameter_size, here.family) == ("Q4_K_M", "4.0B", "qwen3")
    assert (here.native_context_length, here.num_ctx) == (262144, 8192)


async def test_a_native_window_below_the_affordance_is_the_one_reported(db_sessionmaker) -> None:
    # The other half of the same rule: nothing may ask a model for more window
    # than its weights were trained for.
    provider = server(
        CHAT,
        capabilities_by_model={CHAT: ModelCapabilities(completion=True, context_length=4096)},
    )
    settings = make_settings(default_context_window=8192)
    view = await build(db_sessionmaker, provider=provider, settings=settings)
    assert row_for(view, CHAT).num_ctx == 4096


async def test_a_multimodal_tag_is_shown_as_seeing(db_sessionmaker) -> None:
    # The remote-server case: the runtime is the only source there is, and
    # blacking it out would describe a working multimodal model as blind.
    provider = server(
        "vl:7b",
        capabilities_by_model={
            "vl:7b": ModelCapabilities(completion=True, vision=True, template_thinking=True)
        },
    )
    view = await build(db_sessionmaker, provider=provider)
    row = row_for(view, "vl:7b")
    assert (row.vision, row.think) == (True, "hybrid")
    # ...and still not trusted to drive anything: a report describes a
    # template, it does not grant autonomy.
    assert row.agency is not None and row.agency.effective == "deterministic"


async def test_a_row_whose_probe_failed_keeps_the_safe_floor(db_sessionmaker) -> None:
    view = await build(db_sessionmaker, provider=SilentProvider(models=installed(UNKNOWN)))
    row = row_for(view, UNKNOWN)
    # Every column at its floor, and none of them a measurement: "the runtime
    # would not say" must never read as "the model cannot" OR as consent.
    assert (row.role, row.tools, row.think, row.vision) == ("", "none", "unknown", False)
    assert (row.quantization, row.native_context_length) == (None, None)
    assert row.checkable is False


async def test_a_card_names_the_window_a_turn_on_it_would_request(db_sessionmaker) -> None:
    # The card and the turn share one window rule so the number on the page is
    # the number that runs, and the workspace ceiling is part of that rule -
    # for a tag whose native window is unknown it is not a clamp at all, it is
    # the deployment's affordance. So a row that withheld the ceiling would not
    # merely skip a clamp, it would resolve a DIFFERENT window than the turn.
    #
    # Driven at a FAILED probe on purpose: that is the one input where the row
    # cannot confirm the tag chats, and the turn applies the ceiling anyway
    # (it consults no report). A blipped probe must not move the number.
    await store_ceiling(db_sessionmaker, 16384)
    settings = make_settings(default_context_window=8192)
    provider = SilentProvider(models=installed(UNKNOWN))

    view = await build(db_sessionmaker, provider=provider, settings=settings)
    requested = await resolve_num_ctx(
        provider,
        "fake",
        UNKNOWN,
        default_window=settings.default_context_window,
        ceiling=16384,
    )

    assert row_for(view, UNKNOWN).num_ctx == requested == 16384


async def test_an_embedder_is_sized_without_the_chat_turn_ceiling(db_sessionmaker) -> None:
    # The ceiling is a chat-turn budget: it bounds what one QUESTION may spend
    # of the machine. An embedder is driven by the ingest path and never reads
    # the setting, so a card showing it clamped would report a limit that does
    # not apply to it.
    await store_ceiling(db_sessionmaker, 16384)
    settings = make_settings(default_context_window=8192)
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED), settings=settings)
    assert row_for(view, CHAT).num_ctx == 16384
    assert row_for(view, EMBED).num_ctx == 8192


# --- checkable: whether the tool-check button may be offered ------------------------


async def test_only_chat_capable_rows_are_checkable(db_sessionmaker) -> None:
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED))
    assert row_for(view, CHAT).checkable is True
    # The verify endpoint refuses a tag the runtime reports as an embedder, so
    # the page must not offer it: a check against one holds the card for
    # minutes to learn nothing.
    assert row_for(view, EMBED).checkable is False


async def test_a_row_the_runtime_would_not_describe_is_not_checkable(db_sessionmaker) -> None:
    # The one place the page is STRICTER than the endpoint, which accepts an
    # unreadable report. Enabling a control on a server that will not describe
    # its own models would spend GPU minutes on a guess; the endpoint stays
    # permissive so a deliberate request still goes through.
    view = await build(db_sessionmaker, provider=SilentProvider(models=installed(UNKNOWN)))
    assert row_for(view, UNKNOWN).checkable is False


# --- role gaps: what the inventory cannot do ----------------------------------------


def gap_for(view: ModelManagerResponse, role: str):
    return next((gap for gap in view.role_gaps if gap.role == role), None)


async def test_a_healthy_inventory_reports_no_gaps(db_sessionmaker) -> None:
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED))
    assert view.role_gaps == []


async def test_a_missing_embedder_is_fatal_and_names_what_stops_working(
    db_sessionmaker,
) -> None:
    # The failure this exists for is SILENT: uploads still accept files,
    # searches still return, and nothing ever grounds anything.
    view = await build(db_sessionmaker, provider=server(CHAT))
    gap = gap_for(view, "embedding")
    assert gap is not None
    assert (gap.severity, gap.code) == ("fatal", "embedder_missing")
    # Nothing to suggest: naming a model nobody here holds would be a curated
    # list by another name, so the message points at the download box instead.
    assert gap.suggested_model is None
    assert "uploaded files" in gap.message and "ground" in gap.message
    # Worst first, so a client showing one shows the one that matters.
    assert view.role_gaps[0] is gap


async def test_a_chosen_embedder_the_server_does_not_hold_is_fatal(db_sessionmaker) -> None:
    # The other half of the same gap, and the one whose fix is a DOWNLOAD: the
    # workspace named an embedder and the server does not have it, so retrieval
    # is dead even though another embedding-capable tag is sitting right there.
    # Nothing selects that one, and choosing between installed candidates would
    # be a ranking.
    await choose_embedder(db_sessionmaker, "gone:1b")
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED))
    gap = gap_for(view, "embedding")
    assert gap is not None
    assert (gap.severity, gap.code) == ("fatal", "embedder_missing")
    assert gap.suggested_model == "gone:1b"
    assert EMBED in gap.message


async def test_several_embedders_and_nothing_chosen_is_a_choose_not_a_download(
    db_sessionmaker,
) -> None:
    view = await build(db_sessionmaker, provider=embed_server(CHAT, EMBED, EMBED_ALT))
    gap = gap_for(view, "embedding")
    assert gap is not None
    # FATAL, even though the embedders are sitting right there: retrieval
    # embeds through the SELECTED model and nothing else, so an unselected one
    # on disk does not make the feature work less well - it makes it not work.
    # The code, not the severity, is what carries "a choice fixes this".
    assert (gap.severity, gap.code) == ("fatal", "embedder_unselected")
    # No suggestion: the fix is a choice, and choosing between two would be a
    # ranking. Every suggestion this surface makes is a tag to DOWNLOAD.
    assert gap.suggested_model is None
    assert EMBED in gap.message and EMBED_ALT in gap.message


async def test_a_chat_model_that_can_also_embed_is_not_named_as_an_embedder(
    db_sessionmaker,
) -> None:
    # It renders as a CHAT card (completion wins the role derivation), so
    # naming it in the embedding banner would put the banner and the cards
    # beneath it in contradiction - the exact failure the gaps are built from
    # the finished rows to avoid.
    provider = server(
        "dual:7b",
        capabilities_by_model={"dual:7b": ModelCapabilities(completion=True, embedding=True)},
    )
    view = await build(db_sessionmaker, provider=provider)
    gap = gap_for(view, "embedding")
    assert gap is not None
    assert "dual:7b" not in gap.message


async def test_a_server_whose_models_cannot_take_tools_says_so(db_sessionmaker) -> None:
    view = await build(db_sessionmaker, provider=server(EMBED, PROSE))
    gap = gap_for(view, "chat")
    assert gap is not None
    assert (gap.severity, gap.code) == ("note", "tools_incapable")


async def test_capable_models_held_at_deterministic_are_a_different_gap(
    db_sessionmaker,
) -> None:
    # The distinction the banner exists for: this deployment needs a
    # DECLARATION, a passing check, or a change to its own clamp - and
    # downloading anything would be wasted effort. The deployment clamp is the
    # cleanest case of it, because it holds a model whose template the runtime
    # vouches for.
    settings = make_settings(agency_mode="deterministic")
    view = await build(db_sessionmaker, provider=server(EMBED, CHAT), settings=settings)
    gap = gap_for(view, "chat")
    assert gap is not None
    assert (gap.severity, gap.code) == ("note", "tools_clamped")
    assert gap.suggested_model is None


async def test_a_vouched_tag_counts_as_cleared_for_tools(db_sessionmaker) -> None:
    # The banner reads the finished rows, so a tag nobody declared anything
    # about still clears it: a server holding weights whose template takes
    # tools is not a deployment with "nothing cleared for tools", and telling
    # an operator otherwise sends them looking for a fix that is not needed.
    view = await build(db_sessionmaker, provider=server(EMBED, CHAT))
    assert gap_for(view, "chat") is None


async def test_a_server_with_nothing_to_chat_with_gets_no_tools_note(db_sessionmaker) -> None:
    # There is a much larger problem than tools in that state, and a note about
    # them underneath it would be noise.
    view = await build(db_sessionmaker, provider=server(EMBED))
    assert gap_for(view, "chat") is None


# --- management verbs -------------------------------------------------------------


async def test_verbs_follow_what_is_in_use(db_sessionmaker) -> None:
    # CHAT is what this deployment serves: refused for deletion (see below),
    # but the pull stays - on installed weights it fetches whatever the
    # registry now serves under the tag, which is how a model takes updates.
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT))
    assert (row_for(view, CHAT).can_pull, row_for(view, CHAT).can_delete) == (True, False)
    # An installed alternate nothing is serving keeps every verb.
    alternate = row_for(view, OTHER_CHAT)
    assert (alternate.can_pull, alternate.can_delete) == (True, True)


async def test_the_served_model_is_not_offered_for_deletion(db_sessionmaker) -> None:
    # On every deployment the served model is a row on this page, so that is
    # the modal row - and offering Delete there renders a button whose every
    # press returns 409 from the delete path.
    await arm(db_sessionmaker)
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.can_delete is False
    # The reason rides beside the verb, not inside the agency report: a disabled
    # control explains itself here, and the agency block keeps its one meaning.
    # Hence the deep pass above - an unclamped row is what makes "the agency
    # block said nothing about the delete" observable at all.
    assert "active chat model" in row.delete_blocked_reason
    assert row.agency.clamp_detail is None


async def test_a_declared_partner_is_protected_from_delete(db_sessionmaker) -> None:
    """The two-gate shape: routing and the delete guard have to move together.

    Declaring a pair makes turns route thinking questions to the partner. If
    only the picker learned that and the guard did not, the console would
    happily delete the model half its turns depend on - the exact half-wired
    failure that makes fixing one gate worse than fixing neither.

    MUTATION: drop `declared_thinking_model` from the `load_model_usage` call
    in `build_manager_view` and this row goes back to offering Delete.
    """
    # No environment sibling, so the stored declaration is the ONLY thing that
    # can pair this tag - which is what makes the refusal evidence that it was
    # read.
    await store_pair(db_sessionmaker, model=CHAT, thinking_model=OTHER_CHAT)
    view = await build(
        db_sessionmaker,
        provider=server(CHAT, OTHER_CHAT),
        settings=make_settings(thinking_chat_model=None),
    )
    partner = row_for(view, OTHER_CHAT)
    assert partner.can_delete is False
    assert "active thinking model" in partner.delete_blocked_reason
    # And it names the place that actually decides it: a stored pairing is not
    # deployment configuration, so pointing at the environment would be a dead
    # end.
    #
    # MUTATION: drop the `declared_partner` branch from `delete_block_reason`
    # and this reads back the environment sentence.
    assert "model manager" in partner.delete_blocked_reason
    # And the card names it, so the reason the button is gone is on screen.
    paired = row_for(view, CHAT)
    assert (paired.thinking_model, paired.thinking_model_declared) == (OTHER_CHAT, True)


async def test_the_environment_sibling_is_named_when_nothing_was_declared(
    db_sessionmaker,
) -> None:
    # The control for the row above: with no declaration the card falls back to
    # the deployment's own override, and says so by leaving the flag false.
    view = await build(db_sessionmaker, provider=server(CHAT, THINK))
    row = row_for(view, CHAT)
    assert (row.thinking_model, row.thinking_model_declared) == (THINK, False)


async def test_exactly_one_row_is_marked_the_default(db_sessionmaker) -> None:
    """The grid's badge answers "which model replies when nobody chose?".

    Always answerable, which is why it is not the stored-default flag: with
    nothing stored, `workspace_default` is false on every row and one model
    plainly still answers. Exactly one row carries this, either way.

    MUTATION: render the badge from `workspace_default` instead and the
    no-stored-default half of this fails with zero marked rows.
    """
    provider = server(CHAT, THINK, OTHER_CHAT)
    # Nothing stored: the deployment's configured tag wears it.
    view = await build(db_sessionmaker, provider=provider)
    assert [row.model for row in every_row(view) if row.is_default] == [CHAT]
    assert row_for(view, CHAT).workspace_default is False

    # Stored: it moves, and it still lands on exactly one row.
    await set_workspace_default(db_sessionmaker, OTHER_CHAT)
    view = await build(db_sessionmaker, provider=provider)
    assert [row.model for row in every_row(view) if row.is_default] == [OTHER_CHAT]


async def test_a_locked_role_never_wears_the_default_badge(db_sessionmaker) -> None:
    # An embedder answers nothing, so the badge must not be reachable by it
    # however the chain resolves.
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED))
    assert [row.model for row in every_row(view) if row.is_default] == [CHAT]


async def test_the_workspace_default_pair_is_protected_too(db_sessionmaker) -> None:
    """Two routable defaults, two pairs, and BOTH of their siblings matter.

    The guard used to read the pairing under the workspace-routed tag and then
    apply it to the DEPLOYMENT's pair. With a stored default that differs from
    the configured one - which set-as-default makes ordinary - that is wrong
    twice over: the workspace default's own sibling, which every thinking turn
    actually lands on, was left deletable, while the tag it did protect was one
    nothing routed through.

    MUTATION: drop the second `pair_for` call and reuse the binding's pairing
    for both, and the declared partner becomes deletable while turns route to
    it.
    """
    partner = "reasoner:8b"
    await set_workspace_default(db_sessionmaker, OTHER_CHAT)
    # The workspace default is paired with a partner of its own - one the
    # deployment's own configuration knows nothing about.
    await store_pair(db_sessionmaker, model=OTHER_CHAT, thinking_model=partner)
    view = await build(
        db_sessionmaker,
        provider=server(
            CHAT, THINK, OTHER_CHAT, partner, capabilities_by_model={partner: THINKING_REPORT}
        ),
        settings=make_settings(thinking_chat_model=None),
    )
    # The workspace default itself keeps its Delete - that one is repairable by
    # naming a successor, and collapsing it into a flat refusal was the
    # regression the first attempt at this introduced.
    routed = row_for(view, OTHER_CHAT)
    assert (routed.can_delete, routed.workspace_default) == (True, True)
    # Its SIBLING is not repairable that way: a thinking turn lands there and
    # there is no successor to name, so it is refused like any active sibling.
    row = row_for(view, partner)
    assert row.can_delete is False
    assert "active thinking model" in row.delete_blocked_reason
    assert "model manager" in row.delete_blocked_reason


async def test_an_environment_override_outranks_a_declared_pair(db_sessionmaker) -> None:
    """A declaration that LOST must not be named as the place to change things.

    `THINKING_CHAT_MODEL` is the deployment's kill switch and its verbatim
    override, so it wins. A refusal pointing at a console pairing that no turn
    routes through would send an operator to change something and watch nothing
    happen.

    MUTATION: set `declared_partner` from the argument instead of from the
    RESOLVED pair and this names the model manager for a pairing the
    environment overrode.
    """
    await store_pair(db_sessionmaker, model=CHAT, thinking_model=OTHER_CHAT)
    view = await build(db_sessionmaker, provider=server(CHAT, THINK, OTHER_CHAT))
    # The environment's tag is what the turn routes to, so the declared partner
    # is not protected - it is not part of the active pair at all.
    partner = row_for(view, OTHER_CHAT)
    assert (partner.can_delete, partner.delete_blocked_reason) == (True, None)
    # And the card names what a turn would actually use.
    assert row_for(view, CHAT).thinking_model == THINK


async def test_a_row_that_can_be_deleted_carries_no_reason(db_sessionmaker) -> None:
    # The positive control: the field is populated only where it applies, so a
    # client can key a tooltip on its presence.
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT))
    row = row_for(view, OTHER_CHAT)
    assert (row.can_delete, row.delete_blocked_reason) == (True, None)


async def test_a_provider_without_verbs_offers_none(db_sessionmaker) -> None:
    # The process-per-model runtime (vLLM-like): models change by redeployment,
    # so every row must read as unmanageable rather than showing dead buttons.
    provider = server(CHAT, UNKNOWN, management=ManagementCapabilities())
    view = await build(db_sessionmaker, provider=provider)
    assert not any(row.can_pull or row.can_delete for row in every_row(view))
    assert (view.provider.can_pull, view.provider.can_delete) == (False, False)


async def test_provider_info_reports_the_runtime(db_sessionmaker) -> None:
    provider = server(CHAT, runtime_version="0.32.0")
    view = await build(db_sessionmaker, provider=provider)
    assert view.provider.name == "fake"
    assert view.provider.runtime_version == "0.32.0"
    assert view.provider.can_list_installed is True


# --- a tag nothing declares anything about -----------------------------------------


async def test_an_undeclared_tag_is_listed_at_the_reported_ceiling(db_sessionmaker) -> None:
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT))
    row = row_for(view, OTHER_CHAT)
    # The MECHANICAL columns come from the runtime, and so - one rung short -
    # does the policy one: a vouched template earns ASSISTED, the mode where
    # tools are offered alongside the usual grounding.
    assert (row.role, row.tools) == ("chat", "native")
    assert row.agency is not None
    assert (row.agency.ceiling, row.agency.earned, row.agency.effective) == (
        "assisted",
        "assisted",
        "assisted",
    )
    # Blamed on the missing DECLARATION, because that is what stands between it
    # and agentic - a report describes a template, it does not grant autonomy.
    assert row.agency.clamp_code == "uncatalogued"
    assert row.agency.clamp_detail is not None
    assert "declaration" in row.agency.clamp_detail
    # Re-pullable (a pull on an installed tag fetches updates) and removable.
    assert (row.can_pull, row.can_delete) == (True, True)


async def test_a_tag_the_runtime_will_not_vouch_for_keeps_the_safe_floor(
    db_sessionmaker,
) -> None:
    # The other half of the pivot: with no declaration AND no report of a
    # tool-taking template there is nothing to lift the floor.
    view = await build(db_sessionmaker, provider=server(PROSE))
    row = row_for(view, PROSE)
    assert row.agency is not None
    assert (row.agency.ceiling, row.agency.effective) == ("deterministic", "deterministic")


async def test_a_failed_check_clamps_a_row_the_way_it_clamps_the_turn(db_sessionmaker) -> None:
    # The page/turn agreement this is really about: a verdict bound to a tag's
    # installed weights has to be able to take the report-derived ceiling back
    # on the CARD too - otherwise the row advertises assisted for a model no
    # turn will hand a tool.
    await seed_probe(
        db_sessionmaker,
        model=OTHER_CHAT,
        digest=f"sha256:{OTHER_CHAT}",
        depth="fast",
        earned="deterministic",
    )
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT))
    row = row_for(view, OTHER_CHAT)
    assert row.agency is not None
    assert (row.agency.ceiling, row.agency.earned) == ("assisted", "deterministic")
    assert row.agency.effective == "deterministic"
    assert row.agency.clamp_code == "failed_case"


async def _turn_offer(provider, settings, db_sessionmaker, *, model: str):
    """Resolve one turn's tool offer for ``model``, through the real planner."""

    async def fetch_report(tag: str) -> ModelCapabilities | None:
        return await runtime_capabilities(provider, "fake", tag)

    async def fetch_verdicts(tag: str, ceiling):
        return await effective_verdicts(
            provider=provider,
            provider_name="fake",
            model=tag,
            settings=settings,
            require_verification=settings.agency_require_verification,
            sessionmaker=db_sessionmaker,
            effective_ceiling=ceiling,
        )

    async def route(tag: str, think: bool) -> tuple[str, None]:
        return tag, None

    async def resolve_ctx(tag: str) -> int:
        return 8192

    return await plan_turn(
        req=ChatRequest(message="hello", model=model),
        history=[],
        settings=settings,
        chat_pair=resolve_chat_models(settings),
        search_wired=True,
        rag_wired=True,
        fetch_verdicts=fetch_verdicts,
        route=route,
        resolve_ctx=resolve_ctx,
        fetch_report=fetch_report,
    )


async def test_the_card_and_the_turn_agree_about_an_undeclared_tag(db_sessionmaker) -> None:
    """The cross-surface pin: one resolver, so the page cannot promise what a turn refuses.

    Both surfaces run for real here, over ONE provider - the view through its
    own join, the turn through plan_turn with the real earned lookup and the
    real facts cache behind its boundaries. A second copy of the ceiling rule
    anywhere is a card advertising tools on a model the chat path silently
    declines to hand any.
    """
    settings = make_settings()
    provider = server(CHAT, OTHER_CHAT)
    view = await build(db_sessionmaker, provider=provider, settings=settings)
    row = row_for(view, OTHER_CHAT)

    plan = await _turn_offer(provider, settings, db_sessionmaker, model=OTHER_CHAT)
    assert row.agency is not None
    assert row.agency.ceiling == "assisted"
    assert plan.offer.mode == row.agency.effective == "assisted"
    assert plan.offer_tools is True


async def test_the_card_and_the_turn_agree_across_spellings(db_sessionmaker) -> None:
    """The same agreement when the configured tag and the installed one differ.

    The test above uses ONE string everywhere, so it cannot see this: probe
    rows are written under the name the provider LISTS (the card sends
    ``info.name`` and the verify endpoint records it verbatim), while the turn
    arrives holding whatever the deployment configured. The probe query matches
    exactly. Un-canonicalized, a failed check clamps the card and waves the
    turn through - which breaks the rule that a probe can only ever clamp.
    """
    settings = make_settings()
    provider = FakeProvider(
        models=[ModelInfo(name="mistral:latest", size=1_000, digest="sha256:m")],
        capabilities_by_model={"mistral:latest": CHAT_REPORT},
    )
    # A failed FAST check on the installed weights, recorded under the name the
    # provider lists - exactly what the verify endpoint writes for this row.
    await seed_probe(
        db_sessionmaker,
        model="mistral:latest",
        digest="sha256:m",
        depth="fast",
        earned="deterministic",
    )

    view = await build(db_sessionmaker, provider=provider, settings=settings)
    row = row_for(view, "mistral:latest")

    # The CONFIGURED spelling, one alias away from the installed name.
    plan = await _turn_offer(provider, settings, db_sessionmaker, model="mistral")
    assert row.agency is not None
    assert (row.agency.ceiling, row.agency.earned) == ("assisted", "deterministic")
    assert row.agency.effective == "deterministic"
    # ...and the turn asked under the other spelling reaches the same verdict.
    assert plan.offer.mode == "deterministic"
    assert plan.offer_tools is False


async def test_the_active_chat_pair_is_never_offered_for_deletion(db_sessionmaker) -> None:
    # DEFAULT_CHAT_MODEL is honoured verbatim by the chat path, so a deployment
    # on a custom tag would otherwise be invited to delete the very model its
    # next turn requests.
    settings = make_settings(default_chat_model=UNKNOWN, thinking_chat_model=None)
    provider = server(CHAT, UNKNOWN, capabilities_by_model={UNKNOWN: CHAT_REPORT})
    view = await build(db_sessionmaker, provider=provider, settings=settings)
    row = row_for(view, UNKNOWN)
    assert row.installed is True
    assert row.can_delete is False
    assert "active chat model" in row.delete_blocked_reason


# --- the stored tiers, rendered as facts the confirm dialog discloses ---


async def test_a_pinned_model_keeps_its_delete_and_carries_the_count(db_sessionmaker) -> None:
    # A pin no longer withdraws the verb - the delete clears pins itself - but
    # the person deleting should read the consequence first, so the count rides
    # on the row for the confirm dialog to say.
    await pin_conversations(db_sessionmaker, OTHER_CHAT, count=2)
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT))
    row = row_for(view, OTHER_CHAT)
    assert (row.can_delete, row.delete_blocked_reason) == (True, None)
    assert row.pinned_conversations == 2


@pytest.mark.parametrize("stored", [f"library/{OTHER_CHAT}", f"  {OTHER_CHAT}  "])
async def test_a_pin_in_another_spelling_is_still_counted(db_sessionmaker, stored: str) -> None:
    # A pin is stored verbatim from the request; nothing trims or canonicalizes
    # it on write, so every comparison downstream has to.
    await pin_conversations(db_sessionmaker, stored)
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT))
    assert row_for(view, OTHER_CHAT).pinned_conversations == 1


async def test_the_workspace_default_keeps_its_delete_and_is_flagged(db_sessionmaker) -> None:
    # Deletable, but the dialog must offer the successor choice - the flag is
    # what tells it to.
    await set_workspace_default(db_sessionmaker, OTHER_CHAT)
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT))
    row = row_for(view, OTHER_CHAT)
    assert (row.can_delete, row.delete_blocked_reason) == (True, None)
    assert row.workspace_default is True
    # And only that row: the flag is a per-tag fact, not a page-wide banner.
    assert row_for(view, CHAT).workspace_default is False


async def test_a_pin_on_another_model_leaves_this_row_deletable(db_sessionmaker) -> None:
    # The clamp must not become "nothing is deletable once anyone pins
    # anything": only the model actually named loses its verb.
    await pin_conversations(db_sessionmaker, "something-else:1b")
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT))
    assert row_for(view, OTHER_CHAT).can_delete is True


# --- agency reporting -------------------------------------------------------------


async def test_agency_is_reported_for_chat_rows_only(db_sessionmaker) -> None:
    provider = server(CHAT, EMBED, UNKNOWN, silent={UNKNOWN})
    view = await build(db_sessionmaker, provider=provider)
    assert row_for(view, CHAT).agency is not None
    assert row_for(view, EMBED).agency is None
    # No turn routes to a row the runtime would not describe, so there is no
    # agency question to answer about it.
    assert row_for(view, UNKNOWN).agency is None


async def test_grace_with_no_probe_reports_the_missing_certificate(db_sessionmaker) -> None:
    # No measurement happened, so nothing may be dressed up as one. Under the
    # re-anchored grace that means the row sits at the floor with the honest
    # blame - "nothing has certified this", not "something measured and failed".
    await arm(db_sessionmaker)
    view = await build(db_sessionmaker, provider=server(CHAT))
    agency = row_for(view, CHAT).agency
    assert (agency.ceiling, agency.earned, agency.effective) == ("agentic", "assisted", "assisted")
    assert agency.clamp_code == "unverified_weights"
    assert "No tool-check certifies agentic" in agency.clamp_detail
    # And it must not read as a hardened-regime message: this deployment has
    # AGENCY_REQUIRE_VERIFICATION off, so a sentence about verification being
    # required would send a reader to a switch nobody set.
    assert "required" not in agency.clamp_detail


async def test_a_stored_ceiling_is_the_row_the_grid_reports(db_sessionmaker) -> None:
    # An operator clamped a model whose report would otherwise allow assisted.
    # The grid has to report the clamp and BLAME the declaration, which is the
    # thing a reader would change to lift it.
    await store_policy(db_sessionmaker, ceiling="deterministic")
    view = await build(db_sessionmaker, provider=server(CHAT))
    agency = row_for(view, CHAT).agency
    assert agency.ceiling == "deterministic"
    assert agency.clamp_code == "declared_ceiling"
    assert "stored declaration" in agency.clamp_detail
    assert row_for(view, CHAT).ceiling_declared is True


async def test_a_stored_ceiling_declared_on_other_weights_is_retired(db_sessionmaker) -> None:
    # The re-pull rule, on the grid: the declaration described weights this
    # server no longer holds, so the card falls back to the runtime's report.
    await store_policy(db_sessionmaker, digest="sha256:previous", ceiling="deterministic")
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.agency.ceiling == "assisted"
    assert row.ceiling_declared is False


async def test_a_stored_think_mode_is_the_mode_the_row_shows(db_sessionmaker) -> None:
    # The runtime's template report says this tag never reasons; a confirmed
    # declaration says otherwise, and a declaration outranks an inference.
    await store_policy(db_sessionmaker, think_mode="hybrid")
    view = await build(db_sessionmaker, provider=server(CHAT))
    assert row_for(view, CHAT).think == "hybrid"


async def test_a_stored_grant_shows_on_the_row(db_sessionmaker) -> None:
    await store_policy(db_sessionmaker, thinking_with_tools=True)
    view = await build(db_sessionmaker, provider=server(CHAT))
    assert row_for(view, CHAT).thinking_with_tools is True


async def test_a_deep_pass_reports_no_clamp_at_all(db_sessionmaker) -> None:
    # The positive control: once the declaration AND the certificate exist, the
    # row is fully clear and says nothing.
    await arm(db_sessionmaker)
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")
    view = await build(db_sessionmaker, provider=server(CHAT))
    agency = row_for(view, CHAT).agency
    assert (agency.ceiling, agency.earned, agency.effective) == ("agentic", "agentic", "agentic")
    assert (agency.clamp_code, agency.clamp_detail) == (None, None)


async def test_a_failing_probe_is_blamed(db_sessionmaker) -> None:
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    view = await build(db_sessionmaker, provider=server(CHAT))
    agency = row_for(view, CHAT).agency
    assert agency.earned == "deterministic"
    assert agency.clamp_code == "failed_case"
    assert "fast" in agency.clamp_detail


# --- which cases a clamp names ----------------------------------------------------


def probe_view(cases: list[tuple[str, int, int]], *, depth: str = "deep") -> ProbeView:
    """A probe row as the reader sees it, carrying only the tallies under test."""
    return ProbeView(
        depth=depth,
        earned="assisted",
        thinking_with_tools=False,
        tokens_per_second=None,
        vram_bytes=None,
        runtime_version=None,
        harness_version=harness_version(),
        at=datetime(2026, 7, 1, tzinfo=UTC),
        imperfect_cases=tuple(cases),
    )


def test_the_worst_case_is_named_first() -> None:
    # Worst FRACTION, not worst count: a case that lost every sample is a
    # different problem from one that lost a single sample to noise, and the
    # reader is looking for the first. `selection 2/3` is absent because a
    # deep run scores the deciding cases by majority - it PASSED.
    detail = _failed_case_detail(
        probe_view([("selection", 2, 3), ("must-not-call", 0, 3), ("continuation", 1, 3)])
    )
    assert detail == "must-not-call 0/3, continuation 1/3"


def test_equal_fractions_fall_back_to_the_order_the_check_runs() -> None:
    # Both lost two of three; the harness runs freshness first, so it reads
    # first. Without the tie-break this is whatever order the JSON serialized.
    detail = _failed_case_detail(probe_view([("loop-guard", 1, 3), ("must-call-freshness", 1, 3)]))
    assert detail == "must-call-freshness 1/3, loop-guard 1/3"


def test_only_the_cases_that_held_the_verdict_down_are_named() -> None:
    # The regression this whole filter exists for. A deep run that dropped one
    # sample almost everywhere earns `deterministic` because of the
    # hallucination guard ALONE - every deciding case at 2/3 passed by
    # majority. Naming the imperfect ones would print five cases that passed
    # and bury the one that caused the clamp inside "+1 more".
    detail = _failed_case_detail(
        probe_view(
            [
                ("must-call-freshness", 2, 3),
                ("must-call-workspace", 2, 3),
                ("must-not-call", 2, 3),
                ("selection", 2, 3),
                ("hallucination-guard", 2, 3),
            ]
        )
    )
    assert detail == "hallucination-guard 2/3"


def test_a_guard_case_blames_on_one_lost_sample_where_a_deciding_case_does_not() -> None:
    # Same numbers, two thresholds - which is exactly why the rule cannot be
    # inferred from the fraction here.
    assert _failed_case_detail(probe_view([("loop-guard", 2, 3)])) == "loop-guard 2/3"
    assert _failed_case_detail(probe_view([("selection", 2, 3)])) is None


def test_a_fast_run_blames_any_lost_sample() -> None:
    # One sample per case, so there is no majority to take and a loss is a
    # failure. Read at deep's threshold the same row would blame nothing.
    view = probe_view([("must-not-call", 0, 1)], depth="fast")
    assert _failed_case_detail(view) == "must-not-call 0/1"


def test_a_long_list_is_capped_and_counted() -> None:
    detail = _failed_case_detail(
        probe_view(
            [
                ("must-call-freshness", 0, 3),
                ("must-call-workspace", 0, 3),
                ("must-not-call", 1, 3),
                ("selection", 1, 3),
                ("continuation", 1, 3),
                ("hallucination-guard", 2, 3),
            ]
        )
    )
    assert detail == (
        "must-call-freshness 0/3, must-call-workspace 0/3, must-not-call 1/3, "
        "selection 1/3, +2 more"
    )


def test_the_reasoning_case_is_never_named() -> None:
    # Case (h) decides thinking_with_tools and nothing about agency, so it is
    # in neither scored set and can never blame. It is excluded BEFORE the
    # count too - "+1 more" pointing at it would smuggle the same misdirection
    # back in.
    assert _failed_case_detail(probe_view([("thinking-with-tools", 0, 3)])) is None
    detail = _failed_case_detail(probe_view([("thinking-with-tools", 0, 3), ("selection", 1, 3)]))
    assert detail == "selection 1/3"


def test_a_row_with_nothing_imperfect_adds_no_parenthetical() -> None:
    assert _failed_case_detail(probe_view([])) is None
    assert _failed_case_detail(None) is None


async def test_a_clamp_names_the_cases_the_probe_missed(db_sessionmaker) -> None:
    await seed_probe(
        db_sessionmaker,
        depth="fast",
        earned="deterministic",
        tallies={"must-call-freshness": [1, 1], "must-not-call": [0, 1]},
    )
    view = await build(db_sessionmaker, provider=server(CHAT))
    detail = row_for(view, CHAT).agency.clamp_detail
    assert "did not pass (must-not-call 0/1), so" in detail
    # The case it DID pass is not named: a pointer that lists everything the
    # check ran is not a pointer.
    assert "must-call-freshness" not in detail


async def test_a_clamp_with_no_recorded_tallies_reads_as_it_always_did(db_sessionmaker) -> None:
    # Rows predate the tallies key, and a row that recorded none is not
    # evidence that nothing failed - so the sentence loses its parenthetical
    # rather than gaining an empty one.
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    view = await build(db_sessionmaker, provider=server(CHAT))
    assert row_for(view, CHAT).agency.clamp_detail == (
        "A fast tool-check on the installed weights did not pass, so the measured "
        "verdict clamps this model to deterministic."
    )


async def test_all_perfect_tallies_leave_the_sentence_alone(db_sessionmaker) -> None:
    # The edge the resolution allows and the tallies contradict: a verdict
    # below its depth's cap with every case passing. Whatever produced it,
    # there is no case to point at, so nothing is invented.
    await seed_probe(
        db_sessionmaker,
        depth="fast",
        earned="deterministic",
        tallies={"must-call-freshness": [1, 1], "must-not-call": [1, 1]},
    )
    view = await build(db_sessionmaker, provider=server(CHAT))
    assert row_for(view, CHAT).agency.clamp_detail == (
        "A fast tool-check on the installed weights did not pass, so the measured "
        "verdict clamps this model to deterministic."
    )


async def test_the_cases_named_are_the_clamping_rows_not_the_newest_rows(db_sessionmaker) -> None:
    # A model can hold a clean verdict at one depth and the failure at
    # another. The older DEEP is what clamps (assisted is below deep's cap);
    # the newer FAST passed at its own cap and clamps nothing. Naming the
    # newest row's cases - or the DEEP-preferred row the measured facts show,
    # which is right here only by luck - would send a reader to a case that
    # passed.
    await arm(db_sessionmaker)
    await seed_probe(
        db_sessionmaker,
        depth="deep",
        earned="assisted",
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
        tallies={"must-not-call": [1, 3], "selection": [3, 3]},
    )
    await seed_probe(
        db_sessionmaker,
        depth="fast",
        earned="assisted",
        created_at=datetime(2026, 7, 9, tzinfo=UTC),
        tallies={"must-call-freshness": [1, 1], "must-not-call": [1, 1]},
    )
    view = await build(db_sessionmaker, provider=server(CHAT))
    agency = row_for(view, CHAT).agency
    assert (agency.earned, agency.clamp_code) == ("assisted", "failed_case")
    assert agency.clamp_detail == (
        "A deep tool-check on the installed weights did not pass (must-not-call 1/3), "
        "so the measured verdict clamps this model to assisted."
    )


async def test_a_newer_fast_failure_names_its_own_cases_over_a_bound_deep_pass(
    db_sessionmaker,
) -> None:
    # The mirror, and the case the measured-facts line gets "wrong" by design:
    # that line prefers DEEP as the fuller measurement, while the clamp rests
    # on the newer FAST failure. Two different questions about the same
    # weights, answered from two different rows - so reusing the measured
    # row here would name a case that passed.
    await seed_probe(
        db_sessionmaker,
        depth="deep",
        earned="agentic",
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
        tallies={"selection": [2, 3]},
    )
    await seed_probe(
        db_sessionmaker,
        depth="fast",
        earned="deterministic",
        created_at=datetime(2026, 7, 9, tzinfo=UTC),
        tallies={"must-not-call": [0, 1]},
    )
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.measured.depth == "deep"
    assert row.agency.clamp_detail == (
        "A fast tool-check on the installed weights did not pass (must-not-call 0/1), "
        "so the measured verdict clamps this model to deterministic."
    )


async def test_unverified_weights_are_blamed_under_the_hardened_regime(db_sessionmaker) -> None:
    # A verdict bound to weights that are no longer installed proves nothing.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", digest="sha256:old")
    settings = make_settings(agency_require_verification=True)
    view = await build(db_sessionmaker, provider=server(CHAT), settings=settings)
    agency = row_for(view, CHAT).agency
    assert (agency.earned, agency.effective) == ("deterministic", "deterministic")
    assert agency.clamp_code == "unverified_weights"


async def test_a_verdict_from_another_server_is_blamed_on_the_endpoint(db_sessionmaker) -> None:
    """The card names the SERVER, not a missing check, when one is missing.

    The two clamps read identically to the resolver and differently to a
    person: "run a check" is wrong advice when a passing check already exists
    and simply describes another machine. The copy therefore has to say where
    the verdict came from and what would recover it.

    MUTATION: delete the ``endpoint_mismatch`` branch from ``_clamp_blame`` and
    the code falls back to ``unverified_weights``, whose sentence tells an
    operator nothing has measured weights that plainly have been.
    """
    await arm(db_sessionmaker)
    await seed_probe(
        db_sessionmaker,
        depth="deep",
        earned="agentic",
        serving_endpoint="https://pod-abc.proxy.runpod.net",
    )
    view = await build(db_sessionmaker, provider=server(CHAT))
    agency = row_for(view, CHAT).agency
    assert (agency.ceiling, agency.earned) == ("agentic", "assisted")
    assert agency.clamp_code == "endpoint_mismatch"
    assert "different or unrecorded server" in agency.clamp_detail


async def test_a_template_that_takes_no_tool_list_floors_the_row(db_sessionmaker) -> None:
    # The mechanical floor the turn resolver applies (runtime_floors) has to
    # apply here too, or the grid advertises an autonomy no turn could grant:
    # a model whose chat template cannot take a tool list runs deterministic
    # however agentic an operator declared it.
    await arm(db_sessionmaker, model=PROSE)
    view = await build(db_sessionmaker, provider=server(PROSE))
    row = row_for(view, PROSE)
    assert row.tools == "none"
    # The template floor is the blame even though the earned axis is ALSO below
    # the ceiling: an unproven model is something a check can fix, a template
    # that takes no tool list is not, so the mechanical fact is ranked first.
    assert (row.agency.ceiling, row.agency.earned) == ("agentic", "assisted")
    assert row.agency.effective == "deterministic"
    assert row.agency.clamp_code == "tools_none"
    assert "tool list" in row.agency.clamp_detail


async def test_the_env_clamp_is_blamed_when_nothing_else_holds(db_sessionmaker) -> None:
    # "Nothing else" includes the earned axis, so the model has to have
    # actually earned its ceiling before AGENCY_MODE can be the thing holding
    # it back - otherwise the honest blame is the missing certificate.
    await arm(db_sessionmaker)
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")
    settings = make_settings(agency_mode="assisted")
    view = await build(db_sessionmaker, provider=server(CHAT), settings=settings)
    agency = row_for(view, CHAT).agency
    assert (agency.ceiling, agency.earned, agency.env_clamp) == ("agentic", "agentic", "assisted")
    assert agency.effective == "assisted"
    assert agency.clamp_code == "env_clamp"


async def test_a_measured_failure_outranks_the_env_clamp(db_sessionmaker) -> None:
    # Both clamp; the actionable fact is the case that failed.
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    settings = make_settings(agency_mode="assisted")
    view = await build(db_sessionmaker, provider=server(CHAT), settings=settings)
    assert row_for(view, CHAT).agency.clamp_code == "failed_case"


async def test_auto_agency_mode_is_not_a_clamp(db_sessionmaker) -> None:
    settings = make_settings(agency_mode="auto")
    view = await build(db_sessionmaker, provider=server(CHAT), settings=settings)
    assert row_for(view, CHAT).agency.env_clamp is None


async def test_reported_effective_is_always_what_the_resolver_grants(db_sessionmaker) -> None:
    # The invariant that makes the whole report trustworthy: whatever a row
    # displays, re-resolving its OWN reported axes - including the tool-support
    # floor the row also reports - must give the same mode. A grid can never
    # show an autonomy the turn resolver would not grant.
    await arm(db_sessionmaker)
    await seed_probe(db_sessionmaker, depth="deep", earned="assisted")
    settings = make_settings(agency_mode="agentic")
    view = await build(db_sessionmaker, provider=server(CHAT, OTHER_CHAT, PROSE), settings=settings)
    rows = [row for row in every_row(view) if row.agency is not None]
    assert rows, "expected at least one agency report to check"
    for row in rows:
        agency = row.agency
        assert agency.effective == resolve_agency(
            ceiling=agency.ceiling,
            earned=agency.earned,
            env_mode=agency.env_clamp,
            runtime_floors=() if row.tools == "native" else ("deterministic",),
        )
        # ...and the reported clamp is really the configured one.
        assert agency.env_clamp == parse_agency_mode(settings.agency_mode)


# --- measured facts ---------------------------------------------------------------


async def test_unmeasured_stays_null(db_sessionmaker) -> None:
    view = await build(db_sessionmaker, provider=server(CHAT))
    assert row_for(view, CHAT).measured is None


async def test_measured_facts_come_from_the_probe_row(db_sessionmaker) -> None:
    await seed_probe(
        db_sessionmaker,
        depth="deep",
        earned="agentic",
        runtime_version="0.32.0",
        tokens_per_second=11.5,
        vram_bytes=2_500_000_000,
    )
    view = await build(db_sessionmaker, provider=server(CHAT, runtime_version="0.32.0"))
    measured = row_for(view, CHAT).measured
    assert measured is not None
    assert (measured.depth, measured.tokens_per_second) == ("deep", 11.5)
    assert measured.vram_bytes == 2_500_000_000
    assert isinstance(measured.measured_at, datetime)
    assert (measured.runtime_stale, measured.harness_stale) == (False, False)
    # Every API timestamp carries an explicit offset: sqlite drops the zone on
    # readback, and a reader comparing a stamp against other timestamps in the
    # payload would silently read a naive one as local time.
    assert measured.measured_at.tzinfo is not None
    serialized = measured.model_dump(mode="json")["measured_at"]
    assert serialized.endswith(("Z", "+00:00"))
    assert MeasuredFacts.model_validate_json(measured.model_dump_json()).measured_at.tzinfo


async def test_a_spilled_probe_says_so_beside_its_numbers(db_sessionmaker) -> None:
    # A run the model never reached the GPU for records a VRAM figure of
    # literally zero and no throughput at all. Those two alone read as a broken
    # page; the marker beside them is what makes them mean "it ran on the CPU".
    await seed_probe(
        db_sessionmaker,
        depth="deep",
        earned="agentic",
        vram_bytes=0,
        representative=False,
        gpu_fraction=0.0,
    )
    view = await build(db_sessionmaker, provider=server(CHAT))
    measured = row_for(view, CHAT).measured
    assert measured is not None
    assert measured.representative is False
    assert measured.gpu_fraction == 0.0
    assert (measured.vram_bytes, measured.tokens_per_second) == (0, None)


async def test_an_ordinary_probe_reports_itself_representative(db_sessionmaker) -> None:
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", gpu_fraction=0.6)
    view = await build(db_sessionmaker, provider=server(CHAT))
    measured = row_for(view, CHAT).measured
    # A partial split is ordinary on a small card and is not a fault.
    assert (measured.representative, measured.gpu_fraction) == (True, 0.6)


async def test_a_probe_without_a_recorded_runtime_is_not_stale(db_sessionmaker) -> None:
    # Unknown is not changed: a probe that recorded no runtime version says
    # nothing about whether the serving stack moved, and a banner that cried
    # staleness at every such row would train a reader to ignore the real ones.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", runtime_version=None)
    view = await build(db_sessionmaker, provider=server(CHAT, runtime_version="0.32.0"))
    assert row_for(view, CHAT).measured.runtime_stale is False


async def test_a_runtime_upgrade_marks_the_measurement_stale(db_sessionmaker) -> None:
    # Stale is a BANNER, not a clamp: the verdict still stands, it just
    # describes a different serving stack.
    await arm(db_sessionmaker)
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", runtime_version="0.31.0")
    view = await build(db_sessionmaker, provider=server(CHAT, runtime_version="0.32.0"))
    row = row_for(view, CHAT)
    assert row.measured.runtime_stale is True
    assert row.agency.effective == "agentic"


async def test_a_verdict_from_another_server_is_shown_as_stale_not_dropped(
    db_sessionmaker,
) -> None:
    """The card has to explain a verdict the resolution just discounted.

    A deep pass earned against a rented pod no longer certifies this
    deployment, but the row still exists and its numbers are still the record
    of what that pod did. Dropping it would look like a page that lost data,
    and showing it with no flag would look like a bug: a passing deep probe
    beside "no check has proven this". The flag is what makes the pair read as
    one sentence - measured, elsewhere.
    """
    await arm(db_sessionmaker)
    await seed_probe(
        db_sessionmaker,
        depth="deep",
        earned="agentic",
        serving_endpoint="https://pod-abc.proxy.runpod.net",
    )
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.measured is not None
    assert row.measured.endpoint_stale is True
    assert row.measured.depth == "deep"  # still reported
    # ...and unlike a runtime upgrade, this one DOES clamp: it measured another
    # machine, so it certifies nothing about the one being served from.
    assert row.agency.effective == "assisted"


async def test_a_probe_without_a_recorded_server_is_marked_stale_too(db_sessionmaker) -> None:
    """The unnamed host is the COMMON case, and it must not be the silent one.

    Every row written before the endpoint was tracked has no host, and every
    one of them stops certifying - so on the first page load after the upgrade
    a deployment's passing deep verdicts all drop to the grace floor at once.
    Nothing else marks them: the harness fingerprint hashes the case specs and
    the served prompt stack, so a row written minutes before still carries the
    CURRENT fingerprint and reads as fresh. This flag is the only thing
    standing between an operator and an unexplained demotion.

    Deliberately NOT the runtime-version rule ("changed, never merely
    unknown"): that one describes a banner, and this one describes a verdict
    the turn has already refused.
    """
    await arm(db_sessionmaker)
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", serving_endpoint=None)
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.measured is not None
    assert row.measured.endpoint_stale is True
    # The pairing that makes the flag load-bearing: the verdict really is gone.
    assert row.measured.harness_stale is False
    assert row.agency.effective == "assisted"


async def test_edited_cases_mark_the_measurement_stale(db_sessionmaker) -> None:
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", harness="h-old")
    view = await build(db_sessionmaker, provider=server(CHAT))
    assert row_for(view, CHAT).measured.harness_stale is True


async def test_a_deep_probe_is_preferred_over_a_fast_one(db_sessionmaker) -> None:
    # DEEP re-runs every FAST case plus the guards, so where both exist its
    # numbers are the fuller measurement - even when the FAST row is newer.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", tokens_per_second=9.0)
    await seed_probe(db_sessionmaker, depth="fast", earned="assisted", tokens_per_second=1.0)
    view = await build(db_sessionmaker, provider=server(CHAT))
    measured = row_for(view, CHAT).measured
    assert (measured.depth, measured.tokens_per_second) == ("deep", 9.0)


# --- the weights-change notice ------------------------------------------------------

# The weights a re-pull replaced: every digest-bound read drops verdicts
# measured against these, which is exactly the state the notice exists for.
OLD_WEIGHTS = "sha256:previous"
EARLIER = datetime(2026, 7, 1, tzinfo=UTC)
LATER = datetime(2026, 7, 20, tzinfo=UTC)


async def test_a_re_pull_leaves_the_verdicts_behind_and_the_row_says_so(
    db_sessionmaker,
) -> None:
    # THE default regime, and the reason this notice exists: with the grace in
    # force the earned axis simply returns to the ceiling when a re-pull
    # orphans the log, so the page went from "checked, agentic" to a silent
    # "not checked" with nothing saying why.
    await seed_probe(
        db_sessionmaker, digest=OLD_WEIGHTS, depth="fast", earned="assisted", created_at=EARLIER
    )
    await seed_probe(
        db_sessionmaker, digest=OLD_WEIGHTS, depth="deep", earned="agentic", created_at=LATER
    )
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    note = row.weights_change
    assert note is not None
    assert note.depths == ["fast", "deep"]
    # The fullest orphaned measurement describes the note, as DEEP does the
    # measured facts: it is the verdict a reader stands to lose.
    assert note.earned == "agentic"
    assert note.measured_at.tzinfo is not None
    assert note.model_dump(mode="json")["measured_at"].endswith(("Z", "+00:00"))
    # Nothing bound to the installed weights, said in the row's own words.
    assert row.measured is None


async def test_the_notice_is_display_and_never_policy(db_sessionmaker) -> None:
    # An orphaned FAILURE is the sharpest version of the question: it warns,
    # and it must still not clamp. The verdict was measured on weights this
    # server no longer holds, so feeding it back into the resolution would
    # punish a model for what a different set of weights did.
    await arm(db_sessionmaker)
    await seed_probe(db_sessionmaker, digest=OLD_WEIGHTS, depth="deep", earned="deterministic")
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.weights_change is not None
    # Assisted, not deterministic: the row sits exactly where an un-probed
    # model sits. The orphaned failure informed the reader and moved nothing.
    assert (row.agency.earned, row.agency.effective) == ("assisted", "assisted")
    # And the blame is the MISSING certificate, never the orphaned failure: a
    # clamp code naming that row would be the notice becoming policy in the one
    # place a reader would believe it.
    assert row.agency.clamp_code == "unverified_weights"


async def test_only_the_deep_verdict_can_be_the_orphaned_one(db_sessionmaker) -> None:
    # A fast check ran on the new weights (a pull runs one automatically), so
    # the model is measured - just not at the depth that earns agentic.
    await seed_probe(
        db_sessionmaker, digest=OLD_WEIGHTS, depth="deep", earned="agentic", created_at=EARLIER
    )
    await seed_probe(db_sessionmaker, depth="fast", earned="assisted", created_at=LATER)
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.weights_change is not None
    assert row.weights_change.depths == ["deep"]
    assert row.weights_change.earned == "agentic"
    # The bound FAST verdict is still reported as the measured fact, so the
    # two lines say different true things rather than contradicting.
    assert row.measured is not None and row.measured.depth == "fast"


async def test_a_bound_deep_verdict_supersedes_an_orphaned_fast_one(db_sessionmaker) -> None:
    # A DEEP run re-measures every FAST case, so once one is bound to the
    # installed weights an older orphaned FAST row has nothing left to warn
    # about - the same superseding the earned resolution already applies.
    await seed_probe(
        db_sessionmaker, digest=OLD_WEIGHTS, depth="fast", earned="assisted", created_at=EARLIER
    )
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", created_at=LATER)
    view = await build(db_sessionmaker, provider=server(CHAT))
    assert row_for(view, CHAT).weights_change is None


async def test_a_depth_still_covered_by_a_bound_verdict_raises_no_notice(
    db_sessionmaker,
) -> None:
    # The tag came back to weights a check already covered: a DEEP verdict is
    # bound to what is installed, and a NEWER deep row measured the weights in
    # between. Nothing at that depth is missing, so nothing is warned about -
    # and the row cannot report a bound measurement above a warning that
    # contradicts it at the same depth.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", created_at=EARLIER)
    await seed_probe(
        db_sessionmaker, digest=OLD_WEIGHTS, depth="deep", earned="assisted", created_at=LATER
    )
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.weights_change is None
    assert row.measured is not None and row.measured.depth == "deep"


async def test_a_never_checked_model_raises_no_notice(db_sessionmaker) -> None:
    # The quiet "no tool check has run against these weights" state is what a
    # fresh model looks like, not a warning - and warning about it would make
    # the notice meaningless on the rows that need it.
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert (row.weights_change, row.measured) == (None, None)


async def test_a_check_on_the_installed_weights_clears_the_notice(db_sessionmaker) -> None:
    # Self-clearing is the whole reason nothing is stored: the state ends when
    # a check runs against what is installed, and so does the note.
    await seed_probe(
        db_sessionmaker, digest=OLD_WEIGHTS, depth="deep", earned="agentic", created_at=EARLIER
    )
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", created_at=LATER)
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.weights_change is None
    assert row.measured is not None and row.measured.depth == "deep"


async def test_a_probe_that_recorded_no_digest_is_not_a_weights_change(db_sessionmaker) -> None:
    # Unknown weights are not DIFFERENT weights. A runtime that exposed no
    # digest when the probe ran says nothing about whether the weights moved,
    # and a notice here would send a reader hunting a re-pull that never was.
    await seed_probe(db_sessionmaker, digest=None, depth="deep", earned="agentic")
    view = await build(db_sessionmaker, provider=server(CHAT))
    assert row_for(view, CHAT).weights_change is None


async def test_weights_the_runtime_cannot_identify_raise_no_notice(db_sessionmaker) -> None:
    # Nothing to compare against: with no installed digest no verdict could be
    # bound to these weights in the first place, so "the weights changed" is a
    # claim this view has no evidence for.
    await seed_probe(db_sessionmaker, digest=OLD_WEIGHTS, depth="deep", earned="agentic")
    provider = server(CHAT, models=[ModelInfo(name=CHAT, size=1_000, digest=None)])
    view = await build(db_sessionmaker, provider=provider)
    assert row_for(view, CHAT).weights_change is None


async def test_roles_the_harness_never_measures_carry_no_notice(db_sessionmaker) -> None:
    # Only a chat row is explained at all - an embedder has no tool-check
    # story, so it has no verdict to be orphaned from either.
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED))
    assert row_for(view, EMBED).weights_change is None


# --- the active-pair guard, reused by the delete path ---


@pytest.mark.parametrize(
    ("configured", "asked", "expected"),
    [
        # The tag convention makes these four pairs name the same two models,
        # and the guard has to answer the same for all of them. Expanding only
        # one side would leave one spelling able to walk past it.
        (("chat:4b", "chat:4b-think"), "chat:4b", "chat"),
        (("chat:4b", "chat:4b-think"), "chat:4b-think", "thinking"),
        (("chat", None), "chat", "chat"),
        (("chat", None), "chat:latest", "chat"),
        (("chat:latest", None), "chat", "chat"),
        (("chat:latest", None), "chat:latest", "chat"),
        (("chat", "think:latest"), "think", "thinking"),
        # ...and the publisher namespace, the other implicit default.
        (("library/chat", None), "chat", "chat"),
        (("chat", None), "library/chat", "chat"),
        (("library/chat:latest", None), "chat", "chat"),
        (("chat", None), "something-else", None),
        ((None, None), "chat", None),
    ],
)
def test_the_active_pair_is_recognised_under_either_spelling(
    configured: tuple[str | None, str | None], asked: str, expected: str | None
) -> None:
    pair = ChatModelPair(default=configured[0] or "", thinking=configured[1])
    assert active_chat_role(asked, pair) == expected


_BARE_ALIASES = {"chat", "chat:latest", "library/chat", "library/chat:latest"}


@pytest.mark.parametrize(
    ("tag", "aliases"),
    [
        # Both defaults a registry name leaves implicit - the publisher
        # namespace and the version - expand, because a deployment is free to
        # configure any of these four and mean the same model.
        ("chat", _BARE_ALIASES),
        ("chat:latest", _BARE_ALIASES),
        ("library/chat", _BARE_ALIASES),
        ("library/chat:latest", _BARE_ALIASES),
        # An explicit version pins only the version: "chat:4b" and "chat" are
        # different models, but the namespace is still implicit.
        ("chat:4b", {"chat:4b", "library/chat:4b"}),
        # A real publisher namespace is not the implicit one, so it stays put -
        # prefixing it would invent a tag nobody could pull. Its version still
        # expands.
        ("acme/chat:4b", {"acme/chat:4b"}),
        ("acme/chat", {"acme/chat", "acme/chat:latest"}),
        ("", set()),
    ],
)
def test_tag_aliases_expand_every_implicit_default(tag: str, aliases: set[str]) -> None:
    assert tag_aliases(tag) == aliases


# --- the stranded-vectors report on the payload ---


async def test_no_store_leaves_the_vector_report_absent(db_sessionmaker) -> None:
    # Null is "not asked", never "nothing wrong": a caller with no store to
    # question must not be able to publish an all-clear.
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED))
    assert view.vectors is None


async def test_the_view_reports_stranded_vectors(db_sessionmaker) -> None:
    # The embedding role is LOCKED against a swap from this page, but a swap
    # made elsewhere is invisible to that lock - the report is what catches it,
    # and this page is where the embedder is looked at.
    #
    # The workspace names its embedder, which is what makes the alarm possible
    # at all: it is the one rung chosen INDEPENDENTLY of the corpus, so it is
    # the only one that can disagree with what the corpus holds.
    await choose_embedder(db_sessionmaker, EMBED)
    store = make_memory_store()
    await store.ensure_ready("some-old-embedder", 32)
    await store.upsert(
        "kb",
        "some-old-embedder",
        "00000000-0000-0000-0000-0000000000aa",
        [
            ChunkPoint(
                id=str(uuid.uuid4()),
                vector=[0.1] * 32,
                payload=ChunkPayload(
                    tenant_id="00000000-0000-0000-0000-0000000000aa",
                    doc_id="doc-1",
                    text="stranded passage",
                    created_at=1000.0,
                    embed_model="some-old-embedder",
                ),
            )
        ],
    )
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED), store=store)
    await store.aclose()
    assert view.vectors is not None
    assert view.vectors.known is True
    assert view.vectors.active_model == EMBED
    assert view.vectors.stranded is True
    assert view.vectors.stranded_points == 1
    assert view.vectors.stranded_models == ["some-old-embedder"]
    assert "some-old-embedder" in view.vectors.message
    assert EMBED in view.vectors.message


async def test_with_no_choice_stored_the_page_follows_the_corpus(db_sessionmaker) -> None:
    # The same corpus, with NOTHING stored: the chain falls through to what the
    # passages themselves were written by, so the page reports that model as
    # active and nothing is stranded. This is the interim cost of letting a
    # machine's own evidence outrank the inventory - an upgraded deployment
    # keeps reading its corpus instead of pointing at an empty new space.
    store = make_memory_store()
    await store.ensure_ready("some-old-embedder", 32)
    await store.upsert(
        "kb",
        "some-old-embedder",
        "00000000-0000-0000-0000-0000000000aa",
        [
            ChunkPoint(
                id=str(uuid.uuid4()),
                vector=[0.1] * 32,
                payload=ChunkPayload(
                    tenant_id="00000000-0000-0000-0000-0000000000aa",
                    doc_id="doc-1",
                    text="a passage nobody has disowned",
                    created_at=1000.0,
                    embed_model="some-old-embedder",
                ),
            )
        ],
    )
    view = await build(db_sessionmaker, provider=server(CHAT, EMBED), store=store)
    await store.aclose()
    assert view.vectors is not None
    assert view.vectors.active_model == "some-old-embedder"
    assert view.vectors.stranded is False


# --- the think-mode note ----------------------------------------------------------


async def test_a_measured_mode_that_matches_the_resolution_raises_nothing(
    db_sessionmaker,
) -> None:
    # Agreement is the ordinary case and is not a warning. The runtime reports
    # a template with no think plumbing, and the check measured exactly that.
    await seed_probe(db_sessionmaker, think_mode="never")
    view = await build(db_sessionmaker, provider=server(CHAT))
    row = row_for(view, CHAT)
    assert row.think_mode is None
    assert row.think == "never"


async def test_a_measured_always_over_a_resolved_never_flags_and_clamps(
    db_sessionmaker,
) -> None:
    # The 2507 re-pull detector. Everything about this tag says it never
    # reasons; a check watched it reason with the flag CLEARED, which is what a
    # tag re-pointed at a differently-behaving checkpoint looks like. Turns
    # already force the flag on, and the row says so.
    await seed_probe(db_sessionmaker, think_mode="always")
    view = await build(db_sessionmaker, provider=server(CHAT))
    note = row_for(view, CHAT).think_mode
    assert note is not None
    assert (note.code, note.measured, note.declared) == ("measured_mismatch", "always", "never")
    assert note.routing_clamped is True
    assert note.confirmable is True
    assert "re-pulled" in note.message
    # The displayed mode is what a TURN will send, not what the template infers.
    assert row_for(view, CHAT).think == "always"


async def test_the_clamped_note_says_the_model_also_loses_its_tools(db_sessionmaker) -> None:
    # The clamp forces the flag on, and the reasoning-with-tools gate then
    # drops the tools of any turn that will reason on a model without the
    # grant - which nothing declares by default. That is the consequence a
    # reader OBSERVES, so the note has to say it rather than leaving it to be
    # discovered.
    await seed_probe(db_sessionmaker, think_mode="always")
    view = await build(db_sessionmaker, provider=server(CHAT))
    note = row_for(view, CHAT).think_mode
    assert note is not None and note.routing_clamped is True
    assert "offered no tools" in note.message
    # And it does not leave the reader thinking the answers go ungrounded.
    assert "pre-fetch" in note.message


async def test_a_granted_and_proven_model_is_not_told_it_loses_its_tools(
    db_sessionmaker,
) -> None:
    # The other direction: this row's grant is stored AND certified, so the
    # gate never fires for it and the sentence would be false. A note that said
    # it anyway would be the page inventing a limitation.
    #
    # Both halves are needed because the note keys on the FOLD: a
    # granted-but-unproven model reads as capable in the declaration and is
    # offered no tools by the chat path, so the claim alone would write exactly
    # the wrong sentence.
    await store_policy(db_sessionmaker, thinking_with_tools=True)
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", ttw=True, think_mode="always")
    # Nothing pairs the tag, so the run drove it alone and both sides of the
    # reasoning binding are None - which is what lets the certificate count.
    view = await build(
        db_sessionmaker,
        provider=server(CHAT),
        settings=make_settings(thinking_chat_model=None),
    )
    note = row_for(view, CHAT).think_mode
    assert note is not None and note.routing_clamped is True
    assert "offered no tools" not in note.message


async def test_a_measured_never_over_a_declared_always_flags_without_clamping(
    db_sessionmaker,
) -> None:
    # The safe-direction asymmetry, on the card: the disagreement is reported
    # and offered for confirmation, but nothing about routing moves until a
    # human agrees. Widen the clamp and both `routing_clamped` and `think` go
    # wrong here.
    await store_policy(db_sessionmaker, think_mode="always")
    await seed_probe(db_sessionmaker, think_mode="never")
    view = await build(db_sessionmaker, provider=server(CHAT))
    note = row_for(view, CHAT).think_mode
    assert note is not None
    assert (note.code, note.measured, note.declared) == ("measured_mismatch", "never", "always")
    assert note.routing_clamped is False
    assert note.confirmable is True
    assert row_for(view, CHAT).think == "always"  # the declaration still routes


async def test_a_measurement_on_other_weights_raises_nothing(db_sessionmaker) -> None:
    # Digest-bound, like every other measured fact: the weights-change note is
    # the one that speaks for a verdict a re-pull left behind, and a think-mode
    # flag read off those same weights would be a second, contradictory story.
    await seed_probe(db_sessionmaker, digest="sha256:old", think_mode="always")
    view = await build(db_sessionmaker, provider=server(CHAT))
    assert row_for(view, CHAT).think_mode is None


async def test_the_dangerous_triple_is_flagged_when_nothing_measured_it(
    db_sessionmaker,
) -> None:
    # Nothing declares that it reasons (so a turn omits the flag entirely), the
    # runtime advertises think plumbing, and the model's own template carries
    # it - the combination a serving stack can read as "reason by default", with
    # nothing parsing the channel. Nothing has measured these weights, so it is
    # a hazard rather than an observation.
    #
    # `thinking: true` with a template that carries plumbing resolves to
    # `hybrid`, so the DECLARED mode has to be pushed back to "never" for the
    # branch to be reachable at all - which is exactly the shipped shape: a
    # confirmed never over a template nobody trusts.
    await store_policy(db_sessionmaker, think_mode="never")
    provider = server(
        CHAT,
        capabilities_by_model={
            CHAT: ModelCapabilities(completion=True, thinking=True, template_thinking=True)
        },
    )
    view = await build(db_sessionmaker, provider=provider)
    note = row_for(view, CHAT).think_mode
    assert note is not None
    assert note.code == "nil_promotion"
    assert note.confirmable is False  # there is nothing measured to confirm
    assert note.routing_clamped is False


async def test_an_advertised_capability_with_no_template_plumbing_is_not_a_hazard(
    db_sessionmaker,
) -> None:
    # Measured on this project's own instruct tag: it advertises thinking and
    # its template can do nothing with it, so a promoted flag would drive
    # nothing. Flagging that would train a reader to ignore the real one.
    provider = server(
        CHAT,
        capabilities_by_model={
            CHAT: ModelCapabilities(completion=True, thinking=True, template_thinking=False)
        },
    )
    view = await build(db_sessionmaker, provider=provider)
    assert row_for(view, CHAT).think_mode is None


async def test_a_row_floored_at_deterministic_still_reports_its_think_mode(
    db_sessionmaker,
) -> None:
    # The gap that reading this off the earned explanation opens. A tag whose
    # template takes no tool list is floored at deterministic, and the earned
    # lookup takes an early-out there and never reads the log - which is right
    # for a verdict a probe could not change, and wrong for this axis: the
    # ROUTING clamp applies to such a model just the same. Resolve it off the
    # explanation and the card shows the mode the turn is already overriding,
    # with no note anywhere.
    await seed_probe(db_sessionmaker, model=PROSE, digest=f"sha256:{PROSE}", think_mode="always")
    view = await build(db_sessionmaker, provider=server(PROSE))
    row = row_for(view, PROSE)
    assert row.agency is not None and row.agency.ceiling == "deterministic"
    assert row.think == "always"  # what a turn will actually send
    note = row.think_mode
    assert note is not None
    assert (note.code, note.measured, note.declared) == ("measured_mismatch", "always", "never")
    assert note.routing_clamped is True


async def test_a_row_no_turn_can_route_to_raises_no_think_note(db_sessionmaker) -> None:
    # An embedder distilled from a chat family ships that family's chat
    # template, so a runtime can report BOTH think plumbing flags for it - the
    # exact shape of the hazard note. No chat turn ever routes to an embedder,
    # so no think flag ever reaches it and there is nothing to warn about.
    # Gated on the same answer the verify endpoint gives, so a card cannot
    # raise a note about a row it also says nothing can check.
    provider = server(
        EMBED,
        capabilities_by_model={
            EMBED: ModelCapabilities(embedding=True, thinking=True, template_thinking=True)
        },
    )
    view = await build(db_sessionmaker, provider=provider)
    row = row_for(view, EMBED)
    assert row.checkable is False
    assert row.think_mode is None


async def test_a_measurement_retires_the_hazard_note(db_sessionmaker) -> None:
    # A check that measured the installed weights settles the question either
    # way: it agrees (no note at all) or it disagrees (the mismatch note, which
    # is the better sentence). Either way the suspicion has been answered.
    await store_policy(db_sessionmaker, think_mode="never")
    provider = server(
        CHAT,
        capabilities_by_model={
            CHAT: ModelCapabilities(completion=True, thinking=True, template_thinking=True)
        },
    )
    await seed_probe(db_sessionmaker, think_mode="never")
    view = await build(db_sessionmaker, provider=provider)
    assert row_for(view, CHAT).think_mode is None
