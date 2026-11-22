"""The earned-verdict lookup: grace vs AGENCY_REQUIRE_VERIFICATION, digest-bound.

Two things this module resolves, and they are asymmetric on purpose. The
EARNED agency axis has a grace: a ceiling the caller resolved carries a model
as far as ``assisted`` with nothing measured, because that is the mode where
tools are merely offered alongside the pre-fetch that already ran. The
REASONING-WITH-TOOLS axis has none, in either regime - it starts from a stored
operator grant and nothing else, and a grant reaches a turn only on a deep
certificate bound to the installed weights, because its failure mode is silent
EMPTY OUTPUT rather than an error.

The ceiling itself is resolved by the CALLER now (see
``app.services.model_facts.effective_agency_ceiling``) and handed in as a
required argument, so a card and a turn cannot explain one model against two
different ceilings.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import Settings
from app.models.entities import ModelProbe
from app.providers.base import ModelInfo
from app.providers.fake_provider import FAKE_SERVING_ENDPOINT, FakeProvider
from app.services.earned import (
    _imperfect_cases,
    _ProbeFacts,
    _ProbePair,
    _resolve_earned,
    _resolve_thinking_with_tools,
    _supersede_stale_fast,
    clear_earned_cache,
    declared_thinking_with_tools,
    effective_verdicts,
    explain_earned,
    measured_think_mode,
)
from app.services.model_policy import PolicyView
from app.services.tool_check import harness_version

CHAT = "chat-model"
THINK = "think-model"
CHAT_DIGEST = "sha256:chat"
THINK_DIGEST = "sha256:think"


HARNESS = "h-current"

# The host the deployment under test serves from, and one it does not. A
# certificate only ever counts on the first; a failure counts on either.
ENDPOINT = FAKE_SERVING_ENDPOINT
OTHER_ENDPOINT = "https://pod-abc.proxy.runpod.net"

# The one stored declaration this axis has: an operator granting the claim.
GRANTED = PolicyView(thinking_with_tools=True)


def facts(
    earned: str = "assisted",
    *,
    ttw: bool = False,
    thinking_digest: str | None = None,
    harness: str = HARNESS,
    at: datetime = datetime(2026, 7, 1, tzinfo=UTC),
    serving_endpoint: str | None = ENDPOINT,
):
    return _ProbeFacts(
        earned=earned,
        thinking_with_tools=ttw,
        thinking_digest=thinking_digest,
        harness_version=harness,
        at=at,
        serving_endpoint=serving_endpoint,
    )


def pair(fast: _ProbeFacts | None = None, deep: _ProbeFacts | None = None) -> _ProbePair:
    return _ProbePair(fast=fast, deep=deep)


def make_settings(*, thinking_chat_model: str | None = None) -> Settings:
    """Settings carrying the one rung these tests vary: the env pair override.

    The sibling a verdict binds to is resolved rather than passed in, so the
    lookup needs the environment rung of that resolution - and the stored
    pairing is supplied through ``policy``.
    """
    return Settings(model_provider="fake", thinking_chat_model=thinking_chat_model, _env_file=None)


def pair_provider(**kwargs) -> FakeProvider:
    return FakeProvider(
        models=[
            ModelInfo(name=CHAT, digest=CHAT_DIGEST),
            ModelInfo(name=THINK, digest=THINK_DIGEST),
        ],
        **kwargs,
    )


async def seed_probe(
    db_sessionmaker,
    *,
    depth: str = "fast",
    earned: str = "assisted",
    digest: str = CHAT_DIGEST,
    ttw: bool = False,
    thinking_digest: str | None = None,
    harness: str | None = None,
    tokens_per_second: float | None = None,
    vram_bytes: int | None = None,
    runtime_version: str | None = None,
    serving_endpoint: str | None = ENDPOINT,
    representative: bool = True,
    model: str = CHAT,
    tallies: object = None,
    think_mode: object = None,
    at: datetime | None = None,
) -> None:
    cases: dict = {
        "thinking_digest": thinking_digest,
        "representative": representative,
    }
    # Absent unless a test is about it, so every other row keeps the shape of a
    # probe written before the thinking group existed.
    if think_mode is not None:
        cases["measured_think_mode"] = think_mode
    # Absent unless a test is about them, so every other row here keeps the
    # shape of a probe written before tallies were recorded.
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
        tokens_per_second=tokens_per_second,
        vram_bytes=vram_bytes,
        runtime_version=runtime_version,
        serving_endpoint=serving_endpoint,
        cases=cases,
    )
    if at is not None:
        row.created_at = at
    async with db_sessionmaker() as session:
        session.add(row)
        await session.commit()


# --- what DECLARES the reasoning-with-tools claim ---------------------------------


def test_only_a_stored_grant_declares_the_reasoning_claim() -> None:
    # The whole source list, and it is one item long. Nothing infers this
    # capability from a report or a family, because guessing it optimistically
    # returns blank answers with nothing anywhere reporting a fault - so it has
    # to be asked for by name and then proven.
    assert declared_thinking_with_tools(GRANTED) is True
    assert declared_thinking_with_tools(PolicyView(thinking_with_tools=False)) is False
    assert declared_thinking_with_tools(PolicyView()) is False
    assert declared_thinking_with_tools(None) is False


# --- the pure earned resolution -------------------------------------------------


@pytest.mark.parametrize(
    ("probes", "expected"),
    [
        # Never probed on these weights: the resolved ceiling carries the model
        # to the grace FLOOR and no further. An agentic ceiling with no
        # measurement behind it runs assisted.
        (pair(), "assisted"),
        # A FAST pass at its depth CAP certifies assisted - which is exactly
        # the floor, so it neither lifts nor clamps.
        (pair(fast=facts("assisted")), "assisted"),
        # A FAST guard failure IS failure evidence and clamps below the floor.
        (pair(fast=facts("deterministic")), "deterministic"),
        # A DEEP verdict below ITS cap is failure evidence too.
        (pair(deep=facts("assisted")), "assisted"),
        (pair(deep=facts("deterministic")), "deterministic"),
        # A DEEP pass at cap is the ONLY thing that reaches past the floor.
        (pair(deep=facts("agentic")), "agentic"),
        # ...and only when it is still a valid certificate: an edited case set
        # voids what the old cases earned, so the floor stands alone.
        (pair(deep=facts("agentic", harness="h-old")), "assisted"),
        # A DEEP pass that measured a spilled model cannot certify either.
        (
            pair(deep=dataclasses.replace(facts("agentic"), representative=False)),
            "assisted",
        ),
        # Per-depth reads: a FAST pass beside a DEEP failure keeps the failure.
        (pair(fast=facts("assisted"), deep=facts("assisted")), "assisted"),
        # A newer FAST guard failure still clamps a DEEP certificate.
        (
            pair(
                fast=facts("deterministic", at=datetime(2026, 7, 2, tzinfo=UTC)),
                deep=facts("agentic"),
            ),
            "deterministic",
        ),
    ],
)
def test_grace_floors_at_assisted_and_clamps_on_failure_evidence(
    probes: _ProbePair, expected: str
) -> None:
    resolved, _reason = _resolve_earned(
        ceiling="agentic",
        require_verification=False,
        digest=CHAT_DIGEST,
        probes=probes,
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert resolved == expected


@pytest.mark.parametrize("ceiling", ["deterministic", "assisted"])
def test_the_grace_floor_never_exceeds_the_resolved_ceiling(ceiling: str) -> None:
    # The floor is min(ceiling, assisted), so a deployment whose ceiling is LESS
    # than assisted keeps exactly that - the grace lifts nothing above the
    # ceiling the caller resolved.
    resolved, reason = _resolve_earned(
        ceiling=ceiling,  # type: ignore[arg-type]
        require_verification=False,
        digest=CHAT_DIGEST,
        probes=pair(),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert (resolved, reason) == (ceiling, None)


@pytest.mark.parametrize(
    ("probes", "expected"),
    [
        # No grace: never probed on these weights is deterministic.
        (pair(), "deterministic"),
        # A FAST pass earns exactly its cap, nothing above it.
        (pair(fast=facts("assisted")), "assisted"),
        (pair(deep=facts("agentic")), "agentic"),
        (pair(deep=facts("assisted")), "assisted"),
        # A newer guard failure clamps an older DEEP certificate.
        (pair(fast=facts("deterministic"), deep=facts("agentic")), "deterministic"),
    ],
)
def test_strict_earns_only_what_the_log_proves(probes: _ProbePair, expected: str) -> None:
    resolved, _reason = _resolve_earned(
        ceiling="agentic",
        require_verification=True,
        digest=CHAT_DIGEST,
        probes=probes,
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert resolved == expected


@pytest.mark.parametrize(
    ("probes", "expected"),
    [
        # A row claiming a verdict ABOVE its depth's cap certifies only the
        # cap: the log is machine-writable, and the hardened regime must not
        # accept a hand-inserted fast/agentic row as an agentic proof.
        (pair(fast=facts("agentic")), "assisted"),
        # A certificate from an edited (obsolete) case set is void: what old
        # cases earned does not carry over...
        (pair(deep=facts("agentic", harness="h-old")), "deterministic"),
        # ...but failure evidence they produced still clamps - invalidation
        # only ever points the safe way.
        (pair(deep=facts("deterministic", harness="h-old")), "deterministic"),
        (
            pair(
                fast=facts("deterministic", harness="h-old"),
                deep=facts("agentic"),
            ),
            "deterministic",
        ),
    ],
)
def test_strict_caps_and_versions_certificates(probes: _ProbePair, expected: str) -> None:
    resolved, _reason = _resolve_earned(
        ceiling="agentic",
        require_verification=True,
        digest=CHAT_DIGEST,
        probes=probes,
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert resolved == expected


def test_a_newer_deep_run_supersedes_a_stale_fast_verdict() -> None:
    # DEEP re-measures every FAST case plus the guards, so a DEEP row at least
    # as new as the FAST one retires it - a transient guard failure must be
    # recoverable by the very re-check that proves the model healthy...
    old = datetime(2026, 7, 1, tzinfo=UTC)
    superseded = _supersede_stale_fast(
        pair(
            fast=facts("deterministic", at=old), deep=facts("agentic", at=old + timedelta(hours=1))
        )
    )
    assert superseded.fast is None and superseded.deep is not None
    # ...while a FAST row NEWER than the DEEP one still stands (it measures a
    # subset, but its own failure is the freshest evidence there is).
    kept = _supersede_stale_fast(
        pair(
            fast=facts("deterministic", at=old + timedelta(hours=2)), deep=facts("agentic", at=old)
        )
    )
    assert kept.fast is not None


def test_grace_keeps_old_harness_failure_clamps() -> None:
    # The default regime honours failure evidence even from an older case set.
    resolved, _reason = _resolve_earned(
        ceiling="agentic",
        require_verification=False,
        digest=CHAT_DIGEST,
        probes=pair(fast=facts("deterministic", harness="h-old")),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert resolved == "deterministic"


# --- the serving-endpoint axis ---------------------------------------------------


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "measured_on",
    [
        # Earned against a rented pod this deployment no longer talks to.
        OTHER_ENDPOINT,
        # A row written before the column existed. It says "measured somewhere
        # unrecorded", which is not evidence that it was measured HERE.
        None,
    ],
)
def test_a_certificate_from_another_server_does_not_lift(strict: bool, measured_on) -> None:
    """The whole point of the axis: a verdict must not migrate between hosts.

    Throughput is the host's hardware and tool behaviour belongs to the runtime
    that host runs, so a deep pass earned elsewhere proves nothing here. The
    grace holds at its floor; a hardened deployment has no certificate at all.

    MUTATION: drop ``endpoint_certifies`` from the ``certs`` filter in
    ``_resolve_earned`` and both parametrizations resolve to ``agentic``.
    """
    resolved, _reason = _resolve_earned(
        ceiling="agentic",
        require_verification=strict,
        digest=CHAT_DIGEST,
        probes=pair(deep=facts("agentic", serving_endpoint=measured_on)),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert resolved == ("deterministic" if strict else "assisted")


@pytest.mark.parametrize("measured_on", [OTHER_ENDPOINT, None])
def test_a_discounted_certificate_is_blamed_on_the_endpoint(measured_on) -> None:
    """A pass measured elsewhere reads as "somewhere else", not as "never".

    The two sentences send a reader to different buttons: "nothing has measured
    this" is answered by running a check, and "this passed on another server"
    is answered by running it HERE. Before the code split, a deployment that had
    just moved to a rented pod read as though its whole inventory were unprobed.

    MUTATION: delete the ``reaching`` branch from ``_resolve_earned`` and the
    code falls back to ``unverified_weights``.
    """
    _resolved, reason = _resolve_earned(
        ceiling="agentic",
        require_verification=False,
        digest=CHAT_DIGEST,
        probes=pair(deep=facts("agentic", serving_endpoint=measured_on)),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert reason is not None
    assert reason.code == "endpoint_mismatch"
    # The depth is what names the check to re-run, so it has to be the
    # discounted row's own and not whichever depth happened to be read last.
    assert reason.depth == "deep"


def test_a_certificate_that_reaches_no_higher_is_not_an_endpoint_complaint() -> None:
    """A discounted row only explains what it would actually have recovered.

    A FAST row caps at ``assisted``, which is exactly where the grace already
    sits - so discounting it costs the deployment nothing, and blaming the
    endpoint would point a reader at a re-check that would change no number.

    MUTATION: drop the ``_RANK[entry[0]] > _RANK[earned]`` filter and this
    reports ``endpoint_mismatch`` for a verdict that was never being missed.
    """
    _resolved, reason = _resolve_earned(
        ceiling="agentic",
        require_verification=False,
        digest=CHAT_DIGEST,
        probes=pair(fast=facts("assisted", serving_endpoint=OTHER_ENDPOINT)),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert reason is not None and reason.code == "unverified_weights"


def test_a_failure_outranks_a_discounted_certificate() -> None:
    """Failure evidence stays the blame: it is the actionable half.

    A deployment can hold both at once - a guard that failed here, and a deep
    pass from the pod it used to run on. The failure is what a reader can act
    on, and it is what the fold acted on, so it must not be displaced by an
    explanation about hosts.

    MUTATION: move the ``reaching`` branch above the ``evidence`` branch and
    the reported code becomes ``endpoint_mismatch``.
    """
    _resolved, reason = _resolve_earned(
        ceiling="agentic",
        require_verification=False,
        digest=CHAT_DIGEST,
        probes=pair(
            fast=facts("deterministic", at=datetime(2026, 7, 2, tzinfo=UTC)),
            deep=facts("agentic", serving_endpoint=OTHER_ENDPOINT),
        ),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert reason is not None and reason.code == "failed_case"


@pytest.mark.parametrize("strict", [False, True])
def test_a_failure_from_another_server_still_clamps(strict: bool) -> None:
    """The asymmetry, and it is the half that keeps the change safe.

    Certification is bound to a host; INVALIDATION is not. A model that failed
    a guard somewhere failed a guard, and moving the deployment to a different
    server is not a way to rehabilitate it - it is the same weights making the
    same mistake, and only a fresh check on this host may say otherwise.

    MUTATION: add the endpoint condition to the ``clamps`` comprehension in
    ``_resolve_earned`` and this resolves to ``assisted`` (grace) instead of
    ``deterministic``.
    """
    resolved, reason = _resolve_earned(
        ceiling="agentic",
        require_verification=strict,
        digest=CHAT_DIGEST,
        probes=pair(fast=facts("deterministic", serving_endpoint=OTHER_ENDPOINT)),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert resolved == "deterministic"
    assert reason is not None and reason.code == "failed_case"


def test_a_deployment_that_names_no_host_holds_no_certificate() -> None:
    # The other side of the same strictness. A backend that cannot say where it
    # serves from has nothing to match a row against, so even a row carrying an
    # endpoint cannot be shown to describe THIS deployment.
    resolved, _reason = _resolve_earned(
        ceiling="agentic",
        require_verification=True,
        digest=CHAT_DIGEST,
        probes=pair(deep=facts("agentic")),
        current_harness=HARNESS,
        serving_endpoint=None,
    )
    assert resolved == "deterministic"


@pytest.mark.parametrize("measured_on", [OTHER_ENDPOINT, None])
def test_ttw_positive_requires_the_serving_host(measured_on) -> None:
    """Reasoning-with-tools is per (model, runtime, HOST) like everything else.

    MUTATION: drop the ``endpoint_certifies`` term from the strict branch of
    ``_resolve_thinking_with_tools`` and this returns True.
    """
    assert not _resolve_thinking_with_tools(
        declared=True,
        require_verification=True,
        digest=CHAT_DIGEST,
        deep=facts(ttw=True, thinking_digest=THINK_DIGEST, serving_endpoint=measured_on),
        installed_thinking_digest=THINK_DIGEST,
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )


def test_ttw_negative_from_another_server_still_revokes() -> None:
    # The matching asymmetry on this axis: a measured "it went blank" revokes
    # the declaration whatever host produced it, in either regime. The failure
    # mode here is silent EMPTY OUTPUT, so the safe direction is the only one.
    for strict in (False, True):
        assert not _resolve_thinking_with_tools(
            declared=True,
            require_verification=strict,
            digest=CHAT_DIGEST,
            deep=facts(ttw=False, thinking_digest=THINK_DIGEST, serving_endpoint=OTHER_ENDPOINT),
            installed_thinking_digest=THINK_DIGEST,
            current_harness=HARNESS,
            serving_endpoint=ENDPOINT,
        )


def test_ttw_positive_requires_the_running_harness() -> None:
    # A deep pass measured by obsolete cases cannot prove the capability.
    resolved = _resolve_thinking_with_tools(
        declared=True,
        require_verification=True,
        digest=CHAT_DIGEST,
        deep=facts(ttw=True, thinking_digest=THINK_DIGEST, harness="h-old"),
        installed_thinking_digest=THINK_DIGEST,
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert resolved is False


@pytest.mark.parametrize(("strict", "expected"), [(False, "assisted"), (True, "deterministic")])
def test_undigested_weights_cannot_verify(strict: bool, expected: str) -> None:
    # A runtime that exposes no digest gives a verdict nothing to bind to. The
    # grace keeps its floor and nothing above it - a certificate is
    # digest-bound, so these weights can never earn one however long they run -
    # and a hardened deployment fails closed.
    resolved, _reason = _resolve_earned(
        ceiling="agentic",
        require_verification=strict,
        digest=None,
        probes=pair(),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert resolved == expected


# --- the pure thinking-with-tools resolution -------------------------------------


def test_ttw_needs_the_declaration_first() -> None:
    # A DEEP pass alone never enables it: only an explicit stored grant counts,
    # and a probe can only revoke, never grant.
    assert not _resolve_thinking_with_tools(
        declared=False,
        require_verification=False,
        digest=CHAT_DIGEST,
        deep=facts("agentic", ttw=True, thinking_digest=THINK_DIGEST),
        installed_thinking_digest=THINK_DIGEST,
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )


@pytest.mark.parametrize(
    ("strict", "deep", "installed", "expected"),
    [
        # Unmeasured: the forgiving reading keeps the declaration, the strict
        # one treats it as unproven. Only the strict reading is reachable from
        # the public entry points, where a grant is evidence-gated outright -
        # the forgiving branch is what those callers ask ABOUT, through
        # `_measured_reasoning_case`, to tell "measured absent" from "silence".
        (False, None, THINK_DIGEST, True),
        (True, None, THINK_DIGEST, False),
        # A still-bound negative revokes the declaration in BOTH regimes.
        (False, facts(ttw=False, thinking_digest=THINK_DIGEST), THINK_DIGEST, False),
        (True, facts(ttw=False, thinking_digest=THINK_DIGEST), THINK_DIGEST, False),
        # A verdict measured on re-pulled thinking weights is VOID: it neither
        # revokes (grace) nor proves (strict).
        (False, facts(ttw=False, thinking_digest="sha256:old"), THINK_DIGEST, True),
        (True, facts(ttw=True, thinking_digest="sha256:old"), THINK_DIGEST, False),
        # The proven case: bound positive under strict.
        (True, facts(ttw=True, thinking_digest=THINK_DIGEST), THINK_DIGEST, True),
    ],
)
def test_ttw_verdicts_bind_to_the_thinking_weights(
    strict: bool, deep: _ProbeFacts | None, installed: str, expected: bool
) -> None:
    resolved = _resolve_thinking_with_tools(
        declared=True,
        require_verification=strict,
        digest=CHAT_DIGEST,
        deep=deep,
        installed_thinking_digest=installed,
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert resolved is expected


# --- the async lookup ------------------------------------------------------------


class CountingProvider(FakeProvider):
    """A fake that counts list_models calls, to pin the cache behaviour."""

    def __init__(self, *args, **kwargs) -> None:
        """Wrap FakeProvider with a list_models call counter."""
        super().__init__(*args, **kwargs)
        self.list_calls = 0

    async def list_models(self):
        """Count, then defer to the seeded models."""
        self.list_calls += 1
        return await super().list_models()


class BrokenProvider(FakeProvider):
    """A fake whose model listing is down."""

    async def list_models(self):
        """Raise, as a dead runtime would."""
        raise RuntimeError("runtime down")


async def lookup(
    provider,
    db_sessionmaker,
    *,
    strict: bool = False,
    model: str = CHAT,
    ceiling: str = "agentic",
    policy: PolicyView | None = None,
    settings: Settings | None = None,
):
    """Resolve the chat path's verdicts, with the shared arguments named.

    Args:
        provider: The model seam, for the installed-digest read.
        db_sessionmaker: Session factory for the probe-log read.
        strict: The ``AGENCY_REQUIRE_VERIFICATION`` regime.
        model: The tag to resolve for.
        ceiling: The ceiling the CALLER resolved - a required input now, since
            nothing inside this module derives one.
        policy: The stored declarations for the tag, or None.
        settings: Settings carrying the environment pair override.

    Returns:
        The :class:`EffectiveVerdicts` for the tag.
    """
    return await effective_verdicts(
        provider=provider,
        provider_name="fake",
        model=model,
        settings=make_settings() if settings is None else settings,
        require_verification=strict,
        sessionmaker=db_sessionmaker,
        effective_ceiling=ceiling,  # type: ignore[arg-type]
        policy=policy,
    )


async def test_a_floored_ceiling_stays_io_free(db_sessionmaker) -> None:
    # The early-out, keyed on the CEILING: a tag whose ceiling is already the
    # floor, with no reasoning-with-tools claim to revoke, has nothing a probe
    # could change - so the provider is never listed and the log never opened.
    #
    # It used to be keyed on whether a committed file DESCRIBED the tag, which
    # is how a tool-capable model on an undescribed server could never consult
    # its own verdict.
    provider = CountingProvider()
    verdicts = await lookup(provider, db_sessionmaker, ceiling="deterministic")
    assert verdicts.earned == "deterministic"
    assert verdicts.thinking_with_tools is False
    assert provider.list_calls == 0


async def test_a_vouched_ceiling_does_consult_the_log(db_sessionmaker) -> None:
    # The control for the early-out above: anything ABOVE the floor reads its
    # verdicts, whether or not anybody has ever written the tag down.
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    provider = CountingProvider(models=await pair_provider().list_models())
    assert (await lookup(provider, db_sessionmaker, ceiling="assisted")).earned == "deterministic"
    assert provider.list_calls == 1


async def test_digestless_provider_keeps_the_grace_floor_and_fails_strict(db_sessionmaker) -> None:
    # The default FakeProvider reports no digests at all.
    assert (await lookup(FakeProvider(), db_sessionmaker)).earned == "assisted"
    assert (await lookup(FakeProvider(), db_sessionmaker, strict=True)).earned == "deterministic"


async def test_a_fast_pass_lands_exactly_on_the_grace_floor(db_sessionmaker) -> None:
    # The common real data shape: a FAST row earning assisted exists for the
    # installed digest. FAST caps at assisted, which IS the grace floor, so it
    # neither lifts the model nor clamps it - both regimes read assisted.
    await seed_probe(db_sessionmaker, depth="fast", earned="assisted")
    assert (await lookup(pair_provider(), db_sessionmaker)).earned == "assisted"
    assert (await lookup(pair_provider(), db_sessionmaker, strict=True)).earned == "assisted"


async def test_a_deep_pass_is_what_lifts_the_default_to_agentic(db_sessionmaker) -> None:
    # The direction the re-anchor creates: agentic is EARNED, in both regimes,
    # by a deep certificate bound to the installed weights.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")
    assert (await lookup(pair_provider(), db_sessionmaker)).earned == "agentic"
    assert (await lookup(pair_provider(), db_sessionmaker, strict=True)).earned == "agentic"


async def test_a_stale_harness_deep_pass_does_not_lift_the_default(db_sessionmaker) -> None:
    # ...and only a CURRENT certificate does it: edited cases void what the old
    # ones earned, so the model falls back to the floor rather than the ceiling.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", harness="h-obsolete")
    assert (await lookup(pair_provider(), db_sessionmaker)).earned == "assisted"


async def test_each_tag_reads_its_own_probe_key(db_sessionmaker) -> None:
    # A verdict is a fact about specific WEIGHTS, and a pair is two sets of
    # them. Nothing groups the two members under one governing row any more, so
    # a failure measured on the chat tag must not clamp its sibling - and vice
    # versa. Key the read on anything other than the tag itself and one of
    # these two answers changes.
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    on_chat = await lookup(pair_provider(), db_sessionmaker, model=CHAT)
    on_sibling = await lookup(pair_provider(), db_sessionmaker, model=THINK)
    assert on_chat.earned == "deterministic"
    assert on_sibling.earned == "assisted"


async def test_a_run_the_model_spilled_off_the_gpu_certifies_nothing(db_sessionmaker) -> None:
    # The verify job marks a run the model never reached the GPU for. It still
    # CLAMPS if it failed something - a spill does not make a model's tool
    # calls better - but it cannot be the proof that lifts one, because it
    # measured a machine the deployment does not serve from.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", representative=False)
    assert (await lookup(pair_provider(), db_sessionmaker, strict=True)).earned == "deterministic"


async def test_a_spilled_run_that_failed_still_clamps(db_sessionmaker) -> None:
    # The other half of the asymmetry, on the real read path: the marker
    # withholds a certificate, it does not excuse a failure.
    await seed_probe(db_sessionmaker, depth="deep", earned="assisted", representative=False)
    assert (await lookup(pair_provider(), db_sessionmaker)).earned == "assisted"


def test_an_unrepresentative_row_still_clamps_but_never_certifies() -> None:
    spilled = dataclasses.replace(facts("agentic"), representative=False)
    strict, _ = _resolve_earned(
        ceiling="agentic",
        require_verification=True,
        digest=CHAT_DIGEST,
        probes=pair(deep=spilled),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert strict == "deterministic"  # a pass at cap, but not a certificate
    clamping = dataclasses.replace(facts("deterministic"), representative=False)
    graced, _ = _resolve_earned(
        ceiling="agentic",
        require_verification=False,
        digest=CHAT_DIGEST,
        probes=pair(deep=clamping),
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )
    assert graced == "deterministic"  # failure evidence still counts


async def test_repull_reverts_to_unverified(db_sessionmaker) -> None:
    # The verdict is bound to OLD weights; the installed digest differs. The
    # hardened regime falls to the floor of all floors, and the grace falls to
    # its own - the agentic certificate is invisible to both.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", digest="sha256:old")
    assert (await lookup(pair_provider(), db_sessionmaker, strict=True)).earned == "deterministic"
    assert (await lookup(pair_provider(), db_sessionmaker)).earned == "assisted"


async def test_a_stored_grant_plus_a_deep_pass_turns_the_capability_on(db_sessionmaker) -> None:
    # The declaration is only the STARTING point the measurement rules work
    # from: here the deep verdict bound to the installed weights is what
    # actually turns it on, and the same lookup without a grant answers False.
    #
    # The row records NO partner digest, which is what an unpaired deployment's
    # deep run genuinely writes: with nothing pairing the tag, the reasoning
    # case drove the tag itself. Both sides of the binding are then None, which
    # is a match.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", ttw=True)
    plain = await lookup(pair_provider(), db_sessionmaker)
    assert plain.thinking_with_tools is False
    clear_earned_cache()
    with_grant = await lookup(pair_provider(), db_sessionmaker, policy=GRANTED)
    assert with_grant.thinking_with_tools is True


async def test_a_stored_grant_does_not_survive_an_unreadable_moment(db_sessionmaker) -> None:
    # The failure path may not be the one way to get an evidence-gated grant
    # UNgated. A lookup that could not read the weights is holding no
    # certificate - and this axis is the one where believing an unproven claim
    # sends think:true plus a tool list to a runtime that then answers with
    # nothing at all.
    healthy = await lookup(pair_provider(), db_sessionmaker, policy=GRANTED)
    assert healthy.thinking_with_tools is False
    clear_earned_cache()
    broken = await lookup(BrokenProvider(), db_sessionmaker, policy=GRANTED)
    assert broken.thinking_with_tools is False


async def test_a_stored_grant_is_revoked_by_a_deep_negative(db_sessionmaker) -> None:
    # A measurement bound to these weights that says the model could NOT reason
    # and call tools revokes the grant outright.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", ttw=False)
    verdicts = await lookup(pair_provider(), db_sessionmaker, policy=GRANTED)
    assert verdicts.thinking_with_tools is False


async def test_deep_verdict_carries_thinking_with_tools(db_sessionmaker) -> None:
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", ttw=True)
    verdicts = await lookup(pair_provider(), db_sessionmaker, strict=True, policy=GRANTED)
    assert verdicts.earned == "agentic"
    assert verdicts.thinking_with_tools is True


async def test_lookup_is_cached(db_sessionmaker) -> None:
    provider = pair_provider()
    counting = CountingProvider(models=await provider.list_models())
    await lookup(counting, db_sessionmaker)
    await lookup(counting, db_sessionmaker)
    assert counting.list_calls == 1
    clear_earned_cache()
    await lookup(counting, db_sessionmaker)
    assert counting.list_calls == 2


async def test_an_env_paired_certificate_is_bindable(db_sessionmaker) -> None:
    # Both probe writers drive the tag ``THINKING_CHAT_MODEL`` names, because
    # the environment is the strongest rung of the one shared pair ranking; the
    # consumer resolves the same tag, so a certificate recorded against that
    # sibling's weights is the one this read binds against. Skip the
    # environment rung here and the binding compares None with a real digest -
    # the grant then stays inert with nothing anywhere saying why.
    await seed_probe(
        db_sessionmaker, depth="deep", earned="agentic", ttw=True, thinking_digest=THINK_DIGEST
    )
    verdicts = await lookup(
        pair_provider(),
        db_sessionmaker,
        policy=GRANTED,
        settings=make_settings(thinking_chat_model=THINK),
    )
    assert verdicts.thinking_with_tools is True


async def test_a_declared_pairing_is_bindable_too(db_sessionmaker) -> None:
    # The second rung of the same ranking, and the only one an operator can
    # reach without touching the deployment's configuration. The stored pairing
    # and the grant ride on the SAME policy row, which is how the console
    # writes them.
    await seed_probe(
        db_sessionmaker, depth="deep", earned="agentic", ttw=True, thinking_digest=THINK_DIGEST
    )
    paired = PolicyView(thinking_with_tools=True, thinking_model=THINK)
    verdicts = await lookup(pair_provider(), db_sessionmaker, policy=paired)
    assert verdicts.thinking_with_tools is True


async def test_lookup_failure_degrades_by_regime_and_never_upward(db_sessionmaker) -> None:
    # Grace: the FLOOR, not the ceiling. A read that failed is holding no
    # certificate, so it must not be worth more autonomy than a healthy turn -
    # otherwise a timeout on the model listing becomes the one reliable way to
    # reach agentic on a deployment nobody has deep-checked.
    assert (await lookup(BrokenProvider(), db_sessionmaker)).earned == "assisted"
    # Strict: fail closed, all the way.
    strict = await lookup(BrokenProvider(), db_sessionmaker, strict=True, policy=GRANTED)
    assert strict.earned == "deterministic"
    assert strict.thinking_with_tools is False


# --- a tag whose ceiling came from the runtime's report ---------------------------
#
# The modal row now: a server holds a tag nobody has declared anything about,
# and its ceiling is whatever the runtime's report allows - ``assisted`` for a
# tool-taking template. Read at that ceiling, a verdict about it is reachable
# exactly as one about a declared tag is; before the early-out was re-keyed
# onto the ceiling, such a tag was floored before its own verdict could be read.

STRAY = "stray:7b"
STRAY_DIGEST = "sha256:stray"


def stray_provider(**kwargs) -> FakeProvider:
    return FakeProvider(models=[ModelInfo(name=STRAY, digest=STRAY_DIGEST)], **kwargs)


async def stray_lookup(
    provider, db_sessionmaker, *, strict: bool = False, ceiling: str = "assisted", policy=None
):
    """Resolve verdicts for the undeclared tag, at its report-derived ceiling."""
    return await effective_verdicts(
        provider=provider,
        provider_name="fake",
        model=STRAY,
        settings=make_settings(),
        require_verification=strict,
        sessionmaker=db_sessionmaker,
        effective_ceiling=ceiling,  # type: ignore[arg-type]
        policy=policy,
    )


async def test_an_undeclared_tag_keeps_its_report_derived_ceiling(db_sessionmaker) -> None:
    # The grace regime with no probe rows: the ceiling the caller resolved
    # stands, which for a vouched template is assisted rather than the floor.
    assert (await stray_lookup(stray_provider(), db_sessionmaker)).earned == "assisted"


async def test_a_failed_verdict_clamps_an_undeclared_tag(db_sessionmaker) -> None:
    # A probe on the installed weights failed a guard, so it clamps - the same
    # rule every other tag lives under. Before the early-out was re-keyed this
    # row could not be read at all, and a failing model kept its tools.
    await seed_probe(
        db_sessionmaker, model=STRAY, digest=STRAY_DIGEST, depth="fast", earned="deterministic"
    )
    assert (await stray_lookup(stray_provider(), db_sessionmaker)).earned == "deterministic"


async def test_an_undeclared_tag_under_verification_needs_a_bound_certificate(
    db_sessionmaker,
) -> None:
    # Nothing measured on these weights: the hardened regime floors it...
    assert (await stray_lookup(stray_provider(), db_sessionmaker, strict=True)).earned == (
        "deterministic"
    )
    clear_earned_cache()
    # ...and a FAST pass at its own cap certifies exactly assisted, which is
    # all a report-ceilinged tag could reach anyway.
    await seed_probe(
        db_sessionmaker, model=STRAY, digest=STRAY_DIGEST, depth="fast", earned="assisted"
    )
    assert (await stray_lookup(stray_provider(), db_sessionmaker, strict=True)).earned == "assisted"


async def test_an_undeclared_tag_never_earns_thinking_with_tools(db_sessionmaker) -> None:
    # Structural, not an oversight: the capability starts from a stored
    # operator grant a measurement may only REVOKE, and nothing is stored here.
    # The failure mode is empty output with no error (ollama/ollama#10976), so
    # the optimistic default is the one this axis cannot afford.
    await seed_probe(
        db_sessionmaker,
        model=STRAY,
        digest=STRAY_DIGEST,
        depth="deep",
        earned="agentic",
        ttw=True,
    )
    assert (await stray_lookup(stray_provider(), db_sessionmaker)).thinking_with_tools is False


async def test_a_verdict_is_found_through_the_installed_spelling(db_sessionmaker) -> None:
    # The two sides of a verdict meet at the INSTALLED name: the management card
    # sends `info.name` to the verify endpoint, which records the tag verbatim,
    # while this read arrives holding whatever spelling the deployment
    # CONFIGURED. The probe query matches exactly, so DEFAULT_CHAT_MODEL=mistral
    # against an installed mistral:latest is one model written two ways and,
    # without canonicalization, two different keys.
    provider = FakeProvider(models=[ModelInfo(name="mistral:latest", digest="sha256:m")])
    await seed_probe(
        db_sessionmaker,
        model="mistral:latest",
        digest="sha256:m",
        depth="fast",
        earned="deterministic",
    )
    verdicts = await lookup(
        provider,
        db_sessionmaker,
        model="mistral",  # the configured spelling, one alias away from the row
        ceiling="assisted",
    )
    # Without the canonicalization the digest resolves (alias-aware) and the
    # verdict does not, so a FAILED check would clamp the card and not the turn.
    assert verdicts.earned == "deterministic"


async def test_a_certificate_is_found_through_the_installed_spelling(db_sessionmaker) -> None:
    # The mirror, and the one that fails the other way: under the hardened
    # regime a passing row that the read cannot find floors a turn whose
    # weights were in fact certified.
    provider = FakeProvider(models=[ModelInfo(name="mistral:latest", digest="sha256:m")])
    await seed_probe(
        db_sessionmaker,
        model="mistral:latest",
        digest="sha256:m",
        depth="fast",
        earned="assisted",
    )
    verdicts = await lookup(
        provider, db_sessionmaker, model="mistral", ceiling="assisted", strict=True
    )
    assert verdicts.earned == "assisted"


async def test_earned_never_reads_above_the_resolved_ceiling(db_sessionmaker) -> None:
    # The hardened regime builds earned from the best certificate the log
    # holds, which knows nothing about the ceiling resolved beside it. Harmless
    # for the decision (resolve_agency mins them anyway) and wrong for a
    # reader: the card would show ceiling assisted, earned agentic, effective
    # assisted - a row arguing with itself.
    await seed_probe(
        db_sessionmaker, model=STRAY, digest=STRAY_DIGEST, depth="deep", earned="agentic"
    )
    verdicts = await stray_lookup(stray_provider(), db_sessionmaker, strict=True)
    assert verdicts.earned == "assisted"


async def test_an_unpaired_certificate_binds_against_no_partner(db_sessionmaker) -> None:
    # The shape of a remote server's whole inventory: nothing is paired with
    # the tag and the environment names no override. It therefore has NO
    # thinking partner, the deep run drove the tag itself and recorded no
    # partner digest, and both sides of the binding are None - which is what
    # lets that certificate count. A resolution that reached for some other
    # tag's partner would compare None against a real digest and hold the grant
    # inert with nothing saying why.
    await seed_probe(
        db_sessionmaker,
        model=STRAY,
        digest=STRAY_DIGEST,
        depth="deep",
        earned="agentic",
        ttw=True,
        thinking_digest=None,
    )
    # Another pair's weights are INSTALLED here on purpose: a resolution that
    # reached for them would find a real digest to compare against, which is the
    # only way this assertion can tell the two behaviours apart.
    provider = FakeProvider(
        models=[
            ModelInfo(name=STRAY, digest=STRAY_DIGEST),
            ModelInfo(name=THINK, digest=THINK_DIGEST),
        ]
    )
    verdicts = await stray_lookup(provider, db_sessionmaker, policy=GRANTED)
    assert verdicts.thinking_with_tools is True


# --- the clamp reason ------------------------------------------------------------


def resolve(
    probes: _ProbePair,
    *,
    strict: bool = False,
    digest: str | None = CHAT_DIGEST,
    ceiling: str = "agentic",
):
    return _resolve_earned(
        ceiling=ceiling,  # type: ignore[arg-type]
        require_verification=strict,
        digest=digest,
        probes=probes,
        current_harness=HARNESS,
        serving_endpoint=ENDPOINT,
    )


def test_no_clamp_reports_no_reason() -> None:
    assert resolve(pair(deep=facts("agentic")))[1] is None
    assert resolve(pair(deep=facts("agentic")), strict=True)[1] is None


def test_failure_evidence_is_blamed_with_its_depth_and_time() -> None:
    when = datetime(2026, 7, 4, tzinfo=UTC)
    _earned, reason = resolve(pair(fast=facts("deterministic", at=when)))
    assert reason is not None
    assert (reason.code, reason.depth, reason.probe_at) == ("failed_case", "fast", when)


def test_unproven_weights_are_blamed_under_strict() -> None:
    # No certificate at all...
    assert resolve(pair(), strict=True)[1].code == "unverified_weights"
    # ...and a certificate that simply does not reach the ceiling.
    assert resolve(pair(fast=facts("assisted")), strict=True)[1].code == "unverified_weights"


def test_unproven_weights_are_blamed_under_the_grace_too() -> None:
    # The re-anchor makes this the ORDINARY case, not a hardened-only one: a
    # model sitting at the grace floor under an agentic ceiling is unproven,
    # not failing, and the reason has to say which.
    earned, reason = resolve(pair())
    assert (earned, reason.code) == ("assisted", "unverified_weights")
    assert reason.depth is None  # no probe to point at


def test_failure_evidence_outranks_unproven() -> None:
    # Both apply under strict; the measured failure is the honest blame.
    _earned, reason = resolve(pair(fast=facts("deterministic")), strict=True)
    assert reason.code == "failed_case"


def test_undigestable_weights_are_blamed_only_when_they_clamp() -> None:
    # Under an agentic ceiling both regimes land below it, so both say why.
    assert resolve(pair(), digest=None)[1].code == "no_digest"
    assert resolve(pair(), strict=True, digest=None)[1].code == "no_digest"
    # An assisted ceiling IS the grace floor, so nothing is held back and
    # there is nothing to explain.
    assert resolve(pair(), digest=None, ceiling="assisted")[1] is None


# --- explain_earned --------------------------------------------------------------


# The digest snapshot a grid reads once and hands to every row it explains.
SNAPSHOT = {CHAT: CHAT_DIGEST, THINK: THINK_DIGEST}


def broken_sessionmaker():
    """A session factory that is down, as a dead database would be."""
    raise RuntimeError("database down")


async def explain(
    db_sessionmaker,
    *,
    strict: bool = False,
    model: str = CHAT,
    digests: dict[str, str] | None = None,
    ceiling: str = "agentic",
    serving_endpoint: str | None = ENDPOINT,
    policy: PolicyView | None = None,
    settings: Settings | None = None,
):
    """Resolve the reader's explanation, with the shared arguments named.

    Args:
        db_sessionmaker: Session factory for the (uncached) probe-log read.
        strict: The ``AGENCY_REQUIRE_VERIFICATION`` regime.
        model: The tag to explain.
        digests: The caller's installed name -> digest snapshot.
        ceiling: The ceiling the CALLER resolved, exactly as on
            :func:`lookup` - the two must be handed the same one or the pair
            would be explaining one model against two rules.
        serving_endpoint: The host this deployment serves from.
        policy: The stored declarations for the tag, or None.
        settings: Settings carrying the environment pair override.

    Returns:
        The :class:`EarnedExplanation` for the tag.
    """
    return await explain_earned(
        provider_name="fake",
        model=model,
        settings=make_settings() if settings is None else settings,
        require_verification=strict,
        sessionmaker=db_sessionmaker,
        digests=SNAPSHOT if digests is None else digests,
        serving_endpoint=serving_endpoint,
        effective_ceiling=ceiling,  # type: ignore[arg-type]
        policy=policy,
    )


async def test_the_reader_finds_a_verdict_through_the_installed_spelling(
    db_sessionmaker,
) -> None:
    # The reader's whole contract is "same inputs, same answer as the chat
    # lookup", so it has to canonicalize the same way. Pinned here rather than
    # only through the page, because the page happens to explain every row under
    # `info.name` - which is already the installed spelling, and would therefore
    # pass with no canonicalization at all.
    await seed_probe(
        db_sessionmaker,
        model="mistral:latest",
        digest="sha256:m",
        depth="fast",
        earned="deterministic",
    )
    explanation = await explain(
        db_sessionmaker,
        model="mistral",
        digests={"mistral:latest": "sha256:m"},
        ceiling="assisted",
    )
    assert explanation.earned == "deterministic"
    assert explanation.reason is not None and explanation.reason.code == "failed_case"
    assert explanation.fast is not None


async def test_explain_agrees_with_the_chat_path(db_sessionmaker) -> None:
    # The two must never disagree - a grid that displays a mode the resolver
    # would not grant is a lie about what a turn will do.
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    verdicts = await lookup(pair_provider(), db_sessionmaker)
    explained = await explain(db_sessionmaker)
    assert explained.earned == verdicts.earned == "deterministic"
    assert explained.reason.code == "failed_case"


@pytest.mark.parametrize("ceiling", ["deterministic", "assisted", "agentic"])
@pytest.mark.parametrize("granted", [False, True])
@pytest.mark.parametrize("strict", [False, True])
async def test_explain_and_the_chat_path_never_disagree(
    db_sessionmaker, ceiling: str, granted: bool, strict: bool
) -> None:
    # Both functions early-out before touching the log, and they must take the
    # SAME early-outs: a deterministic ceiling with no reasoning-with-tools
    # claim is resolved from the ceiling alone by the chat path, so a reader
    # that instead read the log could report a HIGHER earned mode than the turn
    # will ever use.
    await seed_probe(db_sessionmaker, depth="fast", earned="assisted")
    policy = GRANTED if granted else None
    verdicts = await lookup(
        pair_provider(), db_sessionmaker, strict=strict, ceiling=ceiling, policy=policy
    )
    explained = await explain(db_sessionmaker, strict=strict, ceiling=ceiling, policy=policy)
    assert explained.earned == verdicts.earned
    assert explained.thinking_with_tools == verdicts.thinking_with_tools


async def test_explain_surfaces_the_measured_facts(db_sessionmaker) -> None:
    await seed_probe(
        db_sessionmaker,
        depth="deep",
        earned="agentic",
        tokens_per_second=11.5,
        vram_bytes=2_500_000_000,
        runtime_version="0.31.0",
    )
    explained = await explain(db_sessionmaker)
    assert explained.fast is None
    assert explained.deep is not None
    assert explained.deep.depth == "deep"
    assert explained.deep.tokens_per_second == 11.5
    assert explained.deep.vram_bytes == 2_500_000_000
    # Carried, never acted on: staleness is the reader's comparison to make.
    assert explained.deep.runtime_version == "0.31.0"
    assert explained.deep.harness_version == harness_version()


async def test_explain_reports_verdicts_the_installed_weights_outran(db_sessionmaker) -> None:
    # The digest-bound halves go quiet after a re-pull, which is the rule
    # working. The reader still needs to know those verdicts EXIST, so the
    # newest row per depth is reported whatever weights it measured, carrying
    # the digest that lets a caller tell the difference.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic", digest="sha256:previous")
    explained = await explain(db_sessionmaker)
    assert (explained.fast, explained.deep) == (None, None)
    assert [(view.depth, view.digest) for view in explained.latest_any_weights] == [
        ("deep", "sha256:previous")
    ]
    assert explained.latest_any_weights[0].earned == "agentic"
    # And none of it moves the verdict: an unbound row binds to nothing, so the
    # model sits at the grace floor exactly as if no probe had ever run.
    assert explained.earned == "assisted"


async def test_a_bound_verdict_is_reported_in_both_halves(db_sessionmaker) -> None:
    # No comparison happens here - the reader gets the row and its digest and
    # decides for itself, exactly as it does with the runtime version.
    await seed_probe(db_sessionmaker, depth="fast", earned="assisted")
    explained = await explain(db_sessionmaker)
    assert explained.fast is not None and explained.fast.digest == CHAT_DIGEST
    assert [view.digest for view in explained.latest_any_weights] == [CHAT_DIGEST]


async def test_a_row_the_parser_rejects_does_not_hide_the_bound_verdict(db_sessionmaker) -> None:
    # A row occupying the newest slot with a verdict this build cannot read is
    # skipped by the parser but is NOT "no rows at this depth". If the unbound
    # read branched on the parsed facts it would stop there, drop the valid
    # bound verdict underneath, and report a mode the turn will never run in.
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    await seed_probe(db_sessionmaker, depth="fast", earned="not-a-mode", digest="sha256:previous")
    explained = await explain(db_sessionmaker)
    verdicts = await lookup(pair_provider(), db_sessionmaker)
    assert explained.earned == verdicts.earned == "deterministic"
    assert explained.fast is not None and explained.fast.earned == "deterministic"
    # The unreadable row contributes nothing at all - not even a reader fact.
    assert explained.latest_any_weights == ()


async def test_unidentifiable_weights_are_not_read_at_all(db_sessionmaker) -> None:
    # With no installed digest there is nothing to compare a row against, so
    # the unbound read would be a query whose answer no caller could use.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")
    explained = await explain(db_sessionmaker, digests={})
    assert explained.latest_any_weights == ()


async def test_explain_reads_the_log_fresh(db_sessionmaker) -> None:
    # A page load must see a probe that finished seconds ago, so the reader
    # cannot be served from the chat path's TTL cache...
    assert (await explain(db_sessionmaker)).deep is None
    await seed_probe(db_sessionmaker, depth="deep", earned="assisted")
    explained = await explain(db_sessionmaker)
    assert explained.deep is not None
    assert explained.earned == "assisted"


async def test_explain_does_not_poison_the_chat_cache(db_sessionmaker) -> None:
    # ...and must not pin what it read onto later turns either.
    await explain(db_sessionmaker)
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    assert (await lookup(pair_provider(), db_sessionmaker)).earned == "deterministic"


# --- the cases a row missed ------------------------------------------------------


def test_only_the_imperfect_tallies_are_carried() -> None:
    # A case that passed every sample is not a case to point anybody at, so it
    # never leaves the parser.
    assert _imperfect_cases({"selection": [3, 3], "must-not-call": [1, 3]}) == (
        ("must-not-call", 1, 3),
    )


def test_the_parser_keeps_every_imperfect_case_including_the_reasoning_one() -> None:
    # Neutral by design: which cases a given report should NAME is a display
    # judgement, and a filter here would hide that rule from the reader.
    assert _imperfect_cases({"thinking-with-tools": [0, 3]}) == (("thinking-with-tools", 0, 3),)


@pytest.mark.parametrize(
    "tallies",
    [
        None,  # a row written before tallies existed
        "not-a-map",
        [["selection", 1, 3]],
        {"selection": [1]},  # not a pair
        {"selection": [1, 3, 5]},
        {"selection": "1/3"},
        {"selection": [1.5, 3]},
        # bool is an int in Python, so [false, true] would otherwise read as a
        # case that failed its one sample.
        {"selection": [False, True]},
        {"selection": [0, 0]},  # measured nothing, so it cannot have failed
        {"selection": [-1, 0]},  # and the shape that divides by zero downstream
        {7: [1, 3]},
    ],
)
def test_a_tally_shape_the_parser_cannot_read_is_skipped(tallies: object) -> None:
    # The column is plain JSON written by a job whose payload may widen, and
    # no explanation is worth failing a page load over.
    assert _imperfect_cases(tallies) == ()


async def test_the_reader_sees_the_cases_a_row_missed(db_sessionmaker) -> None:
    await seed_probe(
        db_sessionmaker,
        depth="fast",
        earned="deterministic",
        tallies={"must-call-freshness": [1, 1], "must-not-call": [0, 1]},
    )
    explained = await explain(db_sessionmaker)
    assert explained.fast is not None
    assert explained.fast.imperfect_cases == (("must-not-call", 0, 1),)


async def test_a_row_with_no_tallies_reports_no_cases(db_sessionmaker) -> None:
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    explained = await explain(db_sessionmaker)
    assert explained.fast is not None and explained.fast.imperfect_cases == ()


async def test_explain_takes_the_callers_digest_snapshot(db_sessionmaker) -> None:
    # The snapshot is REQUIRED, so the reader can never fall back to the
    # shared digest cache - which is what makes "always fresh, never pinning"
    # unconditional rather than true only on the paths that passed one.
    await seed_probe(db_sessionmaker, depth="fast", earned="deterministic")
    assert (await explain(db_sessionmaker)).earned == "deterministic"
    # A snapshot naming different weights binds nothing, so the clamp vanishes
    # and the grace floor stands - proof the passed map is the only digest
    # source (a read off the shared cache would have found the failing row).
    assert (await explain(db_sessionmaker, digests={CHAT: "sha256:other"})).earned == "assisted"


async def test_explain_floors_a_deterministic_ceiling_without_io(db_sessionmaker) -> None:
    # Nothing a probe could change, so nothing to query - which a dead database
    # proves. The reader takes the SAME early-out the chat path does.
    explained = await explain(broken_sessionmaker, ceiling="deterministic")
    assert explained.earned == "deterministic"
    assert (explained.reason, explained.fast, explained.deep) == (None, None, None)


async def test_explain_failure_degrades_by_regime(db_sessionmaker) -> None:
    # The same floor the chat lookup degrades to, so a dead database cannot make
    # the card claim an autonomy the turn would refuse.
    assert (await explain(broken_sessionmaker)).earned == "assisted"
    assert (await explain(broken_sessionmaker, strict=True)).earned == "deterministic"


async def test_the_turn_and_the_card_agree_about_another_server(db_sessionmaker) -> None:
    """The serving-endpoint axis end to end, through both entry points.

    ``effective_verdicts`` and ``explain_earned`` may never disagree about one
    model, and this is the axis where they could most easily drift: the turn
    reads the endpoint off the provider seam, while the card takes the
    caller's one snapshot of it.
    """
    await seed_probe(
        db_sessionmaker, depth="deep", earned="agentic", serving_endpoint=OTHER_ENDPOINT
    )
    verdicts = await lookup(pair_provider(), db_sessionmaker)
    explanation = await explain(db_sessionmaker)
    assert verdicts.earned == "assisted"  # the grace floor, not the pod's agentic
    assert explanation.earned == "assisted"
    # Named as an ENDPOINT problem rather than as "nothing has measured this":
    # the verdict exists and passed, so the action is a re-check here.
    assert explanation.reason is not None and explanation.reason.code == "endpoint_mismatch"
    # The row is still REPORTED - it is the record of what that server did, and
    # a card that silently dropped it would look like a page that lost data.
    assert explanation.deep is not None
    assert explanation.deep.serving_endpoint == OTHER_ENDPOINT


# --- the two-key rule on the reasoning-with-tools axis ------------------------
#
# The axis the console writes, and the one whose failure is SILENT: a grant a
# measurement has not backed produces EMPTY answers, not errors
# (ollama/ollama#10976). So the rule these tests exist for is the strict one -
# a declaration alone may never escalate what a turn does, on either entry
# point and in EITHER regime.


async def deep_reasoning_pass(db_sessionmaker, **kwargs) -> None:
    """Seed the one row that can certify this axis: a deep run that reasoned."""
    await seed_probe(
        db_sessionmaker,
        **{
            "depth": "deep",
            "earned": "agentic",
            "ttw": True,
            # No partner digest by default: with nothing pairing the tag, the
            # reasoning case drove the tag itself and recorded none, so both
            # sides of the binding are None. The paired tests pass one.
            "thinking_digest": None,
            **kwargs,
        },
    )


@pytest.mark.parametrize("strict", [False, True])
async def test_a_stored_grant_alone_never_reaches_a_turn(db_sessionmaker, strict: bool) -> None:
    """min(declared, measured): the declaration is present, the measurement is not.

    An operator granted the claim, which is the exact state the console's
    writer creates. Nothing has measured these weights, so BOTH the turn and
    the card must answer False - and in BOTH regimes, because there is no
    grace on this axis: it is a click somebody made, and the way it fails is
    an empty answer with nothing reporting a fault.

    MUTATION: make the fold read the declaration alone - in
    `effective_verdicts`, pass `require_verification=require_verification`
    instead of OR-ing `_grant_needs_evidence(...)`, or return `declared` from
    `_resolve_thinking_with_tools` - and the un-strict row flips to True.
    """
    turn = await lookup(pair_provider(), db_sessionmaker, strict=strict, policy=GRANTED)
    card = await explain(db_sessionmaker, strict=strict, policy=GRANTED)
    assert turn.thinking_with_tools is False
    assert card.thinking_with_tools is False


@pytest.mark.parametrize("strict", [False, True])
async def test_a_stored_grant_plus_a_deep_pass_is_what_enables_it(
    db_sessionmaker, strict: bool
) -> None:
    """The other half of the same rule: both keys present, so the turn gets it.

    Without this control the test above would also pass on a fold that returned
    False unconditionally, which would be a guard that guards by breaking the
    feature.
    """
    await deep_reasoning_pass(db_sessionmaker)
    turn = await lookup(pair_provider(), db_sessionmaker, strict=strict, policy=GRANTED)
    card = await explain(db_sessionmaker, strict=strict, policy=GRANTED)
    assert turn.thinking_with_tools is True
    assert card.thinking_with_tools is True


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("re-pull", {"digest": "sha256:old"}),
        ("harness edit", {"harness": "h-obsolete"}),
        ("another server", {"serving_endpoint": OTHER_ENDPOINT}),
        ("a spilled run", {"representative": False}),
        ("a re-pulled sibling", {"thinking_digest": "sha256:other-think"}),
    ],
)
async def test_a_stale_deep_pass_reverts_a_stored_grant(db_sessionmaker, label, kwargs) -> None:
    """Every staleness axis auto-reverts the grant THROUGH THE FOLD.

    No second staleness path exists or should: each of these makes the deep
    certificate stop counting for the reasons the module already documents, and
    an evidence-gated grant with no evidence is False. The sibling case is the
    one unique to this axis - the reasoning run drives the PAIR's thinking tag,
    so re-pulling the sibling voids the verdict even though the chat weights
    never moved.

    MUTATION: drop the matching condition from `_resolve_thinking_with_tools`'s
    hardened branch and that row starts certifying again - for four of the five.
    The re-pull case is structural rather than conditional: the digest-bound
    read never returns the row at all, so no branch here could have saved it.
    """
    await deep_reasoning_pass(db_sessionmaker, **kwargs)
    turn = await lookup(pair_provider(), db_sessionmaker, policy=GRANTED)
    card = await explain(db_sessionmaker, policy=GRANTED)
    assert turn.thinking_with_tools is False, label
    assert card.thinking_with_tools is False, label


async def test_the_card_binds_the_grant_to_the_same_sibling_the_turn_does(
    db_sessionmaker,
) -> None:
    """The reader's sibling resolution earns its keep, or the card lies.

    A reasoning verdict binds to the PAIR's thinking weights, so a reader that
    resolved its own answer - or none - would compare the deep row's
    ``thinking_digest`` against the wrong thing and withhold a capability the
    turn grants, which is the one disagreement this pair may never have.

    Asserted on a PASSING run, deliberately: a failing one revokes whether or
    not the binding matched, so it could not tell the two behaviours apart.

    MUTATION: pass ``installed_thinking_digest=None`` inside `explain_earned`
    (or resolve the sibling without the environment rung) and the card reads
    False while the turn reads True.
    """
    await deep_reasoning_pass(db_sessionmaker, thinking_digest=THINK_DIGEST)
    settings = make_settings(thinking_chat_model=THINK)
    turn = await lookup(pair_provider(), db_sessionmaker, policy=GRANTED, settings=settings)
    card = await explain(db_sessionmaker, policy=GRANTED, settings=settings)
    assert turn.thinking_with_tools is True
    assert card.thinking_with_tools is turn.thinking_with_tools


async def test_the_measured_reasoning_case_is_reported_apart_from_the_claim(
    db_sessionmaker,
) -> None:
    """The console's "what would declaring do?" answer, which effective cannot give.

    The bug this pins: ``thinking_with_tools`` is False the moment nothing
    DECLARES the capability, so a reader inferring the check's outcome from it
    reports a deep run that PASSED - and is merely waiting on a click - as a
    check that failed. Here nothing declares the claim and the deep run passed,
    which is the ordinary earn-then-declare state.

    MUTATION: return `explanation.thinking_with_tools` (or `view.thinking_with_
    tools`) for the measured field and the first pair of assertions collapses
    to False/False, losing the distinction entirely.
    """
    await deep_reasoning_pass(db_sessionmaker)
    card = await explain(db_sessionmaker)
    # Not in force - nothing declares it...
    assert card.thinking_with_tools is False
    # ...and yet the measurement plainly established it, so a grant stored now
    # would take effect at once.
    assert card.thinking_with_tools_measured is True


@pytest.mark.parametrize(
    ("label", "kwargs", "expected"),
    [
        ("a passing deep run", {}, True),
        ("a deep run that measured it absent", {"ttw": False}, False),
        ("another server", {"serving_endpoint": OTHER_ENDPOINT}, None),
        ("an edited harness", {"harness": "h-obsolete"}, None),
        ("a spilled run", {"representative": False}, None),
        ("a re-pulled sibling", {"thinking_digest": "sha256:other-think"}, None),
        ("a re-pull", {"digest": "sha256:old"}, None),
    ],
)
async def test_the_measured_reasoning_case_says_established_absent_or_unknown(
    db_sessionmaker, label, kwargs, expected
) -> None:
    """Three answers, and the difference between False and None is the point.

    False is EVIDENCE - a run bound to these weights watched the capability not
    happen. None is SILENCE - nothing this deployment can count measured it,
    which is not the same claim and must not be worded as one. Every axis that
    discounts a certificate lands in None rather than False, because a verdict
    earned elsewhere says nothing about here in either direction.

    MUTATION: collapse `_measured_reasoning_case`'s None branch into False
    (return the hardened answer directly) and the five None rows fail.
    """
    await deep_reasoning_pass(db_sessionmaker, **kwargs)
    card = await explain(db_sessionmaker)
    assert card.thinking_with_tools_measured is expected, label


async def test_nothing_measured_leaves_the_reasoning_case_unknown(db_sessionmaker) -> None:
    # The commonest state on a fresh deployment, and the one a caution must not
    # word as a failure.
    card = await explain(db_sessionmaker)
    assert card.thinking_with_tools_measured is None
    # A FAST row is still silence: that depth never runs the reasoning case.
    await seed_probe(db_sessionmaker, depth="fast", earned="assisted")
    assert (await explain(db_sessionmaker)).thinking_with_tools_measured is None


async def test_a_failed_read_does_not_ungate_a_stored_grant(db_sessionmaker) -> None:
    """A database blip must not become the one reliable way to get the grant.

    It is evidence-gated on the healthy path, so the degraded answer holds at
    False on both entry points.

    MUTATION: return `declared_ttw` on the failure path and this flips to True.
    """
    card = await explain(broken_sessionmaker, policy=GRANTED)
    assert card.thinking_with_tools is False
    turn = await lookup(BrokenProvider(), db_sessionmaker, policy=GRANTED)
    assert turn.thinking_with_tools is False


async def test_a_certificate_from_this_server_still_lifts(db_sessionmaker) -> None:
    # The control for the endpoint pair above: the same row measured HERE does
    # exactly what it always did, so the axis narrows nothing it should not.
    await seed_probe(db_sessionmaker, depth="deep", earned="agentic")
    assert (await lookup(pair_provider(), db_sessionmaker)).earned == "agentic"
    assert (await explain(db_sessionmaker)).earned == "agentic"


# --- the measured think mode (the pre-confirm routing read) -------------------


async def measured(db_sessionmaker, *, model: str = CHAT, provider: FakeProvider | None = None):
    return await measured_think_mode(
        provider=provider if provider is not None else pair_provider(),
        provider_name="fake",
        model=model,
        sessionmaker=db_sessionmaker,
    )


async def test_a_measured_mode_is_read_off_the_installed_weights(db_sessionmaker) -> None:
    await seed_probe(db_sessionmaker, depth="fast", think_mode="always")
    found = await measured(db_sessionmaker)
    assert found is not None
    assert (found.mode, found.digest) == ("always", CHAT_DIGEST)
    assert found.probe_id  # the row a confirmation would rest on


async def test_a_repull_retires_the_measured_mode(db_sessionmaker) -> None:
    # Digest-bound like every other fact on the row: measured on weights this
    # machine no longer holds says nothing about the ones it does.
    await seed_probe(db_sessionmaker, depth="fast", think_mode="always", digest="sha256:old")
    assert await measured(db_sessionmaker) is None


async def test_a_measurement_from_an_older_harness_is_not_read(db_sessionmaker) -> None:
    # Adding or editing a thinking case changes what "always" meant, and this
    # value forces a turn's think flag.
    await seed_probe(db_sessionmaker, depth="fast", think_mode="always", harness="h-old")
    assert await measured(db_sessionmaker) is None


async def test_the_newest_measurement_wins_across_depths(db_sessionmaker) -> None:
    # Every depth runs the thinking group identically, so there is no fuller
    # measurement to prefer - newer is simply truer. A deep-preferred read
    # would answer with the older row here.
    old = datetime(2026, 7, 1, tzinfo=UTC)
    await seed_probe(db_sessionmaker, depth="deep", think_mode="hybrid", at=old)
    await seed_probe(
        db_sessionmaker, depth="fast", think_mode="always", at=old + timedelta(hours=1)
    )
    found = await measured(db_sessionmaker)
    assert found is not None and found.mode == "always"


async def test_a_cpu_bound_run_still_measures_the_think_mode(db_sessionmaker) -> None:
    # Where the model SAT decides whether a throughput number describes the
    # deployment; it has nothing to do with which channel reasoning came out
    # on. Withholding this would leave the leak uncorrected on exactly the
    # machines that spill.
    await seed_probe(db_sessionmaker, depth="fast", think_mode="always", representative=False)
    found = await measured(db_sessionmaker)
    assert found is not None and found.mode == "always"


async def test_an_unknown_stored_mode_is_dropped(db_sessionmaker) -> None:
    # A plain JSON column, and this value can force a turn's think flag.
    await seed_probe(db_sessionmaker, depth="fast", think_mode="sometimes")
    assert await measured(db_sessionmaker) is None


async def test_the_pair_members_are_measured_separately(db_sessionmaker) -> None:
    # The two pair members emphatically do NOT share a think mode - the
    # instruct sibling never reasons and the thinking one always does - so a
    # read keyed on anything but the tag itself would answer about the wrong
    # weights.
    await seed_probe(db_sessionmaker, depth="fast", think_mode="never", digest=CHAT_DIGEST)
    await seed_probe(
        db_sessionmaker, depth="fast", think_mode="always", model=THINK, digest=THINK_DIGEST
    )
    chat = await measured(db_sessionmaker, model=CHAT)
    think = await measured(db_sessionmaker, model=THINK)
    assert chat is not None and chat.mode == "never"
    assert think is not None and think.mode == "always"


async def test_a_row_written_under_the_configured_spelling_is_still_found(db_sessionmaker) -> None:
    # The verify job records the tag it was HANDED, while every store keys on
    # the installed spelling. Both are tried, so a deployment that configures
    # `mistral` against an installed `mistral:latest` still gets its clamp.
    await seed_probe(
        db_sessionmaker, depth="fast", think_mode="always", model="mistral", digest="d"
    )
    provider = FakeProvider(models=[ModelInfo(name="mistral:latest", digest="d")])
    found = await measured(db_sessionmaker, model="mistral", provider=provider)
    assert found is not None and found.mode == "always"


async def test_a_failed_read_leaves_the_mode_unmeasured(db_sessionmaker) -> None:
    # None reproduces the behaviour that shipped before the measurement
    # existed, which is the only safe direction available here.
    assert await measured(broken_sessionmaker) is None
