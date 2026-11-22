"""The model_policy store: machine-scoped, append-only, folded per axis."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.models.entities import ModelPolicy
from app.providers.base import ModelInfo
from app.providers.fake_provider import FakeProvider
from app.repositories.model_policy import ModelPolicyRepository
from app.services.model_policy import (
    _PolicyRow,
    _row_of,
    clear_policy_cache,
    fold_policy,
    policy_for,
    resolve_policy,
)

MODEL = "chat:4b"
DIGEST_A = "sha256:weights-a"
DIGEST_B = "sha256:weights-b"

BASE = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _fresh_policy_cache():
    clear_policy_cache()
    yield
    clear_policy_cache()


def row(
    *,
    digest: str | None = DIGEST_A,
    ceiling: str | None = None,
    thinking_with_tools: bool | None = None,
    think_mode: str | None = None,
    num_ctx: int | None = None,
    thinking_model: str | None = None,
) -> _PolicyRow:
    return _PolicyRow(
        digest=digest,
        ceiling=ceiling,  # type: ignore[arg-type]
        thinking_with_tools=thinking_with_tools,
        think_mode=think_mode,  # type: ignore[arg-type]
        num_ctx=num_ctx,
        thinking_model=thinking_model,
    )


async def store_row(db_sessionmaker, *, model: str = MODEL, at: datetime = BASE, **fields) -> None:
    """Append one declaration row exactly as a future write endpoint would."""
    async with db_sessionmaker() as session:
        await ModelPolicyRepository(session).add(
            ModelPolicy(provider="fake", model=model, created_at=at, **fields)
        )
        await session.commit()


# --- the fold ---------------------------------------------------------------


def test_each_axis_reads_the_newest_row_that_sets_it() -> None:
    view = fold_policy(
        [row(num_ctx=16384), row(ceiling="agentic"), row(num_ctx=4096, ceiling="assisted")],
        installed_digest=DIGEST_A,
    )
    # Newest-first order, so the later rows never overwrite what an earlier one
    # already answered - and an axis nobody set stays silent.
    assert (view.num_ctx, view.ceiling) == (16384, "agentic")
    assert view.thinking_with_tools is None


def test_a_trust_axis_declared_against_other_weights_is_retired() -> None:
    view = fold_policy([row(digest=DIGEST_A, ceiling="agentic")], installed_digest=DIGEST_B)
    assert view.ceiling is None


def test_a_trust_axis_survives_while_its_weights_are_installed() -> None:
    # The positive control for the guard above: same row, matching digest.
    view = fold_policy([row(digest=DIGEST_A, ceiling="agentic")], installed_digest=DIGEST_A)
    assert view.ceiling == "agentic"


@pytest.mark.parametrize("axis", ["ceiling", "thinking_with_tools", "think_mode"])
def test_no_trust_axis_binds_without_an_installed_digest(axis: str) -> None:
    # A runtime that exposes no digest gives a claim about WEIGHTS nothing to
    # attach to, so none of them may stand. This is also the case a naive
    # `row.digest == installed_digest` gets wrong: None == None is True.
    values = {"ceiling": "agentic", "thinking_with_tools": True, "think_mode": "hybrid"}
    view = fold_policy([row(digest=None, **{axis: values[axis]})], installed_digest=None)
    assert getattr(view, axis) is None


def test_preference_axes_follow_the_tag_across_a_repull() -> None:
    # "Give this model a 16k window" is not a claim about a checkpoint, so it
    # survives weights it was not declared against.
    view = fold_policy(
        [row(digest=DIGEST_A, num_ctx=16384, thinking_model="chat:4b-think")],
        installed_digest=DIGEST_B,
    )
    assert (view.num_ctx, view.thinking_model) == (16384, "chat:4b-think")


def test_a_retired_trust_axis_does_not_fall_back_to_an_older_matching_row() -> None:
    # An operator who re-declared an axis SUPERSEDED what came before.
    # Resurrecting the older row because a re-pull invalidated the replacement
    # would restore a decision somebody explicitly withdrew - here, restoring
    # `agentic` from a deployment that had since clamped itself down.
    view = fold_policy(
        [row(digest=DIGEST_B, ceiling="deterministic"), row(digest=DIGEST_A, ceiling="agentic")],
        installed_digest=DIGEST_A,
    )
    assert view.ceiling is None


def test_a_later_row_on_new_weights_never_carries_an_old_ceiling_forward() -> None:
    # THE escalation scenario, and the reason merge-on-write is banned: an
    # agentic ceiling is declared against digest A, the weights are re-pulled
    # to B, and somebody then sets a context window on B. A store that copied
    # prior axes into each new row would have written that ceiling onto a row
    # carrying digest B, silently re-arming a grant the re-pull retired.
    view = fold_policy(
        [row(digest=DIGEST_B, num_ctx=8192), row(digest=DIGEST_A, ceiling="agentic")],
        installed_digest=DIGEST_B,
    )
    assert view.ceiling is None
    assert view.num_ctx == 8192


def test_an_unrecognized_stored_value_is_dropped_rather_than_trusted() -> None:
    # The columns are plain strings a future endpoint (or a hand edit) writes.
    parsed = _row_of(
        ModelPolicy(provider="fake", model=MODEL, ceiling="godmode", think_mode="sometimes")
    )
    assert (parsed.ceiling, parsed.think_mode) == (None, None)


def test_a_nonsense_window_is_dropped() -> None:
    assert _row_of(ModelPolicy(provider="fake", model=MODEL, num_ctx=0)).num_ctx is None
    # bool is an int in Python, so an unguarded check reads True as a window of
    # ONE token - and every budget downstream would treat it as a real number.
    assert _row_of(ModelPolicy(provider="fake", model=MODEL, num_ctx=True)).num_ctx is None


# --- the repository ---------------------------------------------------------


async def test_the_store_is_append_only_and_newest_first(db_sessionmaker) -> None:
    await store_row(db_sessionmaker, at=BASE, ceiling="assisted", digest=DIGEST_A)
    await store_row(db_sessionmaker, at=BASE + timedelta(minutes=5), num_ctx=4096, digest=DIGEST_A)
    async with db_sessionmaker() as session:
        rows = await ModelPolicyRepository(session).recent("fake", MODEL)
    assert [r.num_ctx for r in rows] == [4096, None]  # newest first, nothing overwritten
    assert rows[0].ceiling is None  # each row states only what its action set


async def test_declarations_for_another_model_are_not_read(db_sessionmaker) -> None:
    await store_row(db_sessionmaker, model="other:8b", ceiling="agentic", digest=DIGEST_A)
    async with db_sessionmaker() as session:
        assert await ModelPolicyRepository(session).recent("fake", MODEL) == []


# --- resolution -------------------------------------------------------------


async def test_a_declared_ceiling_resolves_for_the_installed_weights(db_sessionmaker) -> None:
    await store_row(db_sessionmaker, ceiling="agentic", digest=DIGEST_A)
    view = await policy_for(
        provider_name="fake",
        model=MODEL,
        digests={MODEL: DIGEST_A},
        sessionmaker=db_sessionmaker,
    )
    assert view.ceiling == "agentic"


async def test_a_repull_retires_the_declaration_on_the_real_read_path(db_sessionmaker) -> None:
    await store_row(db_sessionmaker, ceiling="agentic", digest=DIGEST_A)
    view = await policy_for(
        provider_name="fake",
        model=MODEL,
        digests={MODEL: DIGEST_B},
        sessionmaker=db_sessionmaker,
    )
    assert view.ceiling is None


async def test_nothing_declared_reads_as_no_policy(db_sessionmaker) -> None:
    assert (
        await policy_for(
            provider_name="fake",
            model=MODEL,
            digests={MODEL: DIGEST_A},
            sessionmaker=db_sessionmaker,
        )
        is None
    )


async def test_a_read_finds_a_row_written_under_the_installed_spelling(db_sessionmaker) -> None:
    # ONE matcher answers both sides. A writer records the name the provider
    # listed (`mistral:latest`); a reader arrives holding whatever the
    # deployment configured (`mistral`). A spelling-blind key would resolve the
    # digest and then find no declaration bound to it - an operator's clamp
    # applying to the management card and not to the turn.
    await store_row(db_sessionmaker, model="mistral:latest", ceiling="deterministic", digest="d1")
    view = await policy_for(
        provider_name="fake",
        model="mistral",
        digests={"mistral:latest": "d1"},
        sessionmaker=db_sessionmaker,
    )
    assert view is not None
    assert view.ceiling == "deterministic"


async def test_the_provider_variant_lists_for_its_own_digests(db_sessionmaker) -> None:
    await store_row(db_sessionmaker, ceiling="agentic", digest=DIGEST_A)
    provider = FakeProvider(models=[ModelInfo(name=MODEL, digest=DIGEST_A)])
    view = await resolve_policy(
        provider=provider,
        provider_name="fake",
        model=MODEL,
        sessionmaker=db_sessionmaker,
    )
    assert view.ceiling == "agentic"


async def test_a_failed_read_degrades_to_no_policy(db_sessionmaker) -> None:
    # The store sits ABOVE the catalog, so "no policy" is exactly the
    # pre-store behaviour rather than a new failure mode.
    def broken():
        raise RuntimeError("database down")

    assert (
        await policy_for(
            provider_name="fake",
            model=MODEL,
            digests={MODEL: DIGEST_A},
            sessionmaker=broken,
        )
        is None
    )


async def test_reads_are_cached_and_the_cache_can_be_dropped(db_sessionmaker) -> None:
    await store_row(db_sessionmaker, ceiling="assisted", digest=DIGEST_A)
    digests = {MODEL: DIGEST_A}
    first = await policy_for(
        provider_name="fake", model=MODEL, digests=digests, sessionmaker=db_sessionmaker
    )
    assert first.ceiling == "assisted"
    # A write another process made is invisible until the cache lets go...
    await store_row(
        db_sessionmaker, at=BASE + timedelta(minutes=1), ceiling="agentic", digest=DIGEST_A
    )
    cached = await policy_for(
        provider_name="fake", model=MODEL, digests=digests, sessionmaker=db_sessionmaker
    )
    assert cached.ceiling == "assisted"
    # ...which is what clear_policy_cache is for, and what a writer must call.
    clear_policy_cache()
    fresh = await policy_for(
        provider_name="fake", model=MODEL, digests=digests, sessionmaker=db_sessionmaker
    )
    assert fresh.ceiling == "agentic"
