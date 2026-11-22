"""Fold the stored operator declarations into one answer per axis.

``model_policy`` is append-only: each row records the axes ONE action set and
stays silent about every other. The effective policy is therefore not "the
newest row" but a per-axis fold - for each axis, the newest row that speaks to
it wins, carrying that row's own digest with it.

That per-axis digest is the whole point, and it is why merge-on-write is not
an option. The obvious implementation of a declaration store copies the
previous row's values into each new row so the newest row is always complete.
Do that and a ceiling declared against one set of weights rides forward onto a
re-pull as a side effect of somebody setting a context window - the exact
silent re-arm that ``min(CEILING, EARNED)`` exists to make impossible. Folding
instead keeps each statement attached to the weights it was made about.

Two kinds of axis, two rules:

- **TRUST axes** - ``ceiling``, ``thinking_with_tools``, ``think_mode`` - are
  claims about WEIGHTS. They are honoured only while the declaring row's
  digest is still what the runtime holds. A re-pull retires them silently and
  the resolution falls back to the runtime's report, the auto-re-clamp rule
  falling out of the fold rather than being enforced anywhere.
- **PREFERENCE axes** - ``num_ctx``, ``thinking_model`` - are claims about how
  this DEPLOYMENT wants a tag run. They follow the tag across a re-pull,
  because a chosen context window is not a statement about a checkpoint.

Reads are cached in-process per (provider, model) with a short TTL, so the
chat path costs at most one query per model per window. A writer must call
:func:`clear_policy_cache` after it commits, exactly as the verify job drops
the earned cache - a declaration nobody can see yet is not much of a
declaration.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.entities import ModelPolicy
from app.providers.base import ModelProvider
from app.repositories.model_policy import ModelPolicyRepository
from app.services.model_identity import digest_of, installed_digests, installed_name_of
from app.services.model_resolution import KNOWN_THINK_MODES, AgencyMode, ThinkMode

logger = logging.getLogger(__name__)

# Declarations change only when an operator makes one, and the writer drops
# this cache when that happens - so the TTL is a backstop against a write from
# another process, not the primary freshness mechanism.
_TTL_SECONDS = 300.0

# How many rows back a fold reads. Deliberately far past what any operator
# would produce, because the direction the bound fails in is the wrong one: an
# axis pushed out of the window reads as "never declared", so a down-clamp
# buried under enough later rows would UNCLAMP rather than persist. Each row is
# one operator action on one model, so five hundred is several lifetimes of
# clicking; a store that ever approached it wants a per-axis read instead.
_HISTORY_LIMIT = 500

# Runtime membership set for validating a stored agency value. The columns are
# plain strings, and a row can be written by an endpoint or edited by hand, so
# an unrecognized value is DROPPED rather than trusted into a typed slot. The
# think-mode half of the same rule reads KNOWN_THINK_MODES, so
# the store and its writer validate against one list.
_AGENCY_MODES = frozenset({"deterministic", "assisted", "agentic"})

# How long a FAILED read is remembered, on exactly the precedent
# :mod:`app.services.model_facts` set for a failed capability probe and for the
# same reason: without it, a store that is down is re-asked on every read, so a
# page rendering a dozen models pays a dozen connect timeouts - and pays them
# again on the next render. Deliberately far shorter than the positive TTL,
# because the direction it fails in is "an operator's declaration is briefly
# invisible" and a recovering database should be believed again quickly.
_FAILURE_TTL_SECONDS = 30.0

# (provider_name, installed model name) -> (fetched_at, rows newest-first, or
# None for a remembered FAILURE). One dict for both, so a successful read
# replaces a remembered failure outright rather than racing a second table.
_rows_cache: dict[tuple[str, str], tuple[float, tuple[_PolicyRow, ...] | None]] = {}

# One axis's value type, so the fold's per-axis helpers stay typed rather than
# reaching for attribute names as strings.
_T = TypeVar("_T")


def clear_policy_cache() -> None:
    """Drop cached declaration reads (test isolation, and after any write)."""
    _rows_cache.clear()


@dataclass
class ReadHealth:
    """Whether the declaration store answered, for ONE caller's batch of reads.

    Mutable and caller-owned on purpose. The reads themselves degrade to "no
    policy" on failure and must keep doing so - the store sits above the
    report, so a database blip has to read as the pre-store behaviour rather
    than as an outage. But a caller resolving MANY models in one request wants
    to know that the first read never reached the database, so it can stop
    asking instead of paying a connect timeout per row.

    Passing a record in is what makes that possible without changing what a
    single read returns. A caller that hands over nothing is unaffected, which
    is every caller on the chat path.

    Attributes:
        failed: A read behind this record could not reach the store.
    """

    failed: bool = False

    def mark(self) -> None:
        """Record that a read failed."""
        self.failed = True


@dataclass(frozen=True)
class _PolicyRow:
    """The fold-relevant slice of one ``model_policy`` row.

    Detached from the session before it is cached, exactly as the probe log's
    facts are: a cached ORM instance whose session has closed is a lazy-load
    waiting to raise on a path that must not fail.
    """

    digest: str | None
    ceiling: AgencyMode | None
    thinking_with_tools: bool | None
    think_mode: ThinkMode | None
    num_ctx: int | None
    thinking_model: str | None


@dataclass(frozen=True)
class PolicyView:
    """The effective declaration for one model, folded and already digest-gated.

    Every attribute is None when nothing declared it, or when a TRUST axis was
    declared against weights the runtime no longer holds. Callers therefore
    never compare digests themselves - by the time they hold a view, a retired
    axis is indistinguishable from one nobody ever set, which is precisely how
    a re-pull is supposed to read.

    Attributes:
        ceiling: The strongest agency mode an operator declared this model may
            attempt. Overrides the report in BOTH directions: a down-clamp
            takes effect immediately, and an up-arm raises only this CEILING -
            the earned axis still gates what a turn actually gets.
        thinking_with_tools: A granted reason-and-call-tools claim. A grant is
            a DECLARATION, never a measurement, and a stored one is held to the
            stricter standard: :mod:`app.services.earned` requires a deep
            verdict bound to the pair's installed weights before a turn reasons
            with tools offered, in EVERY regime. A stored ``False`` revokes on
            its own and needs no evidence at all.
        think_mode: A confirmed think mode, from a check that measured the
            weights rather than read their template.
        num_ctx: A context window chosen for this tag.
        thinking_model: The sibling tag that should serve this one's thinking
            turns.
    """

    ceiling: AgencyMode | None = None
    thinking_with_tools: bool | None = None
    think_mode: ThinkMode | None = None
    num_ctx: int | None = None
    thinking_model: str | None = None


def _row_of(row: ModelPolicy) -> _PolicyRow:
    """Project one stored row, dropping values this build cannot interpret."""
    ceiling = row.ceiling if row.ceiling in _AGENCY_MODES else None
    think_mode = row.think_mode if row.think_mode in KNOWN_THINK_MODES else None
    if row.ceiling is not None and ceiling is None:
        logger.warning("Ignoring model_policy row %s: unknown ceiling %r", row.id, row.ceiling)
    if row.think_mode is not None and think_mode is None:
        logger.warning(
            "Ignoring model_policy row %s: unknown think mode %r", row.id, row.think_mode
        )
    # bool is an int in Python, so a stored True would otherwise read as a
    # context window of one token - a value every budget downstream would then
    # treat as real. The probe log's tally parser guards the same trap.
    num_ctx = (
        row.num_ctx
        if isinstance(row.num_ctx, int) and not isinstance(row.num_ctx, bool) and row.num_ctx > 0
        else None
    )
    return _PolicyRow(
        digest=row.digest,
        ceiling=ceiling,  # type: ignore[arg-type]  # membership narrows to AgencyMode
        thinking_with_tools=row.thinking_with_tools,
        think_mode=think_mode,  # type: ignore[arg-type]  # membership narrows to ThinkMode
        num_ctx=num_ctx,
        thinking_model=(row.thinking_model or None),
    )


def fold_policy(rows: Sequence[_PolicyRow], *, installed_digest: str | None) -> PolicyView:
    """Fold newest-first declaration rows into one view (pure).

    For each axis the NEWEST row that sets it answers, and then a trust axis
    additionally has to have been declared against the installed weights. The
    fold deliberately does NOT fall back to an older matching row when the
    newest one is retired: an operator who re-declared an axis superseded what
    came before, and resurrecting it because a re-pull invalidated the
    replacement would restore a decision somebody explicitly withdrew.

    Args:
        rows: The model's declarations, newest first.
        installed_digest: The digest the runtime holds for this tag right now,
            or None when it exposes none. None retires every trust axis -
            there is nothing to bind a claim about weights to.

    Returns:
        The folded :class:`PolicyView`.
    """

    def newest(pick: Callable[[_PolicyRow], _T | None]) -> tuple[_PolicyRow, _T] | None:
        for row in rows:
            value = pick(row)
            if value is not None:
                return row, value
        return None

    def trusted(pick: Callable[[_PolicyRow], _T | None]) -> _T | None:
        found = newest(pick)
        if found is None:
            return None
        row, value = found
        if installed_digest is None or row.digest != installed_digest:
            return None
        return value

    def preferred(pick: Callable[[_PolicyRow], _T | None]) -> _T | None:
        found = newest(pick)
        return None if found is None else found[1]

    return PolicyView(
        ceiling=trusted(lambda row: row.ceiling),
        thinking_with_tools=trusted(lambda row: row.thinking_with_tools),
        think_mode=trusted(lambda row: row.think_mode),
        num_ctx=preferred(lambda row: row.num_ctx),
        thinking_model=preferred(lambda row: row.thinking_model),
    )


async def _rows_for(
    sessionmaker: async_sessionmaker[AsyncSession], provider_name: str, model: str
) -> tuple[_PolicyRow, ...] | None:
    """Return the model's declaration rows newest-first, cached.

    Returns None when the read could not be made - including when a recent
    failure is still remembered, in which case no query is attempted at all.
    A caller must not read None as "nothing declared": that is the empty tuple,
    and the difference is the whole point of remembering a failure.
    """
    key = (provider_name, model)
    now = time.monotonic()
    hit = _rows_cache.get(key)
    if hit is not None:
        cached_at, cached = hit
        ttl = _TTL_SECONDS if cached is not None else _FAILURE_TTL_SECONDS
        if now - cached_at < ttl:
            return cached
    try:
        async with sessionmaker() as session:
            stored = await ModelPolicyRepository(session).recent(
                provider_name, model, limit=_HISTORY_LIMIT
            )
            rows = tuple(_row_of(row) for row in stored)
    except Exception:
        # Remembered, so a store that is down costs one attempt per model per
        # half-minute rather than one per read. Logged by the caller, which is
        # where the model being asked about is known.
        _rows_cache[key] = (now, None)
        raise
    _rows_cache[key] = (now, rows)
    return rows


async def policy_for(
    *,
    provider_name: str,
    model: str,
    digests: dict[str, str],
    sessionmaker: async_sessionmaker[AsyncSession],
    health: ReadHealth | None = None,
) -> PolicyView | None:
    """Resolve the effective declaration for ``model``, given an installed listing.

    The variant for a caller that already listed the provider - a management
    grid explains many tags from one listing, and re-listing per row would put
    a round trip on each of them.

    Rows are keyed on the name the PROVIDER lists, because that is the spelling
    a writer had in hand; a read arriving with the configured spelling is
    canonicalized here through the one matcher
    (:mod:`app.services.model_identity`) so both sides ask the same question.

    Args:
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        model: The tag, as configured, stored, or listed.
        digests: The installed name -> digest map the caller already read.
        sessionmaker: Session factory for the declaration read (cached).
        health: A caller-owned record marked when the read fails, so a caller
            resolving many models can stop asking instead of paying a connect
            timeout per row (see :class:`ReadHealth`). Omitted everywhere the
            answer for one model is the whole request.

    Returns:
        The folded view, or None when nothing is declared for this tag - and
        also when the read failed, which is the same answer for a reason worth
        naming: the store sits ABOVE the report in the resolution chain, so
        "no policy" degrades to exactly the pre-store behaviour.

        That is a NAMED trade rather than a safe default, and it points the
        wrong way: a database blip turns an operator's down-clamp into "nothing
        declared" for the length of the outage. The alternative - failing the
        trust axes closed - would clamp every model on the deployment to
        deterministic whenever the database hiccuped, which is a far larger
        outage for a far rarer mistake. What bounds the exposure is the layer
        beneath: the earned axis still holds at its own floor, so the worst a
        lost clamp yields is the no-regression mode, never a driving turn. The
        failure is logged rather than swallowed.
    """
    if not model:
        return None
    try:
        name = installed_name_of(model, digests) or model
        rows = await _rows_for(sessionmaker, provider_name, name)
        if rows is None:
            # A remembered failure, or one this read just made. Same answer as
            # "nothing declared" to the caller, and a marked record to a caller
            # that asked to know the difference.
            if health is not None:
                health.mark()
            return None
        if not rows:
            return None
        return fold_policy(rows, installed_digest=digest_of(model, digests))
    except Exception:
        if health is not None:
            health.mark()
        logger.warning(
            "Model-policy lookup failed for %r (%s); falling back to the report",
            model,
            provider_name,
            exc_info=True,
        )
        return None


async def resolve_policy(
    *,
    provider: ModelProvider,
    provider_name: str,
    model: str,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> PolicyView | None:
    """Resolve the effective declaration for ``model``, listing the provider itself.

    The variant for the chat path, which holds no listing of its own. Both the
    listing and the rows are cached, so a steady-state turn pays neither.

    Args:
        provider: The model seam, for the installed-digest read (cached).
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        model: The tag the turn would route to.
        sessionmaker: Session factory for the declaration read (cached).

    Returns:
        The folded view, or None - see :func:`policy_for`.
    """
    if not model:
        return None
    try:
        digests = await installed_digests(provider, provider_name)
    except Exception:
        logger.warning(
            "Could not list %s to resolve the policy for %r; falling back to the report",
            provider_name,
            model,
            exc_info=True,
        )
        return None
    return await policy_for(
        provider_name=provider_name, model=model, digests=digests, sessionmaker=sessionmaker
    )
