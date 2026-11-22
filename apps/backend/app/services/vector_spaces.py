"""The stranded-vectors alarm: is anything stored still reachable?

The embedding model DEFINES the vector space. Collections are named per model
(``kb_chunks__{model}``, ``session_chunks__{model}``), so changing the
configured embedder does not corrupt anything - it simply points every query
at a fresh, empty space while the old one sits there, full and unread.
Retrieval then returns nothing, no error is raised anywhere, and the
deployment looks exactly as though it forgot everything it was taught.

This module is the detector. It takes the ACTIVE embedding tag its caller
resolved, enumerates the store's spaces through the seam, and compares the
active tag against the provenance SAMPLED from real stored points - because
the slug in a collection name is lossy and cannot be turned back into the tag
that wrote it.

Four rules shape what it says:

- **An empty store is not stranded.** Neither is a store whose points were all
  written by the model in use. The alarm fires on evidence, not on absence.
- **Out-of-reach points are only an ALARM when nothing readable is left.** The
  two shapes look identical if the readable side is never counted, and the
  difference is everything: a deployment that changed embedder and re-ingested
  - the remediation this module's own message prescribes - ends up with a full
  new collection beside a full old one, because collections linger until they
  are removed by hand and nothing here removes them. Reporting that as "stored
  knowledge cannot be searched" is false, never clears, and (riding a standing
  notice) would print onto every reloaded answer forever. So: nothing readable
  plus stranded points is the alarm; something readable plus stranded points is
  a CLEANUP note that leaves ``stranded`` False.
- **Only the persistent scope alarms.** Session collections carry a ``ttl_at``
  and the sweep clears them within hours, so a stranded session space is a
  fact that fixes itself; it is REPORTED (a reader may still wonder where a
  conversation's fetched pages went) but never raises ``stranded``. Alarming
  on it would mean crying wolf every time the embedder changes, for a
  condition that resolves on its own before anyone could act on it.
- **A store that cannot be enumerated is UNKNOWN, never fine.** The report
  says so, and callers that act on the alarm act only on a positive. A
  deployment with no embedder resolved is unknown for a different reason - the
  store answered, there is simply nothing to compare its contents against - so
  that report still carries the full inventory: an unresolved embedder is
  exactly when someone needs to see what is actually stored.

This report is DEPLOYMENT-SCOPED, not tenant-scoped, and that is deliberate:
which embedding model wrote which collection is a property of the machine, and
the seam method it reads is one of the sanctioned tenant-free maintenance
calls. The consequence has to be respected by every caller, though - the point
COUNTS here are summed across every tenant. They belong on an operator's page,
never in a sentence shown to one user, which is why :func:`_alarm_summary` (the
one shape a chat turn may show) carries the model tags - deployment
configuration, and what a reader can act on - and leaves the numbers to the
payload, while :func:`_cleanup_summary` is free to state them.

The ENUMERATION is cached in process for :data:`_CACHE_TTL_SECONDS`, because it
costs a round trip per collection and what it finds changes only when someone
edits configuration or re-ingests. An UNKNOWN reading is cached too,
deliberately: a store that is down fails for every caller that asks, and
re-asking it once a turn would add a failing round trip to turns that are
already degraded. The management view passes ``refresh=True``, so a recovered
store is picked up the moment anyone looks.
:func:`clear_vector_space_cache` forces the next read to re-enumerate.

That one cached reading now serves two questions, which is why it is the
enumeration rather than the verdict that is kept. This module asks whether the
stored corpus is still reachable; :func:`observed_embed_model` asks the
inverse - which embedder wrote it - as the evidence rung of the embedding
resolution chain, for a deployment that has named none. The two compose with a
caveat worth stating plainly: where the chain FALLS BACK to that evidence, the
active embedder is by construction one that wrote something, so the alarm can
only ever fire about the OTHER collections. A workspace that names its
embedder is judged against a tag chosen independently of the corpus, which is
where the alarm has its teeth.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.providers.rag.base import VectorSpace, VectorStore, VectorStoreUnavailableError
from app.schemas.rag import StrandedVectorsReport, VectorSpaceRow

logger = logging.getLogger(__name__)

# How long a computed report is reused. Long, because the condition it detects
# is created by an operator editing configuration and cleared by a deliberate
# re-ingest - neither of which happens between two chat turns - and because the
# turn-side consultation must not turn into a per-turn enumeration.
_CACHE_TTL_SECONDS = 900.0

# How long the enumeration may take before the report gives up and says
# "unknown". Deliberately far below the vector-store client's own 30-second
# timeout: this is a status line, and a status line must never be the slowest
# thing on the page it annotates. A store this slow cannot serve retrieval
# either, so the answer would be worthless by the time it arrived.
_ENUMERATION_TIMEOUT_SECONDS = 3.0

# (read_at, spaces), process-wide: vector spaces are a deployment fact, not a
# tenant one, so nothing about a caller belongs in the key. A ``None`` payload
# is a store that could not be enumerated, cached exactly like a success -
# a store that is down fails for every caller that asks, and re-asking it once
# a turn would add a failing round trip to turns that are already degraded.
#
# The ENUMERATION is what is cached, not the report built from it: building is
# a pure pass over a handful of rows, and two readers now judge the same
# inventory against different questions (is anything stranded, and which
# embedder wrote what is here). Caching the verdict instead would have meant
# either a second cache or a second round trip.
_cache: tuple[float, list[VectorSpace] | None] | None = None


def clear_vector_space_cache() -> None:
    """Drop the cached enumeration (test isolation, and a forced recompute)."""
    global _cache
    _cache = None


async def _enumerate(store: VectorStore, *, refresh: bool) -> list[VectorSpace] | None:
    """Return the store's collections, or None when it could not be asked.

    The one query shape both readers share. Never raises: every caller here is
    building a status line or picking a fallback, and neither may be able to
    fail the thing it describes.

    Args:
        store: The vector-store seam.
        refresh: Ignore a fresh cache entry and ask the store again.

    Returns:
        The enumerated spaces, or None when the store did not answer in time.
    """
    global _cache
    now = time.monotonic()
    if not refresh and _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
        return _cache[1]
    spaces: list[VectorSpace] | None
    try:
        # Bounded independently of the store's own client timeout, which is 30
        # seconds. This feeds a STATUS LINE on a page (and, cold, a degraded
        # turn): a black-holed Qdrant that accepts the connection and never
        # answers would otherwise hold the whole management view for half a
        # minute, exactly contradicting the promise that an unreachable store
        # leaves the rest of the page useful.
        async with asyncio.timeout(_ENUMERATION_TIMEOUT_SECONDS):
            spaces = await store.list_spaces()
    except (VectorStoreUnavailableError, TimeoutError) as exc:
        logger.warning("Cannot enumerate vector spaces: %s", exc)
        spaces = None
    # Unsynchronized on purpose: two concurrent first callers both enumerate and
    # the second simply overwrites an identical result. A lock here would be a
    # module-level asyncio primitive shared across event loops for no gain.
    _cache = (now, spaces)
    return spaces


async def refresh_vector_spaces(store: VectorStore) -> None:
    """Re-read the store's collections now, so every reader below sees one answer.

    The single point at which an operator surface pays for freshness. Both
    readers of this inventory - the stranded-vectors verdict and
    :func:`observed_embed_model` - are then served from the same reading, which
    is the property that matters: a corpus judged against an embedder resolved
    from a DIFFERENT reading of that same corpus is two answers pretending to
    be one. Never raises; a store that will not answer leaves the readers with
    their honest "unknown" and "nothing observed".

    Args:
        store: The vector-store seam.
    """
    await _enumerate(store, refresh=True)


async def observed_embed_model(store: VectorStore) -> str | None:
    """Return the embedder this deployment's stored corpus was written by.

    The evidence rung of the embedding resolution chain (see
    :mod:`app.services.embedding_resolution`): where no one has SAID which
    embedder is in use, the passages themselves are a statement of which one
    was, and it is a better answer than any written-down one that ships
    identically to every machine.

    Read from the PERSISTENT scope only. Session collections carry a ``ttl_at``
    and are swept within hours, so a corpus that has expired by tomorrow is not
    something to configure a deployment from. Where several tags are observed -
    an embedder change with a re-ingest behind it leaves two full collections,
    because nothing removes the old one - the one holding the most points wins,
    and ties break on the tag so the answer cannot depend on enumeration order.

    Reads whatever the cached enumeration holds. A caller that needs the answer
    to be current calls :func:`refresh_vector_spaces` first - once, for both
    readers - rather than refreshing from here.

    Args:
        store: The vector-store seam.

    Returns:
        The observed tag, or None when nothing is stored, the points carry no
        provenance, or the store could not be reached. Every one of those is a
        fall-through to the next rung, never a failure.
    """
    spaces = await _enumerate(store, refresh=False)
    if not spaces:
        return None
    points: dict[str, int] = {}
    for space in spaces:
        if space.kind != "kb" or space.points == 0:
            continue
        for tag in space.embed_models:
            # The sample names who wrote SOME of the points, and the count is
            # for the whole collection, so a mixed collection credits its full
            # size to each tag found in it. Fine for a tie-break between an
            # abandoned collection and a live one, which is the only question
            # asked here; it is not a per-tag figure and must not be shown as
            # one (see :class:`~app.providers.rag.base.VectorSpace`).
            points[tag] = points.get(tag, 0) + space.points
    if not points:
        return None
    return max(sorted(points), key=lambda tag: points[tag])


def _readable(space: VectorSpace, active_model: str) -> bool:
    """True when the active embedder wrote what is in this collection.

    Keyed on SAMPLED provenance rather than on the collection name: the name
    carries a lossy slug of the model tag, so two different tags can render to
    one name and the name alone cannot prove who wrote the points. An empty
    collection is readable by definition - a query against it returns nothing
    because there is nothing there, which is not a fault.
    """
    if space.points == 0:
        return True
    return active_model in space.embed_models


def _is_stranded(space: VectorSpace, active_model: str) -> bool:
    """True when this collection holds points the active embedder cannot reach.

    A non-empty collection whose sample yielded NO provenance at all is not
    counted: every point written by this codebase carries ``embed_model``, so
    an empty sample means something unexpected wrote them, and "I cannot tell"
    must not be reported as "these are lost".
    """
    return space.points > 0 and bool(space.embed_models) and not _readable(space, active_model)


def _alarm_summary(*, active_model: str, stranded_models: list[str]) -> str:
    """The sentence for the real emergency: nothing readable, and points stranded.

    States what was OBSERVED - the tag the stored points carry and the tag in
    use - and names the two things that actually fix it. No guess about which
    fix is wanted: restoring the old embedder and re-ingesting against the new
    one are both legitimate, and only an operator knows which.

    Carries NO point count, deliberately. This is the one shape a CHAT TURN may
    show, and the enumeration behind the numbers is tenant-free (see the module
    docstring), so every count here is summed across the whole deployment -
    fine on an operator's page, but telling one user how many passages exist
    would be reporting other tenants' data to them. The two model tags are
    deployment configuration and leak nothing.

    TODO(8b): once tenants are real, this sentence needs a TENANT-SCOPED basis,
    not merely a count-free rendering of a deployment-wide one. Whether the
    reader's OWN corpus is reachable is the honest question, and answering it
    needs a per-tenant count behind a `UserScopedRepository`-style boundary
    rather than the machine-wide enumeration this seam method performs.
    """
    written_by = _join_models(stranded_models)
    return (
        f"Stored knowledge cannot be searched: it was written by {written_by}, and the "
        f"active embedding model is {active_model}. Re-ingest those documents with the "
        "active model, or restore the embedding model that wrote them."
    )


def _join_models(tags: list[str]) -> str:
    """Render one or several model tags as readable English.

    Both summaries are shown verbatim, and both can be reached with more than
    one tag: two embedder changes leave two abandoned collections behind, and a
    single collection can carry mixed provenance. A bare comma join renders
    that as "written by A, B", which reads like a truncated list.
    """
    if len(tags) <= 1:
        return tags[0] if tags else ""
    return f"{', '.join(tags[:-1])} and {tags[-1]}"


def _cleanup_summary(
    *,
    active_model: str,
    stranded_models: list[str],
    stranded_points: int,
    stranded_collections: int,
    readable_points: int,
) -> str:
    """The sentence for leftovers beside a working corpus: housekeeping, not an alarm.

    This is what a deployment looks like AFTER the alarm's own remediation: the
    embedder was changed, everything was re-ingested, and the old collection is
    still sitting there because nothing removes it (collections linger until
    removed by hand). Saying "stored knowledge cannot be searched" then would be
    plainly false, would never clear, and - since it rides a standing notice -
    would persist onto every reloaded answer.

    Counts ARE included here. This shape is operator-facing only: the chat path
    keys its notice on ``stranded``, which this shape leaves False, so a
    deployment-wide number never reaches a single user's transcript.

    It also does not offer to clean anything up, because the app cannot: no
    endpoint or screen deletes a collection, and the platform's own rule is
    that old spaces linger until removed by hand. A sentence implying an action
    that does not exist would send an operator looking for a button.
    """
    written_by = _join_models(stranded_models)
    leftovers = "passage" if stranded_points == 1 else "passages"
    searchable = "passage" if readable_points == 1 else "passages"
    # The counts sum across however many abandoned collections there are, so the
    # sentence has to survive more than one of them - two embedder changes leave
    # two behind, and "a collection written by A, B" would be simply wrong.
    where = (
        f"a collection written by {written_by}"
        if stranded_collections == 1
        else f"{stranded_collections} collections written by {written_by}"
    )
    return (
        f"{stranded_points} {leftovers} remain in {where} and are no longer searchable; the "
        f"active embedding model {active_model} has {readable_points} {searchable} of its own. "
        "Retrieval is working - these are leftovers from an embedding model change, and can "
        "only be removed directly against the vector store."
    )


def _build(spaces: list[VectorSpace], active_model: str | None) -> StrandedVectorsReport:
    """Turn the enumerated spaces into the report, applying the rules above.

    The load-bearing distinction is between the two shapes an out-of-reach
    collection can take, which the earlier version of this function could not
    tell apart because it never asked what the READABLE side held:

    - nothing readable, and points stranded: the alarm. Retrieval genuinely
      returns nothing, and the deployment looks like it forgot everything.
    - something readable, and points stranded: leftovers. Retrieval works; the
      old collection is debris from a completed migration.

    ``active_model`` of None (nothing resolved an embedder) yields an
    UNKNOWN verdict that still carries the inventory. An unresolved embedder is
    precisely when someone needs to see what is actually stored, and the store
    answered perfectly well - the only missing piece is the thing to compare
    its contents against.
    """
    rows = [
        VectorSpaceRow(
            name=space.name,
            kind=space.kind,
            points=space.points,
            embed_models=sorted(space.embed_models),
            # With no embedder configured there is nothing for a point to be
            # readable BY: False here describes the missing configuration, not
            # a judgement about the points.
            readable=active_model is not None and _readable(space, active_model),
        )
        for space in spaces
    ]
    if active_model is None:
        return StrandedVectorsReport(
            known=False,
            spaces=rows,
            message=(
                "No embedding model is configured for this workspace, so there is "
                "nothing to compare the stored collections against. They are listed as "
                "found; whether any of them is still searchable cannot be determined."
            ),
        )
    stranded = [space for space in spaces if _is_stranded(space, active_model)]
    kb_stranded = [space for space in stranded if space.kind == "kb"]
    stranded_points = sum(space.points for space in kb_stranded)
    stranded_models = sorted({tag for space in kb_stranded for tag in space.embed_models})
    # Only the PERSISTENT scope counts as "the corpus still works". A session
    # collection holding readable points is one conversation's scratch space and
    # says nothing about whether the knowledge base survived.
    readable_points = sum(
        space.points for space in spaces if space.kind == "kb" and _readable(space, active_model)
    )
    alarm = bool(kb_stranded) and readable_points == 0
    if alarm:
        message = _alarm_summary(active_model=active_model, stranded_models=stranded_models)
    elif kb_stranded:
        message = _cleanup_summary(
            active_model=active_model,
            stranded_models=stranded_models,
            stranded_points=stranded_points,
            stranded_collections=len(kb_stranded),
            readable_points=readable_points,
        )
    else:
        message = None
    return StrandedVectorsReport(
        known=True,
        active_model=active_model,
        stranded=alarm,
        stranded_points=stranded_points,
        stranded_models=stranded_models,
        stranded_session_points=sum(space.points for space in stranded if space.kind == "session"),
        readable_points=readable_points,
        spaces=rows,
        message=message,
    )


async def vector_space_report(
    *, store: VectorStore, refresh: bool = False, active_model: str | None = None
) -> StrandedVectorsReport:
    """Report whether stored vectors are still reachable, off a cached enumeration.

    Args:
        store: The vector-store seam, for the space enumeration.
        refresh: Re-enumerate even when a cached reading is still fresh. Page
            loads pass True (an operator who just changed the embedder is
            exactly who reads this); the chat path never does.
        active_model: The tag retrieval actually queries with, resolved through
            :func:`app.services.embedding_resolution.resolve_embedding`. None
            means the caller could not resolve one, which is itself a state the
            report has to describe honestly: with no active embedder, nothing
            stored is reachable, and the alarm says so rather than measuring
            against a guess.

    Returns:
        The :class:`StrandedVectorsReport`. An unreachable store yields a
        report with ``known=False`` rather than an exception: this is a status
        line, and a status line must never be able to fail the thing it
        describes.
    """
    active = active_model or None
    spaces = await _enumerate(store, refresh=refresh)
    if spaces is None:
        return StrandedVectorsReport(
            known=False,
            active_model=active,
            message=(
                "The vector store could not be reached, so whether stored passages are "
                "still searchable is unknown."
            ),
        )
    return _build(spaces, active)
