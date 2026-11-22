"""Clean up after uploads that went wrong: leaked bytes, and rows left hanging.

Most of this module reclaims storage. An upload leaves two things behind - a row
in ``attachments`` and a blob in object storage - and deleting the row is easy;
the blob is the part that leaks, and it leaks in two distinct ways.

**A deleted conversation.** Its attachments' ``conversation_id`` becomes NULL
(the FK is ON DELETE SET NULL), which deliberately keeps the row - and with it
the ``storage_key`` that says where the bytes are. That row is a tombstone: it
exists only so the blob can still be found and removed. :func:`purge_attachments`
does exactly that, driven from the delete endpoint for a prompt cleanup;
:func:`purge_orphaned_rows` is the same work as a sweep, for the tombstones that
a queue outage (or a worker that was simply down) left behind.

**A commit that never landed.** The upload writes the bytes before the row, so a
crash in between leaves a blob no row will ever reference. Nothing in the
database records its existence, so the only way to find it is to walk the store
and ask which keys are unaccounted for - :func:`sweep_orphan_blobs`.

That sweep deletes files based on their absence from a table, which is a
dangerous shape, so it is fenced three ways: it only considers keys that match
the ``{tenant}/{conversation}/{attachment}`` UUID triple this app mints (a
stray file put there by a human is never touched), it ignores anything modified
recently (an upload in flight has written its bytes but not yet its row), and it
deletes nothing while the database is unreachable (every key would look
unreferenced).

**A row nobody will ever finish.** The other half of the module, and the only
part a user can see. An attachment leaves ``pending``/``parsing`` only when the
job processing it says so, so a worker that crashes - or that was simply down
when the file was enqueued - leaves a row claiming to be in progress with nobody
left to progress it. It is not a leak; it is worse, because the client is still
watching it. :func:`fail_stalled_attachments` is what eventually tells the truth.

**A file nobody ever chose to keep.** Everything above cleans up after a
failure. :func:`expire_aged_attachments` is the one sweep that acts on files
that worked exactly as intended, and it is a POLICY rather than a repair, so its
two tiers are deliberately asymmetric:

* *Generated files expire by default* (``GENERATED_FILE_RETENTION_DAYS``, 30).
  Nobody asked for them: a tool renders a document as a side effect of
  answering, the next question renders another, and they accumulate for as long
  as the conversation lives. This is also what finally collects the one row the
  generated path can strand - a file stored for a turn that then persisted no
  message, so no message references it and no other sweep is looking for it.
  Deliberately no special case for that row: it is a generated file past its
  window like any other, and a second rule would be a second thing to keep true.
* *Uploads and pastes do not* (``ATTACHMENT_RETENTION_DAYS``, unset). A person
  chose to send those, so their lifetime stays the conversation's, and a
  deployment with a real retention policy opts in by naming a number of days.

What expiry does NOT do is delete the row. It removes the indexed text and the
bytes and leaves the row as a TOMBSTONE at ``expired`` - filename, size and type
intact - so the interface can say "this file was here and has aged out" instead
of a transcript quietly losing a document it still refers to. That is the same
disclose-rather-than-erase reasoning the ``conversation_id`` tombstone above
follows, applied to a file that ran out of time rather than one whose thread was
deleted.

Tenancy: the sweeps span tenants on purpose. They are maintenance, not a
request - no client picks what they touch, and the tenant on each row is read
from the row itself. This is the same sanctioned exception the vector TTL sweep
takes (``QdrantVectorStore.sweep_expired``); the request-driven
:func:`purge_attachments` stays behind the tenant-scoped repository like
everything else.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import and_, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    ATTACHMENT_EXPIRED,
    ATTACHMENT_FAILED,
    ATTACHMENT_ORIGIN_GENERATED,
    ATTACHMENT_ORIGIN_PASTE,
    ATTACHMENT_ORIGIN_UPLOAD,
    ATTACHMENT_PARSING,
    ATTACHMENT_PENDING,
    ATTACHMENT_READY,
    Attachment,
    utcnow,
)
from app.providers.rag.base import VectorStore, VectorStoreUnavailableError
from app.repositories.attachments import AttachmentRepository
from app.storage.base import (
    InvalidStorageKeyError,
    StorageBackend,
    StorageUnavailableError,
)

logger = logging.getLogger(__name__)

# How many rows one pass of a row sweep claims at a time. Bounded so a backlog (a
# worker that was down for a day) is worked off in steady chunks rather than
# loaded whole. The tombstone purge and the retention sweep each loop until they
# run dry; the stalled-upload sweep takes one batch per tick and lets the next
# tick continue, because its rows are not going anywhere.
_ROW_BATCH = 200

# How many keys are checked against the database in one query. Bounded because
# it becomes an IN (...) list, and a query with fifty thousand parameters is
# how a maintenance job takes the database down with it.
_KEY_BATCH = 500

# A defensive bound on the two looping sweeps: batches * this is the most rows
# one run can touch. It exists so a bug that leaves a row in its own selection -
# failing to delete it, failing to move its status - cannot spin forever, not
# because the number itself is meaningful.
_MAX_ROW_BATCHES = 100

# What a stalled row says once the sweep gives up on it. It names re-uploading
# because that genuinely works: the dedupe check skips failed rows, so the same
# bytes uploaded again get a new row and a real second attempt.
_STALLED_MESSAGE = (
    "Processing stopped before this file was read, and did not resume. Try uploading it again."
)


class GarbageCollected(BaseModel):
    """What one garbage-collection pass reclaimed.

    ``expired`` is counted apart from ``rows`` because it is not the same event:
    a reclaimed row is gone, while an expired one is still there and still
    listed - only its contents were reclaimed.
    """

    rows: int = 0
    blobs: int = 0
    expired: int = 0


# The origins a PERSON produces, and so the ones the upload retention window
# governs. Named as the complement of "generated" rather than tested with a
# ``!=``, so an origin added later has to be placed in a tier on purpose instead
# of silently inheriting the human one.
_HUMAN_ORIGINS = (ATTACHMENT_ORIGIN_UPLOAD, ATTACHMENT_ORIGIN_PASTE)

# The statuses a retention window may act on: settled, and still sitting on the
# bytes it was given. Both are files somebody can fetch - a failed row's blob is
# written before its row is committed and no failure path removes it - and being
# fetchable is what puts a file inside a retention policy's reach. Listed rather
# than expressed as "not pending, parsing or expired" so a status added later has
# to be admitted on purpose.
_EXPIRABLE_STATUSES = (ATTACHMENT_READY, ATTACHMENT_FAILED)


def retention_window(days: int | None) -> timedelta | None:
    """Turn a configured retention in days into a window, or None when off.

    One reading of "off" for both tiers: unset and zero both mean keep forever.
    Zero has to be off rather than "expire immediately" - the other reading makes
    a single mistyped digit delete every stored file on the next tick.

    Args:
        days: The configured window, as an operator writes it. None or 0 is off.

    Returns:
        The window to measure a file's age against, or None to keep forever.
    """
    if not days:
        return None
    return timedelta(days=days)


def _is_minted_key(key: str) -> bool:
    """Return whether ``key`` has the exact shape this app mints for a blob.

    ``{tenant_id}/{conversation_id}/{attachment_id}``, all three canonical UUID
    spellings. The blob sweep deletes only what matches, so anything else that
    ever ends up under the storage root - a stray file, a future feature's
    objects, a human's scratch copy - is safe from it by construction rather
    than by luck.

    The canonical check is ``str(UUID(part)) == part``, not merely "``UUID()``
    parses it": UUID also accepts the braced and ``urn:uuid:`` spellings, whose
    punctuation :func:`~app.storage.base.validate_key` rejects. Accepting a key
    the seam then refuses to address would put this fence and that one in
    disagreement, which is precisely how a sweep starts throwing on its own
    candidates.
    """
    parts = key.split("/")
    if len(parts) != 3:
        return False
    for part in parts:
        try:
            if str(UUID(part)) != part:
                return False
        except ValueError:
            return False
    return True


async def _delete_blob(storage: StorageBackend, key: str) -> bool:
    """Delete one blob; returns False if the key was not addressable at all.

    An unaddressable key cannot be reached through the seam, so there is nothing
    to delete and no amount of retrying will change that. It is logged and
    stepped over rather than raised, because one such key raised is one key that
    stops every other file from ever being collected.

    A key can be unaddressable even when it looks minted: a symlink named as a
    UUID resolves out of the storage root, which the local backend refuses.

    Raises:
        StorageUnavailableError: The store is down. This one IS worth retrying,
            so it propagates.
    """
    try:
        await storage.delete(key)
    except InvalidStorageKeyError:
        logger.error("Storage key %r is not addressable; skipping the blob", key)
        return False
    return True


async def _delete_vectors(store: VectorStore, tenant_id: UUID, attachment: Attachment) -> None:
    """Drop an attachment's indexed text.

    Addressed by ``record_id`` (the attachment's own id), the ONE payload field
    that names a whole file - a document id addresses a single page of it. A
    tombstone has no conversation left, so there is no session to scope by, and
    the tenant plus the record is exactly the right net.

    Raises:
        VectorStoreUnavailableError: The store is down. This propagates: it is the
            one failure here that must NOT be shrugged off, because the row is
            about to be deleted and the row is what made these chunks findable.
    """
    session_id = str(attachment.conversation_id) if attachment.conversation_id else None
    dropped = await store.delete_record(str(tenant_id), str(attachment.id), session_id=session_id)
    if dropped:
        logger.info("Purged %d indexed chunk(s) of attachment %s", dropped, attachment.id)


async def purge_attachments(
    *,
    attachment_ids: list[UUID],
    repo: AttachmentRepository,
    storage: StorageBackend,
    vector_store: VectorStore,
) -> int:
    """Delete the given attachments' indexed text, blobs and rows; returns how many.

    Only tombstones are purged: a row whose conversation is still there is
    skipped, so a job replayed long after its conversation was deleted (or one
    carrying a stale id) can never take a live file with it.

    Order matters twice over. The indexed text goes FIRST, because it is the part
    that can still answer questions: a file whose row is gone but whose chunks
    survive is invisible to its owner and yet still quotable, which is the worst
    of the three states. The blob goes next and the row last, because the row is
    the only record of where the blob is - dropping it first would turn a
    recoverable failure into a permanently unfindable file.

    An unreachable vector store therefore ABORTS the purge rather than being
    stepped over. The conversation delete has usually dropped these chunks already
    (they are session-scoped, and it drops the whole session), so most of the time
    this finds nothing - but "usually" is doing real work in that sentence: that
    drop is best-effort and may have failed, and an ingest still in flight can
    write its chunks in behind it. This is what catches both. Stepping over them
    used to look safe on the reasoning that the TTL sweep would collect them
    anyway - and for a web page it would. An attachment's chunks carry no TTL at
    all (they have to outlive the session, or a file would stop grounding two
    hours after it was uploaded), so nothing else is coming back for them, and
    once the row is gone nothing can even name them.

    Args:
        attachment_ids: The attachments to reclaim.
        repo: Tenant-scoped repository; an id belonging to another tenant simply
            resolves to nothing.
        storage: Where the bytes live.
        vector_store: Where the file's indexed text lives.

    Returns:
        The number of attachments fully reclaimed.

    Raises:
        StorageUnavailableError: The blob store could not be reached. Nothing is
            committed, so the caller may retry the whole call.
        VectorStoreUnavailableError: Likewise, for the indexed text.
    """
    purged = 0
    for attachment_id in attachment_ids:
        attachment = await repo.get(attachment_id)
        if attachment is None:
            continue
        if attachment.conversation_id is not None:
            logger.warning(
                "Attachment %s still belongs to conversation %s; not purging it",
                attachment.id,
                attachment.conversation_id,
            )
            continue
        await _delete_vectors(vector_store, repo.tenant.tenant_id, attachment)
        await _delete_blob(storage, attachment.storage_key)
        await repo.delete(attachment)
        purged += 1
    if purged:
        await repo.commit()
    return purged


async def purge_orphaned_rows(
    session: AsyncSession, storage: StorageBackend, vector_store: VectorStore
) -> int:
    """Reclaim every attachment whose conversation is gone; returns how many.

    The sweep half of :func:`purge_attachments`, for tombstones whose targeted
    purge never ran or never finished. Spans tenants (see the module docstring),
    and reclaims the same three things in the same order, for the same reasons.

    An outage in EITHER store ends the pass rather than skipping the row: if one
    is down, every subsequent delete would fail too, and these rows are the only
    thing that will ever bring the job back to what they point at. The next run
    retries.
    """
    purged = 0
    for _ in range(_MAX_ROW_BATCHES):
        rows = (
            (
                await session.execute(
                    select(Attachment).where(Attachment.conversation_id.is_(None)).limit(_ROW_BATCH)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            break
        in_batch = 0
        try:
            for attachment in rows:
                await _delete_vectors(vector_store, attachment.tenant_id, attachment)
                await _delete_blob(storage, attachment.storage_key)
                await session.delete(attachment)
                in_batch += 1
        except (StorageUnavailableError, VectorStoreUnavailableError):
            # Undo the batch whole. The count must not credit rows this rollback
            # is about to restore, so it is only added once the commit lands.
            await session.rollback()
            logger.warning(
                "Stopping the attachment sweep after %d row(s): a store is unavailable",
                purged,
                exc_info=True,
            )
            return purged
        # Commit per batch: a long backlog makes progress that survives a crash
        # mid-sweep, instead of rolling all of it back.
        await session.commit()
        purged += in_batch
    if purged:
        logger.info("Reclaimed %d attachment(s) whose conversation was deleted", purged)
    return purged


async def sweep_orphan_blobs(
    session: AsyncSession,
    storage: StorageBackend,
    *,
    grace: timedelta,
    now: datetime | None = None,
) -> int:
    """Delete stored blobs that no attachment row references; returns how many.

    The last resort for bytes written by an upload whose row never committed.
    See the module docstring for the three fences that keep it from deleting
    anything live.

    Args:
        session: A database session (the referenced-keys query spans tenants).
        storage: The store to walk.
        grace: How long a blob must have sat untouched to count as garbage.
        now: The instant to measure age against; defaults to the current time.

    Returns:
        The number of blobs deleted.

    Raises:
        StorageUnavailableError: The store could not be walked. Raised rather
            than swallowed: a listing that fails halfway has seen an arbitrary
            subset, and the caller decides whether to retry.
    """
    cutoff = (now or datetime.now(UTC)) - grace
    deleted = 0
    skipped = 0
    batch: list[str] = []

    async def flush(keys: list[str]) -> int:
        """Delete whichever of ``keys`` no row claims."""
        # Spans tenants deliberately: a per-tenant query would report every
        # OTHER tenant's blobs as unreferenced, and this loop would delete them.
        referenced = set(
            (
                await session.execute(
                    select(Attachment.storage_key).where(Attachment.storage_key.in_(keys))
                )
            )
            .scalars()
            .all()
        )
        gone = 0
        for key in keys:
            if key in referenced:
                continue
            # Through _delete_blob, not storage.delete: an unaddressable key
            # raised here would abort the walk and truncate every orphan behind
            # it, hourly and forever.
            if await _delete_blob(storage, key):
                logger.info("Deleted orphaned blob %s (no attachment row references it)", key)
                gone += 1
        return gone

    async for info in storage.list_objects():
        if info.modified_at >= cutoff:
            continue
        if not _is_minted_key(info.key):
            skipped += 1
            continue
        batch.append(info.key)
        if len(batch) >= _KEY_BATCH:
            deleted += await flush(batch)
            batch = []
    if batch:
        deleted += await flush(batch)

    if skipped:
        # Never silently: an operator who finds fewer files than expected should
        # be able to see that the sweep chose not to touch them.
        logger.info("Blob sweep left %d object(s) alone: not a key this app mints", skipped)
    return deleted


async def fail_stalled_attachments(
    session: AsyncSession,
    *,
    stall: timedelta,
    has_live_job: Callable[[UUID], Awaitable[bool]] | None = None,
    now: datetime | None = None,
) -> int:
    """Close out attachments no job will ever finish; returns how many.

    An upload's row is only ever moved off ``pending``/``parsing`` by the job
    that processes it. When no such job exists - the queue was down at upload
    time and the enqueue was lost, or the job vanished with the queue's state -
    the row keeps saying "processing" with nothing left to finish it, and the
    client polling it waits forever on a spinner that will never resolve. This is
    the only thing that ever closes that row.

    Age alone does not mean dead, and this is the crux of it. A row can sit
    untouched for an hour simply because it is tenth in a queue the worker is
    still working through. Failing it there would be a lie with teeth: the row
    tells the user to upload the file again, re-uploading skips the dedupe check
    for failed rows BY DESIGN, and the queued job then lands anyway - leaving two
    rows, two blobs and one file citable twice as two separate sources. So
    ``has_live_job`` is asked first, and anything the queue still intends to run
    is left alone. It will be picked up; nothing is lost by waiting.

    That check is also why the sweep can be sure of itself. It only ever runs
    inside a live worker, so a job the queue still holds is a job that WILL run.

    Tombstones are skipped: a row whose conversation is gone is about to be
    deleted outright by :func:`purge_orphaned_rows`, and dressing it up as a
    failure first would be pure noise. Spans tenants, like the other sweeps (see
    the module docstring).

    Args:
        session: A database session (the query spans tenants).
        stall: How long a row must have sat unchanged before it is even a
            candidate. Generous, because one parse attempt may legitimately run
            for the whole job timeout without touching the row.
        has_live_job: Asks whether the queue still holds a job for an attachment.
            Omitted (in tests, and where no queue is reachable), age is the only
            evidence there is.
        now: The instant to measure age against; defaults to the current time.

    Returns:
        The number of attachments marked failed.
    """
    cutoff = (now or utcnow()) - stall
    stmt = (
        select(Attachment)
        .where(
            Attachment.status.in_((ATTACHMENT_PENDING, ATTACHMENT_PARSING)),
            Attachment.updated_at < cutoff,
            Attachment.conversation_id.is_not(None),
        )
        .limit(_ROW_BATCH)
    )
    rows = (await session.execute(stmt)).scalars().all()
    failed = 0
    for attachment in rows:
        if has_live_job is not None and await has_live_job(attachment.id):
            logger.info(
                "Attachment %s has been %s since %s, but its job is still queued; leaving it",
                attachment.id,
                attachment.status,
                attachment.updated_at,
            )
            continue
        logger.warning(
            "Attachment %s (%s) has been %s since %s and no job is left to finish it; "
            "marking it failed",
            attachment.id,
            attachment.filename,
            attachment.status,
            attachment.updated_at,
        )
        attachment.status = ATTACHMENT_FAILED
        attachment.error = _STALLED_MESSAGE
        attachment.updated_at = utcnow()
        session.add(attachment)
        failed += 1
    if failed:
        await session.commit()
    return failed


async def expire_aged_attachments(
    session: AsyncSession,
    storage: StorageBackend,
    vector_store: VectorStore,
    *,
    generated_retention: timedelta | None,
    upload_retention: timedelta | None,
    now: datetime | None = None,
) -> int:
    """Age files past their retention window out of storage; returns how many.

    The two-tier retention policy, in one query. See the module docstring for why
    the tiers are asymmetric; this is the mechanism.

    Two things are reclaimed and the row is NOT one of them. The indexed text
    goes first, for the same reason every other path here drops it first: it is
    the part that can still answer questions, and a file whose bytes are gone but
    whose passages survive would go on grounding - and citing - a document nobody
    can open. The bytes go next, and the status is COMMITTED last.

    The commit is what carries the guarantee, not the order of the statements: a
    row is only ever visible as ``expired`` once its contents are actually gone,
    because until the commit lands nothing outside this transaction can see the
    flip at all. That matters because the status is the only thing that takes a
    row out of the selection - a row marked expired over a blob still sitting in
    the store would have nothing left that would ever come back for it, and
    unlike a purged tombstone its ``storage_key`` is still there, so the orphan
    blob sweep reads those bytes as referenced and steps over them forever.

    Interrupted the other way round is NOT free, and it is worth naming rather
    than glossing. The deletes are not transactional with the row, so a failure
    between them - passages already dropped, blob delete raising - rolls the row
    back to the status it came in with, its indexed text genuinely gone. That
    file is listed, downloadable and inside its retention window while grounding
    and citing nothing: a real degradation, not a no-op. It is bounded (the row
    stays selectable, so the next successful pass finishes it, and every step is
    idempotent - a missing blob deletes as a no-op by the storage seam's
    contract, and matching no chunks costs one count) and it is the least bad of
    the three orderings: dropping the passages last would leave an ``expired``
    row still quotable, and committing first would strand the bytes forever.

    Which is also why the commit is per ROW rather than per batch: it bounds that
    window to the ONE row a failure interrupted, where a batch-wide rollback
    would put every row the pass had already emptied into it.

    An unaddressable key is the single case where a row expires without its bytes
    having gone (:func:`_delete_blob` logs and returns False). It still expires,
    for the same reason :func:`purge_orphaned_rows` still drops such a row: no
    path in this app can reach those bytes through the seam - not a download, not
    a later sweep - so ``expired`` is the honest status, and a row that could
    never leave the selection would be retried every hour forever.

    ``ready`` and ``failed`` rows are both candidates, and the selection is one
    rule for every origin rather than a branch: a file is a candidate once it has
    settled and its bytes are still there. ``failed`` qualifies because those
    bytes ARE still there and, since the download route began serving a failed
    upload back, they are reachable - a file somebody can still fetch is a file a
    retention policy has to be able to reach. ``generated`` never occupies that
    state (it is born ``ready`` and nothing parses it), so including it costs
    nothing and buys a selection with no special case in it.

    ``pending``/``parsing`` are excluded: they belong to
    :func:`fail_stalled_attachments` and to the job still working on them, and a
    row mid-ingest that could be selected here would race an ingest into writing
    chunks nothing would collect. ``expired`` is where this sweep leaves rows.
    Tombstones are skipped too - a row whose conversation is gone is about to be
    deleted outright by :func:`purge_orphaned_rows`, and an expired tombstone is
    a disclosure with nowhere to appear.

    **A row expiring from ``failed`` has its ``error`` cleared, deliberately.**
    That sentence is real information right up until the bytes go, and then it
    stops being: what is true afterwards is that the file is gone by policy,
    which is exactly what a readable file's tombstone says, and one line is all a
    chip has. Keeping it would do worse than clutter - the tray reads ANY row
    with a surviving ``error`` as a PROBLEM rather than a tombstone, so the chip
    would go on presenting a months-old parse verdict as news about a file that
    aged out normally. After the window the parse verdict has nothing left to
    describe.

    Spans tenants, reading the tenant off each row (see the module docstring). An
    outage in either store - or in the database itself, at the commit - ends the
    pass rather than skipping the row, as in :func:`purge_orphaned_rows`:
    whatever the pass finished is kept and reported, and the next tick continues
    from there.

    Args:
        session: A database session (the selection spans tenants).
        storage: Where the bytes live.
        vector_store: Where the indexed text lives.
        generated_retention: How long a tool-generated file is kept, or None to
            keep them forever.
        upload_retention: How long a file a person sent is kept, or None to keep
            them for as long as their conversation.
        now: The instant to measure age against; defaults to the current time.

    Returns:
        The number of attachments expired.
    """
    moment = now or utcnow()
    tiers = (
        (Attachment.origin == ATTACHMENT_ORIGIN_GENERATED, generated_retention),
        (Attachment.origin.in_(_HUMAN_ORIGINS), upload_retention),
    )
    aged = [
        and_(origin_matches, Attachment.created_at < moment - retention)
        for origin_matches, retention in tiers
        if retention is not None
    ]
    if not aged:
        # Both tiers off: the shipped default for uploads, and what an operator
        # who wants nothing deleted configures. Skip the query entirely.
        return 0

    expired = 0
    for _ in range(_MAX_ROW_BATCHES):
        stmt = (
            select(Attachment)
            .where(
                Attachment.status.in_(_EXPIRABLE_STATUSES),
                Attachment.conversation_id.is_not(None),
                or_(*aged),
            )
            # Oldest first, and not only for tidiness: without an ORDER BY the
            # database may hand back a LIMITed batch in any order it likes, so
            # which row an interrupted pass stops on would be unpredictable -
            # untestable, and awkward to reason about from a log.
            .order_by(Attachment.created_at, Attachment.id)
            .limit(_ROW_BATCH)
        )
        rows = (await session.execute(stmt)).scalars().all()
        if not rows:
            break
        for attachment in rows:
            try:
                # A `failed` row has no chunks - its ingestion never got that far
                # - so this is a no-op for it. Run anyway: it is idempotent, it
                # keeps one flow for both statuses, and it collects any
                # historical row that reached `failed` with passages already
                # written.
                await _delete_vectors(vector_store, attachment.tenant_id, attachment)
                await _delete_blob(storage, attachment.storage_key)
                attachment.status = ATTACHMENT_EXPIRED
                # Written straight to the session rather than through
                # AttachmentRepository.set_status, like every sweep in this
                # module: that method ASSERTS the row belongs to the bound
                # principal, and this is cross-tenant maintenance with no
                # principal to bind. See the module docstring on tenancy.
                #
                # CLEARED, not merely left alone - the row may be arriving from
                # `failed` with a parse error on it. See the docstring: a chip
                # has one line, an `error` makes it read as a fault, and "gone by
                # policy" is what is true of this file now.
                attachment.error = None
                attachment.updated_at = utcnow()
                session.add(attachment)
                await session.commit()
            except (StorageUnavailableError, VectorStoreUnavailableError, SQLAlchemyError):
                # Undo only this row's pending flip. The count credits none of it,
                # since it is incremented after the commit that made it real.
                #
                # SQLAlchemyError covers the commit itself failing. Without it a
                # database blip escapes as an unhandled job error, past a
                # docstring in collect_garbage that promises the opposite - and
                # past the rollback that leaves the session usable.
                await session.rollback()
                logger.warning(
                    "Stopping the retention sweep after %d attachment(s): a store or the "
                    "database is unavailable",
                    expired,
                    exc_info=True,
                )
                return expired
            # Logged AFTER the commit that made it true. Before it, the line
            # announced an expiry that a failing commit could still undo, which
            # is the one log an operator would use to explain a missing file.
            logger.info(
                "Expired attachment %s (%s, %s) stored %s: past its retention window",
                attachment.id,
                attachment.filename,
                attachment.origin,
                attachment.created_at,
            )
            expired += 1
    return expired


async def collect_garbage(
    session: AsyncSession,
    storage: StorageBackend,
    vector_store: VectorStore,
    *,
    grace: timedelta,
    # Required, not defaulted to None: a caller that forgot them would silently
    # run with retention off, which looks exactly like a working sweep.
    generated_retention: timedelta | None,
    upload_retention: timedelta | None,
) -> GarbageCollected:
    """Run all three reclaims; returns what each one freed.

    Tombstones, then retention, then blobs - and the order is not arbitrary. The
    tombstone purge goes first because it deletes rows AND their blobs, so
    expiring one of them first would empty a row that is about to be dropped
    whole. The blob walk goes last for the reason it always has: both row passes
    have already removed their keys, so it never has to reason about a key whose
    row is changing underneath it.

    A storage outage aborts the pass and is logged rather than raised: this is
    the periodic cron, and its whole contract is that the next tick tries again.
    Whatever the pass managed before the outage is still reported. The database
    going away is caught on the same terms - the passes above each read and write
    through this one session, so a connection that dies is exactly as transient
    as a store that does, and letting it escape would fail the arq job over a
    blip the next tick would sail through.

    Args:
        session: A database session (all three passes span tenants).
        storage: Where the bytes live.
        vector_store: Where indexed text lives.
        grace: How long an unreferenced blob must sit untouched to count as
            garbage.
        generated_retention: How long a tool-generated file is kept; None keeps
            them forever.
        upload_retention: How long a file a person sent is kept; None keeps them
            for as long as their conversation.
    """
    rows = 0
    blobs = 0
    expired = 0
    try:
        rows = await purge_orphaned_rows(session, storage, vector_store)
        expired = await expire_aged_attachments(
            session,
            storage,
            vector_store,
            generated_retention=generated_retention,
            upload_retention=upload_retention,
        )
        blobs = await sweep_orphan_blobs(session, storage, grace=grace)
    except (StorageUnavailableError, VectorStoreUnavailableError, SQLAlchemyError):
        logger.warning(
            "Attachment garbage collection cut short: a store or the database is unavailable",
            exc_info=True,
        )
    return GarbageCollected(rows=rows, blobs=blobs, expired=expired)
