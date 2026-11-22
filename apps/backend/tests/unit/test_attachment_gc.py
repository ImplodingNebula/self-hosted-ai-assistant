"""Attachment garbage collection: tombstone purges, orphan blobs, and the fences."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from arq import Retry
from arq.jobs import JobStatus
from sqlalchemy import select

from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import (
    ATTACHMENT_FAILED,
    ATTACHMENT_PARSING,
    ATTACHMENT_PENDING,
    ATTACHMENT_READY,
    Attachment,
    Conversation,
)
from app.providers.rag.base import VectorStore, VectorStoreUnavailableError
from app.providers.rag.embedder import FakeEmbedder
from app.repositories.attachments import AttachmentRepository
from app.schemas.rag import RagIngestItem
from app.services.attachment_gc import (
    GarbageCollected,
    _is_minted_key,
    collect_garbage,
    fail_stalled_attachments,
    purge_attachments,
    purge_orphaned_rows,
    sweep_orphan_blobs,
)
from app.services.rag_service import ingest_structured
from app.storage.base import InvalidStorageKeyError, StorageUnavailableError
from app.storage.local import FakeStorageBackend
from app.workers.main import (
    _LIVE_JOB_STATUSES,
    MAX_TRIES,
    _has_live_ingest_job,
    collect_attachment_garbage,
    purge_deleted_attachments,
    reap_stalled_attachments,
)
from tests.conftest import make_memory_store

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")
OTHER_TENANT = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")

GRACE = timedelta(minutes=60)
DATA = b"the bytes of an uploaded file"

# The text an indexed attachment contributes, and the embedder that gives it a
# vector. Both only have to be deterministic: what the tests ask is whether the
# chunks are still there, never what they say.
INDEXED_TEXT = "The turbine was taken offline after the bearing temperature spiked."
EMBEDDER = FakeEmbedder(dim=32)


class _DeadStorage(FakeStorageBackend):
    """A store whose deletes always fail, standing in for one that is down."""

    async def delete(self, key: str) -> None:
        raise StorageUnavailableError("storage is down")


class _DeadVectors:
    """A vector store that cannot be reached (only the purge's one call is needed)."""

    async def delete_record(
        self, tenant_id: str, record_id: str, *, session_id: str | None = None
    ) -> int:
        raise VectorStoreUnavailableError("the vector store is unavailable")


async def _index(store: VectorStore, attachment: Attachment) -> None:
    """Index the file's text exactly as the parse job does: session-scoped, no TTL."""
    await ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=[
            RagIngestItem(
                id=f"attachment:{attachment.id}:0:0",
                text=INDEXED_TEXT,
                record_id=str(attachment.id),
            )
        ],
        embedder=EMBEDDER,
        store=store,
        settings=get_settings(),
        session_id=str(attachment.conversation_id),
        persist=True,
    )


async def _indexed_chunks(store: VectorStore, conversation_id: UUID) -> int:
    """How many of that conversation's chunks the store still holds."""
    hits = await store.query(
        "session",
        EMBEDDER.model,
        str(TENANT.tenant_id),
        await EMBEDDER.embed_query(INDEXED_TEXT),
        top_k=50,
        session_id=str(conversation_id),
    )
    return len(hits)


async def _orphan_the_conversation(db_sessionmaker, conversation_id: UUID) -> None:
    """Delete the conversation, leaving its attachments as tombstones (FK SET NULL)."""
    async with db_sessionmaker() as session:
        conversation = await session.get(Conversation, conversation_id)
        await session.delete(conversation)
        await session.commit()


async def _conversation(db_sessionmaker, tenant: TenantContext = TENANT) -> UUID:
    async with db_sessionmaker() as session:
        conversation = Conversation(tenant_id=tenant.tenant_id, user_id=tenant.user_id)
        session.add(conversation)
        await session.commit()
        return conversation.id


async def _attachment(
    db_sessionmaker,
    storage: FakeStorageBackend,
    conversation_id: UUID | None,
    *,
    tenant: TenantContext = TENANT,
    key: str | None = None,
) -> Attachment:
    """Store bytes and commit the row that points at them, as an upload does."""
    attachment_id = uuid4()
    storage_key = key or f"{tenant.tenant_id}/{conversation_id or uuid4()}/{attachment_id}"
    await storage.put(storage_key, DATA, content_type="text/plain")
    async with db_sessionmaker() as session:
        attachment = Attachment(
            id=attachment_id,
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            conversation_id=conversation_id,
            filename="incident.txt",
            content_type="text/plain",
            size_bytes=len(DATA),
            storage_key=storage_key,
        )
        session.add(attachment)
        await session.commit()
        return attachment


async def _rows(db_sessionmaker) -> list[Attachment]:
    async with db_sessionmaker() as session:
        return list((await session.execute(select(Attachment))).scalars().all())


async def _age(storage: FakeStorageBackend, key: str, *, minutes: int) -> None:
    """Backdate an object so the grace window has passed for it."""
    storage.modified_at[key] = datetime.now(UTC) - timedelta(minutes=minutes)


# ---------------------------------------------------------------- targeted purge


async def test_purging_a_tombstone_removes_both_the_blob_and_the_row(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    attachment = await _attachment(db_sessionmaker, storage, None)

    async with db_sessionmaker() as session:
        purged = await purge_attachments(
            attachment_ids=[attachment.id],
            repo=AttachmentRepository(session, TENANT),
            storage=storage,
            vector_store=make_memory_store(),
        )

    assert purged == 1
    assert storage.objects == {}
    assert await _rows(db_sessionmaker) == []


async def test_an_attachment_whose_conversation_still_exists_is_never_purged(
    db_sessionmaker,
) -> None:
    # The one rule that keeps a stale or replayed job from taking a live file
    # with it: only a row whose conversation is already gone may be reclaimed.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)

    async with db_sessionmaker() as session:
        purged = await purge_attachments(
            attachment_ids=[attachment.id],
            repo=AttachmentRepository(session, TENANT),
            storage=storage,
            vector_store=make_memory_store(),
        )

    assert purged == 0
    assert attachment.storage_key in storage.objects
    assert len(await _rows(db_sessionmaker)) == 1


async def test_purging_another_tenants_attachment_does_nothing(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    attachment = await _attachment(db_sessionmaker, storage, None, tenant=OTHER_TENANT)

    async with db_sessionmaker() as session:
        purged = await purge_attachments(
            attachment_ids=[attachment.id],
            repo=AttachmentRepository(session, TENANT),
            storage=storage,
            vector_store=make_memory_store(),
        )

    assert purged == 0
    assert attachment.storage_key in storage.objects


async def test_purging_a_vanished_attachment_is_a_no_op(db_sessionmaker) -> None:
    storage = FakeStorageBackend()

    async with db_sessionmaker() as session:
        purged = await purge_attachments(
            attachment_ids=[uuid4()],
            repo=AttachmentRepository(session, TENANT),
            storage=storage,
            vector_store=make_memory_store(),
        )

    assert purged == 0


async def test_a_storage_outage_keeps_the_row_so_the_purge_can_be_retried(
    db_sessionmaker,
) -> None:
    # The row is the only record of where the blob is. Dropping it while the
    # store is down would make the bytes unfindable forever.
    storage = _DeadStorage()
    attachment = await _attachment(db_sessionmaker, storage, None)

    async with db_sessionmaker() as session:
        with pytest.raises(StorageUnavailableError):
            await purge_attachments(
                attachment_ids=[attachment.id],
                repo=AttachmentRepository(session, TENANT),
                storage=storage,
                vector_store=make_memory_store(),
            )

    assert len(await _rows(db_sessionmaker)) == 1


async def test_purging_a_tombstone_also_drops_the_text_it_indexed(db_sessionmaker) -> None:
    # The state worth fearing: the row is gone, so the file is invisible to the
    # person who uploaded it - and its passages are still there, still answering
    # questions, with nothing left in the system that could name them. Attachment
    # chunks carry no TTL, so no sweep is coming for them either.
    storage = FakeStorageBackend()
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _index(store, attachment)
    assert await _indexed_chunks(store, conversation_id) == 1
    await _orphan_the_conversation(db_sessionmaker, conversation_id)

    async with db_sessionmaker() as session:
        purged = await purge_attachments(
            attachment_ids=[attachment.id],
            repo=AttachmentRepository(session, TENANT),
            storage=storage,
            vector_store=store,
        )

    assert purged == 1
    assert await _indexed_chunks(store, conversation_id) == 0
    assert storage.objects == {}


async def test_a_vector_outage_keeps_everything_so_the_purge_can_be_retried(
    db_sessionmaker,
) -> None:
    # The indexed text is deleted FIRST, and an outage there stops the purge dead:
    # the row and the blob must both survive, because the row is the only thing
    # that still knows which chunks belong to this file.
    storage = FakeStorageBackend()
    attachment = await _attachment(db_sessionmaker, storage, None)

    async with db_sessionmaker() as session:
        with pytest.raises(VectorStoreUnavailableError):
            await purge_attachments(
                attachment_ids=[attachment.id],
                repo=AttachmentRepository(session, TENANT),
                storage=storage,
                vector_store=_DeadVectors(),
            )

    assert len(await _rows(db_sessionmaker)) == 1
    assert attachment.storage_key in storage.objects


# ------------------------------------------------------------------- row sweep


async def test_the_sweep_reclaims_every_conversationless_row(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    await _attachment(db_sessionmaker, storage, None)
    await _attachment(db_sessionmaker, storage, None, tenant=OTHER_TENANT)

    async with db_sessionmaker() as session:
        purged = await purge_orphaned_rows(session, storage, make_memory_store())

    # Deliberately cross-tenant: this is maintenance, and no client chooses what
    # it touches. A per-tenant sweep would leave every other tenant's files.
    assert purged == 2
    assert storage.objects == {}
    assert await _rows(db_sessionmaker) == []


async def test_the_sweep_leaves_live_attachments_alone(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    live = await _attachment(db_sessionmaker, storage, conversation_id)
    await _attachment(db_sessionmaker, storage, None)

    async with db_sessionmaker() as session:
        purged = await purge_orphaned_rows(session, storage, make_memory_store())

    assert purged == 1
    assert list(storage.objects) == [live.storage_key]


async def test_deleting_a_conversation_leaves_its_attachments_collectable(
    db_sessionmaker,
) -> None:
    # The FK is ON DELETE SET NULL, and that NULL is the tombstone the sweep
    # looks for. If the cascade ever changed to delete the row, the blob would
    # become unfindable and this test would fail loudly rather than silently
    # leaking the bytes.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)

    async with db_sessionmaker() as session:
        conversation = await session.get(Conversation, conversation_id)
        await session.delete(conversation)
        await session.commit()

    async with db_sessionmaker() as session:
        purged = await purge_orphaned_rows(session, storage, make_memory_store())

    assert purged == 1
    assert attachment.storage_key not in storage.objects


async def test_the_sweep_drops_the_indexed_text_of_the_rows_it_reclaims(
    db_sessionmaker,
) -> None:
    # The sweep is the path that runs when the targeted purge never did (the queue
    # was down, the worker was restarting), so it has to reclaim the same three
    # things - and the passages are the one a leak of which is still quotable.
    storage = FakeStorageBackend()
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _index(store, attachment)
    await _orphan_the_conversation(db_sessionmaker, conversation_id)

    async with db_sessionmaker() as session:
        purged = await purge_orphaned_rows(session, storage, store)

    assert purged == 1
    assert await _indexed_chunks(store, conversation_id) == 0


async def test_the_sweep_stops_rather_than_dropping_rows_it_cannot_clean(
    db_sessionmaker,
) -> None:
    storage = _DeadStorage()
    await _attachment(db_sessionmaker, storage, None)

    async with db_sessionmaker() as session:
        purged = await purge_orphaned_rows(session, storage, make_memory_store())

    # Nothing reclaimed, and - crucially - nothing reported as reclaimed: the
    # rollback undid the batch, so counting it would be a lie.
    assert purged == 0
    assert len(await _rows(db_sessionmaker)) == 1


async def test_the_sweep_stops_rather_than_dropping_a_row_whose_chunks_it_cannot_reach(
    db_sessionmaker,
) -> None:
    # The twin of the storage outage above, and the more dangerous of the two: a
    # row deleted while the vector store is down leaves passages nothing can ever
    # name again. Stepping over the row keeps the one handle that addresses them.
    storage = FakeStorageBackend()
    await _attachment(db_sessionmaker, storage, None)

    async with db_sessionmaker() as session:
        purged = await purge_orphaned_rows(session, storage, _DeadVectors())

    assert purged == 0
    assert len(await _rows(db_sessionmaker)) == 1


async def test_an_unaddressable_key_does_not_wedge_the_sweep(db_sessionmaker) -> None:
    # A key no backend can address has no blob to delete and never will. If that
    # row were kept, it would fail forever and block every file behind it.
    class _RefusingStorage(FakeStorageBackend):
        async def delete(self, key: str) -> None:
            raise InvalidStorageKeyError(key)

    storage = _RefusingStorage()
    await _attachment(db_sessionmaker, storage, None)

    async with db_sessionmaker() as session:
        purged = await purge_orphaned_rows(session, storage, make_memory_store())

    assert purged == 1
    assert await _rows(db_sessionmaker) == []


# ------------------------------------------------------------------ blob sweep


async def test_an_orphaned_blob_past_the_grace_window_is_deleted(db_sessionmaker) -> None:
    # The bytes an upload wrote just before failing to commit its row: nothing
    # in the database will ever mention them, so only a walk of the store finds
    # them.
    storage = FakeStorageBackend()
    key = f"{TENANT.tenant_id}/{uuid4()}/{uuid4()}"
    await storage.put(key, DATA)
    await _age(storage, key, minutes=90)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE)

    assert deleted == 1
    assert storage.objects == {}


async def test_an_upload_in_flight_is_protected_by_the_grace_window(db_sessionmaker) -> None:
    # An upload writes its bytes and commits the row moments later. Between the
    # two it looks exactly like garbage, so a sweep with no grace window would
    # delete the file out from under a request that is about to succeed.
    storage = FakeStorageBackend()
    key = f"{TENANT.tenant_id}/{uuid4()}/{uuid4()}"
    await storage.put(key, DATA)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE)

    assert deleted == 0
    assert key in storage.objects


async def test_a_referenced_blob_survives_however_old_it_is(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _age(storage, attachment.storage_key, minutes=60 * 24 * 365)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE)

    assert deleted == 0
    assert attachment.storage_key in storage.objects


async def test_a_referenced_blob_of_another_tenant_survives(db_sessionmaker) -> None:
    # The referenced-key query spans tenants on purpose. If it did not, the
    # sweep would consider every other tenant's files unreferenced and delete
    # all of them - the single worst thing this job could do.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker, OTHER_TENANT)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id, tenant=OTHER_TENANT)
    await _age(storage, attachment.storage_key, minutes=90)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE)

    assert deleted == 0
    assert attachment.storage_key in storage.objects


async def test_the_sweep_never_touches_a_key_this_app_did_not_mint(db_sessionmaker) -> None:
    # Deleting files purely because a table does not mention them is a dangerous
    # shape. Restricting it to the exact {uuid}/{uuid}/{uuid} keys the app mints
    # means a stray file under the storage root is safe by construction.
    storage = FakeStorageBackend()
    for key in (
        "notes.txt",
        f"{TENANT.tenant_id}/loose.bin",
        f"{TENANT.tenant_id}/{uuid4()}/{uuid4()}/deeper.bin",
        f"{TENANT.tenant_id}/not-a-uuid/{uuid4()}",
    ):
        await storage.put(key, DATA)
        await _age(storage, key, minutes=90)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE)

    assert deleted == 0
    assert len(storage.objects) == 4


async def test_one_unaddressable_object_does_not_truncate_the_sweep(db_sessionmaker) -> None:
    # A symlink named as a UUID resolves out of the storage root, so it looks
    # minted but the backend refuses to address it. Raised, that would abort the
    # walk and strand every orphan behind it - hourly, forever.
    poison = f"{TENANT.tenant_id}/{uuid4()}/{uuid4()}"

    class _OnePoisonKey(FakeStorageBackend):
        async def delete(self, key: str) -> None:
            if key == poison:
                raise InvalidStorageKeyError(key)
            await super().delete(key)

    storage = _OnePoisonKey()
    good = f"{TENANT.tenant_id}/{uuid4()}/{uuid4()}"
    for key in (poison, good):
        await storage.put(key, DATA)
        await _age(storage, key, minutes=90)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE)

    assert deleted == 1
    assert list(storage.objects) == [poison]


def test_only_the_canonical_uuid_spelling_counts_as_a_minted_key() -> None:
    # UUID() also parses the braced and urn: spellings, whose punctuation the
    # storage seam's key validator rejects. If this fence accepted them, the
    # sweep would nominate candidates the seam then refuses to delete.
    raw = "0d8f2b1e-4c3a-4f9e-9a1b-2c3d4e5f6071"

    assert _is_minted_key(f"{raw}/{raw}/{raw}") is True
    assert _is_minted_key(f"{raw}/{raw}/urn:uuid:{raw}") is False
    assert _is_minted_key(f"{raw}/{raw}/{{{raw}}}") is False
    assert _is_minted_key(f"{raw}/{raw}/{raw.replace('-', '')}") is False


async def test_the_blob_sweep_batches_without_losing_any(db_sessionmaker) -> None:
    # More keys than one IN (...) query is allowed to carry, so the batching
    # boundary is exercised rather than assumed.
    storage = FakeStorageBackend()
    for _ in range(505):
        key = f"{TENANT.tenant_id}/{uuid4()}/{uuid4()}"
        await storage.put(key, DATA)
        await _age(storage, key, minutes=90)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE)

    assert deleted == 505
    assert storage.objects == {}


# ----------------------------------------------------------------- both, together


async def test_a_collection_pass_reclaims_rows_and_blobs(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    tombstone = await _attachment(db_sessionmaker, storage, None)
    orphan_key = f"{TENANT.tenant_id}/{uuid4()}/{uuid4()}"
    await storage.put(orphan_key, DATA)
    await _age(storage, orphan_key, minutes=90)
    conversation_id = await _conversation(db_sessionmaker)
    live = await _attachment(db_sessionmaker, storage, conversation_id)

    async with db_sessionmaker() as session:
        collected = await collect_garbage(
            session,
            storage,
            make_memory_store(),
            grace=GRACE,
            # Retention off: these two cases are about the reclaim passes.
            generated_retention=None,
            upload_retention=None,
        )

    assert collected == GarbageCollected(rows=1, blobs=1)
    assert list(storage.objects) == [live.storage_key]
    assert tombstone.storage_key not in storage.objects


async def test_a_collection_pass_survives_a_storage_outage(db_sessionmaker) -> None:
    # The cron's contract is that the next tick tries again, so an outage must
    # be reported and shrugged off, never raised into arq as a job failure.
    storage = _DeadStorage()
    await _attachment(db_sessionmaker, storage, None)

    async with db_sessionmaker() as session:
        collected = await collect_garbage(
            session,
            storage,
            make_memory_store(),
            grace=GRACE,
            # Retention off: these two cases are about the reclaim passes.
            generated_retention=None,
            upload_retention=None,
        )

    assert collected == GarbageCollected(rows=0, blobs=0)
    assert len(await _rows(db_sessionmaker)) == 1


# ---------------------------------------------------------------- the arq adapters


def _ctx(db_sessionmaker, storage, *, job_try: int = 1) -> dict:
    return {
        "settings": get_settings(),
        "db_sessionmaker": db_sessionmaker,
        "storage": storage,
        "vector_store": make_memory_store(),
        "job_try": job_try,
    }


async def test_the_purge_job_reclaims_the_ids_it_was_given(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    attachment = await _attachment(db_sessionmaker, storage, None)

    purged = await purge_deleted_attachments(
        _ctx(db_sessionmaker, storage),
        [str(attachment.id)],
        str(TENANT.tenant_id),
        str(TENANT.user_id),
    )

    assert purged == 1
    assert storage.objects == {}


async def test_the_purge_job_retries_a_storage_outage(db_sessionmaker) -> None:
    # arq requeues a job ONLY when it raises Retry: a plain exception ends it for
    # good. With attempts left, a transient outage must raise Retry or the files
    # would silently never be collected by this path.
    storage = _DeadStorage()
    attachment = await _attachment(db_sessionmaker, storage, None)

    with pytest.raises(Retry):
        await purge_deleted_attachments(
            _ctx(db_sessionmaker, storage),
            [str(attachment.id)],
            str(TENANT.tenant_id),
            str(TENANT.user_id),
        )


async def test_the_last_purge_attempt_gives_up_and_leaves_it_to_the_sweep(
    db_sessionmaker,
) -> None:
    storage = _DeadStorage()
    attachment = await _attachment(db_sessionmaker, storage, None)

    with pytest.raises(StorageUnavailableError):
        await purge_deleted_attachments(
            _ctx(db_sessionmaker, storage, job_try=MAX_TRIES),
            [str(attachment.id)],
            str(TENANT.tenant_id),
            str(TENANT.user_id),
        )

    # Nothing is lost by giving up: the row still has no conversation, which is
    # precisely what the periodic sweep looks for.
    rows = await _rows(db_sessionmaker)
    assert [row.conversation_id for row in rows] == [None]


async def test_the_gc_cron_reclaims_rows_and_blobs(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    await _attachment(db_sessionmaker, storage, None)
    orphan_key = f"{TENANT.tenant_id}/{uuid4()}/{uuid4()}"
    await storage.put(orphan_key, DATA)
    await _age(storage, orphan_key, minutes=60 * 24)

    reclaimed = await collect_attachment_garbage(_ctx(db_sessionmaker, storage))

    assert reclaimed == 2
    assert storage.objects == {}


# ------------------------------------------------------------- stalled uploads


STALL = timedelta(minutes=30)


async def _stall(db_sessionmaker, attachment: Attachment, *, status: str, minutes: int) -> None:
    """Leave a row mid-flight and backdate it, exactly as a killed worker does."""
    async with db_sessionmaker() as session:
        row = await session.get(Attachment, attachment.id)
        assert row is not None
        row.status = status
        row.updated_at = datetime.now(UTC) - timedelta(minutes=minutes)
        session.add(row)
        await session.commit()


async def _status(db_sessionmaker, attachment: Attachment) -> tuple[str, str | None]:
    async with db_sessionmaker() as session:
        row = await session.get(Attachment, attachment.id)
        assert row is not None
        return row.status, row.error


@pytest.mark.parametrize("status", [ATTACHMENT_PENDING, ATTACHMENT_PARSING])
async def test_an_upload_nothing_is_working_on_is_eventually_failed(
    db_sessionmaker, status: str
) -> None:
    # Both halves of the mid-flight window strand a client: `pending` is a job no
    # worker ever claimed, `parsing` is one that died holding the row.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _stall(db_sessionmaker, attachment, status=status, minutes=45)

    async with db_sessionmaker() as session:
        failed = await fail_stalled_attachments(session, stall=STALL)

    assert failed == 1
    state, error = await _status(db_sessionmaker, attachment)
    assert state == ATTACHMENT_FAILED
    # The reason has to be worth showing: it is the only thing the person who
    # uploaded the file ever sees, and it must point at the way out.
    assert error is not None
    assert "again" in error


async def test_an_upload_still_within_the_window_is_left_alone(db_sessionmaker) -> None:
    # A single parse attempt may legitimately run for the whole job timeout, and
    # reaping one mid-embed would report a failure that never happened.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _stall(db_sessionmaker, attachment, status=ATTACHMENT_PARSING, minutes=10)

    async with db_sessionmaker() as session:
        assert await fail_stalled_attachments(session, stall=STALL) == 0

    assert (await _status(db_sessionmaker, attachment))[0] == ATTACHMENT_PARSING


@pytest.mark.parametrize("status", [ATTACHMENT_READY, ATTACHMENT_FAILED])
async def test_a_settled_upload_is_never_reopened(db_sessionmaker, status: str) -> None:
    # Only the mid-flight statuses are the sweep's business. A file that finished
    # is finished no matter how long ago, and re-failing an old `ready` row would
    # silently strip a working attachment out of a conversation.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _stall(db_sessionmaker, attachment, status=status, minutes=600)

    async with db_sessionmaker() as session:
        assert await fail_stalled_attachments(session, stall=STALL) == 0

    assert (await _status(db_sessionmaker, attachment))[0] == status


async def test_a_tombstone_is_left_for_the_purge(db_sessionmaker) -> None:
    # Its conversation is gone, so the row is about to be deleted outright.
    # Dressing it up as a failure first would be noise nobody will ever read.
    storage = FakeStorageBackend()
    attachment = await _attachment(db_sessionmaker, storage, None)
    await _stall(db_sessionmaker, attachment, status=ATTACHMENT_PENDING, minutes=600)

    async with db_sessionmaker() as session:
        assert await fail_stalled_attachments(session, stall=STALL) == 0

    assert (await _status(db_sessionmaker, attachment))[0] == ATTACHMENT_PENDING


async def test_a_stalled_row_that_finishes_late_heals(db_sessionmaker) -> None:
    # The property that makes reaping safe to get wrong. A job that lands after
    # the sweep gave up on it still sets `ready` and clears the error, so an
    # early failure costs a moment of a wrong label - never a lost file.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _stall(db_sessionmaker, attachment, status=ATTACHMENT_PARSING, minutes=45)

    async with db_sessionmaker() as session:
        assert await fail_stalled_attachments(session, stall=STALL) == 1

    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        row = await repo.get(attachment.id)
        assert row is not None
        await repo.set_status(row, ATTACHMENT_READY)
        await repo.commit()

    assert await _status(db_sessionmaker, attachment) == (ATTACHMENT_READY, None)


async def test_a_row_whose_job_is_still_queued_is_left_alone(db_sessionmaker) -> None:
    # THE case that must not misfire. A worker coming back to a backlog works it
    # off a few jobs at a time, so a row can sit untouched for an hour purely
    # because it is tenth in line. Failing it there is not a harmless early label:
    # the row tells the user to upload the file again, re-uploading deliberately
    # skips the dedupe check for failed rows, and the queued job then lands too -
    # leaving two rows, two blobs, and one file citable twice as two sources.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _stall(db_sessionmaker, attachment, status=ATTACHMENT_PENDING, minutes=600)

    async def still_queued(attachment_id: UUID) -> bool:
        return True

    async with db_sessionmaker() as session:
        failed = await fail_stalled_attachments(session, stall=STALL, has_live_job=still_queued)

    assert failed == 0
    assert (await _status(db_sessionmaker, attachment))[0] == ATTACHMENT_PENDING


async def test_a_row_the_queue_has_forgotten_is_failed(db_sessionmaker) -> None:
    # The other side of the same question: no job exists, so nothing is ever
    # coming back to this row. That is the whole reason the sweep exists.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _stall(db_sessionmaker, attachment, status=ATTACHMENT_PENDING, minutes=45)

    async def forgotten(attachment_id: UUID) -> bool:
        return False

    async with db_sessionmaker() as session:
        failed = await fail_stalled_attachments(session, stall=STALL, has_live_job=forgotten)

    assert failed == 1
    assert (await _status(db_sessionmaker, attachment))[0] == ATTACHMENT_FAILED


def test_only_a_job_the_queue_will_still_run_counts_as_live() -> None:
    # The whole guard rests on this mapping. `complete` is as dead as `not_found`:
    # a job that ran and left the row mid-flight is not coming back to it.
    assert JobStatus.queued in _LIVE_JOB_STATUSES
    assert JobStatus.deferred in _LIVE_JOB_STATUSES
    assert JobStatus.in_progress in _LIVE_JOB_STATUSES
    assert JobStatus.complete not in _LIVE_JOB_STATUSES
    assert JobStatus.not_found not in _LIVE_JOB_STATUSES


async def test_an_unreachable_queue_leaves_the_row_alone(
    db_sessionmaker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "Unknown" has to read as "alive". A Redis blip is not evidence that a file
    # was abandoned, and failing rows on the strength of one would turn a
    # momentary outage into a pile of files the user has to upload again.
    async def unreachable(self) -> JobStatus:
        raise ConnectionError("redis is down")

    monkeypatch.setattr("arq.jobs.Job.status", unreachable)

    assert await _has_live_ingest_job(object(), uuid4()) is True


async def test_the_worker_job_reaps_with_the_configured_window(
    db_sessionmaker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End to end through the cron: the queue has no job for this attachment, so
    # nothing is ever going to finish it.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id)
    await _stall(db_sessionmaker, attachment, status=ATTACHMENT_PENDING, minutes=45)

    async def forgotten(redis, attachment_id: UUID) -> bool:
        return False

    monkeypatch.setattr("app.workers.main._has_live_ingest_job", forgotten)
    ctx = {"settings": get_settings(), "db_sessionmaker": db_sessionmaker, "redis": object()}

    assert await reap_stalled_attachments(ctx) == 1
    assert (await _status(db_sessionmaker, attachment))[0] == ATTACHMENT_FAILED
