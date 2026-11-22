"""Two-tier attachment retention: the windows, the tombstone, and the fences.

The sweep under test is the only writer of the ``expired`` status. It is a POLICY
rather than a repair - it acts on files that worked exactly as intended - so what
these tests pin is mostly what it must NOT touch: a file inside its window, a
file whose tier is switched off, a row another sweep owns, and every row at all
while a store is unreachable.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.exc import OperationalError

from app.core.config import Settings, get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
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
    Conversation,
)
from app.providers.rag.base import VectorStore, VectorStoreUnavailableError
from app.providers.rag.embedder import FakeEmbedder
from app.repositories.attachments import AttachmentRepository
from app.schemas.rag import RagIngestItem
from app.services.attachment_gc import (
    GarbageCollected,
    collect_garbage,
    expire_aged_attachments,
    retention_window,
)
from app.services.rag_service import ingest_structured
from app.storage.base import InvalidStorageKeyError, StorageUnavailableError
from app.storage.local import FakeStorageBackend
from app.workers.main import collect_attachment_garbage
from tests.conftest import make_memory_store

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")
OTHER_TENANT = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")

DATA = b"the bytes of a file that has been sitting here a while"
INDEXED_TEXT = "The turbine was taken offline after the bearing temperature spiked."
EMBEDDER = FakeEmbedder(dim=32)

# The two windows these tests measure against. Days, because the policy is
# configured in days and the boundary arithmetic has to be exercised in the same
# unit an operator writes.
GENERATED = timedelta(days=30)
UPLOADS = timedelta(days=90)

# Enough of a gap to give rows a stable order under the sweep's ORDER BY, so a
# test about WHICH row an interrupted pass stopped on means something.
ONE_DAY = timedelta(days=1)


class _DeadStorage(FakeStorageBackend):
    """A store whose deletes always fail, standing in for one that is down."""

    async def delete(self, key: str) -> None:
        raise StorageUnavailableError("storage is down")


class _DeadVectors:
    """A vector store that cannot be reached (only the sweep's one call is used)."""

    async def delete_record(
        self, tenant_id: str, record_id: str, *, session_id: str | None = None
    ) -> int:
        raise VectorStoreUnavailableError("the vector store is unavailable")


class _FailingCommitSession:
    """A real session whose Nth commit raises, standing in for the database going away.

    A proxy rather than a monkeypatch so every other call - the selection query,
    the rollback the sweep must still be able to make - runs against the genuine
    session. Only the one step being tested is broken.
    """

    def __init__(self, session, *, fail_on: int) -> None:
        self._session = session
        self._fail_on = fail_on
        self.commits = 0

    def __getattr__(self, name: str):
        return getattr(self._session, name)

    async def commit(self) -> None:
        self.commits += 1
        if self.commits >= self._fail_on:
            raise OperationalError("COMMIT", None, Exception("the database went away"))
        await self._session.commit()


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
    age: timedelta,
    origin: str = ATTACHMENT_ORIGIN_GENERATED,
    status: str = ATTACHMENT_READY,
    tenant: TenantContext = TENANT,
    content_hash: str | None = None,
    error: str | None = None,
) -> Attachment:
    """Store bytes and commit a row of the given age, as the two write paths do."""
    attachment_id = uuid4()
    storage_key = f"{tenant.tenant_id}/{conversation_id or uuid4()}/{attachment_id}"
    await storage.put(storage_key, DATA, content_type="text/plain")
    async with db_sessionmaker() as session:
        attachment = Attachment(
            id=attachment_id,
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            conversation_id=conversation_id,
            filename="quarterly-report.xlsx",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            size_bytes=len(DATA),
            storage_key=storage_key,
            status=status,
            origin=origin,
            content_hash=content_hash,
            error=error,
            created_at=datetime.now(UTC) - age,
        )
        session.add(attachment)
        await session.commit()
        return attachment


async def _index(
    store: VectorStore, attachment: Attachment, tenant: TenantContext = TENANT
) -> None:
    """Index the file's text exactly as the parse job does: session-scoped, no TTL."""
    await ingest_structured(
        source_type="upload",
        tenant=tenant,
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


async def _indexed_chunks(
    store: VectorStore, conversation_id: UUID, tenant: TenantContext = TENANT
) -> int:
    """How many of that conversation's chunks the store still holds."""
    hits = await store.query(
        "session",
        EMBEDDER.model,
        str(tenant.tenant_id),
        await EMBEDDER.embed_query(INDEXED_TEXT),
        top_k=50,
        session_id=str(conversation_id),
    )
    return len(hits)


async def _reload(db_sessionmaker, attachment: Attachment) -> Attachment:
    async with db_sessionmaker() as session:
        row = await session.get(Attachment, attachment.id)
        assert row is not None
        return row


async def _sweep(
    db_sessionmaker,
    storage,
    store,
    *,
    generated: timedelta | None = GENERATED,
    uploads: timedelta | None = None,
    now: datetime | None = None,
) -> int:
    async with db_sessionmaker() as session:
        return await expire_aged_attachments(
            session,
            storage,
            store,
            generated_retention=generated,
            upload_retention=uploads,
            now=now,
        )


# ------------------------------------------------------------------ the windows


async def test_a_generated_file_past_its_window_expires(db_sessionmaker) -> None:
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED + timedelta(hours=1)
    )

    assert await _sweep(db_sessionmaker, storage, store) == 1
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_EXPIRED


async def test_a_generated_file_inside_its_window_is_untouched(db_sessionmaker) -> None:
    # The whole point of a window: a file that is 29 days old is a file the
    # deployment promised to keep for another day.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED - timedelta(days=1)
    )

    assert await _sweep(db_sessionmaker, storage, store) == 0
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_READY
    assert attachment.storage_key in storage.objects


async def test_a_file_exactly_at_its_window_survives_one_instant_longer(db_sessionmaker) -> None:
    # The comparison is strict, so a file whose age is EXACTLY the window is
    # kept and the next tick takes it. Measured against the row's own stored
    # `created_at` rather than a clock read beside it, because a boundary this
    # test claims to pin cannot be a few microseconds to one side of it - that
    # would pass just as happily against a `<=`.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, storage, conversation_id, age=GENERATED)
    born = (await _reload(db_sessionmaker, attachment)).created_at

    assert await _sweep(db_sessionmaker, storage, store, now=born + GENERATED) == 0
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_READY

    swept = await _sweep(
        db_sessionmaker, storage, store, now=born + GENERATED + timedelta(microseconds=1)
    )

    assert swept == 1
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_EXPIRED


async def test_an_upload_past_the_configured_window_expires(db_sessionmaker) -> None:
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=UPLOADS + timedelta(days=1),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )

    assert await _sweep(db_sessionmaker, storage, store, uploads=UPLOADS) == 1
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_EXPIRED


async def test_an_upload_inside_the_configured_window_is_untouched(db_sessionmaker) -> None:
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=UPLOADS - timedelta(days=1),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )

    assert await _sweep(db_sessionmaker, storage, store, uploads=UPLOADS) == 0
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_READY


async def test_an_upload_is_kept_forever_while_no_window_is_configured(db_sessionmaker) -> None:
    # THE shipped default, and the one this whole tier exists to protect: a person
    # chose to send this file, so nothing ages it out until an operator says so.
    # Ten years old, and the sweep still has no business with it.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    uploaded = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=timedelta(days=3650),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    pasted = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=timedelta(days=3650),
        origin=ATTACHMENT_ORIGIN_PASTE,
    )

    assert await _sweep(db_sessionmaker, storage, store, uploads=None) == 0
    assert (await _reload(db_sessionmaker, uploaded)).status == ATTACHMENT_READY
    assert (await _reload(db_sessionmaker, pasted)).status == ATTACHMENT_READY
    assert len(storage.objects) == 2


async def test_a_generated_file_is_kept_forever_when_its_window_is_off(db_sessionmaker) -> None:
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, storage, conversation_id, age=timedelta(days=3650)
    )

    assert await _sweep(db_sessionmaker, storage, store, generated=None) == 0
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_READY


async def test_both_windows_off_does_no_work_at_all(db_sessionmaker) -> None:
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    await _attachment(db_sessionmaker, storage, conversation_id, age=timedelta(days=3650))

    assert await _sweep(db_sessionmaker, storage, store, generated=None, uploads=None) == 0
    assert len(storage.objects) == 1


def test_zero_and_unset_both_read_as_no_retention() -> None:
    # Zero has to mean off rather than "expire immediately": the other reading
    # turns one mistyped digit into every stored file deleted on the next tick.
    assert retention_window(None) is None
    assert retention_window(0) is None
    assert retention_window(30) == timedelta(days=30)


# ------------------------------------------------------------ per-origin selection


async def test_an_upload_is_untouched_when_only_the_generated_window_is_set(
    db_sessionmaker,
) -> None:
    # The tiers are separate policies, not one window with a default. With uploads
    # unconfigured, a file a person sent is out of scope however old it is - even
    # while a generated file beside it, of the very same age, is expiring.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    aged = GENERATED + timedelta(days=1)
    generated = await _attachment(db_sessionmaker, storage, conversation_id, age=aged)
    uploaded = await _attachment(
        db_sessionmaker, storage, conversation_id, age=aged, origin=ATTACHMENT_ORIGIN_UPLOAD
    )
    pasted = await _attachment(
        db_sessionmaker, storage, conversation_id, age=aged, origin=ATTACHMENT_ORIGIN_PASTE
    )

    assert await _sweep(db_sessionmaker, storage, store) == 1
    assert (await _reload(db_sessionmaker, generated)).status == ATTACHMENT_EXPIRED
    assert (await _reload(db_sessionmaker, uploaded)).status == ATTACHMENT_READY
    assert (await _reload(db_sessionmaker, pasted)).status == ATTACHMENT_READY
    assert sorted(storage.objects) == sorted([uploaded.storage_key, pasted.storage_key])


async def test_a_generated_file_is_untouched_by_the_upload_window_alone(db_sessionmaker) -> None:
    # And the mirror image, which is the case a dropped origin filter would break
    # in the more surprising direction: a short upload policy must not reach back
    # and shorten the generated tier's own, longer, window.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    generated = await _attachment(db_sessionmaker, storage, conversation_id, age=timedelta(days=10))
    uploaded = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=timedelta(days=10),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )

    swept = await _sweep(db_sessionmaker, storage, store, generated=None, uploads=timedelta(days=7))

    assert swept == 1
    assert (await _reload(db_sessionmaker, uploaded)).status == ATTACHMENT_EXPIRED
    assert (await _reload(db_sessionmaker, generated)).status == ATTACHMENT_READY


async def test_a_generated_file_no_message_references_needs_no_special_case(
    db_sessionmaker,
) -> None:
    # A turn can store a generated file and then persist no message (an empty or
    # abandoned reply), leaving a row nothing points at on a conversation that is
    # still very much alive - so neither the tombstone purge nor the blob walk is
    # ever coming for it. The generated window collects it as an ordinary
    # generated file, which is why there is no rule here about references.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED + timedelta(days=1)
    )

    assert await _sweep(db_sessionmaker, storage, store) == 1
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_EXPIRED
    assert attachment.storage_key not in storage.objects


# ------------------------------------------------------------ what a row it touches keeps


async def test_expiry_removes_the_bytes_and_the_indexed_text_but_keeps_the_row(
    db_sessionmaker,
) -> None:
    # The shape of the tombstone, in one test. The bytes and the passages go - a
    # chunk of an expired file must not be retrievable or citable, or an answer
    # would quote a document nobody can open - and every piece of metadata a UI
    # needs to say "this was here and has aged out" stays exactly as it was.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=UPLOADS + timedelta(days=1),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    await _index(store, attachment)
    assert await _indexed_chunks(store, conversation_id) == 1

    assert await _sweep(db_sessionmaker, storage, store, uploads=UPLOADS) == 1

    assert await _indexed_chunks(store, conversation_id) == 0
    assert storage.objects == {}
    row = await _reload(db_sessionmaker, attachment)
    assert row.status == ATTACHMENT_EXPIRED
    assert row.filename == attachment.filename
    assert row.content_type == attachment.content_type
    assert row.size_bytes == attachment.size_bytes
    assert row.conversation_id == conversation_id
    assert row.origin == ATTACHMENT_ORIGIN_UPLOAD
    # Nothing failed, so nothing is explained away as a failure: the status is
    # the whole story, and `error` is the column a UI reads for a fault.
    assert row.error is None
    assert row.updated_at > row.created_at


async def test_expiry_of_one_tenants_file_leaves_anothers_alone(db_sessionmaker) -> None:
    # The sweep spans tenants on purpose (it is maintenance, and no client picks
    # what it touches), so it reads the tenant off each row - and the tenant is
    # what addresses the passages. Getting it from anywhere else would drop the
    # wrong tenant's chunks, or none at all.
    storage, store = FakeStorageBackend(), make_memory_store()
    mine = await _conversation(db_sessionmaker)
    theirs = await _conversation(db_sessionmaker, OTHER_TENANT)
    aged = GENERATED + timedelta(days=1)
    ours = await _attachment(db_sessionmaker, storage, mine, age=aged)
    hers = await _attachment(db_sessionmaker, storage, theirs, age=aged, tenant=OTHER_TENANT)
    fresh = await _attachment(
        db_sessionmaker, storage, theirs, age=timedelta(days=1), tenant=OTHER_TENANT
    )
    await _index(store, ours)
    await _index(store, hers, OTHER_TENANT)

    assert await _sweep(db_sessionmaker, storage, store) == 2

    assert (await _reload(db_sessionmaker, ours)).status == ATTACHMENT_EXPIRED
    assert (await _reload(db_sessionmaker, hers)).status == ATTACHMENT_EXPIRED
    assert (await _reload(db_sessionmaker, fresh)).status == ATTACHMENT_READY
    assert await _indexed_chunks(store, mine) == 0
    assert await _indexed_chunks(store, theirs, OTHER_TENANT) == 0
    assert list(storage.objects) == [fresh.storage_key]


# ------------------------------------------------------- rows other sweeps own


async def test_a_row_mid_ingest_is_never_expired(db_sessionmaker) -> None:
    # `pending` and `parsing` belong to the job still working on them and to the
    # stalled-upload sweep. Selecting one here is what would let an expiry race an
    # ingest: the sweep would drop the chunks, the ingest would write them a
    # moment later, and nothing would ever come back for them - an attachment's
    # passages carry no TTL.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    aged = GENERATED + timedelta(days=1)
    pending = await _attachment(
        db_sessionmaker, storage, conversation_id, age=aged, status=ATTACHMENT_PENDING
    )
    parsing = await _attachment(
        db_sessionmaker, storage, conversation_id, age=aged, status=ATTACHMENT_PARSING
    )

    assert await _sweep(db_sessionmaker, storage, store) == 0
    assert (await _reload(db_sessionmaker, pending)).status == ATTACHMENT_PENDING
    assert (await _reload(db_sessionmaker, parsing)).status == ATTACHMENT_PARSING
    assert len(storage.objects) == 2


async def test_a_failed_upload_ages_out_on_the_same_clock_as_a_readable_one(
    db_sessionmaker,
) -> None:
    # A failed row is sitting on its bytes - written before the row was committed
    # and never removed by the failure - and those bytes are downloadable. A file
    # somebody can still fetch is a file a retention policy has to be able to
    # reach, so "we kept the ones we could not read" is not a policy this offers.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    readable = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=UPLOADS + timedelta(days=1),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    unreadable = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=UPLOADS + timedelta(days=1),
        origin=ATTACHMENT_ORIGIN_PASTE,
        status=ATTACHMENT_FAILED,
        error="This file could not be read.",
    )

    assert await _sweep(db_sessionmaker, storage, store, uploads=UPLOADS) == 2
    assert (await _reload(db_sessionmaker, readable)).status == ATTACHMENT_EXPIRED
    assert (await _reload(db_sessionmaker, unreadable)).status == ATTACHMENT_EXPIRED
    # Both blobs reclaimed, which is the point of reaching them at all.
    assert storage.objects == {}


async def test_a_failed_upload_at_its_boundary_survives_one_instant_longer(
    db_sessionmaker,
) -> None:
    # The same strict comparison a readable file gets - the widened selection
    # admits a status, it does not soften the window.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=UPLOADS,
        origin=ATTACHMENT_ORIGIN_UPLOAD,
        status=ATTACHMENT_FAILED,
    )
    born = (await _reload(db_sessionmaker, attachment)).created_at

    assert await _sweep(db_sessionmaker, storage, store, uploads=UPLOADS, now=born + UPLOADS) == 0
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_FAILED

    swept = await _sweep(
        db_sessionmaker,
        storage,
        store,
        uploads=UPLOADS,
        now=born + UPLOADS + timedelta(microseconds=1),
    )

    assert swept == 1
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_EXPIRED


async def test_a_failed_upload_is_kept_forever_while_no_window_is_configured(
    db_sessionmaker,
) -> None:
    # The shipped default. Nothing ages out a person's files until an operator
    # says so, and being unreadable does not make one theirs any less.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=timedelta(days=3650),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
        status=ATTACHMENT_FAILED,
        error="This file could not be read.",
    )

    assert await _sweep(db_sessionmaker, storage, store) == 0
    row = await _reload(db_sessionmaker, attachment)
    assert row.status == ATTACHMENT_FAILED
    assert row.error == "This file could not be read."
    assert storage.objects


async def test_expiring_a_failed_row_clears_the_reason_it_failed(db_sessionmaker) -> None:
    # A chip has ONE line. Any row carrying an `error` renders as a PROBLEM
    # rather than a tombstone, so a surviving parse verdict would present
    # months-old news about a file that has just aged out perfectly normally.
    # The verdict described bytes; the bytes are gone; "gone by policy" is what
    # is left to say, and it is what a readable file's tombstone says too.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=UPLOADS + timedelta(days=1),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
        status=ATTACHMENT_FAILED,
        error="The file is not a readable spreadsheet.",
    )

    assert await _sweep(db_sessionmaker, storage, store, uploads=UPLOADS) == 1

    row = await _reload(db_sessionmaker, attachment)
    assert row.status == ATTACHMENT_EXPIRED
    # Indistinguishable from the tombstone a readable file leaves.
    assert row.error is None


async def test_a_tombstone_is_left_for_the_purge(db_sessionmaker) -> None:
    # Its conversation is gone, so the row is about to be deleted outright.
    # Expiring it first would empty a row that is being dropped whole, and an
    # expired tombstone is a disclosure with nowhere left to appear.
    storage, store = FakeStorageBackend(), make_memory_store()
    attachment = await _attachment(
        db_sessionmaker, storage, None, age=GENERATED + timedelta(days=1)
    )

    assert await _sweep(db_sessionmaker, storage, store) == 0
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_READY


# ------------------------------------------------------ re-runnable, and safe to stop


async def test_a_second_pass_finds_nothing_left_to_do(db_sessionmaker) -> None:
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED + timedelta(days=1)
    )

    assert await _sweep(db_sessionmaker, storage, store) == 1
    assert await _sweep(db_sessionmaker, storage, store) == 0
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_EXPIRED


async def test_a_pass_interrupted_after_the_blob_went_finishes_on_the_next_run(
    db_sessionmaker,
) -> None:
    # A crash between deleting the bytes and committing the status leaves a READY
    # row over a blob that is already gone. Because the status flips LAST, the row
    # is still a candidate, and every step ahead of it is idempotent: the missing
    # blob deletes as a no-op by the storage seam's contract, and matching no
    # chunks costs one count.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED + timedelta(days=1)
    )
    await storage.delete(attachment.storage_key)

    assert await _sweep(db_sessionmaker, storage, store) == 1
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_EXPIRED


async def test_a_storage_outage_expires_nothing(db_sessionmaker) -> None:
    # The ordering guarantee, observed: a row is committed as `expired` only once
    # its contents are actually gone. A row flipped over bytes still sitting in
    # the store would be a lie no later pass could find, because the status is the
    # only thing that keeps it in the selection.
    storage, store = _DeadStorage(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED + timedelta(days=1)
    )

    assert await _sweep(db_sessionmaker, storage, store) == 0
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_READY
    assert attachment.storage_key in storage.objects


async def test_a_vector_outage_expires_nothing(db_sessionmaker) -> None:
    # The twin, and the more dangerous of the two: an expired row whose passages
    # survived would still be quotable, and the answer would cite a file the user
    # is told has aged out.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED + timedelta(days=1)
    )

    assert await _sweep(db_sessionmaker, storage, _DeadVectors()) == 0
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_READY
    assert attachment.storage_key in storage.objects


async def test_an_outage_partway_through_a_batch_keeps_what_it_finished(db_sessionmaker) -> None:
    # Why the commit is per ROW and not per batch. The vector and blob deletes are
    # not transactional, so by the time the store fails on the second row the
    # FIRST row has already had its passages destroyed. Rolling that one back to
    # `ready` would leave a live, listed file that grounds and cites nothing while
    # claiming to be fine - so the row that was finished stays finished, and only
    # the interrupted one is left for the next pass.
    conversation_id = await _conversation(db_sessionmaker)
    store = make_memory_store()
    staging = FakeStorageBackend()
    aged = GENERATED + ONE_DAY
    # A day apart, so the sweep's ORDER BY fixes which row it reaches first -
    # rather than the test resting on two clock reads a microsecond apart.
    first = await _attachment(db_sessionmaker, staging, conversation_id, age=aged + ONE_DAY)
    second = await _attachment(db_sessionmaker, staging, conversation_id, age=aged)
    await _index(store, first)
    await _index(store, second)

    class _OneBadKey(FakeStorageBackend):
        async def delete(self, key: str) -> None:
            if key == second.storage_key:
                raise StorageUnavailableError("storage went away mid-batch")
            await super().delete(key)

    failing = _OneBadKey()
    failing.objects.update(staging.objects)

    assert await _sweep(db_sessionmaker, failing, store) == 1
    assert (await _reload(db_sessionmaker, first)).status == ATTACHMENT_EXPIRED
    # The interrupted row keeps its bytes and its status; the next pass finds it
    # again, because only the status takes a row out of the selection.
    assert (await _reload(db_sessionmaker, second)).status == ATTACHMENT_READY
    assert second.storage_key in failing.objects


async def test_the_interrupted_row_is_left_ready_but_unsearchable(db_sessionmaker) -> None:
    # The honest cost of the ordering, pinned rather than glossed. The passages
    # are dropped BEFORE the bytes, so a blob delete that raises rolls the row
    # back to `ready` with its indexed text already gone: a file that is listed,
    # inside its retention window, and grounds and cites NOTHING.
    #
    # The window is bounded - the row is still selectable, so the next successful
    # pass finishes it - and this ordering is the least bad of the three: dropping
    # the passages last would leave an `expired` row still quotable, and
    # committing the status first would strand the bytes forever. It is a real
    # degradation for a real reason, and a test that did not say so would be
    # asserting the docstring rather than the behaviour.
    conversation_id = await _conversation(db_sessionmaker)
    store = make_memory_store()
    staging = FakeStorageBackend()
    only = await _attachment(
        db_sessionmaker, staging, conversation_id, age=GENERATED + timedelta(days=1)
    )
    await _index(store, only)
    assert await _indexed_chunks(store, conversation_id) == 1

    storage = _DeadStorage()
    storage.objects.update(staging.objects)

    assert await _sweep(db_sessionmaker, storage, store) == 0

    row = await _reload(db_sessionmaker, only)
    assert row.status == ATTACHMENT_READY
    # Its bytes survive, so a download would still serve the file...
    assert only.storage_key in storage.objects
    # ...but its passages are gone, so retrieval finds nothing until a later pass
    # finishes the job. THIS is the window the ordering buys, stated out loud.
    assert await _indexed_chunks(store, conversation_id) == 0

    # And it self-heals: the store comes back, the row is still selectable
    # (nothing took it out of the selection), and the next pass completes it.
    healthy = FakeStorageBackend()
    healthy.objects.update(storage.objects)

    assert await _sweep(db_sessionmaker, healthy, store) == 1
    assert (await _reload(db_sessionmaker, only)).status == ATTACHMENT_EXPIRED
    assert healthy.objects == {}


async def test_an_unaddressable_key_does_not_wedge_the_sweep(db_sessionmaker) -> None:
    # A key no backend can address has no blob to delete and never will (a symlink
    # named as a UUID resolves out of the storage root). The row still expires:
    # nothing in this app can reach those bytes through the seam, so `expired` is
    # the honest status - and a row that could never leave the selection would be
    # retried every hour forever. Same call the tombstone purge makes.
    class _RefusingStorage(FakeStorageBackend):
        async def delete(self, key: str) -> None:
            raise InvalidStorageKeyError(key)

    storage = _RefusingStorage()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED + timedelta(days=1)
    )

    assert await _sweep(db_sessionmaker, storage, make_memory_store()) == 1
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_EXPIRED
    assert await _sweep(db_sessionmaker, storage, make_memory_store()) == 0


async def test_a_database_failure_at_the_commit_stops_the_pass_without_raising(
    db_sessionmaker, caplog
) -> None:
    # The commit is the one step whose failure is not a store outage, and it is
    # the step the whole ordering rests on. Unhandled it escapes as an arq job
    # error, past a docstring promising the opposite - over a database blip the
    # next tick would sail through.
    conversation_id = await _conversation(db_sessionmaker)
    storage, store = FakeStorageBackend(), make_memory_store()
    aged = GENERATED + timedelta(days=1)
    first = await _attachment(db_sessionmaker, storage, conversation_id, age=aged)
    second = await _attachment(db_sessionmaker, storage, conversation_id, age=aged + ONE_DAY)
    third = await _attachment(db_sessionmaker, storage, conversation_id, age=aged + 2 * ONE_DAY)

    async with db_sessionmaker() as session:
        # Oldest first (the sweep orders by created_at), so the SECOND commit is
        # `second`, and `third` is never reached.
        guarded = _FailingCommitSession(session, fail_on=2)
        with caplog.at_level(logging.INFO, logger="app.services.attachment_gc"):
            swept = await expire_aged_attachments(
                guarded,
                storage,
                store,
                generated_retention=GENERATED,
                upload_retention=None,
            )

    # Aborted, not raised, and the count credits only what actually committed.
    assert swept == 1
    assert (await _reload(db_sessionmaker, third)).status == ATTACHMENT_EXPIRED
    assert (await _reload(db_sessionmaker, second)).status == ATTACHMENT_READY
    assert (await _reload(db_sessionmaker, first)).status == ATTACHMENT_READY

    expired_lines = [
        r.message for r in caplog.records if r.message.startswith("Expired attachment")
    ]
    # One line, for the one row that really expired. The log is what an operator
    # reads to explain a missing file, so announcing a row whose commit then
    # failed would make it evidence of something that never happened.
    assert len(expired_lines) == 1
    assert str(third.id) in expired_lines[0]
    assert str(second.id) not in " ".join(expired_lines)
    assert any("database is unavailable" in r.message for r in caplog.records)


async def test_the_collection_pass_survives_a_database_failure(db_sessionmaker) -> None:
    # The same failure one layer up, and it has to originate OUTSIDE the retention
    # sweep to test anything: a commit that fails inside the sweep is already
    # handled there. So this one breaks the TOMBSTONE purge's commit, which runs
    # first and does not catch a database error itself - the only thing standing
    # between it and a failed arq job is collect_garbage's own handler.
    conversation_id = await _conversation(db_sessionmaker)
    storage, store = FakeStorageBackend(), make_memory_store()
    await _attachment(db_sessionmaker, storage, None, age=timedelta(minutes=5))
    aged = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED + timedelta(days=1)
    )

    async with db_sessionmaker() as session:
        collected = await collect_garbage(
            _FailingCommitSession(session, fail_on=1),
            storage,
            store,
            grace=timedelta(minutes=60),
            generated_retention=GENERATED,
            upload_retention=None,
        )

    # Reported, not raised, and nothing is credited.
    assert collected == GarbageCollected(rows=0, blobs=0, expired=0)
    assert (await _reload(db_sessionmaker, aged)).status == ATTACHMENT_READY


# ------------------------------------------------------------- and the dedupe it feeds


async def test_a_row_this_sweep_expired_is_no_longer_a_dedupe_match(db_sessionmaker) -> None:
    # The two halves joined, driving the real sweep into the real query rather
    # than a hand-written `expired` row. Once the bytes are gone the row must stop
    # being offered to the upload path: reusing it would hand somebody who just
    # uploaded a file a row whose file does not exist, and the ingest it
    # re-enqueues would find an empty store and mark THAT row failed - erasing the
    # honest "aged out" the tombstone existed to say.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=UPLOADS + timedelta(days=1),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
        content_hash="ab" * 32,
    )
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        assert (
            await repo.get_by_content_hash("ab" * 32, conversation_id=conversation_id) is not None
        )

    assert await _sweep(db_sessionmaker, storage, store, uploads=UPLOADS) == 1

    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        found = await repo.get_by_content_hash("ab" * 32, conversation_id=conversation_id)

    assert found is None
    assert (await _reload(db_sessionmaker, attachment)).status == ATTACHMENT_EXPIRED


# ---------------------------------------------------------- through the cron job


async def test_the_collection_pass_reports_expiry_apart_from_reclaims(db_sessionmaker) -> None:
    # An expired row is not a reclaimed one - it is still there, still listed -
    # so the pass counts them separately or an operator reading the log would
    # think rows had been deleted.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    tombstone = await _attachment(db_sessionmaker, storage, None, age=timedelta(minutes=5))
    aged = await _attachment(
        db_sessionmaker, storage, conversation_id, age=GENERATED + timedelta(days=1)
    )

    async with db_sessionmaker() as session:
        collected = await collect_garbage(
            session,
            storage,
            store,
            grace=timedelta(minutes=60),
            generated_retention=GENERATED,
            upload_retention=None,
        )

    assert collected == GarbageCollected(rows=1, blobs=0, expired=1)
    assert (await _reload(db_sessionmaker, aged)).status == ATTACHMENT_EXPIRED
    assert tombstone.storage_key not in storage.objects
    assert aged.storage_key not in storage.objects


async def test_the_gc_cron_expires_with_the_configured_windows(db_sessionmaker) -> None:
    # End to end through the job the worker actually schedules, so the settings
    # keys are wired to the sweep rather than merely declared.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    generated = await _attachment(db_sessionmaker, storage, conversation_id, age=timedelta(days=8))
    uploaded = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=timedelta(days=8),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    ctx = {
        "settings": Settings(
            _env_file=None,
            generated_file_retention_days=7,
            attachment_retention_days=None,
        ),
        "db_sessionmaker": db_sessionmaker,
        "storage": storage,
        "vector_store": store,
    }

    assert await collect_attachment_garbage(ctx) == 1
    assert (await _reload(db_sessionmaker, generated)).status == ATTACHMENT_EXPIRED
    assert (await _reload(db_sessionmaker, uploaded)).status == ATTACHMENT_READY


async def test_the_gc_cron_leaves_everything_alone_at_the_shipped_defaults(
    db_sessionmaker,
) -> None:
    # A deployment that configures nothing gets the 30-day generated window and no
    # upload policy at all, which is what makes this change non-breaking for
    # every file already in an existing store.
    storage, store = FakeStorageBackend(), make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    settings = Settings(_env_file=None)
    assert settings.generated_file_retention_days == 30
    assert settings.attachment_retention_days is None
    uploaded = await _attachment(
        db_sessionmaker,
        storage,
        conversation_id,
        age=timedelta(days=3650),
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    generated = await _attachment(db_sessionmaker, storage, conversation_id, age=timedelta(days=20))
    ctx = {
        "settings": settings,
        "db_sessionmaker": db_sessionmaker,
        "storage": storage,
        "vector_store": store,
    }

    assert await collect_attachment_garbage(ctx) == 0
    assert (await _reload(db_sessionmaker, uploaded)).status == ATTACHMENT_READY
    assert (await _reload(db_sessionmaker, generated)).status == ATTACHMENT_READY
