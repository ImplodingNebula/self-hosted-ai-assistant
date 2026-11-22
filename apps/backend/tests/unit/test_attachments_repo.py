"""AttachmentRepository tests: tenant scoping, lifecycle, and dedupe."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import (
    ATTACHMENT_EXPIRED,
    ATTACHMENT_FAILED,
    ATTACHMENT_ORIGIN_PASTE,
    ATTACHMENT_ORIGIN_UPLOAD,
    ATTACHMENT_PARSING,
    ATTACHMENT_PENDING,
    ATTACHMENT_READY,
    Attachment,
    Conversation,
)
from app.repositories.attachments import AttachmentRepository

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")
OTHER_TENANT = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")


async def _new_conversation(db_sessionmaker) -> UUID:
    """Create a real conversation: attachments carry a foreign key to one."""
    async with db_sessionmaker() as session:
        conversation = Conversation(tenant_id=TENANT.tenant_id, user_id=TENANT.user_id)
        session.add(conversation)
        await session.commit()
        return conversation.id


def _attachment(conversation_id: UUID, **over) -> Attachment:
    defaults = {
        "tenant_id": TENANT.tenant_id,
        "user_id": TENANT.user_id,
        "conversation_id": conversation_id,
        "filename": "notes.txt",
        "content_type": "text/plain",
        "size_bytes": 5,
        "storage_key": f"{TENANT.tenant_id}/{conversation_id}/{uuid4()}",
    }
    return Attachment(**{**defaults, **over})


async def test_new_attachment_starts_pending_and_uploaded(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        entity = repo.add(_attachment(conversation_id))
        await repo.commit()

    assert entity.status == ATTACHMENT_PENDING
    assert entity.origin == ATTACHMENT_ORIGIN_UPLOAD
    assert entity.error is None
    assert entity.content_hash is None


async def test_list_for_conversation_is_scoped_and_ordered(db_sessionmaker) -> None:
    mine = await _new_conversation(db_sessionmaker)
    theirs = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        repo.add(_attachment(mine, filename="first.txt"))
        repo.add(_attachment(mine, filename="second.txt"))
        repo.add(_attachment(theirs, filename="other-thread.txt"))
        await repo.commit()

    async with db_sessionmaker() as session:
        rows = await AttachmentRepository(session, TENANT).list_for_conversation(mine)

    assert [row.filename for row in rows] == ["first.txt", "second.txt"]


async def test_another_tenant_cannot_see_the_row(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        entity = repo.add(_attachment(conversation_id))
        await repo.commit()
        attachment_id = entity.id

    async with db_sessionmaker() as session:
        # A foreign-tenant row is indistinguishable from a missing one.
        assert await AttachmentRepository(session, OTHER_TENANT).get(attachment_id) is None
        assert (
            await AttachmentRepository(session, OTHER_TENANT).list_for_conversation(conversation_id)
            == []
        )


async def test_set_status_stamps_updated_at(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    # Backdated on purpose: two calls in the same millisecond make `>=` pass even
    # if the stamp is never rewritten, which is the bug this test exists to catch.
    before = datetime.now(UTC) - timedelta(hours=1)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        entity = repo.add(_attachment(conversation_id, updated_at=before))
        await repo.flush()

        await repo.set_status(entity, ATTACHMENT_PARSING)
        await repo.commit()

    assert entity.status == ATTACHMENT_PARSING
    assert entity.updated_at > before


async def test_failure_records_a_reason_and_success_clears_it(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        entity = repo.add(_attachment(conversation_id))
        await repo.flush()

        await repo.set_status(entity, ATTACHMENT_FAILED, error="The PDF is password protected.")
        assert entity.error == "The PDF is password protected."

        # A retry that succeeds must not keep carrying why it once failed.
        await repo.set_status(entity, ATTACHMENT_READY)
        await repo.commit()

    assert entity.status == ATTACHMENT_READY
    assert entity.error is None


async def test_an_error_on_a_non_failed_status_is_dropped(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        entity = repo.add(_attachment(conversation_id))
        await repo.flush()

        await repo.set_status(entity, ATTACHMENT_READY, error="stale message")

    assert entity.error is None


async def test_an_unknown_status_is_a_programming_error(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        entity = repo.add(_attachment(conversation_id))
        await repo.flush()

        with pytest.raises(ValueError, match="Unknown attachment status"):
            await repo.set_status(entity, "done")


async def test_an_unknown_origin_never_reaches_the_database(db_sessionmaker) -> None:
    # `origin` and `status` are plain strings in the schema, so the repository is
    # the only thing standing between a typo and a permanently unreadable row.
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)

        with pytest.raises(ValueError, match="Unknown attachment origin"):
            repo.add(_attachment(conversation_id, origin="dropped-from-the-sky"))

        with pytest.raises(ValueError, match="Unknown attachment status"):
            repo.add(_attachment(conversation_id, status="almost-done"))


async def test_set_status_refuses_a_row_from_another_principal(db_sessionmaker) -> None:
    # A worker holds no request scope, so a row loaded outside a repository must
    # fail loudly here rather than silently write across the tenant OR user
    # boundary. Both a foreign tenant and a same-tenant foreign user are refused.
    conversation_id = await _new_conversation(db_sessionmaker)
    other_user = TenantContext(tenant_id=TENANT.tenant_id, user_id=uuid4(), role="member")
    async with db_sessionmaker() as session:
        entity = AttachmentRepository(session, TENANT).add(_attachment(conversation_id))
        await session.flush()

        for foreign in (
            AttachmentRepository(session, OTHER_TENANT),
            AttachmentRepository(session, other_user),
        ):
            with pytest.raises(ValueError, match="another principal"):
                await foreign.set_status(entity, ATTACHMENT_READY)

    assert entity.status == ATTACHMENT_PENDING


async def test_get_by_content_hash_finds_the_same_bytes_in_one_thread(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    other_conversation = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        repo.add(_attachment(conversation_id, content_hash="abc123"))
        await repo.commit()

    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        found = await repo.get_by_content_hash("abc123", conversation_id=conversation_id)
        assert found is not None

        # Scoped to the conversation: the same file in another thread is its own
        # row, because each thread's ephemeral vectors expire on their own clock.
        assert await repo.get_by_content_hash("abc123", conversation_id=other_conversation) is None
        assert await repo.get_by_content_hash("nope", conversation_id=conversation_id) is None

    async with db_sessionmaker() as session:
        # ... and a hash collision across tenants reveals nothing either.
        foreign = AttachmentRepository(session, OTHER_TENANT)
        assert await foreign.get_by_content_hash("abc123", conversation_id=conversation_id) is None


async def test_a_pasted_block_is_an_ordinary_attachment(db_sessionmaker) -> None:
    # Pasting a long traceback stores a real file that travels the identical
    # pipeline; only the label differs, so a citation can say "Pasted content".
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        entity = repo.add(
            _attachment(
                conversation_id,
                filename="pasted-text.txt",
                origin=ATTACHMENT_ORIGIN_PASTE,
            )
        )
        await repo.commit()

    assert entity.origin == ATTACHMENT_ORIGIN_PASTE
    assert entity.status == ATTACHMENT_PENDING


async def test_a_failed_row_never_matches_the_same_bytes(db_sessionmaker) -> None:
    """A dead end must not be handed back as if it were the file.

    A failed row holds no text and never will. Returning one would mean a file
    that failed once can never be uploaded successfully into that conversation
    again - every retry would be handed the corpse of the first attempt. Failed
    rows are not rare either: one is left behind whenever a parse exhausts its
    retries, or a stalled upload is swept.
    """
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        repo.add(_attachment(conversation_id, content_hash="abc123", status=ATTACHMENT_FAILED))
        await repo.commit()

    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        found = await repo.get_by_content_hash("abc123", conversation_id=conversation_id)

    # No match means "store it fresh", which is exactly right for bytes whose
    # only other copy is broken.
    assert found is None


async def test_a_failed_row_does_not_mask_a_healthy_one_beside_it(db_sessionmaker) -> None:
    # The older-wins tie-break used to pick the failed row purely because it got
    # there first, handing back an unreadable file while a perfectly good copy of
    # the same bytes sat right next to it.
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        repo.add(
            _attachment(
                conversation_id,
                content_hash="abc123",
                status=ATTACHMENT_FAILED,
                filename="first-try.txt",
            )
        )
        repo.add(
            _attachment(
                conversation_id,
                content_hash="abc123",
                status=ATTACHMENT_READY,
                filename="second-try.txt",
            )
        )
        await repo.commit()

    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        found = await repo.get_by_content_hash("abc123", conversation_id=conversation_id)

    assert found is not None
    assert found.filename == "second-try.txt"


async def test_an_expired_row_never_matches_the_same_bytes(db_sessionmaker) -> None:
    """A row whose bytes the retention sweep removed is not a file to reuse.

    Sharper than the failed case: an expired row looks perfectly healthy - it was
    ``ready``, it carries a real hash, it has a filename and a size - and its blob
    is gone. Handing it back would let an upload appear to succeed against a file
    that does not exist, and the ingest it re-enqueues would then find an empty
    store and mark that row ``failed``, overwriting the honest "this aged out"
    with a fault that never happened.
    """
    conversation_id = await _new_conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        repo.add(_attachment(conversation_id, content_hash="abc123", status=ATTACHMENT_EXPIRED))
        await repo.commit()

    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        found = await repo.get_by_content_hash("abc123", conversation_id=conversation_id)

    assert found is None
