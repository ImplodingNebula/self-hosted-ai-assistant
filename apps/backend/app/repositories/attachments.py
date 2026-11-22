"""Tenant-scoped access to uploaded-attachment rows."""

from __future__ import annotations

from typing import ClassVar
from uuid import UUID

from sqlalchemy import func, select

from app.models import (
    ATTACHMENT_EXPIRED,
    ATTACHMENT_FAILED,
    ATTACHMENT_ORIGINS,
    ATTACHMENT_STATUSES,
    Attachment,
    utcnow,
)
from app.repositories.user_scoped import UserScopedRepository


def _check_status(status: str) -> None:
    """Raise ValueError unless ``status`` is a known lifecycle value."""
    if status not in ATTACHMENT_STATUSES:
        raise ValueError(f"Unknown attachment status: {status!r}")


class AttachmentRepository(UserScopedRepository[Attachment]):
    """Bookkeeping for uploaded files; the bytes live behind the storage seam.

    User-scoped: the parse and purge jobs run under a context carrying the
    uploader's own ``user_id`` (the enqueue passes it), so scoping by user
    changes nothing today and keeps a future second user out of another's files.
    The cross-tenant maintenance sweeps in ``attachment_gc`` deliberately bypass
    this repository - they read the tenant and user off each row.
    """

    model: ClassVar[type[Attachment]] = Attachment

    async def still_exists(self, attachment_id: UUID) -> bool:
        """Ask the database whether this row is still there, changing nothing.

        Deliberately NOT :meth:`get`: the caller (the ingest job) is holding a
        loaded, MODIFIED attachment, and any ordinary query would autoflush that
        pending UPDATE first - against a row that may have just been deleted,
        which raises rather than answering the question. A bare count with
        autoflush suppressed asks only what was asked.
        """
        stmt = (
            select(func.count())
            .select_from(Attachment)
            .where(
                Attachment.id == attachment_id,
                Attachment.tenant_id == self.tenant.tenant_id,
                Attachment.user_id == self.tenant.user_id,
            )
        )
        result = await self.session.execute(stmt, execution_options={"autoflush": False})
        return bool(result.scalar_one())

    def add(self, entity: Attachment) -> Attachment:
        """Stage a new attachment after checking its enum-ish columns.

        ``status`` and ``origin`` are plain strings in the schema (portable
        across sqlite and postgres, and migration-free when a value is added),
        which means nothing below this line would reject a typo. This is where
        that check lives.

        Raises:
            ValueError: ``status`` or ``origin`` is not a known value.
        """
        _check_status(entity.status)
        if entity.origin not in ATTACHMENT_ORIGINS:
            raise ValueError(f"Unknown attachment origin: {entity.origin!r}")
        return super().add(entity)

    async def list_for_conversation(self, conversation_id: UUID) -> list[Attachment]:
        """Return this tenant's attachments on ``conversation_id``, oldest first.

        ``id`` breaks ties: two files added in the same microsecond would
        otherwise come back in whatever order the database felt like.
        """
        stmt = (
            select(Attachment)
            .where(
                Attachment.tenant_id == self.tenant.tenant_id,
                Attachment.user_id == self.tenant.user_id,
                Attachment.conversation_id == conversation_id,
            )
            .order_by(Attachment.created_at, Attachment.id)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_content_hash(
        self, content_hash: str, *, conversation_id: UUID
    ) -> Attachment | None:
        """Return a usable identical file already attached to this conversation.

        Scoped to the conversation, not the tenant: the same file attached to two
        threads is two rows, because each thread's ephemeral vectors expire on
        their own clock. What this prevents is re-parsing and re-embedding the
        same bytes when a client uploads them twice into one thread.

        A FAILED row never wins, however old it is. It is a dead end - it holds no
        text and never will - and it is the OLDEST row of those bytes, so as the
        first match it would mask a perfectly good row sitting right beside it and
        hand the caller the dead one instead. (Failed rows are not rare: one is
        left behind whenever a parse exhausts its retries, or a stalled upload is
        swept.) Excluding them here is what makes "no match" mean what the caller
        assumes it means - store these bytes fresh - and keeps that decision in the
        query rather than in every caller that has to remember to re-check.

        An EXPIRED row is excluded for a sharper version of the same reason: the
        retention sweep deleted its bytes and kept the row as a tombstone, so
        reusing it would hand the upload path a row whose file does not exist.
        The upload would look like it succeeded, re-enqueue an ingest that then
        finds nothing in the store, and the person who just uploaded a file would
        watch the row they were given turn ``failed`` - having also overwritten
        the honest "this aged out" the tombstone was there to say. Re-uploading a
        file after it expires has to be a genuinely fresh row.

        Among usable rows the oldest wins: two can legitimately share a hash when
        the same file is uploaded twice before the first finishes, and whichever
        one a parse job is already working on is the one to reuse.
        """
        stmt = (
            select(Attachment)
            .where(
                Attachment.tenant_id == self.tenant.tenant_id,
                Attachment.user_id == self.tenant.user_id,
                Attachment.conversation_id == conversation_id,
                Attachment.content_hash == content_hash,
                Attachment.status.not_in((ATTACHMENT_FAILED, ATTACHMENT_EXPIRED)),
            )
            .order_by(Attachment.created_at, Attachment.id)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def set_status(
        self, attachment: Attachment, status: str, *, error: str | None = None
    ) -> Attachment:
        """Move ``attachment`` to ``status``, stamping ``updated_at``.

        An error message is kept only on the failed status: a row that later
        succeeds on retry must not still be carrying the reason it once failed.

        The tenant and user checks are defence in depth. Every read above already
        filters by both, so a foreign row can only arrive here by being loaded
        outside a repository - which is exactly the mistake worth failing loudly
        on, because a background worker owns no request scope to catch it later.

        Raises:
            ValueError: ``status`` is not a lifecycle value, or ``attachment``
                belongs to another tenant or user. Both are programming errors
                rather than client input, so neither is an HTTP-facing exception.
        """
        _check_status(status)
        if (
            attachment.tenant_id != self.tenant.tenant_id
            or attachment.user_id != self.tenant.user_id
        ):
            raise ValueError("Attachment belongs to another principal.")
        attachment.status = status
        attachment.error = error if status == ATTACHMENT_FAILED else None
        attachment.updated_at = utcnow()
        self.session.add(attachment)
        await self.flush()
        return attachment
