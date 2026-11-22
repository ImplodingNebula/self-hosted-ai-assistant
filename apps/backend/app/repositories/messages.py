"""Message persistence - append, ordered reads, and tail truncation, tenant-scoped."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from itertools import takewhile
from typing import ClassVar
from uuid import UUID

from sqlalchemy import delete, select

from app.models import Message
from app.repositories.user_scoped import UserScopedRepository


class MessageRepository(UserScopedRepository[Message]):
    """Data access for chat messages."""

    model: ClassVar[type[Message]] = Message

    # Deliberately shadows UserScopedRepository.add: appending a message is the
    # only write this repository exposes, and building the entity here keeps the
    # tenant/user stamp and the flush (id generation) in one place.
    async def add(  # type: ignore[override]
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        thinking: str | None = None,
        model: str | None = None,
        meta: dict | None = None,
        created_at: datetime | None = None,
    ) -> Message:
        """Append a message to a conversation and flush so its id is usable.

        ``created_at`` may be supplied to guarantee ordering within one turn
        (user message strictly before the assistant reply).
        """
        message = Message(
            tenant_id=self.tenant.tenant_id,
            user_id=self.tenant.user_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            thinking=thinking,
            model=model,
            meta=meta if meta is not None else {},
        )
        if created_at is not None:
            message.created_at = created_at
        self.session.add(message)
        await self.flush()
        return message

    async def list_for_conversation(self, conversation_id: UUID) -> list[Message]:
        """Return all messages of a conversation, oldest first."""
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.tenant_id == self.tenant.tenant_id,
                Message.user_id == self.tenant.user_id,
            )
            .order_by(Message.created_at.asc())
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def delete_after(self, conversation_id: UUID, message_id: UUID) -> int:
        """Delete the assistant replies that answer ``message_id``, and only those.

        The one write that removes messages rather than adding them, and it exists
        for regenerating an answer: the question stays, the reply it produced goes,
        and the turn is run again.

        The reach is deliberately tiny, and the shape of the stop is the whole
        point. It removes the run of ``assistant`` rows IMMEDIATELY after the
        anchor and STOPS at the first row of any other role - it does not filter
        the tail for assistant rows, which would sail past a later question and
        delete the answer to THAT one instead. The difference is invisible in the
        case this was written for (a thread ending at the anchor) and is silent,
        permanent data loss in the case that arrives on its own: a second tab
        sending a new turn while a regenerate is still streaming. Stopping means
        the worst outcome there is an extra reply appended, not somebody's answer
        destroyed. A caller pointing this at anything but the newest question is
        still a bug - it just is no longer a destructive one.

        Position-based, not timestamp-based: the anchor is located by id inside the
        ordered thread and the run after it is deleted by an explicit id list, so
        the delete can never reach further than the rows this read actually saw. A
        ``created_at >`` predicate would delete by a rule instead of by a list, and
        a row arriving between the read and the write would be swept up by it.
        (The ordering itself is safe because nothing ties: every write clamps
        strictly past its own question - see ``_persist_assistant`` and the
        prepare phase in ``chat_service`` - so ``created_at`` is a total order
        here even though ``ORDER BY`` alone would not guarantee one. It is a
        total order, not necessarily a LOGICAL one: a turn that re-answers a
        question further up the thread writes its reply with a fresh timestamp,
        so if a second tab appended in the meantime, the replacement sorts after
        that new turn rather than beside the question it answers. Nothing is lost
        by it - the anchor guard and the captured id list still hold - but a
        reader of this method should not assume reply-follows-question.)

        In practice it removes zero or one row: a turn has at most one reply, and
        an interrupted turn that saved nothing leaves none. Both are normal.

        Args:
            conversation_id: The conversation whose tail is being rewritten.
            message_id: The question whose answer is being replaced.

        Returns:
            How many replies were deleted; 0 when the anchor is unknown to this
            principal, is already last, or is followed by another question.
        """
        # Scoped here rather than borrowed from ``list_for_conversation``: this is
        # the read that decides what gets DELETED, so it carries its own principal
        # filter and cannot inherit a looser one if that helper ever changes.
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.tenant_id == self.tenant.tenant_id,
                Message.user_id == self.tenant.user_id,
            )
            .order_by(Message.created_at.asc())
        )
        ordered = list((await self.session.execute(stmt)).scalars().all())
        position = next((index for index, row in enumerate(ordered) if row.id == message_id), None)
        if position is None:
            return 0
        doomed = [
            row.id
            for row in takewhile(lambda row: row.role == "assistant", ordered[position + 1 :])
        ]
        if not doomed:
            return 0
        removal = delete(Message).where(
            Message.id.in_(doomed),
            Message.conversation_id == conversation_id,
            Message.tenant_id == self.tenant.tenant_id,
            Message.user_id == self.tenant.user_id,
        )
        result = await self.session.execute(removal)
        return result.rowcount or 0

    async def ids_after(self, conversation_id: UUID, message_id: UUID) -> list[UUID]:
        """Return the ids of every message that follows ``message_id``, in order.

        The reading half of a truncation, split out from the writing half because
        the two happen at different times. Regenerating an OLDER reply has to know
        what it is about to destroy while the user is still deciding (the
        confirmation dialog counts these rows), and then delete exactly that list
        at save time - so anything a second tab appended in between survives,
        because it was never on the list the user was shown.

        Position-based like :meth:`delete_after`, and for the same reason: the
        anchor is found by id inside the ordered thread and everything past it is
        named explicitly, so no rule can reach a row this read did not see.

        Args:
            conversation_id: The conversation being read.
            message_id: The row the tail starts after.

        Returns:
            The following messages' ids, oldest first; empty when the anchor is
            last, or is unknown to this principal.
        """
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.tenant_id == self.tenant.tenant_id,
                Message.user_id == self.tenant.user_id,
            )
            .order_by(Message.created_at.asc())
        )
        ordered = list((await self.session.execute(stmt)).scalars().all())
        position = next((index for index, row in enumerate(ordered) if row.id == message_id), None)
        if position is None:
            return []
        return [row.id for row in ordered[position + 1 :]]

    async def delete_ids(self, conversation_id: UUID, message_ids: Sequence[UUID]) -> int:
        """Delete exactly these messages of this conversation; return how many went.

        Existence-tolerant by design: it is handed a list captured EARLIER (see
        :meth:`ids_after`), and a row already gone in the meantime is not an
        error - the outcome asked for is that it no longer be there. The
        conversation, tenant and user filters are re-applied on the delete rather
        than trusted from the read, so a list that has travelled through a
        checkpoint still cannot reach another thread or another principal.

        Args:
            conversation_id: The conversation the ids must belong to.
            message_ids: The rows to remove; an empty list deletes nothing.

        Returns:
            How many rows were actually deleted.
        """
        if not message_ids:
            return 0
        removal = delete(Message).where(
            Message.id.in_(list(message_ids)),
            Message.conversation_id == conversation_id,
            Message.tenant_id == self.tenant.tenant_id,
            Message.user_id == self.tenant.user_id,
        )
        result = await self.session.execute(removal)
        return result.rowcount or 0

    async def delete_all_after(self, conversation_id: UUID, message_id: UUID) -> int:
        """Delete EVERY message after ``message_id``, whatever its role.

        The sibling of :meth:`delete_after`, and deliberately the opposite shape:
        that one stops at the first row of another role, this one does not stop at
        all. Each fits the turn it serves.

        Regenerating replaces one answer, so it must not reach past the next
        question - a second tab's new turn is somebody else's exchange. EDITING a
        question is the other case: every turn below it was written in reply to
        wording that no longer exists, so keeping any of it would leave a thread
        answering a question nobody asked. The user is told exactly that, and how
        many turns it costs, before this runs.

        Position-based, and the delete names explicit ids, for the same reason
        :meth:`delete_after` is: the read decides the reach, so no predicate can
        sweep up a row that arrived after it.

        Args:
            conversation_id: The conversation whose tail is being removed.
            message_id: The question that has just been rewritten.

        Returns:
            How many messages were deleted; 0 when the anchor is already last or
            is unknown to this principal.
        """
        return await self.delete_ids(
            conversation_id, await self.ids_after(conversation_id, message_id)
        )

    async def first_of_role(self, conversation_id: UUID, role: str) -> Message | None:
        """Return a conversation's earliest message in ``role``, or None.

        The opening exchange, which is what names a thread: the background
        title job reads the first question and the first reply through this.
        Ordered and limited in the database - the (conversation_id, created_at)
        index prunes to one row rather than shipping a whole transcript to pick
        its head.
        """
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.tenant_id == self.tenant.tenant_id,
                Message.user_id == self.tenant.user_id,
                Message.role == role,
            )
            .order_by(Message.created_at.asc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalars().first()

    async def list_recent(self, conversation_id: UUID, limit: int) -> list[Message]:
        """Return the last ``limit`` messages in chronological order.

        Fetch DESC then reverse: the database prunes old rows via the
        (conversation_id, created_at) index instead of shipping the whole
        history for a long conversation.
        """
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.tenant_id == self.tenant.tenant_id,
                Message.user_id == self.tenant.user_id,
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        rows = list((await self.session.execute(stmt)).scalars().all())
        rows.reverse()
        return rows
