"""Conversation persistence - list/read/create/touch/delete, tenant-scoped."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar
from uuid import UUID

from sqlalchemy import delete, func, nulls_last, select, update

from app.models import Conversation, Message
from app.repositories.user_scoped import UserScopedRepository


class ConversationRepository(UserScopedRepository[Conversation]):
    """Data access for conversations and their lifecycle."""

    model: ClassVar[type[Conversation]] = Conversation

    async def list_summaries(self, limit: int, offset: int) -> tuple[list[Conversation], int]:
        """Return one page of the user's conversations (pinned first) plus the total.

        Pinned threads sort ahead of every unpinned one, newest pin first, and
        the rest follow by newest activity. The pin has to win in the QUERY, not
        just in the client's grouping: the sidebar reads the first page only, so
        an ordering by activity alone would drop a pinned thread off the payload
        the moment enough other threads had newer activity - silently losing the
        row the pin exists to keep in view. ``NULLS LAST`` is explicit because
        the dialects disagree about where a descending sort puts a NULL.

        The total is computed in the same session so the sidebar can paginate;
        two queries are fine at this scale and keep both portable.
        """
        tenant_id = self.tenant.tenant_id
        user_id = self.tenant.user_id
        stmt = (
            select(Conversation)
            .where(
                Conversation.tenant_id == tenant_id,
                Conversation.user_id == user_id,
            )
            .order_by(
                nulls_last(Conversation.pinned_at.desc()),
                Conversation.updated_at.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        rows = list((await self.session.execute(stmt)).scalars().all())
        count_stmt = (
            select(func.count())
            .select_from(Conversation)
            .where(
                Conversation.tenant_id == tenant_id,
                Conversation.user_id == user_id,
            )
        )
        total = (await self.session.execute(count_stmt)).scalar_one()
        return rows, total

    async def pinned_model_counts(self) -> dict[str, int]:
        """Return how many of the TENANT's conversations pin each chat model.

        Deliberately tenant-wide, and the only method here that is. The user
        filter exists so one user cannot reach another's rows; this question is
        not about reaching them. Model weights are a workspace resource - one
        delete removes them for everyone - so a guard that counted only the
        acting user's pins would report zero and let an admin break every other
        user's conversations. Nothing about those rows leaves this method: it
        returns tag COUNTS, so the widening cannot become a read of another
        user's titles, messages, or settings.

        One grouped query rather than a count per candidate tag, because both
        callers ask about many models at once - the management view asks for
        every row on the page. Keys are trimmed here: a pin is stored verbatim
        from the request, so a stray space would otherwise be a different model
        to every comparison downstream. A full scan is fine at this scale, and
        both callers run on a deliberate action rather than on a chat turn.

        The pin lives in a JSON column, so the read goes through SQLAlchemy's
        JSON accessor rather than string surgery - the same statement then runs
        on PostgreSQL and on the SQLite the tests use.

        Returns:
            Trimmed pinned tag to the number of conversations pinning it;
            empty when nothing is pinned.
        """
        pin = Conversation.settings["model"].as_string()
        stmt = (
            select(pin, func.count())
            .select_from(Conversation)
            .where(Conversation.tenant_id == self.tenant.tenant_id, pin.is_not(None))
            .group_by(pin)
        )
        counts: dict[str, int] = {}
        for value, count in (await self.session.execute(stmt)).all():
            tag = str(value).strip()
            if tag:
                counts[tag] = counts.get(tag, 0) + count
        return counts

    async def clear_model_pins(self, aliases: set[str]) -> int:
        """Clear every conversation pin naming one of ``aliases``; return the count.

        Tenant-wide like :meth:`pinned_model_counts`, and for the same reason:
        model weights are a workspace resource, so the delete that calls this
        removes them for every user at once, and clearing only the acting
        user's pins would leave everyone else's threads pointed at weights
        that are gone. The write is as narrow as the read: only the
        ``settings["model"]`` key changes, so the widening cannot become an
        edit of another user's titles, messages, or remaining settings.

        Row-by-row through the ORM rather than one JSON-surgery UPDATE: the
        statement must run on PostgreSQL and on the SQLite the tests use, the
        JSON column needs a REASSIGNED dict to mark the row dirty, and the
        caller acts deliberately (a delete) on a set that is small by
        construction. Matching mirrors ``pinned_model_counts`` downstream: a
        pin is stored verbatim, so it is trimmed here and matched as a literal
        against the tag's spellings.

        Args:
            aliases: Every spelling of the deleted tag, from ``tag_aliases``.

        Returns:
            How many conversations had their pin cleared. Staged and flushed;
            the caller commits.
        """
        pin = Conversation.settings["model"].as_string()
        stmt = select(Conversation).where(
            Conversation.tenant_id == self.tenant.tenant_id, pin.is_not(None)
        )
        cleared = 0
        for conversation in (await self.session.execute(stmt)).scalars():
            stored = conversation.settings if isinstance(conversation.settings, dict) else {}
            if str(stored.get("model") or "").strip() not in aliases:
                continue
            remaining = dict(stored)
            remaining.pop("model", None)
            conversation.settings = remaining
            cleared += 1
        if cleared:
            await self.flush()
        return cleared

    async def get_with_messages(
        self, conversation_id: UUID
    ) -> tuple[Conversation, list[Message]] | None:
        """Return the conversation and its messages (created_at ASC), or None."""
        conversation = await self.get(conversation_id)
        if conversation is None:
            return None
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.tenant_id == self.tenant.tenant_id,
                Message.user_id == self.tenant.user_id,
            )
            .order_by(Message.created_at.asc())
        )
        messages = list((await self.session.execute(stmt)).scalars().all())
        return conversation, messages

    async def create(
        self,
        title: str | None = None,
        settings: dict | None = None,
    ) -> Conversation:
        """Create a conversation for the bound principal and flush so its id is usable.

        The owner is the acting user from the bound context - deliberately not a
        parameter, so a conversation cannot be created on another user's behalf.
        """
        conversation = Conversation(
            tenant_id=self.tenant.tenant_id,
            user_id=self.tenant.user_id,
            title=title,
            settings=settings if settings is not None else {},
        )
        self.add(conversation)
        await self.flush()
        return conversation

    async def get_for_update(self, conversation_id: UUID) -> Conversation | None:
        """Fetch one conversation holding a row lock for a read-merge-write.

        The settings PATCH merges the stored JSON with the request and writes
        it back; without the lock two overlapping single-field saves could
        both read the same base and the later commit would silently drop the
        earlier one's axis. FOR UPDATE serializes them on the row (a no-op on
        sqlite, whose single-writer model cannot interleave the writes
        anyway). The caller commits, releasing the lock.

        Returns:
            The conversation, or None when the principal has none by that id.
        """
        stmt = (
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == self.tenant.tenant_id,
                Conversation.user_id == self.tenant.user_id,
            )
            .with_for_update()
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def set_pending_confirmation(self, conversation_id: UUID, payload: dict) -> None:
        """Checkpoint a turn suspended on a side-effecting confirmation onto its conversation.

        Overwrites any earlier pending confirmation - a conversation has at most
        one at a time. Scoped to the acting principal, so one user cannot park a
        checkpoint on another's thread. The caller commits.
        """
        stmt = (
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == self.tenant.tenant_id,
                Conversation.user_id == self.tenant.user_id,
            )
            .values(pending_confirmation=payload)
        )
        await self.session.execute(stmt)

    async def take_pending_confirmation(
        self, conversation_id: UUID, confirmation_id: str
    ) -> dict | None:
        """Atomically read and clear the pending confirmation, if its id matches.

        The row is locked FOR UPDATE for the read-and-clear so two concurrent
        resumes (a double-tapped approve) cannot both take the same checkpoint and
        run a side-effecting action twice - the second finds it already gone. A
        mismatched id leaves the pending confirmation intact and returns None; the
        acting-principal scope means another user's checkpoint is never even
        visible here. The caller commits (releasing the lock).

        Returns:
            The consumed checkpoint payload, or None when nothing matches.
        """
        stmt = (
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == self.tenant.tenant_id,
                Conversation.user_id == self.tenant.user_id,
            )
            .with_for_update()
        )
        conversation = (await self.session.execute(stmt)).scalar_one_or_none()
        if conversation is None:
            return None
        pending = conversation.pending_confirmation
        if not isinstance(pending, dict) or pending.get("id") != confirmation_id:
            return None
        conversation.pending_confirmation = None
        await self.session.flush()
        return pending

    async def touch(self, conversation_id: UUID, when: datetime) -> None:
        """Bump ``updated_at`` so the conversation surfaces at the top of the sidebar."""
        stmt = (
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == self.tenant.tenant_id,
                Conversation.user_id == self.tenant.user_id,
            )
            .values(updated_at=when)
        )
        await self.session.execute(stmt)

    async def delete_with_messages(self, conversation_id: UUID) -> bool:
        """Delete the conversation and its messages; False when it does not exist.

        Messages are deleted explicitly before the parent: the FK is ondelete
        CASCADE on postgres, but sqlite only honors it with a pragma - the
        explicit delete makes behavior identical everywhere.
        """
        conversation = await self.get(conversation_id)
        if conversation is None:
            return False
        stmt = delete(Message).where(
            Message.conversation_id == conversation_id,
            Message.tenant_id == self.tenant.tenant_id,
            Message.user_id == self.tenant.user_id,
        )
        await self.session.execute(stmt)
        await self.delete(conversation)
        await self.flush()
        return True
