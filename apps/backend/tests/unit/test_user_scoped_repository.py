"""User scoping WITHIN a tenant - the boundary the tenant filter alone misses.

``test_db_foundation`` already proves two *tenants* cannot see each other's
rows. This proves two *users of the same tenant* cannot either, which is the
gap that matters the moment real login lands: without it, Bob could name Alice's
``conversation_id`` on a chat request and have his answer cite her private
messages. Everything here shares one tenant id and differs only by user.

Today a single user is seeded, so these filters are no-ops in production; the
test stands them up with a second user so the boundary is exercised now rather
than discovered broken when OIDC arrives.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.models import Attachment, Conversation
from app.repositories.attachments import AttachmentRepository
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository

# One tenant, two users. Only the user id differs.
ALICE = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
BOB = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")


def _attachment(conversation_id, *, content_hash: str = "hash") -> Attachment:
    return Attachment(
        tenant_id=DEFAULT_TENANT_ID,
        user_id=ALICE.user_id,
        conversation_id=conversation_id,
        filename="a.txt",
        content_type="text/plain",
        size_bytes=3,
        storage_key=f"{DEFAULT_TENANT_ID}/{conversation_id}/{uuid4()}",
        content_hash=content_hash,
    )


async def test_a_conversation_is_invisible_to_another_user_of_the_same_tenant(
    db_sessionmaker,
) -> None:
    async with db_sessionmaker() as session:
        alice = ConversationRepository(session, ALICE)
        bob = ConversationRepository(session, BOB)
        conv = await alice.create(title="Alice's thread")
        await session.commit()

        # A foreign-user row is indistinguishable from a missing one.
        assert await bob.get(conv.id) is None
        assert await bob.get_with_messages(conv.id) is None

        # It never appears in Bob's list, and Alice still sees her own.
        bob_rows, bob_total = await bob.list_summaries(limit=50, offset=0)
        assert bob_total == 0
        assert bob_rows == []
        alice_rows, alice_total = await alice.list_summaries(limit=50, offset=0)
        assert alice_total == 1
        assert [c.id for c in alice_rows] == [conv.id]


async def test_a_user_cannot_mutate_another_users_conversation(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        alice = ConversationRepository(session, ALICE)
        bob = ConversationRepository(session, BOB)
        conv = await alice.create(title="hers")
        await session.commit()

        # touch() matches nothing, so the row is untouched. (Compare tz-naive:
        # sqlite hands back a naive datetime on refresh, so a bare == would trip
        # on tzinfo rather than on the value, which is what actually matters.)
        before = conv.updated_at
        await bob.touch(conv.id, when=before.replace(year=2030))
        await session.commit()
        await session.refresh(conv)
        assert conv.updated_at.replace(tzinfo=None) == before.replace(tzinfo=None)

        # ...and delete reports "nothing here" rather than removing it.
        assert await bob.delete_with_messages(conv.id) is False
        await session.commit()
        assert await alice.get(conv.id) is not None


async def test_messages_do_not_leak_across_users_even_with_the_conversation_id(
    db_sessionmaker,
) -> None:
    # Messages have no user_id column of their own; their scope is the owning
    # user, carried on every row the repository writes. Bob holding the right
    # conversation id still reads nothing.
    async with db_sessionmaker() as session:
        alice_conv = ConversationRepository(session, ALICE)
        alice_msg = MessageRepository(session, ALICE)
        bob_msg = MessageRepository(session, BOB)
        conv = await alice_conv.create(title="hers")
        await alice_msg.add(conv.id, role="user", content="secret question")
        await alice_msg.add(conv.id, role="assistant", content="secret answer")
        await session.commit()

        assert len(await alice_msg.list_for_conversation(conv.id)) == 2
        assert await bob_msg.list_for_conversation(conv.id) == []
        assert await bob_msg.list_recent(conv.id, limit=10) == []


async def test_one_user_cannot_truncate_another_users_thread(db_sessionmaker) -> None:
    # delete_after is the one message write that REMOVES rows (a regenerate drops
    # the reply it is replacing), so the boundary matters more here than on any
    # read: an unscoped version would let Bob, holding Alice's conversation and
    # message ids, delete her replies. It reads as an unknown anchor to him -
    # nothing deleted, nothing reported - and still works for Alice.
    async with db_sessionmaker() as session:
        alice_conv = ConversationRepository(session, ALICE)
        alice_msg = MessageRepository(session, ALICE)
        bob_msg = MessageRepository(session, BOB)
        conv = await alice_conv.create(title="hers")
        question = await alice_msg.add(conv.id, role="user", content="her question")
        await alice_msg.add(conv.id, role="assistant", content="her answer")
        await session.commit()

        assert await bob_msg.delete_after(conv.id, question.id) == 0
        assert len(await alice_msg.list_for_conversation(conv.id)) == 2

        assert await alice_msg.delete_after(conv.id, question.id) == 1
        await session.commit()
        assert [m.role for m in await alice_msg.list_for_conversation(conv.id)] == ["user"]


async def test_one_user_cannot_wipe_another_users_thread_by_editing(db_sessionmaker) -> None:
    # delete_all_after is the WIDER of the two truncations - it stops at nothing -
    # so an unscoped version would let Bob empty Alice's thread from its first
    # question down with two ids. The anchor is unknown to him, so it reaches
    # nothing; the read that finds it carries the principal filter itself rather
    # than borrowing a looser one.
    async with db_sessionmaker() as session:
        alice_conv = ConversationRepository(session, ALICE)
        alice_msg = MessageRepository(session, ALICE)
        bob_msg = MessageRepository(session, BOB)
        conv = await alice_conv.create(title="hers")
        question = await alice_msg.add(conv.id, role="user", content="her question")
        await alice_msg.add(conv.id, role="assistant", content="her answer")
        await alice_msg.add(conv.id, role="user", content="her follow-up")
        await session.commit()

        assert await bob_msg.ids_after(conv.id, question.id) == []
        assert await bob_msg.delete_all_after(conv.id, question.id) == 0
        assert len(await alice_msg.list_for_conversation(conv.id)) == 3

        assert await alice_msg.delete_all_after(conv.id, question.id) == 2
        await session.commit()
        assert [m.role for m in await alice_msg.list_for_conversation(conv.id)] == ["user"]


async def test_one_user_cannot_delete_another_users_messages_by_id(db_sessionmaker) -> None:
    # delete_ids is handed a list captured a whole generation earlier, so its
    # filters cannot be inherited from the read that produced it: they are
    # re-applied on the delete. Bob naming Alice's rows verbatim removes nothing.
    async with db_sessionmaker() as session:
        alice_conv = ConversationRepository(session, ALICE)
        alice_msg = MessageRepository(session, ALICE)
        bob_msg = MessageRepository(session, BOB)
        conv = await alice_conv.create(title="hers")
        question = await alice_msg.add(conv.id, role="user", content="her question")
        answer = await alice_msg.add(conv.id, role="assistant", content="her answer")
        await session.commit()

        assert await bob_msg.delete_ids(conv.id, [question.id, answer.id]) == 0
        assert len(await alice_msg.list_for_conversation(conv.id)) == 2

        # And an id that has since vanished is not an error: the list has waited
        # out a whole generation, and the outcome asked for is that it be gone.
        assert await alice_msg.delete_ids(conv.id, [answer.id, uuid4()]) == 1
        await session.commit()
        assert [m.role for m in await alice_msg.list_for_conversation(conv.id)] == ["user"]


async def test_delete_after_stops_at_the_next_question(db_sessionmaker) -> None:
    # Not a scoping test, but the same class of guard: a boundary that is invisible
    # in the case the method was written for and destructive in the case that
    # arrives on its own. delete_after must remove the run of replies immediately
    # after its anchor and STOP at the next question - a version that filters the
    # tail for assistant rows instead sails past q2 and deletes ITS answer, which
    # is somebody else's turn and unrecoverable.
    async with db_sessionmaker() as session:
        conv_repo = ConversationRepository(session, ALICE)
        msg_repo = MessageRepository(session, ALICE)
        conv = await conv_repo.create(title="hers")
        q1 = await msg_repo.add(conv.id, role="user", content="q1")
        await msg_repo.add(conv.id, role="assistant", content="a1")
        await msg_repo.add(conv.id, role="user", content="q2")
        await msg_repo.add(conv.id, role="assistant", content="a2")
        await session.commit()

        assert await msg_repo.delete_after(conv.id, q1.id) == 1
        await session.commit()

    async with db_sessionmaker() as session:
        rows = await MessageRepository(session, ALICE).list_for_conversation(conv.id)
    assert [(m.role, m.content) for m in rows] == [
        ("user", "q1"),
        ("user", "q2"),
        ("assistant", "a2"),
    ]


async def test_attachments_do_not_leak_across_users(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        alice_conv = ConversationRepository(session, ALICE)
        alice_att = AttachmentRepository(session, ALICE)
        bob_att = AttachmentRepository(session, BOB)
        conv = await alice_conv.create(title="hers")
        att = _attachment(conv.id)
        alice_att.add(att)
        await session.commit()

        assert await bob_att.get(att.id) is None
        assert await bob_att.still_exists(att.id) is False
        assert await bob_att.list_for_conversation(conv.id) == []
        assert await bob_att.get_by_content_hash("hash", conversation_id=conv.id) is None

        # Alice, the owner, sees all of it.
        assert await alice_att.get(att.id) is not None
        assert await alice_att.still_exists(att.id) is True
        assert len(await alice_att.list_for_conversation(conv.id)) == 1


async def test_create_owns_the_row_to_the_bound_user_not_an_argument(db_sessionmaker) -> None:
    # create() takes no user_id: the owner is the bound principal, so a row
    # cannot be minted on someone else's behalf.
    async with db_sessionmaker() as session:
        conv = await ConversationRepository(session, ALICE).create(title="mine")
        assert conv.user_id == ALICE.user_id
        assert conv.tenant_id == DEFAULT_TENANT_ID


async def test_add_restamps_a_forged_user_id_to_the_bound_user(db_sessionmaker) -> None:
    # An entity that lies about its owner is corrected on the way in, exactly as
    # the base repository already does for the tenant.
    async with db_sessionmaker() as session:
        alice = ConversationRepository(session, ALICE)
        forged = Conversation(tenant_id=DEFAULT_TENANT_ID, user_id=BOB.user_id, title="forged")
        alice.add(forged)
        await alice.flush()
        assert forged.user_id == ALICE.user_id
        await session.rollback()


@pytest.mark.parametrize("repo_cls", [ConversationRepository, AttachmentRepository])
async def test_user_scoped_repositories_declare_their_user_column(
    db_sessionmaker, repo_cls
) -> None:
    # A guard against a future repository being made user-scoped over a table
    # that has no user_id: the get() filter would then be a silent type error.
    assert hasattr(repo_cls.model, "user_id")
