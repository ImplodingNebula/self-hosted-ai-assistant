"""Editing a question: the fourth request shape, and the tail it takes with it.

An edit rewrites a stored user message and answers the new wording. Two things
make it different from every other shape and are what these drive: the OLD
wording is gone (nothing keeps it), and so is every turn that came after it,
because those replies answered a question that no longer exists.

Like the regenerate tests, these go through the real endpoint - what the model is
handed, what survives in the transcript, and what the stream says are only
observable together.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete

from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import TITLE_ORIGIN_MANUAL, Attachment, Conversation
from app.parsing.base import ParsedDocument
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from app.services import rag_service
from app.services.attachment_ingest import build_ingest_items
from app.workers.main import JOB_GENERATE_TITLE
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store
from tests.unit.test_chat_rag import (
    CHAT_URL,
    citing_provider,
    client_for,
    collect_events,
    make_settings,
)

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")

ASK = "What is the refund window for orders?"
REWORDED = "What is the refund window for subscriptions?"

# Text sharing no vocabulary with the question that names it, so only a
# deliberate re-feed of the stored reference can put it in front of the model.
FILE_TEXT = (
    "Zarquon Industries reported a 14 percent decline in quarterly turbine "
    "shipments, attributing the shortfall to a magnet supplier in Ostrava."
)


async def _messages(db_sessionmaker, conversation_id: UUID):
    async with db_sessionmaker() as session:
        return await MessageRepository(session, TENANT).list_for_conversation(conversation_id)


async def _empty_conversation(db_sessionmaker) -> UUID:
    async with db_sessionmaker() as session:
        conversation = Conversation(tenant_id=TENANT.tenant_id, user_id=TENANT.user_id)
        session.add(conversation)
        await session.commit()
        return conversation.id


async def _attachment(db_sessionmaker, conversation_id: UUID) -> Attachment:
    async with db_sessionmaker() as session:
        attachment = Attachment(
            tenant_id=TENANT.tenant_id,
            user_id=TENANT.user_id,
            conversation_id=conversation_id,
            filename="turbines.txt",
            content_type="text/plain",
            size_bytes=len(FILE_TEXT),
            storage_key=f"{TENANT.tenant_id}/{conversation_id}/{uuid4()}",
            status="ready",
        )
        session.add(attachment)
        await session.commit()
        return attachment


async def _ingest_file(store, attachment: Attachment, conversation_id: UUID) -> None:
    """Put the file's text where the parse worker would have put it."""
    await rag_service.ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=build_ingest_items(attachment, [ParsedDocument(ordinal=0, text=FILE_TEXT)]),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=make_settings(),
        session_id=str(conversation_id),
    )


async def _conversation_row(db_sessionmaker, conversation_id: UUID):
    async with db_sessionmaker() as session:
        return await ConversationRepository(session, TENANT).get(conversation_id)


def _edit(conversation_id: UUID | str, message_id: UUID | str, text: str = REWORDED) -> dict:
    return {
        "conversation_id": str(conversation_id),
        "edit": {"message_id": str(message_id), "message": text},
    }


async def _thread(client, db_sessionmaker, provider: FakeProvider, turns: list[str]):
    """Send ``turns`` in order, re-scripting the fake for each; return the rows."""
    conversation_id: UUID | None = None
    for index, text in enumerate(turns):
        provider._script = [f"Answer {index}."]  # noqa: SLF001 - re-script the same fake
        payload: dict = {"message": text}
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)
        events = await collect_events(client, payload)
        conversation_id = UUID(dict(events)["done"]["conversation_id"])
    assert conversation_id is not None
    return conversation_id, await _messages(db_sessionmaker, conversation_id)


# --------------------------------------------------------------------------- #
# The point: the question changes, and what answered the old one goes          #
# --------------------------------------------------------------------------- #


async def test_editing_rewrites_the_question_and_answers_the_new_wording(
    db_sessionmaker,
) -> None:
    provider = FakeProvider(script=["First answer."])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id, before = await _thread(client, db_sessionmaker, provider, [ASK])
        provider._script = ["Subscriptions are 14 days."]  # noqa: SLF001
        await collect_events(client, _edit(conversation_id, before[0].id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.role for m in after] == ["user", "assistant"]
    # The SAME row, reworded in place - not a new question appended under the old.
    assert after[0].id == before[0].id
    assert after[0].content == REWORDED
    assert after[1].content == "Subscriptions are 14 days."
    assert after[1].id != before[1].id
    # And the model was asked the new wording, exactly once, with nothing before it.
    assert [m.content for m in provider.last_messages if m.role != "system"] == [REWORDED]


async def test_the_old_wording_is_not_kept_anywhere(db_sessionmaker) -> None:
    # Truncate-first, not branch: there is no second version of this question to
    # go back to, which is what the UI told the user before it ran.
    provider = FakeProvider(script=["First answer."])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id, before = await _thread(client, db_sessionmaker, provider, [ASK])
        await collect_events(client, _edit(conversation_id, before[0].id))

    stored = "\n".join(m.content for m in await _messages(db_sessionmaker, conversation_id))
    assert ASK not in stored


async def test_editing_removes_every_turn_below_it_whatever_its_role(db_sessionmaker) -> None:
    # THE headline difference from a regenerate. Everything under the edited
    # question - later questions included, not just replies - answered wording
    # that no longer exists, so it all goes.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id, rows = await _thread(
            client, db_sessionmaker, provider, [ASK, "And for gift cards?", "And digital goods?"]
        )
        assert len(rows) == 6
        provider._script = ["Subscriptions are 14 days."]  # noqa: SLF001
        await collect_events(client, _edit(conversation_id, rows[0].id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.role for m in after] == ["user", "assistant"]
    assert after[0].id == rows[0].id
    assert after[1].content == "Subscriptions are 14 days."


async def test_editing_a_middle_question_keeps_everything_above_it(db_sessionmaker) -> None:
    # The turns ABOVE the edited one are untouched, and they are replayed as this
    # turn's history - the exchange still reads as a conversation.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id, rows = await _thread(
            client, db_sessionmaker, provider, [ASK, "And for gift cards?", "And digital goods?"]
        )
        provider._script = ["Reworded answer."]  # noqa: SLF001
        await collect_events(client, _edit(conversation_id, rows[2].id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.id for m in after[:2]] == [rows[0].id, rows[1].id]
    assert [m.role for m in after] == ["user", "assistant", "user", "assistant"]
    assert after[2].content == REWORDED
    replayed = [m.content for m in provider.last_messages if m.role != "system"]
    assert replayed == [ASK, "Answer 0.", REWORDED]


async def test_an_edit_that_produces_nothing_leaves_the_question_standing(
    db_sessionmaker,
) -> None:
    # The tail is already gone by the time generation starts, deliberately: the
    # reply answered wording that no longer exists. A generation that then fails
    # leaves the rewritten question with no reply - the exact state Regenerate is
    # the retry for - rather than restoring an answer to a question nobody asked.
    provider = FakeProvider(script=["First answer."])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id, before = await _thread(client, db_sessionmaker, provider, [ASK])
        provider._script = []  # noqa: SLF001 - the model returns nothing at all
        await collect_events(client, _edit(conversation_id, before[0].id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.role for m in after] == ["user"]
    assert after[0].content == REWORDED


async def test_editing_counts_as_activity(db_sessionmaker) -> None:
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id, rows = await _thread(client, db_sessionmaker, provider, [ASK])
        before = await _conversation_row(db_sessionmaker, conversation_id)
        provider._script = ["again"]  # noqa: SLF001
        await collect_events(client, _edit(conversation_id, rows[0].id))

    after = await _conversation_row(db_sessionmaker, conversation_id)
    assert after.updated_at > before.updated_at


# --------------------------------------------------------------------------- #
# What it refuses                                                              #
# --------------------------------------------------------------------------- #


async def test_editing_a_reply_is_a_plain_400(db_sessionmaker) -> None:
    # Only a question can be edited. A reply is the model's, and rewriting it
    # would put words in its mouth that a reader would take for its own.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id, rows = await _thread(client, db_sessionmaker, provider, [ASK])
        resp = await client.post(CHAT_URL, json=_edit(conversation_id, rows[1].id))

    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"
    # And the turn it refused is untouched.
    assert [m.content for m in await _messages(db_sessionmaker, conversation_id)] == [
        ASK,
        "Answer 0.",
    ]


async def test_a_missing_message_is_told_apart_from_a_missing_conversation(
    db_sessionmaker,
) -> None:
    # Both are 404, and they call for OPPOSITE reactions - so they carry
    # different codes. A missing conversation means the pane is bound to
    # something deleted and should unbind; a missing MESSAGE happens the moment a
    # second tab rewinds a thread, and a client that read it as "the conversation
    # is gone" would detach a pane from a conversation that is perfectly alive.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id, rows = await _thread(client, db_sessionmaker, provider, [ASK])
        missing_message = await client.post(CHAT_URL, json=_edit(conversation_id, uuid4()))
        missing_thread = await client.post(CHAT_URL, json=_edit(uuid4(), rows[0].id))

    assert missing_message.status_code == missing_thread.status_code == 404
    assert missing_message.json()["code"] == "message_not_found"
    assert missing_thread.json()["code"] == "not_found"


async def test_editing_a_message_of_another_conversation_is_a_plain_404(db_sessionmaker) -> None:
    # Indistinguishable from a message that does not exist, because the read that
    # resolves it is scoped to the thread AND the principal.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        first, first_rows = await _thread(client, db_sessionmaker, provider, [ASK])
        second, _ = await _thread(client, db_sessionmaker, provider, ["Different thread"])
        resp = await client.post(CHAT_URL, json=_edit(second, first_rows[0].id))

    assert resp.status_code == 404
    assert resp.json()["code"] == "message_not_found"
    # Nothing in either thread moved.
    assert len(await _messages(db_sessionmaker, first)) == 2


async def test_one_users_edit_cannot_reach_another_users_thread(db_sessionmaker) -> None:
    # The repository is user-scoped, so a stranger's message id resolves to
    # nothing at all rather than to somebody else's row.
    stranger = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
    async with db_sessionmaker() as session:
        conversation = Conversation(tenant_id=stranger.tenant_id, user_id=stranger.user_id)
        session.add(conversation)
        await session.commit()
        message = await MessageRepository(session, stranger).add(
            conversation.id, role="user", content=ASK
        )
        await session.commit()
        their_conversation, their_message = conversation.id, message.id

    async with client_for(FakeProvider(script=["x"]), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json=_edit(their_conversation, their_message))

    # The CONVERSATION is already invisible, so this stops at the outer gate.
    assert resp.status_code == 404
    async with db_sessionmaker() as session:
        rows = await MessageRepository(session, stranger).list_for_conversation(their_conversation)
    assert [m.content for m in rows] == [ASK]


async def test_an_edit_of_an_unknown_conversation_is_a_plain_404(db_sessionmaker) -> None:
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json=_edit(uuid4(), uuid4()))

    assert resp.status_code == 404


@pytest.mark.parametrize(
    ("extra", "what"),
    [
        ({"message": "and also this"}, "a new message"),
        ({"attachment_ids": [str(uuid4())]}, "attachments"),
        ({"confirmation": {"id": "c1", "decision": "approve"}}, "a confirmation"),
        ({"regenerate": True}, "a regenerate"),
        ({"message_id": str(uuid4())}, "a regenerate target"),
        ({"quote": {"text": "something"}}, "a quote"),
    ],
)
async def test_an_edit_carrying_anything_else_is_rejected(
    db_sessionmaker, extra: dict, what: str
) -> None:
    # An edit supplies the ONE thing that is not already stored - the new
    # wording. Anything else would be silently ignored, and silently ignoring a
    # field is worse than refusing it.
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json={**_edit(uuid4(), uuid4()), **extra})

    assert resp.status_code == 422, f"an edit carrying {what} must be refused"


@pytest.mark.parametrize("blank", [" ", "\n\t "])
async def test_an_edit_carrying_only_whitespace_is_rejected(db_sessionmaker, blank: str) -> None:
    # Same rule as the regenerate: a body of spaces is still words the sender
    # wrote and the backend would still discard, so it is refused rather than
    # silently dropped. The new wording belongs in edit.message.
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json={**_edit(uuid4(), uuid4()), "message": blank})

    assert resp.status_code == 422


async def test_an_edit_must_name_its_conversation(db_sessionmaker) -> None:
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(
            CHAT_URL, json={"edit": {"message_id": str(uuid4()), "message": REWORDED}}
        )

    assert resp.status_code == 422


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
async def test_an_edit_cannot_blank_the_question(db_sessionmaker, blank: str) -> None:
    # An empty question is not an edit, it is a delete - and deleting a turn is
    # not something this shape does.
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json=_edit(uuid4(), uuid4(), blank))

    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# What comes back off the stored row                                           #
# --------------------------------------------------------------------------- #


async def test_the_files_the_question_arrived_with_survive_the_edit(db_sessionmaker) -> None:
    # An edit changes the words, not the files: the references written on the row
    # are re-fed exactly as a regenerate re-feeds them, so "summarise this" can be
    # reworded without re-uploading the file. The relevance floor is pinned
    # impossibly high, so a deliberate re-feed is the ONLY way the file's text can
    # reach the prompt or its title a citation.
    store = make_memory_store()
    settings = make_settings(rag_min_score=0.99)
    conversation_id = await _empty_conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_file(store, attachment, conversation_id)

    provider = citing_provider(["It is about turbines [1]."])
    async with client_for(provider, db_sessionmaker, settings, store=store) as client:
        await collect_events(
            client,
            {
                "message": "Summarise the file I attached.",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        rows = await _messages(db_sessionmaker, conversation_id)
        events = await collect_events(client, _edit(conversation_id, rows[0].id, "What changed?"))

    assert "Zarquon Industries" in "\n".join(m.content for m in provider.last_messages)
    assert [data["title"] for name, data in events if name == "citation"] == ["turbines.txt"]


async def test_a_file_deleted_since_is_skipped_not_fatal(db_sessionmaker) -> None:
    # Same leniency a regenerate gets: the reference outlives the file it names,
    # so an edit routinely meets one pointing at nothing and answers from what
    # remains rather than refusing the turn.
    store = make_memory_store()
    conversation_id = await _empty_conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_file(store, attachment, conversation_id)

    provider = FakeProvider(script=["Read."])
    async with client_for(provider, db_sessionmaker, store=store) as client:
        await collect_events(
            client,
            {
                "message": "Summarise the file I attached.",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        rows = await _messages(db_sessionmaker, conversation_id)
        async with db_sessionmaker() as session:
            await session.execute(delete(Attachment).where(Attachment.id == attachment.id))
            await session.commit()
        provider._script = ["Answered without it."]  # noqa: SLF001
        events = await collect_events(client, _edit(conversation_id, rows[0].id, "What changed?"))

    assert dict(events)["done"]["message_id"], "a stale reference must not fail the turn"
    assert not [name for name, _ in events if name == "error"]
    after = await _messages(db_sessionmaker, conversation_id)
    assert after[-1].content == "Answered without it."


# --------------------------------------------------------------------------- #
# The stream and the sidebar                                                   #
# --------------------------------------------------------------------------- #


async def test_an_edit_stream_names_its_conversation_and_its_question(db_sessionmaker) -> None:
    # meta is the first frame of EVERY stream, and it now names the question the
    # turn is answering - which for an edit is the row that was just rewritten.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id, rows = await _thread(client, db_sessionmaker, provider, [ASK])
        provider._script = ["Reworded answer."]  # noqa: SLF001
        events = await collect_events(client, _edit(conversation_id, rows[0].id))

    assert events[0][0] == "meta"
    assert events[0][1]["conversation_id"] == str(conversation_id)
    assert events[0][1]["user_message_id"] == str(rows[0].id)
    assert [name for name, _ in events].count("meta") == 1


async def test_editing_the_opening_question_asks_for_a_new_title(db_sessionmaker) -> None:
    # The thread is named after its first question. Reword that one and the name
    # is about something nobody asked, so the naming runs again - under a job id
    # of its own, or arq would deduplicate it against the first and it would
    # silently never happen.
    queue = FakeJobQueue()
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker, queue=queue) as client:
        conversation_id, rows = await _thread(client, db_sessionmaker, provider, [ASK])
        provider._script = ["Reworded answer."]  # noqa: SLF001
        await collect_events(client, _edit(conversation_id, rows[0].id))

    titles = [call for call in queue.calls if call[0] == JOB_GENERATE_TITLE]
    assert len(titles) == 2, "the edit must queue a naming job of its own"
    assert len(queue.job_ids) == len(titles), "the two jobs must not share an id"
    # And it is told to replace whatever the thread is called NOW.
    conversation = await _conversation_row(db_sessionmaker, conversation_id)
    assert titles[-1][1][-1] == conversation.title


async def test_editing_a_later_question_leaves_the_title_alone(db_sessionmaker) -> None:
    queue = FakeJobQueue()
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker, queue=queue) as client:
        conversation_id, rows = await _thread(
            client, db_sessionmaker, provider, [ASK, "And for gift cards?"]
        )
        provider._script = ["Reworded answer."]  # noqa: SLF001
        await collect_events(client, _edit(conversation_id, rows[2].id))

    assert [call for call in queue.calls if call[0] == JOB_GENERATE_TITLE].__len__() == 1


async def test_keeping_the_name_a_thread_already_has_still_claims_it(db_sessionmaker) -> None:
    # A rename that submits the SAME words is still a person choosing that name.
    # Recording the choice only when the string moves would leave the provenance
    # saying "generated", and a later edit of the opening question would then
    # overwrite a title the user had just deliberately kept. The thread must not
    # resurface for it, though - confirming a name is not activity in the thread.
    queue = FakeJobQueue()
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker, queue=queue) as client:
        conversation_id, rows = await _thread(client, db_sessionmaker, provider, [ASK])
        before = await _conversation_row(db_sessionmaker, conversation_id)
        resp = await client.patch(
            f"/api/v1/conversations/{conversation_id}", json={"title": before.title}
        )
        assert resp.status_code == 200
        # Measured across the PATCH alone: the edit below is real activity and
        # bumps the clock legitimately.
        renamed = await _conversation_row(db_sessionmaker, conversation_id)
        assert renamed.title_origin == TITLE_ORIGIN_MANUAL, "the choice is recorded"
        assert renamed.updated_at == before.updated_at, "confirming a name is not activity"
        provider._script = ["Reworded answer."]  # noqa: SLF001
        await collect_events(client, _edit(conversation_id, rows[0].id))

    after = await _conversation_row(db_sessionmaker, conversation_id)
    assert after.title == before.title, "and the edit leaves that name alone"
    assert len([call for call in queue.calls if call[0] == JOB_GENERATE_TITLE]) == 1


async def test_a_hand_picked_title_is_not_re_asked_for(db_sessionmaker) -> None:
    # The one thing nothing overrides. A name its owner typed stands even when
    # the question it was chosen for is rewritten underneath it.
    queue = FakeJobQueue()
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker, queue=queue) as client:
        conversation_id, rows = await _thread(client, db_sessionmaker, provider, [ASK])
        resp = await client.patch(
            f"/api/v1/conversations/{conversation_id}", json={"title": "Payments incident"}
        )
        assert resp.status_code == 200
        provider._script = ["Reworded answer."]  # noqa: SLF001
        await collect_events(client, _edit(conversation_id, rows[0].id))

    assert [call for call in queue.calls if call[0] == JOB_GENERATE_TITLE].__len__() == 1
    conversation = await _conversation_row(db_sessionmaker, conversation_id)
    assert conversation.title == "Payments incident"
    assert conversation.title_origin == TITLE_ORIGIN_MANUAL
