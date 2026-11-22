"""Regenerating a reply: the third request shape, and what it re-uses.

A regenerate answers a stored question AGAIN, in place of the reply that question
already produced - the conversation's newest by default, or the one above the
reply it names. It carries nothing of its own - no words, no files, no
confirmation - so everything it needs has to come back off the stored row, and
the reply it replaces has to actually go. Naming an OLDER reply also discards
what came after it, and the ids of exactly those rows are captured before
anything runs, so a turn that arrives in the meantime survives.

These drive the real endpoint end to end, because the interesting parts (what the
model is handed, what survives in the transcript, what the stream says) are only
observable together.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete

from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import Attachment, Conversation
from app.parsing.base import ParsedDocument
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from app.services import rag_service
from app.services.attachment_ingest import build_ingest_items
from tests.conftest import make_memory_store
from tests.unit.test_chat_persistence import HookedProvider
from tests.unit.test_chat_rag import (
    CHAT_URL,
    citing_provider,
    client_for,
    collect_events,
    make_settings,
)

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")

ASK = "What is the refund window for orders?"

# Borrowed wholesale from the attachment tests: a file whose text shares no
# vocabulary with the question that names it, so only a deliberate re-feed of
# the stored reference can put it in front of the model.
FILE_TEXT = (
    "Zarquon Industries reported a 14 percent decline in quarterly turbine "
    "shipments, attributing the shortfall to a magnet supplier in Ostrava."
)
SUMMARISE = "Summarise the file I attached."


async def _conversation(db_sessionmaker) -> UUID:
    async with db_sessionmaker() as session:
        conversation = Conversation(tenant_id=TENANT.tenant_id, user_id=TENANT.user_id)
        session.add(conversation)
        await session.commit()
        return conversation.id


async def _messages(db_sessionmaker, conversation_id: UUID):
    async with db_sessionmaker() as session:
        return await MessageRepository(session, TENANT).list_for_conversation(conversation_id)


async def _conversation_row(db_sessionmaker, conversation_id: UUID):
    async with db_sessionmaker() as session:
        return await ConversationRepository(session, TENANT).get(conversation_id)


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


def _regen(conversation_id: UUID | str) -> dict:
    return {"conversation_id": str(conversation_id), "regenerate": True}


# --------------------------------------------------------------------------- #
# The point: one question, one reply, whichever way it was reached             #
# --------------------------------------------------------------------------- #


async def test_regenerating_replaces_the_reply_rather_than_adding_one(db_sessionmaker) -> None:
    # The headline behaviour. After a regenerate the thread still reads
    # question-then-answer: the first reply is GONE (not archived, not stacked
    # under a second one), and the question it was written for is untouched.
    provider = FakeProvider(script=["First answer."])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": ASK})
        conversation_id = UUID(dict(events)["done"]["conversation_id"])
        first = await _messages(db_sessionmaker, conversation_id)
        assert [m.role for m in first] == ["user", "assistant"]

        provider._script = ["Second answer."]  # noqa: SLF001 - re-script the same fake
        await collect_events(client, _regen(conversation_id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.role for m in after] == ["user", "assistant"]
    assert after[0].id == first[0].id, "the question must survive its answer being replaced"
    assert after[0].content == ASK
    assert after[1].content == "Second answer."
    assert after[1].id != first[1].id, "the old reply row is deleted, not rewritten"


async def test_the_regenerated_reply_is_asked_the_original_question(db_sessionmaker) -> None:
    # Nothing is sent with a regenerate, so the words the model sees can only
    # have come off the stored row - and it must see them exactly once, as the
    # turn's question, not twice (once replayed as history and once re-asked).
    provider = FakeProvider(script=["ok"])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": ASK})
        conversation_id = UUID(dict(events)["done"]["conversation_id"])
        await collect_events(client, _regen(conversation_id))

    replayed = [m.content for m in provider.last_messages if m.role != "system"]
    assert replayed == [ASK]


async def test_regenerating_works_when_the_question_has_no_reply_yet(db_sessionmaker) -> None:
    # An interrupted turn can leave a question with nothing under it (the reply
    # streamed no text, so none was saved). Regenerate is the retry for exactly
    # that case, so deleting zero rows must be a normal outcome, not an error.
    conversation_id = await _conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        await MessageRepository(session, TENANT).add(conversation_id, role="user", content=ASK)
        await session.commit()

    async with client_for(FakeProvider(script=["Now answered."]), db_sessionmaker) as client:
        events = await collect_events(client, _regen(conversation_id))

    assert dict(events)["done"]["message_id"]
    rows = await _messages(db_sessionmaker, conversation_id)
    assert [m.role for m in rows] == ["user", "assistant"]
    assert rows[1].content == "Now answered."


async def test_a_regenerate_that_produces_nothing_keeps_the_old_answer(db_sessionmaker) -> None:
    # The reply being replaced is dropped by the SAVE, not up front - so a
    # regenerate that streams nothing (an empty generation here; a provider
    # failure before the first token and a disconnect land the same way) leaves
    # the previous answer standing. Deleting eagerly would destroy it on the way
    # to failing, and the client puts its transcript back whenever the backend
    # did not take the turn, so it would go on showing an answer that is gone.
    provider = FakeProvider(script=["First answer."])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": ASK})
        conversation_id = UUID(dict(events)["done"]["conversation_id"])
        before = await _messages(db_sessionmaker, conversation_id)

        provider._script = []  # noqa: SLF001 - the model returns nothing at all
        await collect_events(client, _regen(conversation_id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.role for m in after] == ["user", "assistant"]
    assert after[1].id == before[1].id, "a regenerate that answered nothing must destroy nothing"
    assert after[1].content == "First answer."


async def test_regenerating_a_question_whose_only_file_is_gone_is_a_plain_400(
    db_sessionmaker,
) -> None:
    # A question that was a file and nothing else, whose file has since been
    # deleted, has nothing left to ask. Every other request shape is saved from
    # this by the validator; this is the one path that can reach it, and the
    # alternative is handing the model a blank user turn and passing off whatever
    # it says as an answer to a question nobody can read.
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_file(store, attachment, conversation_id)

    async with client_for(FakeProvider(script=["ok"]), db_sessionmaker, store=store) as client:
        # No words at all - the files ARE the question.
        await collect_events(
            client,
            {
                "message": "",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        async with db_sessionmaker() as session:
            await session.execute(delete(Attachment).where(Attachment.id == attachment.id))
            await session.commit()
        resp = await client.post(CHAT_URL, json=_regen(conversation_id))

    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"
    # And the turn it refused is untouched - the refusal costs nothing.
    assert [m.role for m in await _messages(db_sessionmaker, conversation_id)] == [
        "user",
        "assistant",
    ]


async def test_regenerating_an_empty_conversation_is_a_plain_400(db_sessionmaker) -> None:
    # Nothing was ever asked here, so there is no answer to replace. It is
    # refused BEFORE any stream opens (a problem+json, not an error frame), the
    # same as every other prepare-phase refusal.
    conversation_id = await _conversation(db_sessionmaker)
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json=_regen(conversation_id))

    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "bad_request"


async def test_regenerating_an_unknown_conversation_is_a_plain_404(db_sessionmaker) -> None:
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json=_regen(uuid4()))

    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


# --------------------------------------------------------------------------- #
# The exclusivity rules                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("extra", "what"),
    [
        ({"message": "and also this"}, "a new message"),
        ({"attachment_ids": [str(uuid4())]}, "attachments"),
        ({"confirmation": {"id": "c1", "decision": "approve"}}, "a confirmation"),
        ({"quote": {"text": "something"}}, "a quote"),
        (
            {"edit": {"message_id": str(uuid4()), "message": "reworded"}},
            "an edit",
        ),
    ],
)
async def test_a_regenerate_carrying_anything_else_is_rejected(
    db_sessionmaker, extra: dict, what: str
) -> None:
    # Everything a regenerate answers with comes from the stored question, so
    # anything sent alongside it would be silently ignored - and silently
    # ignoring a message is worse than refusing it. Each is a 422 on its own.
    conversation_id = await _conversation(db_sessionmaker)
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json={**_regen(conversation_id), **extra})

    assert resp.status_code == 422, f"a regenerate carrying {what} must be refused"


@pytest.mark.parametrize("blank", [" ", "\n\t "])
async def test_a_regenerate_carrying_only_whitespace_is_rejected(
    db_sessionmaker, blank: str
) -> None:
    # A body of spaces is still words the sender wrote, and the backend would
    # still throw them away - which is the exact thing the exclusivity rule
    # exists to stop. Refused, not quietly dropped.
    conversation_id = await _conversation(db_sessionmaker)
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json={**_regen(conversation_id), "message": blank})

    assert resp.status_code == 422


async def test_a_regenerate_must_name_its_conversation(db_sessionmaker) -> None:
    # It cannot be the first thing a conversation hears: there would be no
    # question to answer again.
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json={"regenerate": True})

    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# What comes back off the stored row                                           #
# --------------------------------------------------------------------------- #


async def test_the_files_the_question_arrived_with_are_re_fed(db_sessionmaker) -> None:
    # The question named a file; the regenerate names nothing. The file still has
    # to reach the model, and only the reference stored on the question can put
    # it there - the relevance floor is pinned impossibly high, so a re-feed is
    # the ONLY way its text can appear in the prompt or its title in a citation.
    store = make_memory_store()
    settings = make_settings(rag_min_score=0.99)
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_file(store, attachment, conversation_id)

    provider = citing_provider(["It is about turbines [1]."])
    async with client_for(provider, db_sessionmaker, settings, store=store) as client:
        await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        events = await collect_events(client, _regen(conversation_id))

    assert "Zarquon Industries" in "\n".join(m.content for m in provider.last_messages)
    cited = [data["title"] for name, data in events if name == "citation"]
    assert cited == ["turbines.txt"]


async def test_a_file_deleted_since_is_skipped_not_fatal(db_sessionmaker) -> None:
    # The reference on a question OUTLIVES the file it names - that is the whole
    # reason it is stored - so a regenerate routinely meets one pointing at
    # nothing. Naming a missing file on a fresh turn is a client mistake worth a
    # 404; a file deleted after the fact is nobody's mistake, so the turn answers
    # from what remains instead of refusing.
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_file(store, attachment, conversation_id)

    provider = FakeProvider(script=["Answered without it."])
    async with client_for(provider, db_sessionmaker, store=store) as client:
        await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        async with db_sessionmaker() as session:
            await session.execute(delete(Attachment).where(Attachment.id == attachment.id))
            await session.commit()

        events = await collect_events(client, _regen(conversation_id))

    assert dict(events)["done"]["message_id"], "a stale reference must not fail the turn"
    assert not [name for name, _ in events if name == "error"]
    rows = await _messages(db_sessionmaker, conversation_id)
    assert [m.role for m in rows] == ["user", "assistant"]
    assert rows[1].content == "Answered without it."


# --------------------------------------------------------------------------- #
# The stream and the sidebar                                                   #
# --------------------------------------------------------------------------- #


async def test_a_regenerate_stream_still_opens_by_naming_its_conversation(
    db_sessionmaker,
) -> None:
    # "meta is the first frame of EVERY stream" is a contract, and a regenerate
    # is a stream like any other - a client that never reaches done must still be
    # able to bind its pane.
    provider = FakeProvider(script=["again"])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": ASK})
        conversation_id = dict(events)["done"]["conversation_id"]
        again = await collect_events(client, _regen(conversation_id))

    assert again[0][0] == "meta"
    assert again[0][1]["conversation_id"] == conversation_id
    assert [name for name, _ in again].count("meta") == 1


async def test_done_carries_the_new_reply_id(db_sessionmaker) -> None:
    provider = FakeProvider(script=["fresh"])
    async with client_for(provider, db_sessionmaker) as client:
        first = await collect_events(client, {"message": ASK})
        conversation_id = UUID(dict(first)["done"]["conversation_id"])
        again = await collect_events(client, _regen(conversation_id))

    done = dict(again)["done"]
    assert done["conversation_id"] == str(conversation_id)
    rows = await _messages(db_sessionmaker, conversation_id)
    # The id names the row that is actually there - not the one just deleted.
    assert done["message_id"] == str(rows[1].id)
    assert done["message_id"] != dict(first)["done"]["message_id"]


async def test_regenerating_counts_as_activity(db_sessionmaker) -> None:
    # Asking again is fresh activity, so the thread rises in the sidebar. The
    # question's own timestamp does not move (the reply still sorts after it),
    # which is why the conversation clock has to be bumped separately.
    provider = FakeProvider(script=["again"])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": ASK})
        conversation_id = UUID(dict(events)["done"]["conversation_id"])
        before = await _conversation_row(db_sessionmaker, conversation_id)
        await collect_events(client, _regen(conversation_id))

    after = await _conversation_row(db_sessionmaker, conversation_id)
    assert after.updated_at > before.updated_at


# --------------------------------------------------------------------------- #
# Naming a reply: regenerating anywhere in the thread                          #
# --------------------------------------------------------------------------- #


def _regen_at(conversation_id: UUID | str, message_id: UUID | str) -> dict:
    return {**_regen(conversation_id), "message_id": str(message_id)}


async def _thread(client, provider: FakeProvider, turns: list[str]) -> UUID:
    """Send ``turns`` in order, re-scripting the fake for each; return the thread id."""
    conversation_id: UUID | None = None
    for index, text in enumerate(turns):
        provider._script = [f"Answer {index}."]  # noqa: SLF001 - re-script the same fake
        payload: dict = {"message": text}
        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)
        events = await collect_events(client, payload)
        conversation_id = UUID(dict(events)["done"]["conversation_id"])
    assert conversation_id is not None
    return conversation_id


async def test_regenerating_an_older_reply_replaces_it_and_drops_what_followed(
    db_sessionmaker,
) -> None:
    # The thread cannot keep turns that were written after an answer that no
    # longer exists in the form they replied to, so the reply and everything
    # below it goes - and the user was shown that count before agreeing.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id = await _thread(client, provider, [ASK, "And gift cards?", "And digital?"])
        rows = await _messages(db_sessionmaker, conversation_id)
        assert len(rows) == 6
        provider._script = ["A better first answer."]  # noqa: SLF001
        await collect_events(client, _regen_at(conversation_id, rows[1].id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.role for m in after] == ["user", "assistant"]
    assert after[0].id == rows[0].id, "the question it answers must survive"
    assert after[1].content == "A better first answer."


async def test_regenerating_an_older_reply_re_asks_the_question_above_it(
    db_sessionmaker,
) -> None:
    # "Regenerate this one" means the same thing wherever it is clicked: the
    # nearest question ABOVE the reply is re-asked, with only what preceded it as
    # history - not the newest question, and not the whole transcript.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id = await _thread(client, provider, [ASK, "And gift cards?", "And digital?"])
        rows = await _messages(db_sessionmaker, conversation_id)
        provider._script = ["again"]  # noqa: SLF001
        await collect_events(client, _regen_at(conversation_id, rows[3].id))

    replayed = [m.content for m in provider.last_messages if m.role != "system"]
    assert replayed == [ASK, "Answer 0.", "And gift cards?"]


async def test_a_targeted_regenerate_that_produces_nothing_destroys_nothing(
    db_sessionmaker,
) -> None:
    # Nothing is deleted at prepare: the ids are captured then, and removed only
    # in the transaction that writes the replacement. A regenerate that answers
    # nothing therefore leaves the WHOLE tail standing, exactly as the
    # newest-reply case leaves its single answer.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id = await _thread(client, provider, [ASK, "And gift cards?"])
        rows = await _messages(db_sessionmaker, conversation_id)
        provider._script = []  # noqa: SLF001 - the model returns nothing at all
        await collect_events(client, _regen_at(conversation_id, rows[1].id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.id for m in after] == [m.id for m in rows]


async def test_a_turn_that_arrived_after_the_count_is_not_swept_up(db_sessionmaker) -> None:
    # THE reason the doomed rows are a captured LIST rather than a rule. Another
    # tab appends a turn while the regenerate is streaming; it was never in the
    # count the user agreed to lose, so it survives. Recomputing "everything after
    # the reply" at save time would destroy it.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id = await _thread(client, provider, [ASK, "And gift cards?"])
        rows = await _messages(db_sessionmaker, conversation_id)

        async def other_tab() -> None:
            async with db_sessionmaker() as session:
                await MessageRepository(session, TENANT).add(
                    conversation_id, role="user", content="a turn from another tab"
                )
                await session.commit()

        hooked = HookedProvider(["A better first answer."], after_first_token=other_tab)
        async with client_for(hooked, db_sessionmaker) as second_tab:
            await collect_events(second_tab, _regen_at(conversation_id, rows[1].id))

    after = await _messages(db_sessionmaker, conversation_id)
    contents = [m.content for m in after]
    assert "a turn from another tab" in contents
    assert "A better first answer." in contents
    # And the rows that WERE counted are gone.
    assert "Answer 0." not in contents
    assert "And gift cards?" not in contents


async def test_only_the_named_reply_goes_not_the_one_above_it(db_sessionmaker) -> None:
    # A question can end up with more than one reply beside it (a partial that
    # landed late, a swap that stopped at a moved thread). Regenerating the
    # SECOND of them must take that one and what follows, and leave the first
    # alone: it is above what was clicked, so it was never part of the count the
    # user agreed to lose. This is why the narrow stop-at-the-next-question sweep
    # is switched OFF whenever an explicit list is carried - it would reach the
    # row the list was written not to touch.
    conversation_id = await _conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        repo = MessageRepository(session, TENANT)
        question = await repo.add(conversation_id, role="user", content=ASK)
        first = await repo.add(conversation_id, role="assistant", content="the earlier reply")
        second = await repo.add(conversation_id, role="assistant", content="the later reply")
        await session.commit()

    async with client_for(FakeProvider(script=["A replacement."]), db_sessionmaker) as client:
        await collect_events(client, _regen_at(conversation_id, second.id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.id for m in after[:2]] == [question.id, first.id]
    assert [m.content for m in after] == [ASK, "the earlier reply", "A replacement."]


async def test_regenerating_a_question_rather_than_a_reply_is_a_plain_400(
    db_sessionmaker,
) -> None:
    # message_id names the REPLY to redo. Pointed at a question it is refused
    # rather than reinterpreted: without the check it would resolve to the
    # question ABOVE that one and quietly re-answer the wrong turn, taking the
    # named question and its answer with it.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id = await _thread(client, provider, [ASK, "And gift cards?"])
        rows = await _messages(db_sessionmaker, conversation_id)
        resp = await client.post(CHAT_URL, json=_regen_at(conversation_id, rows[2].id))

    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"
    # And the turn it refused is untouched.
    assert [m.id for m in await _messages(db_sessionmaker, conversation_id)] == [m.id for m in rows]


async def test_regenerating_a_reply_that_is_not_there_is_a_plain_404(db_sessionmaker) -> None:
    # Its own code, not the conversation's: a second tab rewinding this thread
    # makes live conversations full of ids that are suddenly gone, and a client
    # that read that as "the conversation is deleted" would unbind a live pane.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id = await _thread(client, provider, [ASK])
        resp = await client.post(CHAT_URL, json=_regen_at(conversation_id, uuid4()))

    assert resp.status_code == 404
    assert resp.json()["code"] == "message_not_found"


async def test_regenerating_a_reply_with_no_question_above_it_is_a_plain_400(
    db_sessionmaker,
) -> None:
    # A reply with nothing above it cannot be re-asked - there is no question to
    # ask again. Refused before any stream, like every other prepare-phase refusal.
    conversation_id = await _conversation(db_sessionmaker)
    async with db_sessionmaker() as session:
        orphan = await MessageRepository(session, TENANT).add(
            conversation_id, role="assistant", content="an answer to nothing"
        )
        await session.commit()

    async with client_for(FakeProvider(script=["x"]), db_sessionmaker) as client:
        resp = await client.post(CHAT_URL, json=_regen_at(conversation_id, orphan.id))

    assert resp.status_code == 400


async def test_naming_a_reply_without_a_regenerate_is_rejected(db_sessionmaker) -> None:
    # message_id is the regenerate's one modifier. Anywhere else it would be
    # silently ignored, so it is refused rather than read as a second way to
    # address a turn.
    async with client_for(FakeProvider(), db_sessionmaker) as client:
        resp = await client.post(
            CHAT_URL,
            json={"message": "hi", "conversation_id": str(uuid4()), "message_id": str(uuid4())},
        )

    assert resp.status_code == 422


async def test_regenerating_the_newest_reply_by_name_matches_the_unnamed_form(
    db_sessionmaker,
) -> None:
    # Naming the newest reply is the same turn as not naming one: the question
    # above it is the newest question either way.
    provider = FakeProvider(script=["x"])
    async with client_for(provider, db_sessionmaker) as client:
        conversation_id = await _thread(client, provider, [ASK, "And gift cards?"])
        rows = await _messages(db_sessionmaker, conversation_id)
        provider._script = ["A different second answer."]  # noqa: SLF001
        await collect_events(client, _regen_at(conversation_id, rows[3].id))

    after = await _messages(db_sessionmaker, conversation_id)
    assert [m.role for m in after] == ["user", "assistant", "user", "assistant"]
    assert [m.id for m in after[:3]] == [m.id for m in rows[:3]]
    assert after[3].content == "A different second answer."


async def test_regenerating_does_not_rename_the_conversation(db_sessionmaker) -> None:
    # The thread was named by the exchange being re-run; renaming it now would
    # move a row in the sidebar for a question the reader already asked once. The
    # background title job is likewise not re-queued - there is no seed for it.
    provider = FakeProvider(script=["again"])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": ASK})
        conversation_id = UUID(dict(events)["done"]["conversation_id"])
        titled = await _conversation_row(db_sessionmaker, conversation_id)
        await collect_events(client, _regen(conversation_id))

    after = await _conversation_row(db_sessionmaker, conversation_id)
    assert after.title == titled.title == ASK
