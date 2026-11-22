"""Attachments in a chat turn: naming a file, waiting for it, and honest notices."""

from __future__ import annotations

from uuid import UUID, uuid4

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import (
    ATTACHMENT_EXPIRED,
    ATTACHMENT_FAILED,
    ATTACHMENT_ORIGIN_UPLOAD,
    ATTACHMENT_PARSING,
    ATTACHMENT_PENDING,
    ATTACHMENT_READY,
    Attachment,
    Conversation,
)
from app.parsing.base import ParsedDocument
from app.providers.base import ChatMessage
from app.providers.rag.base import ScoredChunk
from app.providers.rag.embedder import FakeEmbedder
from app.services import chat_service, rag_service
from app.services.attachment_ingest import build_ingest_items
from tests.conftest import make_memory_store
from tests.unit.test_chat_rag import citing_provider, client_for, collect_events, make_settings

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")

# The file's content. Note the question the tests ask ("summarise the file I
# attached") shares NO vocabulary with it - which is the entire point: that is
# how people talk about an attachment, and it is exactly the query a similarity
# floor rejects.
FILE_TEXT = (
    "Zarquon Industries reported a 14 percent decline in quarterly turbine "
    "shipments, attributing the shortfall to a magnet supplier in Ostrava."
)
SUMMARISE = "Summarise the file I attached."

# A question that points at NOTHING: no filename, no "the file", and no word in
# common with what the file says. Nothing but an id can reach the attachment from
# here - which is what makes it the control. SUMMARISE cannot play that part any
# more, because pointing at a file in words is itself a way of reaching it now
# (see test_a_later_turn_reaches_a_file_it_only_points_at).
UNPOINTED = "Give me a short summary."


async def _conversation(db_sessionmaker) -> UUID:
    async with db_sessionmaker() as session:
        conversation = Conversation(tenant_id=TENANT.tenant_id, user_id=TENANT.user_id)
        session.add(conversation)
        await session.commit()
        return conversation.id


async def _attachment(
    db_sessionmaker,
    conversation_id: UUID | None,
    *,
    status: str = ATTACHMENT_READY,
    error: str | None = None,
    filename: str = "turbines.txt",
    content_type: str = "text/plain",
    tenant: TenantContext = TENANT,
) -> Attachment:
    async with db_sessionmaker() as session:
        attachment = Attachment(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            conversation_id=conversation_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(FILE_TEXT),
            storage_key=f"{tenant.tenant_id}/{conversation_id or uuid4()}/{uuid4()}",
            status=status,
            error=error,
        )
        session.add(attachment)
        await session.commit()
        return attachment


async def _ingest_attachment_text(store, attachment: Attachment, conversation_id: UUID) -> None:
    """Put the file's text where the worker would have put it."""
    await rag_service.ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=build_ingest_items(attachment, [ParsedDocument(ordinal=0, text=FILE_TEXT)]),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=make_settings(),
        session_id=str(conversation_id),
    )


def _citations(events: list[tuple[str, dict]]) -> list[dict]:
    return [data for name, data in events if name == "citation"]


def _statuses(events: list[tuple[str, dict]]) -> dict[str, str]:
    return {
        data.get("code", ""): data.get("message", "") for name, data in events if name == "status"
    }


# ------------------------------------------------------------------ the point


async def test_a_named_file_grounds_an_answer_the_relevance_floor_would_reject(
    db_sessionmaker,
) -> None:
    """The headline behaviour of this slice.

    "Summarise the file I attached" shares no vocabulary with the file, so every
    chunk of it scores low and the relevance floor throws it away. Naming the
    attachment says the reference IS the relevance - similarity only chooses
    WHICH parts to bring, never whether to bring any. The floor is pinned
    impossibly high here so the effect is unmistakable rather than a matter of
    which side of 0.35 a fake embedding happens to land on.
    """
    store = make_memory_store()
    settings = make_settings(rag_min_score=0.99)
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        payload = {
            "message": SUMMARISE,
            "conversation_id": str(conversation_id),
            "attachment_ids": [str(attachment.id)],
        }
        with_file = await collect_events(client, payload)
        without_file = await collect_events(
            client, {"message": UNPOINTED, "conversation_id": str(conversation_id)}
        )
        pointed_only = await collect_events(
            client, {"message": SUMMARISE, "conversation_id": str(conversation_id)}
        )

    assert [c["title"] for c in _citations(with_file)] == ["turbines.txt"]
    # Same store, same floor. A question that reaches for nothing gets nothing:
    # the floor really does throw the file away, which is what makes naming it
    # meaningful rather than incidental.
    assert _citations(without_file) == []
    # ...and pointing at the file in words reaches it just as an id does, with no
    # attachment_ids on the turn at all. This is the follow-up case: the tray
    # emptied when the file was sent, so "summarise the file I attached" is all a
    # later turn has left to work with.
    assert [c["title"] for c in _citations(pointed_only)] == ["turbines.txt"]


async def test_a_named_file_grounds_even_with_the_knowledge_base_disabled(
    db_sessionmaker,
) -> None:
    # rag_enabled=False turns off stored-record grounding. A file the user
    # attached to THIS turn is not a stored record - refusing to read it would
    # be answering a question about a document while holding the document.
    store = make_memory_store()
    settings = make_settings(rag_enabled=False, rag_min_score=0.99)
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    assert len(_citations(events)) == 1


# ------------------------------------------------------------------- honesty


async def test_a_failed_file_is_reported_rather_than_silently_ignored(db_sessionmaker) -> None:
    # An answer that quietly ignored the file it was asked about reads exactly
    # like an answer that read it and found nothing. Say which, and why.
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker,
        conversation_id,
        status=ATTACHMENT_FAILED,
        error="No readable text could be extracted from this file.",
    )

    async with client_for(citing_provider(), db_sessionmaker, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    notice = _statuses(events)["attachment_unavailable"]
    assert "turbines.txt" in notice
    assert "No readable text could be extracted from this file" in notice
    assert _citations(events) == []


async def test_an_image_says_only_its_text_was_read(db_sessionmaker) -> None:
    # An image reaches the model as the text OCR read out of it, and nothing
    # else. Say so - otherwise a photo that produced a vague answer looks like a
    # bug, when in truth the picture was never shown to anything.
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, conversation_id, filename="chart.png", content_type="image/png"
    )

    async with client_for(
        citing_provider(script=["A chart", " of sales."]), db_sessionmaker, store=store
    ) as client:
        events = await collect_events(
            client,
            {
                "message": "What does the attached chart show?",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    notice = _statuses(events).get("image_not_seen", "")
    assert "not shown to the model" in notice
    assert "chart.png" in notice


async def test_a_vision_model_does_not_silence_the_notice(db_sessionmaker) -> None:
    """The notice must not key on what the model CAN do, only on what was sent.

    It used to ask the model whether it could see, and stay quiet if it said yes.
    That was honest purely by accident: no tier runs a vision chat model, so the
    answer was always no. The moment one is configured - the obvious next step -
    the probe would say yes, the notice would vanish, and the pixels would STILL
    not be sent, because nothing carries them: ChatMessage has no images field and
    no provider forwards any. The result would be an answer written from OCR text
    alone with nothing left to say so, and it would look like a fix.

    So the notice fires for a seeing model too, and stays honest until the day
    something actually sends the picture.
    """
    from app.providers.fake_provider import FakeProvider

    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker, conversation_id, filename="chart.png", content_type="image/png"
    )

    seeing = FakeProvider(script=["A bar", " chart."], vision=True)
    async with client_for(seeing, db_sessionmaker, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": "What does the attached chart show?",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    assert "chart.png" in _statuses(events)["image_not_seen"]


async def test_no_image_notice_for_a_non_image_file(db_sessionmaker) -> None:
    # A text file attached to a model that cannot see images is not an image, so
    # the "cannot see" notice must not fire for it.
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)  # text/plain
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    assert "image_not_seen" not in _statuses(events)


async def test_a_file_still_parsing_when_the_wait_runs_out_is_reported(db_sessionmaker) -> None:
    store = make_memory_store()
    # A wait of zero: the turn checks once and proceeds. A chat turn may not hang.
    settings = make_settings(attachment_wait_seconds=0.0, attachment_poll_seconds=0.0)
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id, status=ATTACHMENT_PARSING)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    statuses = _statuses(events)
    assert "turbines.txt" in statuses["attachment_processing"]
    assert "still being processed" in statuses["attachment_unavailable"]


async def test_a_file_that_aged_out_is_reported_as_gone_not_as_pending(db_sessionmaker) -> None:
    # An expired file is as settled as a failed one: the retention sweep emptied
    # it and no job is ever coming back to the row. Treated as unfinished it
    # would cost the turn the WHOLE wait - a generous window, on a question that
    # can never be improved by waiting - and then be described as "still being
    # processed", which is untrue twice over. The wait here is deliberately long,
    # so a turn that stalls on it fails this test on the clock as well as on the
    # wording.
    store = make_memory_store()
    settings = make_settings(attachment_wait_seconds=30.0, attachment_poll_seconds=0.5)
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id, status=ATTACHMENT_EXPIRED)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    statuses = _statuses(events)
    # Never announced as processing: nothing is processing it.
    assert "attachment_processing" not in statuses
    notice = statuses["attachment_unavailable"]
    assert "turbines.txt" in notice
    assert "no longer stored" in notice
    assert "still being processed" not in notice


async def test_the_turn_waits_for_a_file_that_is_still_being_parsed(
    db_sessionmaker, monkeypatch
) -> None:
    # The ordinary way to use an attachment: upload it, then immediately ask.
    # The upload responds as soon as the bytes are stored, so the file is still
    # `pending` when the question arrives. Without the wait, the answer would
    # ignore a file that was seconds from being readable.
    #
    # The worker that flips the row lives in ANOTHER PROCESS, so it is stubbed
    # rather than raced: an earlier version of this test ran a real background
    # task against the tests' single shared sqlite connection while the poll loop
    # hammered it, and on a loaded CI runner the writer never got a slot. That is
    # an artefact of one in-memory connection, not of the code (production polls
    # Postgres from its own pool), but a test that fails on a busy machine is
    # worthless either way.
    store = make_memory_store()
    settings = make_settings(
        attachment_wait_seconds=5.0, attachment_poll_seconds=0.0, rag_min_score=0.99
    )
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id, status=ATTACHMENT_PENDING)
    await _ingest_attachment_text(store, attachment, conversation_id)

    polls = 0
    real_load = chat_service._load_attachments

    async def parsing_then_ready(sessionmaker, tenant, attachment_ids):
        nonlocal polls
        rows = await real_load(sessionmaker, tenant, attachment_ids)
        polls += 1
        if polls > 1:  # the worker finished between this poll and the last
            for row in rows:
                row.status = ATTACHMENT_READY
        return rows

    monkeypatch.setattr(chat_service, "_load_attachments", parsing_then_ready)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    statuses = _statuses(events)
    # It saw the file was not ready, said so, and waited...
    assert "attachment_processing" in statuses
    assert polls > 1
    # ...and once it was ready the answer used it. No excuse needed.
    assert "attachment_unavailable" not in statuses
    assert len(_citations(events)) == 1


# ---------------------------------------------------------------- validation


async def test_a_malformed_attachment_id_is_a_400_not_a_broken_stream(db_sessionmaker) -> None:
    conversation_id = await _conversation(db_sessionmaker)

    async with client_for(citing_provider(), db_sessionmaker) as client:
        resp = await client.post(
            "/api/v1/chat/stream",
            json={
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": ["not-a-uuid"],
            },
        )

    # Validated in the awaited prepare phase, so it is a plain problem+json
    # response rather than an error frame inside a stream that already started.
    assert resp.status_code == 400


async def test_an_unknown_attachment_id_is_a_404(db_sessionmaker) -> None:
    conversation_id = await _conversation(db_sessionmaker)

    async with client_for(citing_provider(), db_sessionmaker) as client:
        resp = await client.post(
            "/api/v1/chat/stream",
            json={
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(uuid4())],
            },
        )

    assert resp.status_code == 404
    # A DISTINCT code from a missing conversation: a stale chip on a live thread
    # must not read as "this conversation is gone" and unbind the pane.
    assert resp.json()["code"] == "attachment_not_found"


async def test_a_missing_conversation_is_a_plain_not_found(db_sessionmaker) -> None:
    # The other side of the distinction: a conversation that does not exist
    # answers the generic not_found, so a client can unbind that pane while
    # leaving an attachment_not_found (a live thread, a stale chip) bound.
    async with client_for(citing_provider(), db_sessionmaker) as client:
        resp = await client.post(
            "/api/v1/chat/stream",
            json={
                "message": SUMMARISE,
                "conversation_id": str(uuid4()),
            },
        )

    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


async def test_another_conversations_attachment_is_a_404(db_sessionmaker) -> None:
    # An attachment belongs to exactly one thread. Naming one from a different
    # thread is as wrong as naming one that does not exist, and answers the same
    # way: which it was must not be inferable from the difference.
    mine = await _conversation(db_sessionmaker)
    theirs = await _conversation(db_sessionmaker)
    elsewhere = await _attachment(db_sessionmaker, theirs)

    async with client_for(citing_provider(), db_sessionmaker) as client:
        resp = await client.post(
            "/api/v1/chat/stream",
            json={
                "message": SUMMARISE,
                "conversation_id": str(mine),
                "attachment_ids": [str(elsewhere.id)],
            },
        )

    assert resp.status_code == 404


async def test_a_foreign_tenants_attachment_is_a_404(db_sessionmaker) -> None:
    other = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")
    conversation_id = await _conversation(db_sessionmaker)
    theirs = await _attachment(db_sessionmaker, conversation_id, tenant=other)

    async with client_for(citing_provider(), db_sessionmaker) as client:
        resp = await client.post(
            "/api/v1/chat/stream",
            json={
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(theirs.id)],
            },
        )

    assert resp.status_code == 404


async def test_naming_the_same_file_twice_is_naming_it_once(db_sessionmaker) -> None:
    store = make_memory_store()
    settings = make_settings(rag_min_score=0.99)
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id), str(attachment.id)],
            },
        )

    assert len(_citations(events)) == 1


async def test_a_turn_that_names_nothing_is_unchanged(db_sessionmaker) -> None:
    # The field is additive: every existing client keeps working, and a turn
    # with no attachment_ids must not pay a status frame or a wait for it.
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)

    async with client_for(citing_provider(), db_sessionmaker, store=store) as client:
        events = await collect_events(
            client, {"message": "Hello there.", "conversation_id": str(conversation_id)}
        )

    codes = _statuses(events)
    assert "attachment_processing" not in codes
    assert "attachment_unavailable" not in codes
    assert [name for name, _ in events][-1] == "done"


def test_settings_expose_the_attachment_turn_knobs() -> None:
    settings = Settings(_env_file=None)

    assert settings.attachment_wait_seconds == 30.0
    assert settings.rag_attachment_top_k == 12
    # Far larger than the knowledge base's ceiling, on purpose: a file someone
    # attached and asked about should arrive whole if the window has room.
    assert settings.rag_attachment_token_budget > settings.rag_token_budget


async def test_a_long_file_is_summarised_from_more_than_one_paragraph_of_itself(
    db_sessionmaker,
) -> None:
    """The bug this slice shipped with, and the reason merging exists.

    Every chunk of a file shares ONE doc_id (the only parser returns a single
    document, and splitting starts past 50k chars), so running a named file's
    passages through the per-source dedupe kept the first and threw the rest
    away: a long document was summarised from a single paragraph of itself, and
    rag_attachment_top_k was a knob that did nothing.

    The earlier tests all used files small enough to be one chunk, which is
    precisely why they could not see it.
    """
    store = make_memory_store()
    settings = make_settings(rag_min_score=0.99)
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    # Several distinct paragraphs, each a chunk, all under ONE doc_id.
    long_text = "\n\n".join(
        f"Section {n}. " + " ".join(f"finding-{n}-{w}" for w in range(120)) for n in range(6)
    )
    await rag_service.ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=build_ingest_items(attachment, [ParsedDocument(ordinal=0, text=long_text)]),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=settings,
        session_id=str(conversation_id),
    )

    grouped = await rag_service.retrieve_records(
        query=SUMMARISE,
        tenant=TENANT,
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=settings,
        session_id=str(conversation_id),
        record_ids=[str(attachment.id)],
    )
    merged = rag_service.merge_record_passages(grouped, budget_tokens=10_000)

    # Many passages of the file were retrieved...
    assert len(grouped) == 1
    assert len(grouped[0]) > 1
    # ...and they survive as ONE chunk (so the file is cited once, and the [n]
    # numbering stays aligned) carrying ALL of them, not just the best one.
    assert len(merged) == 1
    assert len(merged[0].text) > len(grouped[0][0].text)
    for passage in grouped[0]:
        assert passage.text in merged[0].text


async def test_a_merged_file_respects_its_share_of_the_context_window(db_sessionmaker) -> None:
    # trim_chunks_to_budget drops a chunk WHOLE. An unbounded merge would build
    # one too big to fit, and the file would then ground nothing at all - worse
    # than the bug it replaced.
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    long_text = "\n\n".join(
        f"Section {n}. " + " ".join(f"finding-{n}-{w}" for w in range(120)) for n in range(6)
    )
    await rag_service.ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=build_ingest_items(attachment, [ParsedDocument(ordinal=0, text=long_text)]),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=settings,
        session_id=str(conversation_id),
    )

    grouped = await rag_service.retrieve_records(
        query=SUMMARISE,
        tenant=TENANT,
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=settings,
        session_id=str(conversation_id),
        record_ids=[str(attachment.id)],
    )
    tiny = rag_service.merge_record_passages(grouped, budget_tokens=1)

    # Even an absurd budget still yields the file's best passage: a named file
    # always contributes SOMETHING rather than vanishing.
    assert len(tiny) == 1
    assert tiny[0].text == max(grouped[0], key=lambda c: c.score).text


async def test_two_named_files_do_not_starve_each_other(db_sessionmaker) -> None:
    # One query over both files would rank them against each other and the
    # better-matching one could take every slot. Each file is searched on its
    # own, so naming two files reads two files.
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    first = await _attachment(db_sessionmaker, conversation_id, filename="a.txt")
    second = await _attachment(db_sessionmaker, conversation_id, filename="b.txt")
    await _ingest_attachment_text(store, first, conversation_id)
    await rag_service.ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=build_ingest_items(
            second, [ParsedDocument(ordinal=0, text="An entirely unrelated note about badgers.")]
        ),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=settings,
        session_id=str(conversation_id),
    )

    grouped = await rag_service.retrieve_records(
        query=SUMMARISE,
        tenant=TENANT,
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=settings,
        session_id=str(conversation_id),
        record_ids=[str(first.id), str(second.id)],
    )

    assert len(grouped) == 2
    assert {group[0].record_id for group in grouped} == {str(first.id), str(second.id)}


async def test_a_named_file_is_retrievable_as_a_whole_not_one_document(db_sessionmaker) -> None:
    # A file becomes MANY rag documents (a page each, a part each). The chunks
    # are grouped by the attachment id, so naming the file addresses all of
    # them - a document id could only ever address one.
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await rag_service.ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=build_ingest_items(
            attachment,
            [
                ParsedDocument(ordinal=0, title="page 1", text=FILE_TEXT),
                ParsedDocument(ordinal=1, title="page 2", text="Ostrava supplies the magnets."),
            ],
        ),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=make_settings(),
        session_id=str(conversation_id),
    )

    grouped = await rag_service.retrieve_records(
        query=SUMMARISE,
        tenant=TENANT,
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=make_settings(),
        session_id=str(conversation_id),
        record_ids=[str(attachment.id)],
    )

    # One group (one file), holding the documents from BOTH of its pages.
    assert len(grouped) == 1
    assert {hit.doc_id for hit in grouped[0]} == {
        f"attachment:{attachment.id}:0:0",
        f"attachment:{attachment.id}:1:0",
    }


async def test_retrieval_by_record_never_reaches_another_source(db_sessionmaker) -> None:
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    mine = await _attachment(db_sessionmaker, conversation_id)
    other = await _attachment(db_sessionmaker, conversation_id, filename="unrelated.txt")
    await _ingest_attachment_text(store, mine, conversation_id)
    await rag_service.ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=build_ingest_items(
            other, [ParsedDocument(ordinal=0, text="Something else entirely.")]
        ),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=make_settings(),
        session_id=str(conversation_id),
    )

    grouped = await rag_service.retrieve_records(
        query=SUMMARISE,
        tenant=TENANT,
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=make_settings(),
        session_id=str(conversation_id),
        record_ids=[str(mine.id)],
    )

    assert len(grouped) == 1
    assert all(hit.doc_id.startswith(f"attachment:{mine.id}:") for hit in grouped[0])


# --------------------------------------------- reading order across documents


def _named_passage(file_id: UUID, ordinal: int, part: int, index: int, score: float, text: str):
    return ScoredChunk(
        id=f"{ordinal}:{part}:{index}",
        score=score,
        text=text,
        doc_id=f"attachment:{file_id}:{ordinal}:{part}",
        record_id=str(file_id),
        title=f"doc.pdf (part {ordinal + 1})",
        chunk_index=index,
    )


def test_merge_reads_a_multi_document_file_in_page_order() -> None:
    """A file that split into several documents merges in READING order.

    chunk_index restarts at 0 for every document a file produces (a page each,
    an oversized part each), so a merge that sorted on chunk_index ALONE left a
    multi-page file in score order with its pages interleaved. Reading order is
    (ordinal, part, chunk_index), read numerically off the doc_id.
    """
    file_id = uuid4()
    # As retrieve_records yields them - best score first: page two outscores one.
    page_two = _named_passage(file_id, ordinal=1, part=0, index=0, score=0.9, text="Page two body.")
    page_one = _named_passage(file_id, ordinal=0, part=0, index=0, score=0.4, text="Page one body.")

    merged = rag_service.merge_record_passages([[page_two, page_one]], budget_tokens=10_000)

    assert len(merged) == 1
    # Reading order wins over score order: page one leads.
    assert merged[0].text == "Page one body.\n\nPage two body."
    # Anchored on the file's FIRST passage, so the citation points at where the
    # file begins rather than at whichever part scored highest...
    assert merged[0].doc_id == f"attachment:{file_id}:0:0"
    assert merged[0].title == "doc.pdf (part 1)"
    # ...while still carrying the strongest score so ranking treats it as the hit.
    assert merged[0].score == 0.9


def test_merge_reads_chunks_within_a_document_in_order() -> None:
    """Within one document, the chunks still merge in chunk_index order."""
    file_id = uuid4()
    middle = _named_passage(file_id, ordinal=0, part=0, index=1, score=0.9, text="B middle.")
    first = _named_passage(file_id, ordinal=0, part=0, index=0, score=0.5, text="A first.")
    last = _named_passage(file_id, ordinal=0, part=0, index=2, score=0.3, text="C last.")

    merged = rag_service.merge_record_passages([[middle, first, last]], budget_tokens=10_000)

    assert len(merged) == 1
    assert merged[0].text == "A first.\n\nB middle.\n\nC last."


# ------------------------------------------------ the honest "unusable" reason


async def _retrieve(store, settings, conversation_id: UUID, record_ids, *, num_ctx: int):
    return await chat_service._retrieve_context(
        settings=settings,
        tenant=TENANT,
        session_id=str(conversation_id),
        user_message=SUMMARISE,
        messages=[ChatMessage(role="user", content=SUMMARISE)],
        embedder=FakeEmbedder(dim=32),
        store=store,
        reranker=None,
        num_ctx=num_ctx,
        reserve_tokens=0,
        include_kb=True,
        record_ids=record_ids,
    )


async def test_a_named_files_absent_text_is_reported_as_missing(db_sessionmaker) -> None:
    # The file whose session-scoped vectors have expired (or never landed): its
    # text simply is not there to retrieve. That is NOT a context-window problem,
    # and the old code that always blamed the window told the user to shrink a
    # file that was never too big.
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    # Deliberately NOT ingested - nothing to find.

    chunks, _citations_out, unfitted = await _retrieve(
        store, settings, conversation_id, [str(attachment.id)], num_ctx=8192
    )

    assert chunks == []
    assert unfitted == {str(attachment.id): "missing"}


async def test_a_named_file_with_no_window_room_is_reported_as_no_room(db_sessionmaker) -> None:
    # A window with no room for grounding at all: the file IS retrieved, then
    # trimmed out for lack of space. Only that reason is the honest one here.
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    chunks, _citations_out, unfitted = await _retrieve(
        store, settings, conversation_id, [str(attachment.id)], num_ctx=8
    )

    assert chunks == []
    assert unfitted == {str(attachment.id): "no_room"}


async def test_the_unusable_notice_states_the_reason_that_actually_applied(db_sessionmaker) -> None:
    # One sentence, one honest reason per file - not a single hard-coded cause.
    conversation_id = await _conversation(db_sessionmaker)
    down = await _attachment(db_sessionmaker, conversation_id, filename="down.txt")
    gone = await _attachment(db_sessionmaker, conversation_id, filename="gone.txt")
    big = await _attachment(db_sessionmaker, conversation_id, filename="big.txt")
    reasons = {
        str(down.id): "unavailable",
        str(gone.id): "missing",
        str(big.id): "no_room",
    }

    notice = chat_service._unfitted_notice([down, gone, big], reasons)

    assert "down.txt (retrieval was unavailable)" in notice
    assert "gone.txt (its stored text could not be found)" in notice
    assert "big.txt (it did not fit in the model's context window)" in notice
    # The bug this replaced stated the window reason for every file; now it is
    # attached to exactly the one file it applied to.
    assert notice.count("did not fit in the model's context window") == 1


# ------------------------------------- what the turn carried, and what qualifies it


async def test_the_files_a_turn_carried_are_recorded_on_its_message(db_sessionmaker) -> None:
    # The composer empties its tray the moment a message is sent, so the message
    # itself is the only thing that can still say which files the question came
    # with. A reloaded transcript reads it back from there.
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, make_settings(), store=store) as c:
        await collect_events(
            c,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        detail = (await c.get(f"/api/v1/conversations/{conversation_id}")).json()

    user_message = detail["messages"][0]
    assert user_message["role"] == "user"
    assert user_message["attachments"] == [
        {
            "id": str(attachment.id),
            "filename": "turbines.txt",
            "content_type": "text/plain",
            "size_bytes": len(FILE_TEXT),
            # How the file got here. An assistant reply can now carry files too
            # (ones a tool generated), so the reference says which kind it is
            # rather than leaving a client to infer it from the message's role.
            "origin": ATTACHMENT_ORIGIN_UPLOAD,
        }
    ]
    # This reply ran no file-producing tool, and an unattached turn records none.
    assert detail["messages"][1]["attachments"] == []


async def test_a_turn_that_names_no_file_records_none(db_sessionmaker) -> None:
    conversation_id = await _conversation(db_sessionmaker)

    async with client_for(citing_provider(), db_sessionmaker, make_settings()) as c:
        await collect_events(c, {"message": "hello", "conversation_id": str(conversation_id)})
        detail = (await c.get(f"/api/v1/conversations/{conversation_id}")).json()

    assert detail["messages"][0]["attachments"] == []


async def test_a_standing_notice_is_saved_with_the_answer_it_explains(db_sessionmaker) -> None:
    # The whole point of the notice is that the reader knows the answer did not
    # use their file. A notice that vanished on reload would leave the reply
    # looking like it had read the file all along.
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(
        db_sessionmaker,
        conversation_id,
        status=ATTACHMENT_FAILED,
        error="The file is password-protected.",
    )

    async with client_for(citing_provider(), db_sessionmaker, make_settings()) as c:
        events = await collect_events(
            c,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        detail = (await c.get(f"/api/v1/conversations/{conversation_id}")).json()

    live = _statuses(events)["attachment_unavailable"]
    notices = detail["messages"][1]["notices"]
    assert [n["code"] for n in notices] == ["attachment_unavailable"]
    # The stored notice is the one the client saw stream, word for word.
    assert notices[0]["message"] == live
    assert "turbines.txt" in notices[0]["message"]


async def test_a_transient_progress_status_is_not_stored_as_a_notice(db_sessionmaker) -> None:
    # "Searching stored records..." describes the wait, not the answer. Keeping it
    # would litter every reloaded transcript with stale progress chatter.
    store = make_memory_store()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, make_settings(), store=store) as c:
        events = await collect_events(
            c,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        detail = (await c.get(f"/api/v1/conversations/{conversation_id}")).json()

    # The transient status did stream...
    assert "retrieving" in _statuses(events)
    # ...but nothing was kept: this turn used its file, so it has nothing to explain.
    assert detail["messages"][1]["notices"] == []


# ------------------------------------------- a turn that is a file and nothing else


async def test_a_file_with_no_words_is_a_question(db_sessionmaker) -> None:
    """Attaching a file and pressing send, having typed nothing, is a real request.

    It is how every mainstream assistant is used, and it is the natural reply to
    "attach the logs and I will take a look". The model cannot be handed an empty
    turn, so the file becomes the ask - and the answer still grounds on it.
    """
    store = make_memory_store()
    settings = make_settings(rag_min_score=0.99)
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": "",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    assert [c["title"] for c in _citations(events)] == ["turbines.txt"]


async def test_the_words_nobody_typed_are_not_put_in_their_mouth(db_sessionmaker) -> None:
    # The model is asked about the file; the TRANSCRIPT keeps the empty message it
    # was actually sent. Storing the synthesized question would show the user
    # saying something they never wrote.
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        await collect_events(
            client,
            {
                "message": "",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        detail = (await client.get(f"/api/v1/conversations/{conversation_id}")).json()

    stored = [m for m in detail["messages"] if m["role"] == "user"]
    assert stored[0]["content"] == ""
    # The chips are what the turn shows instead, so they must be on it.
    assert [a["filename"] for a in stored[0]["attachments"]] == ["turbines.txt"]


async def test_a_wordless_first_turn_still_names_the_conversation(db_sessionmaker) -> None:
    # A title is derived from the first message, and there is no message. Falling
    # back to the filename beats an untitled row in the sidebar.
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        await collect_events(
            client,
            {
                "message": "",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )
        listed = await client.get("/api/v1/conversations")

    titles = {c["id"]: c["title"] for c in listed.json()["items"]}
    assert titles[str(conversation_id)] == "turbines.txt"


async def test_a_turn_with_neither_words_nor_files_is_rejected(db_sessionmaker) -> None:
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)

    async with client_for(citing_provider(), db_sessionmaker, settings) as client:
        resp = await client.post(
            "/api/v1/chat/stream",
            json={"message": "   ", "conversation_id": str(conversation_id)},
        )

    assert resp.status_code == 422


# ------------------------------------------------- saying what is being read


async def test_the_wait_says_which_file_it_is_reading(db_sessionmaker) -> None:
    """A progress line that names the wrong thing is worse than none at all.

    Every retrieval used to announce "Searching stored records", including a turn
    whose whole point was the file attached to it. The wait is the same; what the
    wait is FOR is the part worth knowing.
    """
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    assert "turbines.txt" in _statuses(events)["retrieving"]


async def test_the_wait_falls_back_to_stored_records_when_no_file_is_named(
    db_sessionmaker,
) -> None:
    # Nothing named, nothing fetched: the generic line is the honest one, and it
    # is the one every turn used to get.
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id)
    await _ingest_attachment_text(store, attachment, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client, {"message": UNPOINTED, "conversation_id": str(conversation_id)}
        )

    assert _statuses(events)["retrieving"] == "Searching stored records for relevant context..."


async def test_a_wordless_turn_does_not_drag_in_the_conversations_other_files(
    db_sessionmaker,
) -> None:
    """The turn is asked on the user's behalf; that question must not be read back.

    "I have attached these files: a, b. Read them..." points at files in exactly
    the way the file-reference rule looks for, so feeding it back in would match
    it against the WHOLE conversation and haul in every unrelated older upload.
    The rule reads what the user typed - here, nothing - so only the two files the
    turn actually carries are used.
    """
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    older = await _attachment(db_sessionmaker, conversation_id, filename="unrelated.txt")
    await _ingest_attachment_text(store, older, conversation_id)
    first = await _attachment(db_sessionmaker, conversation_id, filename="a.txt")
    second = await _attachment(db_sessionmaker, conversation_id, filename="b.txt")
    for carried in (first, second):
        await _ingest_attachment_text(store, carried, conversation_id)

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": "",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(first.id), str(second.id)],
            },
        )

    reading = _statuses(events)["retrieving"]
    assert "2 attached files" in reading
    assert "3" not in reading


async def test_a_guess_that_finds_nothing_does_not_accuse(db_sessionmaker) -> None:
    """A file the user never named must not be blamed on the answer.

    An inferred reference is a READING of the words, and readings can be wrong -
    "the file system in linux" looks just like someone pointing at an upload. If
    such a file yields no passages (its session vectors expired, say), announcing
    "this answer does not use screenshot.png" would put a permanent, visibly wrong
    notice on a reply about linux. A guess that misses says nothing; a file the
    user really named is a promise, and stays worth admitting to breaking.
    """
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    # Ready, pointed at by the words - but never ingested, so it holds nothing.
    await _attachment(db_sessionmaker, conversation_id, filename="screenshot.png")

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {"message": "what does the screenshot say?", "conversation_id": str(conversation_id)},
        )

    assert "attachment_unavailable" not in _statuses(events)


async def test_a_file_the_turn_named_is_still_admitted_when_it_yields_nothing(
    db_sessionmaker,
) -> None:
    # The other side of the line: this one the user really did point at, so
    # staying quiet about it would be answering as though the file had been read.
    store = make_memory_store()
    settings = make_settings()
    conversation_id = await _conversation(db_sessionmaker)
    attachment = await _attachment(db_sessionmaker, conversation_id, filename="screenshot.png")

    async with client_for(citing_provider(), db_sessionmaker, settings, store=store) as client:
        events = await collect_events(
            client,
            {
                "message": "what does it say?",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(attachment.id)],
            },
        )

    assert "screenshot.png" in _statuses(events)["attachment_unavailable"]
