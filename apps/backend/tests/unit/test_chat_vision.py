"""Showing a model the actual picture, and staying honest when it was not shown.

Every test here seeds REAL image bytes into storage. That matters: the rest of
the attachment suite seeds rows without bytes, so a passthrough that silently
delivered nothing would leave those tests green on a false premise. The
assertions below read what the provider was actually handed.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest
from PIL import Image

from app.providers.base import ChatMessage, ImageContent, ModelInfo
from app.providers.fake_provider import FakeProvider
from app.services.context_budget import approx_tokens
from app.services.image_prep import normalize
from app.services.model_facts import clear_facts_cache
from app.storage.base import StorageUnavailableError
from app.storage.local import FakeStorageBackend
from tests.conftest import make_memory_store
from tests.unit.test_chat_attachments import (
    SUMMARISE,
    TENANT,
    _attachment,
    _conversation,
    _ingest_attachment_text,
    _statuses,
)
from tests.unit.test_chat_rag import (
    CHAT_MODEL,
    THINKING_MODEL,
    client_for,
    collect_events,
    make_settings,
)

# A window small enough that the CONTEXT WINDOW binds the retrieval budget rather
# than the configured ceiling. With a roomy window min(ceiling, room) returns the
# ceiling either way and the image cost is invisible to the assertion.
TIGHT_WINDOW = 4096

# The configured chat pair. A fake reporting neither tag would have the turn
# substitute onto some other installed model, and the vision question would then
# be answered about weights no test here chose.
DEV_PAIR = [ModelInfo(name=CHAT_MODEL), ModelInfo(name=THINKING_MODEL)]


@pytest.fixture(autouse=True)
def _fresh_vision_cache():
    clear_facts_cache()
    yield
    clear_facts_cache()


def provisioned(**kwargs) -> FakeProvider:
    """A FakeProvider that reports holding the configured chat pair.

    Args:
        **kwargs: Passed through to :class:`FakeProvider`; ``vision`` is the
            one every test here sets, because the runtime's report is now the
            whole answer to whether a model may be shown pixels.

    Returns:
        The fake, listing the pair unless the caller seeded its own models.
    """
    kwargs.setdefault("models", list(DEV_PAIR))
    return FakeProvider(**kwargs)


def png(width: int = 640, height: int = 480, color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


async def seed_image(
    db_sessionmaker,
    storage: FakeStorageBackend,
    conversation_id: UUID,
    *,
    filename: str,
    data: bytes | None = None,
):
    """A READY image row whose bytes really are in the store."""
    row = await _attachment(
        db_sessionmaker, conversation_id, filename=filename, content_type="image/png"
    )
    payload = png() if data is None else data
    await storage.put(row.storage_key, payload, content_type="image/png")
    return row


def sent_images(provider: FakeProvider):
    """The images on the last user message the provider was handed.

    Reads ``calls[0]`` rather than ``last_messages``: the latter is a live alias
    the tool loop appends to, so on a tool turn its final entry is a tool result
    rather than the user message.
    """
    user_message = provider.calls[0][-1]
    return user_message.images or []


async def run_turn(
    provider: FakeProvider,
    db_sessionmaker,
    storage: FakeStorageBackend,
    conversation_id: UUID,
    attachment_ids: list[str],
    *,
    message: str = "What does the attached chart show?",
    **setting_overrides,
):
    """Drive one turn that names attachments, and return its SSE events.

    Args:
        provider: The fake the turn generates on; its report decides vision.
        db_sessionmaker: The test session factory.
        storage: The object store holding the image bytes.
        conversation_id: The conversation the attachments belong to.
        attachment_ids: The attachment ids the request names.
        message: The user's question.
        **setting_overrides: Extra settings fields for this turn.

    Returns:
        The collected (event name, payload) pairs.
    """
    settings = make_settings(**setting_overrides)
    async with client_for(
        provider, db_sessionmaker, settings, store=make_memory_store(), storage=storage
    ) as client:
        return await collect_events(
            client,
            {
                "message": message,
                "conversation_id": str(conversation_id),
                "attachment_ids": attachment_ids,
            },
        )


async def test_a_delivered_picture_reaches_the_model_and_silences_the_notice(
    db_sessionmaker,
) -> None:
    # Nothing here opts this model in, and that is the point. The gate used to
    # need a written-down declaration on top of the runtime's report, and a
    # capable model on a server nobody had described was therefore silently
    # blind. A tag is mutable, so the measurement always had to agree anyway -
    # and with the report deciding alone, agreeing is all it takes.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="chart.png")
    provider = provisioned(script=["A bar", " chart."], vision=True)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    images = sent_images(provider)
    assert len(images) == 1, "the pixels must actually reach the provider"
    assert images[0].data, "the image must carry bytes, not an empty payload"
    assert images[0].media_type.startswith("image/")
    # Nothing to warn about: the picture was shown.
    assert "image_not_seen" not in _statuses(events)


async def test_a_delivered_picture_counts_toward_the_reported_window_use(db_sessionmaker) -> None:
    # The done frame's context accounting must mirror the budget floor's
    # formula, image term included: pixels are 12-49% of a small window (the
    # runtime floors an image around a thousand tokens), so a meter that
    # counted only text would under-report a vision turn by up to half the
    # window while the budgeter trimmed against the true number.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    data = png()
    row = await seed_image(
        db_sessionmaker, storage, conversation_id, filename="chart.png", data=data
    )
    provider = provisioned(script=["A bar", " chart."], vision=True)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    assert len(sent_images(provider)) == 1
    usage = dict(events)["done"]["context_usage"]
    text_only = sum(approx_tokens(m.content) for m in provider.calls[0])
    # The exact cost the delivery charged: the same normalization, same budget.
    _, _, image_cost = normalize(data, budget_cells=1024)
    assert image_cost > 0
    assert usage["prompt_tokens"] == text_only + image_cost


async def test_a_blind_model_keeps_the_notice_and_sends_no_pixels(db_sessionmaker) -> None:
    # The clamp direction that makes the report-only gate safe: nothing can say
    # yes over a runtime that says no. It is the only direction left - there is
    # no opt-in to override, so a model that cannot see is told about, never
    # shown.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="chart.png")
    provider = provisioned(script=["Only", " text."], vision=False)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    assert sent_images(provider) == []
    notice = _statuses(events)["image_not_seen"]
    assert "chart.png" in notice
    assert "cannot look at pictures" in notice


async def test_a_mixed_turn_names_only_the_picture_it_could_not_show(db_sessionmaker) -> None:
    # The case a single per-turn flag cannot express: one delivered, one not.
    # The notice must name the skipped file and NOT the delivered one, or it
    # tells the user their answer ignored a picture the model actually saw.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    first = await seed_image(db_sessionmaker, storage, conversation_id, filename="shown.png")
    second = await seed_image(db_sessionmaker, storage, conversation_id, filename="skipped.png")
    provider = provisioned(script=["One", " chart."], vision=True)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(first.id), str(second.id)],
        vision_max_images=1,
    )

    assert len(sent_images(provider)) == 1
    notice = _statuses(events)["image_not_seen"]
    assert "skipped.png" in notice
    assert "shown.png" not in notice


async def test_unreadable_bytes_keep_the_notice_and_do_not_fail_the_turn(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(
        db_sessionmaker, storage, conversation_id, filename="broken.png", data=b"not an image"
    )
    provider = provisioned(script=["Still", " answered."], vision=True)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    assert sent_images(provider) == []
    notice = _statuses(events)["image_not_seen"]
    assert "broken.png" in notice
    assert "could not be prepared" in notice
    # The answer still arrives: one bad file must not cost the turn its reply.
    assert any(name == "token" for name, _ in events)
    assert not any(name == "error" for name, _ in events)


async def test_a_dead_store_keeps_the_notice_and_does_not_fail_the_turn(db_sessionmaker) -> None:
    class DeadStorage(FakeStorageBackend):
        async def get(self, key: str) -> bytes:
            raise StorageUnavailableError("the store is down")

    storage = DeadStorage()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="chart.png")
    provider = provisioned(script=["Answered", " anyway."], vision=True)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    assert sent_images(provider) == []
    notice = _statuses(events)["image_not_seen"]
    assert "chart.png" in notice
    assert "stored copy could not be read" in notice
    assert not any(name == "error" for name, _ in events)


async def test_a_probe_failure_keeps_the_notice(db_sessionmaker) -> None:
    class UnprobableProvider(FakeProvider):
        async def capabilities(self, model: str):
            raise TimeoutError("the runtime did not answer")

    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="chart.png")
    provider = UnprobableProvider(script=["No", " pixels."], vision=True, models=list(DEV_PAIR))

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    assert sent_images(provider) == []
    assert "chart.png" in _statuses(events)["image_not_seen"]


async def test_a_grounded_turn_still_carries_its_pixels(db_sessionmaker) -> None:
    """The regression guard for the rebuild that drops fields.

    Grounding rebuilds the final user message to fold retrieved context in. Built
    from role and content alone it drops the images - and ONLY on the branch where
    retrieval found something, which for an image turn is the normal branch, since
    what retrieval finds is the OCR text of that same picture. The notice would
    already have been suppressed on the grounds the pixels were delivered.

    This test MUST actually ground, or it guards nothing: the rebuild sits behind
    `if chunks:`, so a turn that retrieves nothing skips it and the guard passes
    for the wrong reason. That is not hypothetical - the first version of this
    test seeded no vectors and survived the bug being reintroduced. Hence the
    ingest below, and the assertion that the text really was wrapped.
    """
    store = make_memory_store()
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="chart.png")
    # Put the picture's OCR text where the worker would have, so retrieval has
    # something to find and the rebuild branch is genuinely taken.
    await _ingest_attachment_text(store, row, conversation_id)
    provider = provisioned(script=["Grounded", " answer [1]."], vision=True)

    settings = make_settings()
    async with client_for(
        provider, db_sessionmaker, settings, store=store, storage=storage
    ) as client:
        events = await collect_events(
            client,
            {
                "message": SUMMARISE,
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(row.id)],
            },
        )

    sent = provider.calls[0][-1]
    # The turn really did ground - otherwise the rebuild never ran and the rest
    # of this test proves nothing.
    assert "Zarquon" in sent.content, "expected the retrieved text to be folded in"
    assert sent.images, "grounding must not drop the pixels it copies forward"
    assert "image_not_seen" not in _statuses(events)


async def test_a_large_picture_is_shrunk_before_it_is_sent(db_sessionmaker) -> None:
    # A screenshot-sized upload must not arrive at its full token cost.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    original = png(2560, 1440, color=(10, 120, 200))
    row = await seed_image(
        db_sessionmaker, storage, conversation_id, filename="big.png", data=original
    )
    provider = provisioned(script=["Shrunk."], vision=True)

    await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    images = sent_images(provider)
    assert len(images) == 1
    with Image.open(io.BytesIO(images[0].data)) as sent:
        assert max(sent.size) < 2560, "an oversized picture must be resized down"


async def test_a_small_picture_is_sent_untouched(db_sessionmaker) -> None:
    # Below the runtime's floor, shrinking costs detail and saves no tokens at
    # all, so an image that already fits must be forwarded byte for byte.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    original = png(320, 240, color=(5, 200, 90))
    row = await seed_image(
        db_sessionmaker, storage, conversation_id, filename="small.png", data=original
    )
    provider = provisioned(script=["Untouched."], vision=True)

    await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    assert sent_images(provider)[0].data == original


async def test_the_count_cap_and_the_byte_cap_give_different_reasons(db_sessionmaker) -> None:
    """Two caps, two explanations - one code for both told the user a falsehood.

    The byte cap used to report "this turn already carries as many pictures as
    it can show", which for a single oversized image is simply untrue (the turn
    carried none) and points at the wrong remedy: send fewer, rather than send
    smaller.
    """
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="huge.png")
    provider = provisioned(script=["Nope."], vision=True)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
        vision_max_source_bytes=1,
    )

    notice = _statuses(events)["image_not_seen"]
    assert "huge.png" in notice
    assert "larger than one turn will read" in notice
    # The count-cap wording must NOT appear: the turn carried no pictures at all.
    assert "already carries as many pictures" not in notice


async def test_a_third_image_is_refused_by_the_count_cap(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    rows = [
        await seed_image(db_sessionmaker, storage, conversation_id, filename=f"p{n}.png")
        for n in range(3)
    ]
    provider = provisioned(script=["Two."], vision=True)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(r.id) for r in rows],
        vision_max_images=2,
    )

    assert len(sent_images(provider)) == 2
    notice = _statuses(events)["image_not_seen"]
    assert "p2.png" in notice
    assert "already carries as many pictures" in notice


async def test_a_decompression_bomb_is_skipped_not_fatal(db_sessionmaker) -> None:
    # The failure this guards was the worst kind: the exception escaped the
    # whole turn, so the user got no answer AND no notice about the image.
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    bomb = png(16000, 12000)
    row = await seed_image(
        db_sessionmaker, storage, conversation_id, filename="bomb.png", data=bomb
    )
    provider = provisioned(script=["Answered", " regardless."], vision=True)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    assert sent_images(provider) == []
    assert "bomb.png" in _statuses(events)["image_not_seen"]
    assert any(name == "token" for name, _ in events), "the turn must still answer"
    assert not any(name == "error" for name, _ in events)


async def test_a_delivered_picture_shrinks_the_room_left_for_retrieved_text(
    db_sessionmaker,
) -> None:
    """The slice's load-bearing safety claim, pinned end to end.

    Image tokens are invisible to approx_tokens, so they are handed to the RAG
    budget explicitly. If that wiring is dropped, retrieval hands back room the
    pixels already took and the runtime silently truncates the prompt from the
    HEAD - losing the system prompt with the tool guidance and injection defense
    in it, with no error anywhere. So: the same turn, same corpus, same tiny
    window, differing only in whether a picture was delivered. The one carrying
    pixels must be given strictly less retrieved text.
    """

    async def prompt_length(*, vision: bool) -> int:
        # The probe cache is keyed on (provider_name, model), which is right in
        # production - same tag, same weights, same answer - but here two
        # differently-capable fakes share one key, so the second run would read
        # the first run's verdict and the comparison would be against itself.
        clear_facts_cache()
        store = make_memory_store()
        storage = FakeStorageBackend()
        conversation_id = await _conversation(db_sessionmaker)
        row = await seed_image(db_sessionmaker, storage, conversation_id, filename="chart.png")
        # A corpus big enough that the budget actually TRIMS. With one short
        # passage nothing is ever dropped and both turns come out identical -
        # the first version of this test did exactly that and proved nothing.
        await _ingest_long_text(store, row, conversation_id)
        provider = provisioned(script=["Answer [1]."], vision=vision)
        settings = make_settings(default_context_window=TIGHT_WINDOW)
        async with client_for(
            provider, db_sessionmaker, settings, store=store, storage=storage
        ) as client:
            await collect_events(
                client,
                {
                    "message": SUMMARISE,
                    "conversation_id": str(conversation_id),
                    "attachment_ids": [str(row.id)],
                },
            )
        return sum(len(m.content) for m in provider.calls[0])

    with_pixels = await prompt_length(vision=True)
    without_pixels = await prompt_length(vision=False)
    assert with_pixels < without_pixels, (
        "a delivered image must reduce the room retrieval is given; "
        f"got {with_pixels} chars with pixels vs {without_pixels} without"
    )


async def _ingest_long_text(store, attachment, conversation_id) -> None:
    """Ingest a document long enough that the retrieval budget has to trim it."""
    from app.parsing.base import ParsedDocument
    from app.providers.rag.embedder import FakeEmbedder
    from app.services import rag_service
    from app.services.attachment_ingest import build_ingest_items
    from tests.unit.test_chat_attachments import TENANT

    body = " ".join(
        f"Passage {n}: Zarquon Industries reported a decline in quarterly turbine "
        f"shipments, attributing the shortfall to a magnet supplier in Ostrava."
        for n in range(120)
    )
    await rag_service.ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=build_ingest_items(attachment, [ParsedDocument(ordinal=0, text=body)]),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=make_settings(),
        session_id=str(conversation_id),
    )


# --- the confirmation checkpoint (slice 8c-B) ---


async def test_a_delivered_image_is_not_also_called_unused(db_sessionmaker) -> None:
    """Two notices must not contradict each other about the same file.

    A text-free image is deliberately marked ready with ZERO chunks, precisely
    because a vision model can still see it - so its stored text is always
    "missing". Without excluding delivered pictures from the unused-file notice,
    the exact image class this feature exists to serve gets "this answer does not
    use chart.png" printed beside a picture the model read.
    """
    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="chart.png")
    # No ingest at all: the row is READY with no stored text, exactly as the
    # worker leaves a text-free image.
    provider = provisioned(script=["A bar chart."], vision=True)

    events = await run_turn(
        provider,
        db_sessionmaker,
        storage,
        conversation_id,
        [str(row.id)],
    )

    statuses = _statuses(events)
    assert sent_images(provider), "the picture must have been delivered for this to be the case"
    assert "chart.png" not in statuses.get("attachment_unavailable", "")
    assert "image_not_seen" not in statuses


async def test_the_checkpoint_stores_image_ids_not_pixels(db_sessionmaker) -> None:
    """Pixels must not land in the confirmation checkpoint.

    It is a JSON column re-read in full on every later turn and locked for update
    on resume, so base64 of a multi-megabyte picture would be paid for again and
    again - and it would put a durable copy of the user's image outside the
    object store, where deleting the attachment could never reach it.
    """
    from app.services.chat_service import _build_confirmation_checkpoint

    checkpoint = _build_confirmation_checkpoint(
        confirmation_id="c1",
        messages=[
            ChatMessage(role="system", content="rules"),
            ChatMessage(
                role="user",
                content="what is this?",
                images=[ImageContent(data=png(64, 64), media_type="image/png")],
            ),
        ],
        round_calls=[],
        confirmed_call_ids=[],
        model="m",
        think=None,
        num_ctx=8192,
        reserve_tokens=1024,
        citations=[],
        notices=[],
        tool_steps=[],
        generated_files=[],
        iteration=1,
        calls_made=1,
        result_tokens=0,
        turn_result_budget=2000,
        announce_tools_not_used=False,
        offered_tools=[],
        user_at=datetime.now(UTC),
        ttl_seconds=900,
        image_attachment_ids=["abc-123"],
    )

    blob = json.dumps(checkpoint)
    assert "images" not in checkpoint["messages"][1]
    assert checkpoint["image_attachment_ids"] == ["abc-123"]
    # The whole serialized checkpoint stays small - a base64 picture would not.
    assert len(blob) < 2000, f"checkpoint unexpectedly large ({len(blob)} bytes)"


async def test_a_resumed_turn_gets_its_picture_back(db_sessionmaker) -> None:
    """The pixels the paused half carried must reach the resumed half too.

    The checkpoint stores ids rather than pixels, so the resume re-reads the
    files. Skipping that would quietly answer the second half of the turn
    without the picture the first half was looking at.
    """
    from app.services.chat_service import _reattach_checkpoint_images

    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="chart.png")

    messages, missing = await _reattach_checkpoint_images(
        messages=[ChatMessage(role="user", content="what is this?")],
        attachment_ids=[str(row.id)],
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        storage=storage,
        settings=make_settings(),
    )

    assert missing == []
    assert messages[-1].images, "the resumed prompt must carry the picture again"
    assert messages[-1].images[0].data


async def test_a_file_deleted_during_the_confirmation_window_is_reported(
    db_sessionmaker,
) -> None:
    # The honest cost of storing ids instead of pixels. It must be reported,
    # never silently dropped - and it must never fail the resume, because the
    # user has already decided about a side-effecting action.
    from app.services.chat_service import _reattach_checkpoint_images

    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="gone.png")
    storage.objects.clear()  # the blob sweep, or a delete, got there first

    messages, missing = await _reattach_checkpoint_images(
        messages=[ChatMessage(role="user", content="what is this?")],
        attachment_ids=[str(row.id)],
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        storage=storage,
        settings=make_settings(),
    )

    assert [r.filename for r in missing] == ["gone.png"]
    assert not (messages[-1].images or []), "no pixels to attach, and none invented"


async def test_a_paused_and_resumed_turn_carries_its_picture_end_to_end(db_sessionmaker) -> None:
    """The whole halt-and-resume path, with a real picture on the turn.

    The unit tests above call the re-attach helper directly, which cannot catch
    the resume path simply not calling it - a mutation proved exactly that. This
    drives the real thing: a turn that delivers a picture, pauses on a
    side-effecting tool, and resumes. The prompt the model is handed AFTER the
    confirmation must still carry the pixels, or the second half of the turn
    answers about an image it cannot see.
    """
    from app.providers.base import ToolCall
    from app.providers.fake_provider import FakeTurn
    from tests.unit.test_chat_mcp_tools import (
        FakeMcpSource,
        RemoteTool,
        mcp_chat_client,
    )
    from tests.unit.test_chat_mcp_tools import make_settings as mcp_settings

    storage = FakeStorageBackend()
    conversation_id = await _conversation(db_sessionmaker)
    row = await seed_image(db_sessionmaker, storage, conversation_id, filename="chart.png")

    provider = provisioned(
        vision=True,
        turns=[
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c1", name="create_record", arguments={"query": "chart bug"})
                ]
            ),
            FakeTurn(tokens=["Filed it."]),
        ],
    )
    danger = RemoteTool("create_record", side_effecting=True, result="RECORD CREATED")
    # The MCP harness's settings, and a fake whose runtime reports vision - the
    # whole of what it now takes for the picture to be delivered on the paused
    # half of the turn.
    settings = mcp_settings()

    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        settings,
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([danger])],
        storage=storage,
    ) as client:
        paused = await collect_events(
            client,
            {
                "message": "file a bug about this chart",
                "conversation_id": str(conversation_id),
                "attachment_ids": [str(row.id)],
                "supports_confirmation": True,
            },
        )
        confirmations = [d for n, d in paused if n == "tool_confirmation"]
        assert len(confirmations) == 1, "the turn should suspend on one confirmation"
        assert provider.calls[0][-1].images, "the paused half must have carried the picture"

        before_resume = len(provider.calls)
        await collect_events(
            client,
            {
                "conversation_id": str(conversation_id),
                "confirmation": {"id": confirmations[0]["id"], "decision": "approve"},
            },
        )

    assert len(provider.calls) > before_resume, "the resume must have called the model again"
    resumed_user = next(
        message
        for message in reversed(provider.calls[-1])
        if message.role == "user" and message.images
    )
    assert resumed_user.images[0].data, "the resumed prompt must still carry the pixels"
