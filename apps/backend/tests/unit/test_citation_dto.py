"""Lenient parsing of stored citations into the MessageOut DTO."""

from datetime import UTC, datetime
from uuid import uuid4

from app.models import Message
from app.schemas.conversations import MessageOut, _citations_from_meta

GOOD = {
    "id": "p1",
    "index": 1,
    "source": "Refund policy",
    "snippet": "Orders can be refunded...",
    "score": 0.91,
    "title": "Refund policy",
    "doc_id": "rec-1001",
    "source_type": "records",
}


def make_message(meta: dict) -> Message:
    return Message(
        id=uuid4(),
        tenant_id=uuid4(),
        user_id=uuid4(),
        conversation_id=uuid4(),
        role="assistant",
        content="answer",
        meta=meta,
        created_at=datetime.now(UTC),
    )


def test_valid_citations_round_trip() -> None:
    out = MessageOut.from_entity(make_message({"citations": [GOOD]}))
    assert len(out.citations) == 1
    assert out.citations[0].index == 1
    assert out.citations[0].doc_id == "rec-1001"


def test_missing_meta_key_yields_empty_list() -> None:
    assert MessageOut.from_entity(make_message({})).citations == []
    assert MessageOut.from_entity(make_message({"interrupted": True})).citations == []


def test_malformed_entries_are_dropped_not_fatal() -> None:
    broken = {"id": "p2"}  # missing required fields
    out = _citations_from_meta({"citations": [GOOD, broken, "not-a-dict"]})
    assert [c.id for c in out] == ["p1"]


def test_none_citations_value_tolerated() -> None:
    assert _citations_from_meta({"citations": None}) == []


def test_interrupted_flag_surfaces_from_meta() -> None:
    # The interrupted flag is the second meta key deliberately re-exposed (after
    # citations), so a reloaded transcript can badge a cut-short reply.
    assert MessageOut.from_entity(make_message({"interrupted": True})).interrupted is True


def test_interrupted_defaults_false_without_flag() -> None:
    assert MessageOut.from_entity(make_message({})).interrupted is False
    assert MessageOut.from_entity(make_message({"citations": [GOOD]})).interrupted is False
