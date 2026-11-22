"""Tests for the conversations REST API (create, list, detail, patch, delete).

Most tests insert rows directly through the sessionmaker fixture - which is
also how the tenant-isolation cases plant foreign-tenant data the API must
never return. The create endpoint mints an empty conversation up front (so a
client can attach a file before the first chat turn exists); the chat flow
still creates one implicitly on the first message when none was named.
Malformed ids map to 404 by design (documented in the router): a value that
cannot be a UUID addresses nothing, and answering exactly as for an absent id
keeps existence unanswerable in every case.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select

from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID
from app.models import Attachment, Conversation, Message
from app.providers.base import ModelInfo
from app.providers.fake_provider import FakeProvider
from app.workers.main import JOB_PURGE_ATTACHMENTS


@pytest.fixture
def fake_provider() -> FakeProvider:
    """The server holds only the thinking sibling of the configured pair.

    Overrides the conftest default for this module so the context-estimate
    tests can show both halves of the default resolution with one inventory:
    a stored pick of `qwen3:4b` is honoured as installed, while the missing
    instruct default routes to that installed partner exactly as a turn
    would. The other tests here never touch the provider.
    """
    return FakeProvider(models=[ModelInfo(name="qwen3:4b")])


# Anchored safely in the past so wall-clock time can never precede it (tests
# compare utcnow()-derived updated_at against BASE).
BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _ts(raw: str) -> datetime:
    """Parse an ISO timestamp, normalizing to naive UTC.

    Values read back from sqlite lose their offset while values serialized
    from in-memory entities keep it; normalizing makes the two comparable.
    """
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


async def _mk_conversation(
    db_sessionmaker,
    *,
    tenant_id: UUID = DEFAULT_TENANT_ID,
    user_id: UUID = DEFAULT_USER_ID,
    title: str | None = None,
    settings: dict | None = None,
    updated_at: datetime | None = None,
    pinned_at: datetime | None = None,
) -> Conversation:
    async with db_sessionmaker() as session:
        conversation = Conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            title=title,
            settings=settings if settings is not None else {},
            pinned_at=pinned_at,
        )
        if updated_at is not None:
            conversation.created_at = updated_at
            conversation.updated_at = updated_at
        session.add(conversation)
        await session.commit()
        return conversation


async def _mk_message(
    db_sessionmaker,
    conversation_id: UUID,
    *,
    role: str,
    content: str,
    thinking: str | None = None,
    model: str | None = None,
    created_at: datetime,
) -> Message:
    async with db_sessionmaker() as session:
        message = Message(
            tenant_id=DEFAULT_TENANT_ID,
            user_id=DEFAULT_USER_ID,
            conversation_id=conversation_id,
            role=role,
            content=content,
            thinking=thinking,
            model=model,
            created_at=created_at,
        )
        session.add(message)
        await session.commit()
        return message


async def _message_count(db_sessionmaker, conversation_id: UUID) -> int:
    async with db_sessionmaker() as session:
        stmt = (
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        )
        return (await session.execute(stmt)).scalar_one()


# ---------------------------------------------------------------- create


async def test_create_returns_a_new_titleless_conversation(client, db_sessionmaker) -> None:
    resp = await client.post("/api/v1/conversations")
    assert resp.status_code == 201
    data = resp.json()
    # Summary shape, no title yet: the first message derives one, and a fresh
    # conversation is never pinned.
    assert set(data) == {"id", "title", "pinned_at", "created_at", "updated_at"}
    assert data["title"] is None
    assert data["pinned_at"] is None
    new_id = data["id"]
    assert isinstance(new_id, str) and UUID(new_id)
    assert resp.headers["Location"] == f"/api/v1/conversations/{new_id}"

    # It is immediately usable: it appears in the list and its detail has no
    # messages (an upload can now name it before any turn exists).
    listed = (await client.get("/api/v1/conversations")).json()
    assert [c["id"] for c in listed["items"]] == [new_id]
    detail = (await client.get(f"/api/v1/conversations/{new_id}")).json()
    assert detail["messages"] == []
    assert detail["settings"] == {}


async def test_create_is_scoped_to_the_acting_tenant(client, db_sessionmaker) -> None:
    new_id = UUID((await client.post("/api/v1/conversations")).json()["id"])
    async with db_sessionmaker() as session:
        row = await session.get(Conversation, new_id)
        assert row is not None
        assert row.tenant_id == DEFAULT_TENANT_ID
        assert row.user_id == DEFAULT_USER_ID


# ---------------------------------------------------------------- list


async def test_list_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/conversations")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0}


async def test_list_orders_by_updated_at_desc_with_total(client, db_sessionmaker) -> None:
    for i, title in enumerate(["oldest", "middle", "newest"]):
        await _mk_conversation(db_sessionmaker, title=title, updated_at=BASE.replace(minute=i))

    resp = await client.get("/api/v1/conversations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert [c["title"] for c in data["items"]] == ["newest", "middle", "oldest"]
    # Summary shape: string id, no settings/messages payload.
    first = data["items"][0]
    assert isinstance(first["id"], str) and UUID(first["id"])
    assert set(first) == {"id", "title", "pinned_at", "created_at", "updated_at"}


async def test_list_limit_and_offset_page_through(client, db_sessionmaker) -> None:
    for i in range(3):
        await _mk_conversation(db_sessionmaker, title=f"c{i}", updated_at=BASE.replace(minute=i))

    resp = await client.get("/api/v1/conversations", params={"limit": 2, "offset": 1})
    assert resp.status_code == 200
    data = resp.json()
    # Page skips the newest (c2); total stays the tenant-wide count.
    assert [c["title"] for c in data["items"]] == ["c1", "c0"]
    assert data["total"] == 3


async def test_list_rejects_out_of_range_paging(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/conversations", params={"limit": 201})).status_code == 422
    assert (await client.get("/api/v1/conversations", params={"limit": 0})).status_code == 422
    assert (await client.get("/api/v1/conversations", params={"offset": -1})).status_code == 422


# ---------------------------------------------------------------- detail


async def test_detail_returns_messages_ascending_with_thinking(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="t", settings={"model": "qwen3:4b"})
    # Insert newest first to prove ordering comes from created_at, not insertion.
    await _mk_message(
        db_sessionmaker,
        conv.id,
        role="assistant",
        content="hi there",
        thinking="the user greeted me",
        model="fake",
        created_at=BASE.replace(minute=1),
    )
    await _mk_message(
        db_sessionmaker, conv.id, role="user", content="hello", created_at=BASE.replace(minute=0)
    )

    resp = await client.get(f"/api/v1/conversations/{conv.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(conv.id)
    assert data["title"] == "t"
    assert data["settings"] == {"model": "qwen3:4b"}
    assert [m["content"] for m in data["messages"]] == ["hello", "hi there"]
    user_msg, assistant_msg = data["messages"]
    assert user_msg["role"] == "user"
    assert user_msg["thinking"] is None and user_msg["model"] is None
    assert assistant_msg["thinking"] == "the user greeted me"
    assert assistant_msg["model"] == "fake"
    assert isinstance(assistant_msg["id"], str) and UUID(assistant_msg["id"])


async def test_detail_unknown_id_404(client: AsyncClient) -> None:
    resp = await client.get(f"/api/v1/conversations/{uuid4()}")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


async def test_detail_malformed_id_404(client: AsyncClient) -> None:
    # Documented choice: malformed ids are indistinguishable from absent ones.
    resp = await client.get("/api/v1/conversations/not-a-uuid")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


# ---------------------------------------------------------------- patch


async def test_patch_title_only_bumps_updated_at(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(
        db_sessionmaker, title="old", settings={"keep": True}, updated_at=BASE
    )

    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"title": "new"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "new"
    assert _ts(data["updated_at"]) > _ts(BASE.isoformat())

    detail = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    assert detail["title"] == "new"
    assert detail["settings"] == {"keep": True}  # untouched by a title-only patch


async def test_patch_settings_merges_per_field(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(
        db_sessionmaker,
        title="t",
        settings={"model": "qwen3:4b", "thinking": "on"},
        updated_at=BASE,
    )

    resp = await client.patch(
        f"/api/v1/conversations/{conv.id}", json={"settings": {"web_search": "off"}}
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "t"  # untouched by a settings-only patch

    detail = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    # The axes the patch did not carry survive: merged, not replaced.
    assert detail["settings"] == {"model": "qwen3:4b", "thinking": "on", "web_search": "off"}


async def test_patch_settings_null_clears_only_that_field(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(
        db_sessionmaker,
        settings={"model": "qwen3:4b", "thinking": "on", "web_search": "off"},
        updated_at=BASE,
    )

    resp = await client.patch(
        f"/api/v1/conversations/{conv.id}", json={"settings": {"thinking": None}}
    )
    assert resp.status_code == 200

    detail = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    assert detail["settings"] == {"model": "qwen3:4b", "web_search": "off"}  # siblings kept

    # The model axis clears the same way - it is the composer's "Auto" model pick.
    resp = await client.patch(
        f"/api/v1/conversations/{conv.id}", json={"settings": {"model": None}}
    )
    assert resp.status_code == 200
    detail = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    assert detail["settings"] == {"web_search": "off"}


async def test_patch_settings_stores_auto_as_a_real_pin(client, db_sessionmaker) -> None:
    """A stored "auto" is a tri-state value distinct from a null clear.

    It pins the heuristic for this thread even where a workspace default says
    on/off - the resolution chain deliberately lets a conversation "auto"
    shadow the workspace tier. The composer never writes it (its Auto is a
    null clear), but the API keeps it expressible and it must round-trip.
    """
    conv = await _mk_conversation(db_sessionmaker, updated_at=BASE)

    resp = await client.patch(
        f"/api/v1/conversations/{conv.id}", json={"settings": {"web_search": "auto"}}
    )
    assert resp.status_code == 200

    detail = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    assert detail["settings"] == {"web_search": "auto"}  # stored, not dropped as a clear


async def test_patch_settings_preserves_unknown_stored_keys(client, db_sessionmaker) -> None:
    """Keys written by another code version ride along untouched through a merge."""
    conv = await _mk_conversation(
        db_sessionmaker, settings={"legacy": True, "model": "old:tag"}, updated_at=BASE
    )

    await client.patch(f"/api/v1/conversations/{conv.id}", json={"settings": {"model": "new:tag"}})

    detail = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    assert detail["settings"] == {"legacy": True, "model": "new:tag"}


async def test_patch_settings_rejects_unknown_and_malformed_fields(client, db_sessionmaker) -> None:
    """Writes are strict where reads are lenient: a typo must not store a dead key."""
    conv = await _mk_conversation(db_sessionmaker, settings={"thinking": "on"}, updated_at=BASE)

    for bad in ({"thinkng": "on"}, {"thinking": "yes"}, {"model": ""}, {"model": "x" * 201}):
        resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"settings": bad})
        assert resp.status_code == 422, bad

    # The envelope is strict too: a typo'd top-level field must not no-op.
    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"setings": {}})
    assert resp.status_code == 422

    detail = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    assert detail["settings"] == {"thinking": "on"}  # no rejected write changed anything


async def test_patch_effectless_settings_is_a_noop(client, db_sessionmaker) -> None:
    """A patch already satisfied by the stored values must not bump updated_at."""
    conv = await _mk_conversation(db_sessionmaker, settings={"thinking": "on"}, updated_at=BASE)

    resp = await client.patch(
        f"/api/v1/conversations/{conv.id}", json={"settings": {"thinking": "on"}}
    )
    assert resp.status_code == 200
    assert _ts(resp.json()["updated_at"]) == _ts(BASE.isoformat())

    # An empty settings object carries no fields, so it changes nothing either.
    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"settings": {}})
    assert resp.status_code == 200
    assert _ts(resp.json()["updated_at"]) == _ts(BASE.isoformat())


async def test_patch_same_title_is_a_noop(client, db_sessionmaker) -> None:
    """A rename to the current title must not bump updated_at or reorder."""
    conv = await _mk_conversation(db_sessionmaker, title="t", updated_at=BASE)

    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"title": "t"})
    assert resp.status_code == 200
    assert _ts(resp.json()["updated_at"]) == _ts(BASE.isoformat())


async def test_settings_change_commits_without_reordering_the_sidebar(
    client, db_sessionmaker
) -> None:
    """Settings are metadata, not activity: a control flip must not resurface a thread.

    Only a title change (and real chat activity) bumps updated_at; a settings
    change persists without touching the sidebar order - otherwise flipping a
    composer toggle would jump an old conversation to the top with no message
    sent.
    """
    conv = await _mk_conversation(db_sessionmaker, title="t", updated_at=BASE)

    resp = await client.patch(
        f"/api/v1/conversations/{conv.id}", json={"settings": {"thinking": "on"}}
    )
    assert resp.status_code == 200
    assert _ts(resp.json()["updated_at"]) == _ts(BASE.isoformat())  # no bump

    detail = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    assert detail["settings"] == {"thinking": "on"}  # yet the change persisted


async def test_patch_title_and_settings_together(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="old", updated_at=BASE)

    resp = await client.patch(
        f"/api/v1/conversations/{conv.id}",
        json={"title": "both", "settings": {"thinking": "on"}},
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "both"

    detail = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    assert detail["title"] == "both"
    assert detail["settings"] == {"thinking": "on"}


async def test_patch_empty_body_is_noop(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="t", updated_at=BASE)

    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "t"
    assert _ts(data["updated_at"]) == _ts(BASE.isoformat())  # no bump on a no-op


async def test_patch_reorders_list(client, db_sessionmaker) -> None:
    await _mk_conversation(db_sessionmaker, title="newer", updated_at=BASE.replace(minute=1))
    older = await _mk_conversation(db_sessionmaker, title="older", updated_at=BASE)

    await client.patch(f"/api/v1/conversations/{older.id}", json={"title": "older-renamed"})
    data = (await client.get("/api/v1/conversations")).json()
    assert [c["title"] for c in data["items"]] == ["older-renamed", "newer"]


async def test_patch_validates_title_bounds(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="t")
    assert (
        await client.patch(f"/api/v1/conversations/{conv.id}", json={"title": ""})
    ).status_code == 422
    assert (
        await client.patch(f"/api/v1/conversations/{conv.id}", json={"title": "x" * 201})
    ).status_code == 422


async def test_patch_unknown_and_malformed_id_404(client: AsyncClient) -> None:
    assert (
        await client.patch(f"/api/v1/conversations/{uuid4()}", json={"title": "x"})
    ).status_code == 404
    assert (
        await client.patch("/api/v1/conversations/nope", json={"title": "x"})
    ).status_code == 404


# ---------------------------------------------------------------- pin / unpin


async def _stored_pinned_at(db_sessionmaker, conversation_id: UUID) -> datetime | None:
    async with db_sessionmaker() as session:
        row = await session.get(Conversation, conversation_id)
        assert row is not None
        return row.pinned_at


async def test_patch_pinned_true_stamps_the_server_clock(client, db_sessionmaker) -> None:
    """The client sends intent; the server owns the timestamp."""
    conv = await _mk_conversation(db_sessionmaker, title="t", updated_at=BASE)

    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"pinned": True})
    assert resp.status_code == 200
    stamped = resp.json()["pinned_at"]
    assert stamped is not None
    # Stamped now, not at the anchor the row was planted with - the value came
    # from the server's clock rather than from anything the request carried.
    assert _ts(stamped) > _ts(BASE.isoformat())
    assert await _stored_pinned_at(db_sessionmaker, conv.id) is not None


async def test_patch_pinned_false_clears_the_timestamp(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="t", pinned_at=BASE, updated_at=BASE)

    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"pinned": False})
    assert resp.status_code == 200
    assert resp.json()["pinned_at"] is None
    assert await _stored_pinned_at(db_sessionmaker, conv.id) is None


async def test_patch_without_pinned_leaves_the_pin_untouched(client, db_sessionmaker) -> None:
    """The per-field contract: a field the request omits is not written.

    A title-only or settings-only save must not silently unpin a thread, which
    is exactly what a patch that read an absent ``pinned`` as ``false`` would do.
    """
    conv = await _mk_conversation(db_sessionmaker, title="t", pinned_at=BASE, updated_at=BASE)

    for body in ({"title": "renamed"}, {"settings": {"thinking": "on"}}, {}):
        resp = await client.patch(f"/api/v1/conversations/{conv.id}", json=body)
        assert resp.status_code == 200, body
        stamped = resp.json()["pinned_at"]
        # Named explicitly: a silent unpin reads as a null here, and comparing
        # timestamps first would fail on the parse instead of on the fact.
        assert stamped is not None, body
        assert _ts(stamped) == _ts(BASE.isoformat()), body

    stored = await _stored_pinned_at(db_sessionmaker, conv.id)
    assert stored is not None and stored.replace(tzinfo=None) == BASE.replace(tzinfo=None)


async def test_pinning_an_already_pinned_conversation_does_not_restamp(
    client, db_sessionmaker
) -> None:
    """Re-pinning must not jump the thread to the head of the pinned group."""
    conv = await _mk_conversation(db_sessionmaker, title="t", pinned_at=BASE, updated_at=BASE)

    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"pinned": True})
    assert resp.status_code == 200
    assert _ts(resp.json()["pinned_at"]) == _ts(BASE.isoformat())


async def test_unpinning_an_unpinned_conversation_is_a_noop(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="t", updated_at=BASE)

    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"pinned": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pinned_at"] is None
    assert _ts(data["updated_at"]) == _ts(BASE.isoformat())


async def test_pinning_does_not_reorder_the_sidebar(client, db_sessionmaker) -> None:
    """A pin is metadata, not activity: it must not resurface an old thread."""
    conv = await _mk_conversation(db_sessionmaker, title="t", updated_at=BASE)

    resp = await client.patch(f"/api/v1/conversations/{conv.id}", json={"pinned": True})
    assert resp.status_code == 200
    assert _ts(resp.json()["updated_at"]) == _ts(BASE.isoformat())


async def test_list_carries_pinned_at(client, db_sessionmaker) -> None:
    pinned = await _mk_conversation(
        db_sessionmaker, title="pinned", pinned_at=BASE, updated_at=BASE
    )
    plain = await _mk_conversation(
        db_sessionmaker, title="plain", updated_at=BASE.replace(minute=1)
    )

    items = (await client.get("/api/v1/conversations")).json()["items"]
    by_id = {c["id"]: c for c in items}
    assert _ts(by_id[str(pinned.id)]["pinned_at"]) == _ts(BASE.isoformat())
    assert by_id[str(plain.id)]["pinned_at"] is None


async def test_list_orders_pinned_first_regardless_of_activity(client, db_sessionmaker) -> None:
    """The pin wins in the QUERY, not only in the client's grouping.

    Newest pin leads the pinned block; the unpinned rows keep their own
    activity order behind it.
    """
    await _mk_conversation(db_sessionmaker, title="busy", updated_at=BASE.replace(minute=9))
    await _mk_conversation(
        db_sessionmaker,
        title="pinned-recently",
        updated_at=BASE,
        pinned_at=BASE.replace(minute=5),
    )
    await _mk_conversation(
        db_sessionmaker,
        title="pinned-long-ago",
        updated_at=BASE.replace(minute=8),
        pinned_at=BASE.replace(minute=1),
    )
    await _mk_conversation(db_sessionmaker, title="quiet", updated_at=BASE.replace(minute=2))

    items = (await client.get("/api/v1/conversations")).json()["items"]
    assert [c["title"] for c in items] == [
        "pinned-recently",
        "pinned-long-ago",
        "busy",
        "quiet",
    ]


async def test_a_pinned_thread_stays_on_the_first_page(client, db_sessionmaker) -> None:
    """The failure pinning exists to prevent: falling off the only page read.

    The sidebar fetches page one and never asks for another, so a pin that only
    grouped client-side would silently disappear once enough threads had newer
    activity - no error, just a row gone from both groups.
    """
    await _mk_conversation(db_sessionmaker, title="pinned-stale", updated_at=BASE, pinned_at=BASE)
    for i in range(1, 4):
        await _mk_conversation(
            db_sessionmaker, title=f"newer{i}", updated_at=BASE.replace(minute=i)
        )

    data = (await client.get("/api/v1/conversations", params={"limit": 2})).json()
    assert data["total"] == 4
    assert [c["title"] for c in data["items"]] == ["pinned-stale", "newer3"]


async def test_patch_rejects_a_client_supplied_pinned_at(client, db_sessionmaker) -> None:
    """Only intent is on the wire - a timestamp is not a field a client may write."""
    conv = await _mk_conversation(db_sessionmaker, title="t", updated_at=BASE)

    resp = await client.patch(
        f"/api/v1/conversations/{conv.id}", json={"pinned_at": BASE.isoformat()}
    )
    assert resp.status_code == 422
    assert await _stored_pinned_at(db_sessionmaker, conv.id) is None


async def test_another_users_conversation_cannot_be_pinned(client, db_sessionmaker) -> None:
    """Scoping runs through the real request path, not a hand-built repository.

    The conversation shares the acting tenant and differs only by owner - the
    boundary the tenant filter alone misses. The request goes through the app's
    own ``get_current_tenant`` dependency, so what is proven here is what a real
    caller gets: a foreign row is indistinguishable from an absent one, and the
    attempt leaves it unpinned.
    """
    theirs = await _mk_conversation(
        db_sessionmaker, user_id=uuid4(), title="theirs", updated_at=BASE
    )

    resp = await client.patch(f"/api/v1/conversations/{theirs.id}", json={"pinned": True})
    assert resp.status_code == 404
    assert await _stored_pinned_at(db_sessionmaker, theirs.id) is None

    # It is not in the acting user's list either, so nothing could have named it.
    assert (await client.get("/api/v1/conversations")).json() == {"items": [], "total": 0}


async def test_a_foreign_tenants_conversation_cannot_be_pinned(client, db_sessionmaker) -> None:
    theirs = await _mk_conversation(
        db_sessionmaker, tenant_id=uuid4(), title="theirs", updated_at=BASE
    )

    resp = await client.patch(f"/api/v1/conversations/{theirs.id}", json={"pinned": True})
    assert resp.status_code == 404
    assert await _stored_pinned_at(db_sessionmaker, theirs.id) is None


# ---------------------------------------------------------------- delete


async def test_delete_cascades_and_disappears(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="doomed")
    await _mk_message(db_sessionmaker, conv.id, role="user", content="hello", created_at=BASE)
    await _mk_message(
        db_sessionmaker, conv.id, role="assistant", content="hi", created_at=BASE.replace(minute=1)
    )
    assert await _message_count(db_sessionmaker, conv.id) == 2

    resp = await client.delete(f"/api/v1/conversations/{conv.id}")
    assert resp.status_code == 204
    assert resp.content == b""

    assert (await client.get("/api/v1/conversations")).json() == {"items": [], "total": 0}
    assert (await client.get(f"/api/v1/conversations/{conv.id}")).status_code == 404
    assert await _message_count(db_sessionmaker, conv.id) == 0


async def test_delete_unknown_and_malformed_id_404(client: AsyncClient) -> None:
    assert (await client.delete(f"/api/v1/conversations/{uuid4()}")).status_code == 404
    assert (await client.delete("/api/v1/conversations/42")).status_code == 404


# ------------------------------------------------------- delete reclaims the files


async def _mk_attachment(db_sessionmaker, conversation_id: UUID) -> Attachment:
    async with db_sessionmaker() as session:
        attachment = Attachment(
            tenant_id=DEFAULT_TENANT_ID,
            user_id=DEFAULT_USER_ID,
            conversation_id=conversation_id,
            filename="notes.txt",
            content_type="text/plain",
            size_bytes=4,
            storage_key=f"{DEFAULT_TENANT_ID}/{conversation_id}/{uuid4()}",
        )
        session.add(attachment)
        await session.commit()
        return attachment


async def test_delete_queues_the_conversations_files_for_removal(
    client, db_sessionmaker, fake_job_queue
) -> None:
    # The ids must be read BEFORE the delete: the FK nulls conversation_id, so
    # afterwards nothing links these rows to the conversation they came from.
    conv = await _mk_conversation(db_sessionmaker, title="doomed")
    attachment = await _mk_attachment(db_sessionmaker, conv.id)

    assert (await client.delete(f"/api/v1/conversations/{conv.id}")).status_code == 204

    name, args, _ = fake_job_queue.calls[0]
    assert name == JOB_PURGE_ATTACHMENTS
    assert args == ([str(attachment.id)], str(DEFAULT_TENANT_ID), str(DEFAULT_USER_ID))
    assert f"attachment-purge:{conv.id}" in fake_job_queue.job_ids


async def test_delete_of_a_conversation_with_no_files_queues_nothing(
    client, db_sessionmaker, fake_job_queue
) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="empty")

    assert (await client.delete(f"/api/v1/conversations/{conv.id}")).status_code == 204

    assert fake_job_queue.calls == []


async def test_delete_still_succeeds_when_the_queue_is_down(
    client, db_sessionmaker, fake_job_queue
) -> None:
    # The conversation is already gone by the time the job is queued, so failing
    # the response would be a lie. The rows now have no conversation, which is
    # exactly what the periodic attachment sweep collects.
    conv = await _mk_conversation(db_sessionmaker, title="doomed")
    await _mk_attachment(db_sessionmaker, conv.id)
    fake_job_queue.unavailable = True

    assert (await client.delete(f"/api/v1/conversations/{conv.id}")).status_code == 204

    assert (await client.get(f"/api/v1/conversations/{conv.id}")).status_code == 404


# ---------------------------------------------------------------- tenant isolation


async def test_foreign_tenant_conversation_is_invisible(client, db_sessionmaker) -> None:
    mine = await _mk_conversation(db_sessionmaker, title="mine", updated_at=BASE)
    foreign = await _mk_conversation(
        db_sessionmaker, tenant_id=uuid4(), title="theirs", updated_at=BASE.replace(minute=1)
    )

    data = (await client.get("/api/v1/conversations")).json()
    assert data["total"] == 1
    assert [c["id"] for c in data["items"]] == [str(mine.id)]

    # Foreign ids behave exactly like absent ones on every verb.
    assert (await client.get(f"/api/v1/conversations/{foreign.id}")).status_code == 404
    assert (
        await client.patch(f"/api/v1/conversations/{foreign.id}", json={"title": "hijack"})
    ).status_code == 404
    assert (await client.delete(f"/api/v1/conversations/{foreign.id}")).status_code == 404

    # And the foreign row is untouched by those attempts.
    async with db_sessionmaker() as session:
        row = await session.get(Conversation, foreign.id)
        assert row is not None and row.title == "theirs"


# ------------------------------------------------- detail carries the conversation's files


async def test_detail_lists_the_conversations_attachments(client, db_sessionmaker) -> None:
    # A file uploaded but never sent is still the server's: the client restores it
    # into the composer on reopen rather than pretending the draft was lost.
    conv = await _mk_conversation(db_sessionmaker, title="t")
    attachment = await _mk_attachment(db_sessionmaker, conv.id)

    data = (await client.get(f"/api/v1/conversations/{conv.id}")).json()

    assert [a["id"] for a in data["attachments"]] == [str(attachment.id)]
    assert data["attachments"][0]["filename"] == "notes.txt"
    # The internal key never leaves the server.
    assert "storage_key" not in data["attachments"][0]


async def test_detail_of_a_conversation_with_no_files_lists_none(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="t")
    data = (await client.get(f"/api/v1/conversations/{conv.id}")).json()
    assert data["attachments"] == []


async def test_detail_does_not_list_another_conversations_files(client, db_sessionmaker) -> None:
    mine = await _mk_conversation(db_sessionmaker, title="mine")
    theirs = await _mk_conversation(db_sessionmaker, title="theirs")
    await _mk_attachment(db_sessionmaker, theirs.id)

    data = (await client.get(f"/api/v1/conversations/{mine.id}")).json()

    assert data["attachments"] == []


# ---------------------------------------------------- context estimate


def _expected_base_tokens(*contents: str) -> int:
    """The estimate's own arithmetic: system prompt plus the replayed texts.

    Computed from the same primitives the service uses so the assertions pin
    WHAT is counted (the system prompt, user/assistant content and nothing
    else) rather than re-deriving the char math by hand.
    """
    from app.core.config import get_settings
    from app.services.context_budget import approx_tokens
    from app.services.prompts import compose_system_prompt

    # No workspace persona is stored by these fixtures, so the estimate under
    # test composes the deployment's own - the tier this helper stands in for.
    system = compose_system_prompt(get_settings(), workspace_prompt=None)
    total = approx_tokens(system) if system else 0
    return total + sum(approx_tokens(text) for text in contents)


async def test_context_estimate_counts_system_prompt_and_history(client, db_sessionmaker) -> None:
    conv = await _mk_conversation(db_sessionmaker, title="t")
    await _mk_message(
        db_sessionmaker, conv.id, role="user", content="What is in the fridge?", created_at=BASE
    )
    await _mk_message(
        db_sessionmaker,
        conv.id,
        role="assistant",
        content="Milk and eggs.",
        thinking="reasoning that must never be counted",
        created_at=BASE.replace(minute=1),
    )

    resp = await client.get(
        f"/api/v1/conversations/{conv.id}/context", params={"model": "qwen3:4b-instruct"}
    )

    assert resp.status_code == 200
    data = resp.json()
    # The dev profile sizes this tag; the window must match what a turn asks for.
    assert data["num_ctx"] == 8192
    assert data["model"] == "qwen3:4b-instruct"
    assert data["prompt_tokens"] == _expected_base_tokens(
        "What is in the fridge?", "Milk and eggs."
    )


async def test_context_estimate_replays_only_user_and_assistant_turns(
    client, db_sessionmaker
) -> None:
    # A planted system row must not inflate the reading: the turn's replay
    # skips every role but user/assistant, so the estimate does too.
    conv = await _mk_conversation(db_sessionmaker, title="t")
    await _mk_message(db_sessionmaker, conv.id, role="system", content="x" * 4000, created_at=BASE)
    await _mk_message(
        db_sessionmaker, conv.id, role="user", content="hello", created_at=BASE.replace(minute=1)
    )

    data = (
        await client.get(
            f"/api/v1/conversations/{conv.id}/context", params={"model": "qwen3:4b-instruct"}
        )
    ).json()

    assert data["prompt_tokens"] == _expected_base_tokens("hello")


async def test_context_estimate_resolves_the_stored_pick_when_no_model_is_named(
    client, db_sessionmaker
) -> None:
    # No query param: the default resolves through the same stored chain a
    # turn uses, so a conversation pinned to the thinking sibling reads THAT
    # model's window, not the workspace default's.
    conv = await _mk_conversation(db_sessionmaker, settings={"model": "qwen3:4b"})

    data = (await client.get(f"/api/v1/conversations/{conv.id}/context")).json()

    assert data["model"] == "qwen3:4b"
    assert data["num_ctx"] == 8192


async def test_context_estimate_routes_the_default_like_a_turn_would(
    client, db_sessionmaker
) -> None:
    # No pick, and the configured default is not on the server: the estimate
    # names the tag a turn would actually run (the installed pair partner),
    # not the configured one. Anything else and the client's post-turn refetch
    # would see a tag mismatch and replace the measured reading with a floor
    # for a model that never runs.
    conv = await _mk_conversation(db_sessionmaker)

    data = (await client.get(f"/api/v1/conversations/{conv.id}/context")).json()

    assert data["model"] == "qwen3:4b"


async def test_context_estimate_of_a_missing_conversation_is_404(client) -> None:
    resp = await client.get(f"/api/v1/conversations/{uuid4()}/context")
    assert resp.status_code == 404


async def test_context_estimate_never_reads_across_tenants(client, db_sessionmaker) -> None:
    foreign = await _mk_conversation(db_sessionmaker, tenant_id=uuid4(), user_id=uuid4())
    resp = await client.get(f"/api/v1/conversations/{foreign.id}/context")
    assert resp.status_code == 404
