"""Chat-path tests for the settings resolution chain (slice 4).

Stored per-conversation settings must actually steer a live turn through the
app: a conversation-level web_search=off keeps the turn fully local (no
pre-fetch, no web tool) even with a search provider wired and the request
deferring with auto; a conversation-level model pick routes the turn. The
no-settings path stays pinned by test_chat_agency.py, which runs every
scenario with empty settings.
"""

from __future__ import annotations

import pytest

from app.providers.base import ModelInfo
from app.providers.fake_provider import FakeProvider
from app.providers.search.base import SearchResult
from app.providers.search.searxng import FakeSearchProvider
from tests.conftest import make_memory_store
from tests.unit.test_chat_agency import (
    CHAT_MODEL,
    THINKING_MODEL,
    agency_client,
    collect_events,
    make_settings,
    offered_names,
    status_codes,
)

FRESH_QUESTION = "What is the latest python version today?"

# The configured pair, so these turns run on the model the settings chain picked
# rather than on whatever the missing-model rescue would have found instead.
DEV_PAIR = [ModelInfo(name=CHAT_MODEL), ModelInfo(name=THINKING_MODEL)]


async def _patch_settings(client, conversation_id: str, settings_blob: dict) -> None:
    resp = await client.patch(
        f"/api/v1/conversations/{conversation_id}", json={"settings": settings_blob}
    )
    assert resp.status_code == 200


async def _create_conversation(client) -> str:
    resp = await client.post("/api/v1/conversations", json={})
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.mark.anyio
async def test_conversation_web_off_keeps_the_turn_fully_local(db_sessionmaker):
    """Stored web_search=off + an auto request = no pre-fetch AND no web tool."""
    provider = FakeProvider(script=["Answered", " locally."], models=list(DEV_PAIR))
    search = FakeSearchProvider(results=[SearchResult(title="t", url="https://x", snippet="s")])
    store = make_memory_store()
    async with agency_client(
        provider, db_sessionmaker, make_settings(), store=store, search=search
    ) as client:
        conversation_id = await _create_conversation(client)
        await _patch_settings(client, conversation_id, {"web_search": "off"})
        events = await collect_events(
            client,
            {"message": FRESH_QUESTION, "conversation_id": conversation_id},
        )
    codes = status_codes(events)
    assert "searching_web" not in codes
    assert "web_search" not in offered_names(provider)
    assert search.last_query is None  # the provider was never even consulted


@pytest.mark.anyio
async def test_conversation_model_pick_routes_the_turn(db_sessionmaker):
    """A stored model pick becomes the model the provider is actually asked for."""
    provider = FakeProvider(script=["Hi."], models=list(DEV_PAIR))
    store = make_memory_store()
    async with agency_client(provider, db_sessionmaker, make_settings(), store=store) as client:
        conversation_id = await _create_conversation(client)
        # Deliberately NOT the configured default: the thinking sibling is
        # the other half of the same pair, so this assertion dies if the stored
        # pick stops being threaded into resolve_chat_models (a pick equal to
        # the default would pass hollowly).
        await _patch_settings(client, conversation_id, {"model": THINKING_MODEL})
        await collect_events(client, {"message": "hello there", "conversation_id": conversation_id})
    assert provider.last_model == THINKING_MODEL
