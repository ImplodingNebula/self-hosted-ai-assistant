"""Tests for DELETE /api/v1/models/{name}: its refusals, and what it moves.

Deleting weights is the one management verb that can break a running
deployment, so every guard gets its own case. The guards are re-derived
server-side on every request: the management view's ``can_delete`` flag is a
snapshot of the moment a page rendered, and a tab left open since this morning
is asserting something about a stack that has moved on.

What each guard reads has changed shape. Which tag a turn RUNS comes from the
deployment's configuration (``DEFAULT_CHAT_MODEL`` and its thinking sibling)
and the stored workspace default; what a tag IS - chat, embedder, or neither -
comes from the runtime's own report about the weights, through the same
``derive_role`` the management view groups by. So the scripted capability
reports below are not scenery: they are what decides whether a tag may take
over as the workspace default and whether the embedding lock applies to it.

The stored tiers are handled rather than guarded: conversation pins on the
deleted tag are cleared in the same request, and deleting the workspace
default requires naming a validated successor - both get their own cases,
including the tenant-wide reach of the pin clearing and the refusals that
protect the successor rule.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import get_model_provider, get_vector_store
from app.core.config import Settings, get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.models import WorkspaceSettings
from app.models.entities import ModelPolicy
from app.providers.base import (
    ManagementCapabilities,
    ModelCapabilities,
    ModelInfo,
    ModelNotFoundError,
    ModelProvider,
    ProviderUnavailableError,
)
from app.providers.fake_provider import FakeProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.rag.base import ChunkPayload, ChunkPoint, VectorStoreUnavailableError
from app.repositories.conversations import ConversationRepository
from app.schemas.settings import WorkspaceDefaultsPatch
from app.services.auth.principal import seeded_principal
from app.services.model_policy import resolve_policy
from app.services.workspace_defaults import apply_defaults
from tests.conftest import make_memory_store

# What the runtime says about the weights under each tag. A role is DERIVED
# from this and nothing else now, so a chat model that also reports `embedding`
# (the FakeProvider's own default) would land in the embedding role and pick up
# its lock - which is why every tag these tests touch is described explicitly.
CHAT_REPORT = ModelCapabilities(completion=True, tools=True)
EMBED_REPORT = ModelCapabilities(embedding=True)

REPORTS: dict[str, ModelCapabilities] = {
    # The deployment's own pair, plus a chat alternate nothing is serving...
    "fake": CHAT_REPORT,
    "fake:latest": CHAT_REPORT,
    "library/fake": CHAT_REPORT,
    "fake-think": CHAT_REPORT,
    "fake-think:latest": CHAT_REPORT,
    "fake-alt": CHAT_REPORT,
    "stray:1b": CHAT_REPORT,
    # ...and the embedders, in every spelling these tests ask about. A real
    # runtime resolves a tag's aliases itself and answers about the same
    # weights whichever way the request spelled them; this fake matches the
    # string exactly, so each spelling has to be described.
    "fake-embed": EMBED_REPORT,
    "fake-embed:1b": EMBED_REPORT,
    "fake-embed:latest": EMBED_REPORT,
    "fake-embed:4b": EMBED_REPORT,
    "library/fake-embed": EMBED_REPORT,
    "library/fake-embed:1b": EMBED_REPORT,
}

# The deployment under test: a chat pair, an embedder, one chat alternate that
# nothing is serving, and a tag nothing has any use for.
INSTALLED = [
    ModelInfo(name="fake"),
    ModelInfo(name="fake-think"),
    ModelInfo(name="fake-alt"),
    ModelInfo(name="fake-embed:1b"),
    ModelInfo(name="stray:1b"),
]


class _UnansweringStore:
    """A store whose enumeration fails - the fail-closed case's other half."""

    async def list_spaces(self):
        raise VectorStoreUnavailableError("The vector store is unavailable.")


def described(models: list[ModelInfo], **kwargs) -> FakeProvider:
    """A fake server holding ``models``, reporting :data:`REPORTS` for each.

    Args:
        models: The installed listing to serve.
        **kwargs: Passed through to :class:`FakeProvider`.

    Returns:
        The configured provider.
    """
    kwargs["capabilities_by_model"] = {**REPORTS, **(kwargs.get("capabilities_by_model") or {})}
    return FakeProvider(models=models, **kwargs)


def installed_provider(**kwargs) -> FakeProvider:
    return described(list(INSTALLED), **kwargs)


@asynccontextmanager
async def management_client(
    provider: ModelProvider,
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    chat: str = "fake",
    thinking: str | None = "fake-think",
    store: object | None = None,
) -> AsyncIterator[AsyncClient]:
    """App-wired client whose deployment configuration names ``chat``/``thinking``.

    Args:
        provider: The model seam standing in for the server.
        sessionmaker: The test session factory.
        chat: ``DEFAULT_CHAT_MODEL`` - the tag the next turn would request.
        thinking: ``THINKING_CHAT_MODEL`` - its reasoning sibling, or None.
        store: A vector store to override the dependency with, when a test is
            about the embedding lock.
    """
    settings = Settings(
        default_chat_model=chat,
        thinking_chat_model=thinking,
        _env_file=None,
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    if store is not None:
        app.dependency_overrides[get_vector_store] = lambda: store
    app.state.db_sessionmaker = sessionmaker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
    app.dependency_overrides.clear()


async def pin_conversations(
    sessionmaker: async_sessionmaker[AsyncSession], model: str, count: int = 1
) -> None:
    """Store ``count`` conversations whose settings pin ``model``, as the API does."""
    async with sessionmaker() as session:
        repo = ConversationRepository(session, TenantContext(DEFAULT_TENANT_ID, DEFAULT_USER_ID))
        for index in range(count):
            await repo.create(title=f"thread {index}", settings={"model": model})
        await session.commit()


async def set_workspace_default(sessionmaker: async_sessionmaker[AsyncSession], model: str) -> None:
    """Store a workspace default chat model, the tier the Settings page writes."""
    async with sessionmaker() as session:
        row = await session.get(WorkspaceSettings, DEFAULT_TENANT_ID)
        assert row is not None, "the default tenant's workspace row should be seeded"
        row.settings = {"model": model}
        session.add(row)
        await session.commit()


async def choose_embedder(sessionmaker: async_sessionmaker[AsyncSession], model: str) -> None:
    """Name the workspace's embedder, the top rung of the resolution chain.

    Needed wherever a server holds MORE than one embedding-capable tag: the
    inventory rung refuses to guess between them, so without a stored choice
    there is no active embedder for a bystander to be a bystander of.
    """
    async with sessionmaker() as session:
        await apply_defaults(
            session, seeded_principal(), WorkspaceDefaultsPatch(embedding_model=model)
        )
        await session.commit()


async def test_deleting_an_unused_model_succeeds(db_sessionmaker) -> None:
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/stray:1b")
        assert resp.status_code == 200
        # Nothing stored named this tag, and the body says so exactly.
        assert resp.json() == {
            "removed": "stray:1b",
            "unpinned_conversations": 0,
            "new_default": None,
        }
        # The follow-up listing proves the provider was actually asked, rather
        # than the endpoint simply answering 200 and doing nothing.
        listing = await client.get("/api/v1/models")
    assert provider.deleted == ["stray:1b"]
    assert "stray:1b" not in {m["name"] for m in listing.json()["models"]}


async def test_an_alternate_nobody_is_serving_is_deletable(db_sessionmaker) -> None:
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt")
    assert resp.status_code == 200
    assert provider.deleted == ["fake-alt"]


async def test_a_model_that_is_not_installed_is_a_404(db_sessionmaker) -> None:
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/ghost:1b")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert "nothing to delete" in resp.json()["detail"]
    assert provider.deleted == []


async def test_the_active_chat_model_is_never_deletable(db_sessionmaker) -> None:
    # Deleting the tag the next turn is about to request breaks that turn, and
    # a client's stale can_delete flag must not be able to authorize it.
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "chat model" in detail
    # The refusal has to say what to change, or people go editing config by hand.
    assert "another model first" in detail
    assert provider.deleted == []


async def test_the_active_thinking_sibling_is_never_deletable(db_sessionmaker) -> None:
    # The pair's other half is reached by routing rather than by being picked,
    # so it is just as load-bearing and just as invisible.
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-think")
    assert resp.status_code == 409
    assert "thinking model" in resp.json()["detail"]
    assert provider.deleted == []


async def test_the_active_embedding_model_is_refused(db_sessionmaker) -> None:
    # It defines the vector space every stored chunk lives in; deleting it
    # breaks ingestion and retrieval outright, and no amount of confirmation
    # makes that recoverable. It is the ACTIVE one here because the server
    # holds exactly one tag the runtime says can embed, which is the rung the
    # resolution chain settles on when nothing else has.
    provider = installed_provider()
    store = make_memory_store()
    async with management_client(provider, db_sessionmaker, store=store) as client:
        resp = await client.delete("/api/v1/models/fake-embed:1b")
    await store.aclose()
    assert resp.status_code == 409
    assert "active embedding model" in resp.json()["detail"]
    assert provider.deleted == []


async def test_an_embedding_alternate_nothing_references_deletes(db_sessionmaker) -> None:
    # The lock is on what STRANDS vectors, and an installed alternate no
    # stored passage was embedded by strands nothing.
    await choose_embedder(db_sessionmaker, "fake-embed:1b")
    provider = described([*INSTALLED, ModelInfo(name="fake-embed:4b")])
    store = make_memory_store()
    async with management_client(provider, db_sessionmaker, store=store) as client:
        resp = await client.delete("/api/v1/models/fake-embed:4b")
    await store.aclose()
    assert resp.status_code == 200
    assert provider.deleted == ["fake-embed:4b"]


async def test_an_embedding_model_with_stored_passages_is_refused(db_sessionmaker) -> None:
    # Stored passages can only be searched through the weights that embedded
    # them; the refusal carries the count and what to do about it.
    #
    # The workspace names its own embedder, so the tag under delete is a
    # BYSTANDER holding passages rather than the active model - a different
    # refusal, tested separately. With nothing stored the chain would follow
    # the corpus and make it active, since the observed rung sits above the
    # inventory one.
    await choose_embedder(db_sessionmaker, "fake-embed:1b")
    provider = described([*INSTALLED, ModelInfo(name="fake-embed:4b")])
    store = make_memory_store()
    await store.ensure_ready("fake-embed-4b", 8)
    await store.upsert(
        "kb",
        "fake-embed-4b",
        str(DEFAULT_TENANT_ID),
        [
            ChunkPoint(
                id="11111111-1111-1111-1111-111111111111",
                vector=[0.1] * 8,
                payload=ChunkPayload(
                    tenant_id=str(DEFAULT_TENANT_ID),
                    doc_id="doc-1",
                    text="a passage in the alternate's space",
                    created_at=1000.0,
                    embed_model="fake-embed:4b",
                ),
            )
        ],
    )
    async with management_client(provider, db_sessionmaker, store=store) as client:
        resp = await client.delete("/api/v1/models/fake-embed:4b")
    await store.aclose()
    assert resp.status_code == 409
    assert "stored passage" in resp.json()["detail"]
    assert provider.deleted == []


async def test_an_embedding_delete_fails_closed_when_the_store_does_not_answer(
    db_sessionmaker,
) -> None:
    # The report gates a destructive verb whose damage is exactly what it
    # would have warned about, so no answer means no delete - never a shrug.
    await choose_embedder(db_sessionmaker, "fake-embed:1b")
    provider = described([*INSTALLED, ModelInfo(name="fake-embed:4b")])
    async with management_client(provider, db_sessionmaker, store=_UnansweringStore()) as client:
        resp = await client.delete("/api/v1/models/fake-embed:4b")
    assert resp.status_code == 409
    assert "did not answer" in resp.json()["detail"]
    assert provider.deleted == []


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        # Configured with ":latest", requested bare - the spelling that used to
        # slip past both guards, because the lookups compared literal strings.
        ("fake", "chat model"),
        ("fake-think", "thinking model"),
        ("fake-embed", "active embedding model"),
        # ...and the same names spelled out, which must stay refused too.
        ("fake:latest", "chat model"),
        ("fake-think:latest", "thinking model"),
        ("fake-embed:latest", "active embedding model"),
    ],
)
async def test_the_latest_convention_cannot_slip_past_a_lock(
    db_sessionmaker, requested: str, expected: str
) -> None:
    provider = described(
        [
            ModelInfo(name="fake:latest"),
            ModelInfo(name="fake-think:latest"),
            ModelInfo(name="fake-embed:latest"),
        ]
    )
    # The store answers (empty), so the embedding rows exercise the ACTIVE
    # refusal - which is the alias-sensitive comparison this test exists for.
    store = make_memory_store()
    async with management_client(
        provider,
        db_sessionmaker,
        chat="fake:latest",
        thinking="fake-think:latest",
        store=store,
    ) as client:
        resp = await client.delete(f"/api/v1/models/{requested}")
    await store.aclose()
    assert resp.status_code == 409
    assert expected in resp.json()["detail"]
    assert provider.deleted == []


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("fake", "chat model"),
        ("library/fake", "chat model"),
        ("fake-think", "thinking model"),
        ("fake-embed:1b", "active embedding model"),
        ("library/fake-embed:1b", "active embedding model"),
    ],
)
async def test_the_publisher_namespace_cannot_slip_past_a_lock(
    db_sessionmaker, requested: str, expected: str
) -> None:
    provider = installed_provider()
    store = make_memory_store()
    async with management_client(
        provider,
        db_sessionmaker,
        chat="library/fake",
        thinking="library/fake-think",
        store=store,
    ) as client:
        resp = await client.delete(f"/api/v1/models/{requested}")
    await store.aclose()
    assert resp.status_code == 409
    assert expected in resp.json()["detail"]
    assert provider.deleted == []


async def test_a_backend_that_cannot_delete_says_so(db_sessionmaker) -> None:
    provider = installed_provider(management=ManagementCapabilities(can_list_installed=True))
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/stray:1b")
    assert resp.status_code == 400
    assert "redeployment" in resp.json()["detail"]
    assert provider.deleted == []


async def test_an_unsafe_name_is_refused_before_the_provider_is_asked(db_sessionmaker) -> None:
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "validation_error"
    assert "'..'" in body["detail"]
    assert provider.deleted == []


async def test_losing_the_race_to_another_caller_is_a_404(db_sessionmaker) -> None:
    # The listing said the weights were here; between then and the delete,
    # somebody else removed them. That is a 404, not a 500.
    provider = installed_provider(delete_error=ModelNotFoundError("Model 'stray:1b' is gone."))
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/stray:1b")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


async def test_an_unreachable_model_server_is_a_503(db_sessionmaker) -> None:
    # Fails at the listing, before any guard runs.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = OllamaProvider(
        base_url="http://mock:11434",
        default_model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/stray:1b")
    assert resp.status_code == 503
    assert resp.json()["code"] == "service_unavailable"


async def test_a_server_that_dies_during_the_delete_is_also_a_503(db_sessionmaker) -> None:
    # The other half: the listing succeeded and every guard passed, so this
    # exercises the router's mapping around the DELETE call itself.
    provider = installed_provider(
        delete_error=ProviderUnavailableError("The model server is unavailable.")
    )
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/stray:1b")
    assert resp.status_code == 503
    assert resp.json()["code"] == "service_unavailable"
    assert provider.deleted == ["stray:1b"]


# --- the stored tiers: what a conversation or the workspace is set to use ---


async def stored_pins(sessionmaker: async_sessionmaker[AsyncSession]) -> list[str]:
    """Every pin still stored on any conversation, in no particular order."""
    async with sessionmaker() as session:
        repo = ConversationRepository(session, TenantContext(DEFAULT_TENANT_ID, DEFAULT_USER_ID))
        return sorted((await repo.pinned_model_counts()).keys())


async def stored_default(sessionmaker: async_sessionmaker[AsyncSession]) -> str | None:
    """The workspace default chat model as stored, or None."""
    async with sessionmaker() as session:
        row = await session.get(WorkspaceSettings, DEFAULT_TENANT_ID)
        assert row is not None
        return (row.settings or {}).get("model")


async def test_deleting_a_pinned_model_clears_the_pins(db_sessionmaker) -> None:
    # A pin used to refuse the delete outright, which made every model anyone
    # had ever picked effectively undeletable - an admin cannot walk every
    # user's threads to unpin them. The delete clears the pins instead, so
    # those threads follow the workspace default, and the response says how
    # many moved so the caller can report it.
    await pin_conversations(db_sessionmaker, "fake-alt", count=3)
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt")
    assert resp.status_code == 200
    assert resp.json()["unpinned_conversations"] == 3
    assert provider.deleted == ["fake-alt"]
    assert await stored_pins(db_sessionmaker) == []


async def test_pin_clearing_reaches_every_user_of_the_tenant(db_sessionmaker) -> None:
    # The delete removes the weights for the whole workspace, so the clearing
    # must be tenant-wide: a user-scoped sweep would leave everyone else's
    # threads pinned to weights that are gone.
    other_user = UUID("00000000-0000-0000-0000-00000000000f")
    async with db_sessionmaker() as session:
        repo = ConversationRepository(session, TenantContext(DEFAULT_TENANT_ID, other_user))
        await repo.create(title="someone else's thread", settings={"model": "fake-alt"})
        await session.commit()
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt")
    assert resp.status_code == 200
    assert resp.json()["unpinned_conversations"] == 1
    assert await stored_pins(db_sessionmaker) == []


@pytest.mark.parametrize("stored", ["library/fake-alt:latest", "  fake-alt  ", "fake-alt:latest"])
async def test_a_pin_in_another_spelling_is_still_cleared(db_sessionmaker, stored: str) -> None:
    # A pin is stored verbatim from the request - nothing trims or canonicalizes
    # it on write - so every comparison downstream has to.
    await pin_conversations(db_sessionmaker, stored)
    async with management_client(installed_provider(), db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt")
    assert resp.status_code == 200
    assert resp.json()["unpinned_conversations"] == 1
    assert await stored_pins(db_sessionmaker) == []


async def test_clearing_touches_only_the_deleted_tag_and_only_the_pin_key(
    db_sessionmaker,
) -> None:
    # Two boundaries in one: another model's pin survives, and a cleared
    # conversation keeps every OTHER setting it had chosen.
    await pin_conversations(db_sessionmaker, "stray:1b")
    async with db_sessionmaker() as session:
        repo = ConversationRepository(session, TenantContext(DEFAULT_TENANT_ID, DEFAULT_USER_ID))
        await repo.create(title="rich", settings={"model": "fake-alt", "web_search": True})
        await session.commit()
    async with management_client(installed_provider(), db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt")
    assert resp.status_code == 200
    assert resp.json()["unpinned_conversations"] == 1
    assert await stored_pins(db_sessionmaker) == ["stray:1b"]
    async with db_sessionmaker() as session:
        repo = ConversationRepository(session, TenantContext(DEFAULT_TENANT_ID, DEFAULT_USER_ID))
        rows, _ = await repo.list_summaries(limit=10, offset=0)
        rich = next(row for row in rows if row.title == "rich")
        assert rich.settings == {"web_search": True}


async def test_a_refusal_is_logged_with_its_actor(db_sessionmaker, caplog) -> None:
    # For a destructive verb the refused attempts are the more interesting half
    # of the audit trail, and only successes were being recorded.
    with caplog.at_level(logging.INFO, logger="app.services.model_delete"):
        async with management_client(installed_provider(), db_sessionmaker) as client:
            await client.delete("/api/v1/models/fake")
    line = next(r.getMessage() for r in caplog.records if "model delete refused" in r.getMessage())
    assert str(DEFAULT_TENANT_ID) in line
    assert str(DEFAULT_USER_ID) in line
    assert "fake" in line


async def test_a_success_is_logged_with_its_actor(db_sessionmaker, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="app.services.model_delete"):
        async with management_client(installed_provider(), db_sessionmaker) as client:
            await client.delete("/api/v1/models/stray:1b")
    line = next(r.getMessage() for r in caplog.records if "model deleted by" in r.getMessage())
    assert str(DEFAULT_TENANT_ID) in line
    assert "stray:1b" in line


# --- the workspace default: replaced in the same request, never orphaned ---


async def test_the_endpoint_refuses_a_declared_thinking_partner(db_sessionmaker) -> None:
    """The guard has to be armed on the ENDPOINT, not only on the page.

    A declaration is machine-scoped, so one admin's pairing invalidates every
    other open console. If the page disables Delete and the endpoint behind the
    button does not refuse, the model half the deployment's thinking turns
    depend on is one stale tab (or one API call) from being removed - and the
    turn that follows silently answers on the never-thinking default instead.

    This is the exact two-gate shape the pair work was supposed to close, and
    it was open: the endpoint's `load_model_usage` call simply omitted the
    declaration, so the parameter defaulted away and the guard protected
    whichever sibling the environment happened to name.

    MUTATION: drop `sessionmaker`/`digests` from the `load_model_usage` call in
    `_delete_model` (or make the pairing read optional again and skip it) and
    this deletes a model the management view refuses to offer.
    """
    async with db_sessionmaker() as session:
        # Keyed on the provider these SETTINGS name, which is the default rather
        # than the fake seam the listing comes from: a row under another
        # provider is a declaration the app would never look up.
        session.add(ModelPolicy(provider="ollama", model="fake", thinking_model="fake-alt"))
        await session.commit()
    provider = installed_provider()
    # No environment sibling, so the stored declaration is the ONLY thing that
    # can pair this tag - which is what makes the refusal below evidence that
    # the declaration was read.
    async with management_client(provider, db_sessionmaker, thinking=None) as client:
        resp = await client.delete("/api/v1/models/fake-alt")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "active thinking model" in detail
    # And it names the console: a stored pairing is not deployment
    # configuration, so pointing at the environment would be a dead end.
    assert "model manager" in detail
    assert provider.deleted == []


async def test_the_endpoint_and_the_page_agree_about_a_declared_partner(db_sessionmaker) -> None:
    # The pairing of the two gates, asserted as a pairing rather than twice
    # over: whatever the row says about Delete is what the endpoint does.
    async with db_sessionmaker() as session:
        session.add(ModelPolicy(provider="ollama", model="fake", thinking_model="fake-alt"))
        await session.commit()
    provider = installed_provider()
    store = make_memory_store()
    async with management_client(provider, db_sessionmaker, thinking=None, store=store) as client:
        view = await client.get("/api/v1/models/manager")
        rows = [row for group in view.json()["groups"] for row in group["rows"]]
        row = next(r for r in rows if r["model"] == "fake-alt")
        resp = await client.delete("/api/v1/models/fake-alt")
    await store.aclose()
    assert row["can_delete"] is False
    assert resp.status_code == 409
    assert row["delete_blocked_reason"] == resp.json()["detail"]


async def test_the_workspace_default_needs_a_successor(db_sessionmaker) -> None:
    # Deleting the default with nothing to take over would leave every
    # conversation that never chose a model - the common case - resolving to
    # nothing. The refusal names the mechanism so an API caller is not left
    # guessing what the dialog would have offered.
    await set_workspace_default(db_sessionmaker, "fake-alt")
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "workspace's default chat model" in detail
    assert "replacement" in detail
    assert provider.deleted == []
    assert await stored_default(db_sessionmaker) == "fake-alt"


async def test_the_default_hands_over_to_the_named_successor(db_sessionmaker) -> None:
    # The successful path: the successor is written, the pins on the deleted
    # default are cleared, and the body reports both halves of the succession.
    await set_workspace_default(db_sessionmaker, "fake-alt")
    await pin_conversations(db_sessionmaker, "fake-alt", count=2)
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt", params={"replacement": "stray:1b"})
    assert resp.status_code == 200
    assert resp.json() == {
        "removed": "fake-alt",
        "unpinned_conversations": 2,
        "new_default": "stray:1b",
    }
    assert provider.deleted == ["fake-alt"]
    assert await stored_default(db_sessionmaker) == "stray:1b"
    assert await stored_pins(db_sessionmaker) == []


async def test_a_successor_the_server_does_not_hold_is_refused(db_sessionmaker) -> None:
    await set_workspace_default(db_sessionmaker, "fake-alt")
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt", params={"replacement": "ghost:1b"})
    assert resp.status_code == 409
    assert "not installed" in resp.json()["detail"]
    assert provider.deleted == []
    assert await stored_default(db_sessionmaker) == "fake-alt"


async def test_the_deleted_model_cannot_succeed_itself(db_sessionmaker) -> None:
    # Under ANY spelling: "fake-alt:latest" names the same weights being
    # deleted, so accepting it would orphan the default with extra steps.
    await set_workspace_default(db_sessionmaker, "fake-alt")
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete(
            "/api/v1/models/fake-alt", params={"replacement": "fake-alt:latest"}
        )
    assert resp.status_code == 409
    assert "model being deleted" in resp.json()["detail"]
    assert provider.deleted == []


async def test_a_model_that_cannot_chat_cannot_succeed_the_default(db_sessionmaker) -> None:
    # The embedder cannot drive a chat turn; a default naming it would break
    # every unpinned conversation just as surely as no default at all. The
    # runtime's report is what says so - nothing declares this role any more.
    await set_workspace_default(db_sessionmaker, "fake-alt")
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete(
            "/api/v1/models/fake-alt", params={"replacement": "fake-embed:1b"}
        )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "embedding" in detail and "cannot drive a chat turn" in detail
    assert provider.deleted == []


async def test_the_last_chat_capable_model_is_guarded(db_sessionmaker) -> None:
    # With no possible successor the delete is refused outright - a workspace
    # whose only chat-capable weights are gone cannot answer anything at all,
    # and no picker can offer a model that does not exist.
    await set_workspace_default(db_sessionmaker, "fake-alt")
    provider = described([ModelInfo(name="fake-alt"), ModelInfo(name="fake-embed:1b")])
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "pull another chat model first" in detail
    assert provider.deleted == []


async def test_a_non_chat_role_cannot_succeed_under_another_spelling(db_sessionmaker) -> None:
    # The server LISTS the embedder as "fake-embed:latest" while the request
    # names it bare. The role map is keyed by the listed names, so the successor
    # check has to run through the same alias conventions the delete guards use
    # - otherwise this spelling writes the EMBEDDER as the workspace default
    # chat model and every default-following conversation breaks.
    await set_workspace_default(db_sessionmaker, "fake-alt")
    provider = described(
        [
            ModelInfo(name="fake-alt"),
            ModelInfo(name="fake-embed:latest"),
            ModelInfo(name="stray:1b"),
        ]
    )
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt", params={"replacement": "fake-embed"})
    assert resp.status_code == 409
    assert "cannot drive a chat turn" in resp.json()["detail"]
    assert provider.deleted == []
    assert await stored_default(db_sessionmaker) == "fake-alt"


async def test_the_last_model_guard_sees_through_spellings_too(db_sessionmaker) -> None:
    # With only the default and the (bare-spelled) embedder installed, there is
    # no possible successor - the refusal must be the pull-another-model guard,
    # not an invitation to name a successor that cannot exist.
    await set_workspace_default(db_sessionmaker, "fake-alt")
    provider = described([ModelInfo(name="fake-alt"), ModelInfo(name="fake-embed")])
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/fake-alt")
    assert resp.status_code == 409
    assert "pull another chat model first" in resp.json()["detail"]
    assert provider.deleted == []


async def test_a_stale_replacement_on_a_non_default_delete_is_ignored(db_sessionmaker) -> None:
    # A dialog rendered before another tab changed the default may still send
    # a successor; with nothing to replace, it must not be written.
    await set_workspace_default(db_sessionmaker, "fake-alt")
    provider = installed_provider()
    async with management_client(provider, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/stray:1b", params={"replacement": "fake-alt"})
    assert resp.status_code == 200
    assert resp.json()["new_default"] is None
    assert await stored_default(db_sessionmaker) == "fake-alt"


async def test_a_delete_drops_the_digest_bound_caches(db_sessionmaker) -> None:
    """The two-clock skew: a delete frees a name a re-pull can immediately reuse.

    A stored declaration binds to the digest a tag resolves to, and the listing
    that answers "resolves to" is itself cached. Leave either behind and, for
    the rest of the window, a turn validates an operator's ceiling against
    weights this server no longer holds - which is the auto-re-clamp failing in
    exactly the minutes an operator was watching it work.
    """
    before = described([ModelInfo(name="stray:1b", digest="sha256:one")])
    async with db_sessionmaker() as session:
        session.add(
            ModelPolicy(provider="fake", model="stray:1b", digest="sha256:one", ceiling="agentic")
        )
        await session.commit()

    async def ceiling_now(provider) -> str | None:
        view = await resolve_policy(
            provider=provider,
            provider_name="fake",
            model="stray:1b",
            sessionmaker=db_sessionmaker,
        )
        return view.ceiling if view is not None else None

    # Warm both caches against the weights the declaration was made about.
    assert await ceiling_now(before) == "agentic"

    async with management_client(before, db_sessionmaker) as client:
        resp = await client.delete("/api/v1/models/stray:1b")
    assert resp.status_code == 200

    # The same NAME, different weights - a re-pull into the space just freed.
    after = described([ModelInfo(name="stray:1b", digest="sha256:two")])
    assert await ceiling_now(after) is None
