"""/rag/ingest + /rag/query API tests (ASGITransport over the in-memory store)."""

from contextlib import asynccontextmanager

from httpx import ASGITransport, AsyncClient

from app.api.deps import get_embedder, get_model_provider
from app.core.concurrency import StreamLimiter
from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID
from app.main import create_app
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.providers.rag.qdrant_store import QdrantVectorStore
from tests.conftest import make_memory_store

INGEST_URL = "/api/v1/rag/ingest"
QUERY_URL = "/api/v1/rag/query"
CONVERSATIONS_URL = "/api/v1/conversations"
# A well-formed UUID that names no conversation of the caller's - what a session
# id belonging to somebody else looks like from this side of the boundary.
FOREIGN_SESSION = "00000000-0000-0000-0000-0000000000aa"


def items(*texts: str, prefix: str = "doc") -> list[dict]:
    return [{"id": f"{prefix}-{i}", "title": f"T{i}", "text": text} for i, text in enumerate(texts)]


async def new_session(client) -> str:
    """Mint a conversation and return its id, the only valid session scope.

    The session scope IS a conversation, and both endpoints now resolve the id
    through the user-scoped conversation repository, so a session test has to
    own one rather than invent a UUID.
    """
    resp = await client.post(CONVERSATIONS_URL)
    assert resp.status_code == 201
    return resp.json()["id"]


@asynccontextmanager
async def client_for(db_sessionmaker, *, embedder=..., store=None):
    """App-wired client mirroring the conftest fixture, with RAG overrides."""
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider()
    if embedder is ...:
        app.state.embedder = FakeEmbedder(dim=32)
    elif embedder is None:
        app.dependency_overrides[get_embedder] = lambda: None
    else:
        app.state.embedder = embedder
    app.state.vector_store = store if store is not None else make_memory_store()
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(get_settings().max_streams_per_tenant)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
    app.dependency_overrides.clear()


async def test_ingest_then_query_roundtrip(client) -> None:
    resp = await client.post(
        INGEST_URL,
        json={"items": items("Refunds are issued within 5 business days of approval.")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["collection_id"] is not None
    assert body["documents"] == 1
    assert body["chunks"] >= 1

    query = await client.post(
        QUERY_URL,
        json={"query": "Refunds are issued within 5 business days of approval."},
    )
    assert query.status_code == 200
    results = query.json()["results"]
    assert results
    assert results[0]["doc_id"] == "doc-0"
    assert results[0]["title"] == "T0"
    assert results[0]["text"]
    assert results[0]["score"] > 0.9


async def test_named_collections_scope_queries(client) -> None:
    await client.post(
        INGEST_URL, json={"collection": "kb-a", "items": items("alpha only fact", prefix="a")}
    )
    await client.post(
        INGEST_URL, json={"collection": "kb-b", "items": items("beta only fact", prefix="b")}
    )
    # Exact text exists in kb-a: visible when scoped there, invisible when
    # the query is scoped to the OTHER collection.
    in_scope = await client.post(QUERY_URL, json={"query": "alpha only fact", "collection": "kb-a"})
    assert [r["doc_id"] for r in in_scope.json()["results"]] == ["a-0"]
    out_of_scope = await client.post(
        QUERY_URL, json={"query": "alpha only fact", "collection": "kb-b"}
    )
    assert out_of_scope.json()["results"] == []
    unknown = await client.post(QUERY_URL, json={"query": "x", "collection": "nope"})
    assert unknown.status_code == 404


async def test_reingest_same_collection_reuses_row(client) -> None:
    first = await client.post(INGEST_URL, json={"items": items("v1 text")})
    second = await client.post(INGEST_URL, json={"items": items("v2 text")})
    assert first.json()["collection_id"] == second.json()["collection_id"]


async def test_session_ingest_is_ephemeral_scope(client) -> None:
    session = await new_session(client)
    resp = await client.post(
        INGEST_URL,
        json={"session_id": session, "items": items("session only fact", prefix="s")},
    )
    assert resp.status_code == 200
    assert resp.json()["collection_id"] is None

    with_session = await client.post(
        QUERY_URL, json={"query": "session only fact", "session_id": session}
    )
    assert [r["doc_id"] for r in with_session.json()["results"]] == ["s-0"]
    without = await client.post(QUERY_URL, json={"query": "session only fact"})
    assert without.json()["results"] == []


async def test_session_id_spellings_canonicalize_to_one_scope(client) -> None:
    # An uppercase UUID spelling on ingest and the lowercase canonical form
    # on query must land in the SAME session scope - otherwise chat turns
    # (which scope by the canonical conversation id) could never see it.
    session = await new_session(client)
    resp = await client.post(
        INGEST_URL,
        json={"session_id": session.upper(), "items": items("case folded fact", prefix="s")},
    )
    assert resp.status_code == 200
    found = await client.post(QUERY_URL, json={"query": "case folded fact", "session_id": session})
    assert [r["doc_id"] for r in found.json()["results"]] == ["s-0"]


async def test_a_session_id_that_is_not_the_callers_conversation_is_not_found(
    db_sessionmaker,
) -> None:
    # The session scope holds uploaded file text, fetched pages and records
    # returned under a per-user credential, and its id is the conversation id -
    # which is not a secret, it sits in the frontend URL. So a UUID check is not
    # a permission check: without resolving the id through the user-scoped
    # conversation repository, anyone holding another member's conversation id
    # could READ that scope, and PLANT chunks their next turn would retrieve and
    # cite as their own stored records. A no-op with one seeded user; the whole
    # boundary the moment login lands.
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store) as client:
        read = await client.post(
            QUERY_URL, json={"query": "anything", "session_id": FOREIGN_SESSION}
        )
        assert read.status_code == 404

        plant = await client.post(
            INGEST_URL,
            json={"session_id": FOREIGN_SESSION, "items": items("planted", prefix="evil")},
        )
        assert plant.status_code == 404

    # And nothing was written on the way to that 404: the guard runs BEFORE the
    # ingest, not as a report after it. Asked of the store directly, under the
    # foreign scope - querying some other session for the text would come back
    # empty whether the plant landed or not, which would pin nothing at all.
    assert (
        await store.count(
            "session", "fake-embed", str(DEFAULT_TENANT_ID), session_id=FOREIGN_SESSION
        )
        == 0
    )


async def test_a_deleted_conversation_is_no_longer_a_usable_session(client) -> None:
    # Ownership is resolved live, not trusted from a first success: once the
    # conversation is gone the id addresses nothing, exactly as a stranger's does.
    session = await new_session(client)
    assert (
        await client.post(INGEST_URL, json={"session_id": session, "items": items("a fact")})
    ).status_code == 200
    assert (await client.delete(f"{CONVERSATIONS_URL}/{session}")).status_code == 204

    assert (
        await client.post(QUERY_URL, json={"query": "a fact", "session_id": session})
    ).status_code == 404


async def test_ingest_validation_rejections(client) -> None:
    both = await client.post(
        INGEST_URL,
        json={"collection": "kb", "session_id": FOREIGN_SESSION, "items": items("x")},
    )
    assert both.status_code == 422

    dup = await client.post(
        INGEST_URL,
        json={"items": [{"id": "same", "text": "a"}, {"id": "same", "text": "b"}]},
    )
    assert dup.status_code == 422

    bad_session = await client.post(
        INGEST_URL, json={"session_id": "not-a-uuid", "items": items("x")}
    )
    assert bad_session.status_code == 422

    empty = await client.post(INGEST_URL, json={"items": []})
    assert empty.status_code == 422


async def test_query_validation_rejections(client) -> None:
    assert (await client.post(QUERY_URL, json={"query": ""})).status_code == 422
    assert (await client.post(QUERY_URL, json={"query": "x", "top_k": 0})).status_code == 422
    assert (await client.post(QUERY_URL, json={"query": "x", "top_k": 21})).status_code == 422


async def test_missing_embedder_is_typed_503(db_sessionmaker) -> None:
    async with client_for(db_sessionmaker, embedder=None) as client:
        resp = await client.post(INGEST_URL, json={"items": items("x")})
        assert resp.status_code == 503
        assert resp.json()["code"] == "service_unavailable"
        assert "embedding model" in resp.json()["detail"].lower()


async def test_unreachable_store_is_typed_503(db_sessionmaker) -> None:
    dead_store = QdrantVectorStore(url="http://127.0.0.1:1", timeout=1)
    async with client_for(db_sessionmaker, store=dead_store) as client:
        resp = await client.post(INGEST_URL, json={"items": items("x")})
        assert resp.status_code == 503
        assert resp.json()["code"] == "service_unavailable"
    await dead_store.aclose()
