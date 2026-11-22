"""Knowledge-console API tests (ASGITransport over the in-memory vector store).

The store is qdrant-client's local mode, so every assertion below runs the real
filter, scroll and delete-by-filter code paths rather than a stand-in - which is
the only way a tenant-scoping or orphaned-vector test proves anything.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_tenant, get_embedder, get_model_provider
from app.core.concurrency import StreamLimiter
from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.providers.fake_provider import FakeProvider
from app.providers.rag import qdrant_store
from app.providers.rag.embedder import FakeEmbedder
from app.providers.rag.qdrant_store import QdrantVectorStore
from app.schemas.rag import RagIngestItem
from app.services.rag_service import ingest_structured
from tests.conftest import make_memory_store

DOCS_URL = "/api/v1/knowledge/documents"
INGEST_URL = "/api/v1/rag/ingest"
QUERY_URL = "/api/v1/rag/query"

# A tenant that is not the seeded default - the "somebody else's workspace" every
# scoping assertion below is written against.
OTHER_TENANT = TenantContext(
    tenant_id=UUID("00000000-0000-0000-0000-0000000000ff"),
    user_id=UUID("00000000-0000-0000-0000-0000000000fe"),
    role="admin",
)

# The same workspace, seen by somebody without the admin role. Same ids as the
# seeded principal on purpose: the ONLY difference between this and an ordinary
# request is the role, so a refusal can only have come from the role gate.
MEMBER = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="member")
# The same principal WITH the role, so the refusals above can be shown to come
# from the role and not from the override itself.
ADMIN = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


@asynccontextmanager
async def client_for(db_sessionmaker, *, store=None, embedder=..., tenant=None):
    """App-wired client mirroring the conftest fixture, with RAG overrides.

    Takes the store from the caller so a test can plant another tenant's chunks
    in the SAME store the app reads, which is the whole point: an isolation test
    against a store the app cannot see would pass no matter what the filter did.

    ``tenant`` overrides the AUTH SEAM only - ``get_current_tenant``, exactly as
    the members API tests do it - so ``require_role`` itself still runs for real.
    That is the level a role test has to sit at: the thing under test is the
    dependency, and substituting the identity it reads is precisely what the
    OIDC cutover does anyway.
    """
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider()
    if tenant is not None:
        app.dependency_overrides[get_current_tenant] = lambda: tenant
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


def upload(name: str = "handbook.txt", body: bytes = b"Refunds are issued within 5 days.") -> dict:
    """One multipart file part, as httpx wants it."""
    return {"file": (name, body, "text/plain")}


async def collection_id_of(client) -> str:
    """Return the default collection's row id, as the payloads record it."""
    resp = await client.get(DOCS_URL)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items, "expected at least one stored source"
    return items[0]["collection_id"]


async def plant_foreign_source(
    store: QdrantVectorStore, *, collection_id: str, doc_id: str = "their-doc"
) -> None:
    """Ingest a document into ANOTHER tenant's knowledge base, same collection id.

    Deliberately the same ``collection_id``: if isolation ever came from the
    collection rather than from the tenant filter, this is the shape that would
    expose it.
    """
    await ingest_structured(
        source_type="records",
        tenant=OTHER_TENANT,
        items=[RagIngestItem(id=doc_id, title="Their private notes", text="Their secret plan.")],
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=get_settings(),
        collection_id=collection_id,
    )


async def test_upload_then_list_shows_the_file_with_provenance(db_sessionmaker) -> None:
    async with client_for(db_sessionmaker) as client:
        added = await client.post(DOCS_URL, files=upload())
        assert added.status_code == 200
        body = added.json()
        assert body["id"].startswith("knowledge:")
        assert body["title"] == "handbook.txt"
        assert body["documents"] == 1
        assert body["chunks"] >= 1

        listed = await client.get(DOCS_URL)
        assert listed.status_code == 200
        page = listed.json()
        assert page["collection"] == "default"
        assert page["active_embedding_model"] == FakeEmbedder(dim=32).model
        assert page["truncated"] is False
        (row,) = page["items"]
        assert row["id"] == body["id"]
        assert row["title"] == "handbook.txt"
        assert row["source_type"] == "upload"
        assert row["documents"] == 1
        assert row["chunks"] == body["chunks"]
        assert row["embed_models"] == [FakeEmbedder(dim=32).model]
        # The active embedder wrote it, so an answer today can actually cite it.
        assert row["reachable"] is True
        assert row["ingested_at"] is not None
        assert row["content_hash"]


async def test_uploaded_text_is_retrievable_from_the_persistent_scope(db_sessionmaker) -> None:
    # The console's whole reason to exist: a file added here has to ground
    # answers, which means landing in the kb scope a chat turn searches - not in
    # a conversation's session scratch.
    async with client_for(db_sessionmaker) as client:
        added = await client.post(
            DOCS_URL, files=upload(body=b"Refunds are issued within 5 business days of approval.")
        )
        assert added.status_code == 200

        found = await client.post(
            QUERY_URL, json={"query": "Refunds are issued within 5 business days of approval."}
        )
        assert found.status_code == 200
        results = found.json()["results"]
        assert results
        assert results[0]["source_type"] == "upload"


async def test_a_multi_part_file_lists_as_ONE_source(db_sessionmaker) -> None:
    # A long file becomes several RAG documents. The console lists SOURCES, so a
    # person who sent one file sees one row - and deletes it in one call.
    long_body = ("The refund policy is reviewed annually. " * 2000).encode()
    async with client_for(db_sessionmaker) as client:
        added = await client.post(DOCS_URL, files=upload("policy.txt", long_body))
        assert added.status_code == 200
        assert added.json()["documents"] > 1

        (row,) = (await client.get(DOCS_URL)).json()["items"]
        assert row["documents"] > 1
        assert row["chunks"] >= row["documents"]


async def test_re_uploading_the_same_bytes_stores_a_second_source(db_sessionmaker) -> None:
    # Each upload mints its own source id, so the same file sent twice is two
    # rows rather than a silent overwrite. Stated by a test because the opposite
    # is just as defensible and somebody will wonder which was chosen.
    async with client_for(db_sessionmaker) as client:
        first = await client.post(DOCS_URL, files=upload())
        second = await client.post(DOCS_URL, files=upload())
        assert first.json()["id"] != second.json()["id"]
        assert len((await client.get(DOCS_URL)).json()["items"]) == 2


async def test_unreadable_file_is_refused_before_anything_is_stored(db_sessionmaker) -> None:
    async with client_for(db_sessionmaker) as client:
        refused = await client.post(
            DOCS_URL, files={"file": ("mystery.bin", b"\x00\x01\x02\x03\xff\xfe", "image/x-oddity")}
        )
        assert refused.status_code == 415
        assert (await client.get(DOCS_URL)).json()["items"] == []


async def test_a_file_with_no_readable_text_is_refused(db_sessionmaker) -> None:
    # Storing it would be a quiet lie: listed as knowledge, grounding nothing.
    async with client_for(db_sessionmaker) as client:
        refused = await client.post(DOCS_URL, files=upload("blank.txt", b"   \n\t  \n"))
        assert refused.status_code == 422
        assert (await client.get(DOCS_URL)).json()["items"] == []


async def test_empty_upload_is_a_bad_request(db_sessionmaker) -> None:
    async with client_for(db_sessionmaker) as client:
        assert (await client.post(DOCS_URL, files=upload("empty.txt", b""))).status_code == 400


async def test_oversized_upload_is_refused(db_sessionmaker) -> None:
    settings = get_settings()
    async with client_for(db_sessionmaker) as client:
        too_big = b"x " * settings.attachment_max_bytes
        assert (await client.post(DOCS_URL, files=upload("big.txt", too_big))).status_code == 413


async def test_list_is_empty_before_anything_is_ingested(db_sessionmaker) -> None:
    # No collection row exists yet. That is an empty knowledge base, not a
    # missing resource, and reading it must not create one either.
    async with client_for(db_sessionmaker) as client:
        resp = await client.get(DOCS_URL)
        assert resp.status_code == 200
        assert resp.json() == {
            "items": [],
            "collection": "default",
            "active_embedding_model": FakeEmbedder(dim=32).model,
            "truncated": False,
        }


async def test_api_ingested_records_are_listed_too(db_sessionmaker) -> None:
    # The console has to show what the raw ingest API put there - that corpus is
    # precisely what had no management surface before.
    async with client_for(db_sessionmaker) as client:
        ingested = await client.post(
            INGEST_URL,
            json={"items": [{"id": "ticket-1", "title": "Ticket 1", "text": "A stored record."}]},
        )
        assert ingested.status_code == 200

        (row,) = (await client.get(DOCS_URL)).json()["items"]
        # No record_id on a directly-ingested record: it IS its own source.
        assert row["id"] == "ticket-1"
        assert row["source_type"] == "records"
        assert row["title"] == "Ticket 1"


async def test_delete_removes_the_vectors_not_just_the_listing(db_sessionmaker) -> None:
    # The guard: a delete that dropped the row from the list while leaving the
    # points behind would keep answering from a document its owner removed.
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store) as client:
        added = (await client.post(DOCS_URL, files=upload())).json()
        collection_id = await collection_id_of(client)
        assert (
            await store.count(
                "kb",
                FakeEmbedder(dim=32).model,
                str(DEFAULT_TENANT_ID),
                collection_id=collection_id,
            )
            > 0
        )

        removed = await client.request("DELETE", DOCS_URL, params={"document_id": added["id"]})
        assert removed.status_code == 200
        assert removed.json() == {"id": added["id"], "chunks_deleted": added["chunks"]}

        # Asked of the STORE, not of the listing: the listing is derived from the
        # points, so it would agree with itself even if nothing had been deleted.
        assert (
            await store.count(
                "kb",
                FakeEmbedder(dim=32).model,
                str(DEFAULT_TENANT_ID),
                collection_id=collection_id,
            )
            == 0
        )
        assert (await client.get(DOCS_URL)).json()["items"] == []


async def test_delete_removes_every_document_of_a_multi_part_file(db_sessionmaker) -> None:
    # One call, whatever a file split into. Addressing by document id would leave
    # every page but the first behind.
    long_body = ("The refund policy is reviewed annually. " * 2000).encode()
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store) as client:
        added = (await client.post(DOCS_URL, files=upload("policy.txt", long_body))).json()
        assert added["documents"] > 1
        collection_id = await collection_id_of(client)

        removed = await client.request("DELETE", DOCS_URL, params={"document_id": added["id"]})
        assert removed.json()["chunks_deleted"] == added["chunks"]
        assert (
            await store.count(
                "kb",
                FakeEmbedder(dim=32).model,
                str(DEFAULT_TENANT_ID),
                collection_id=collection_id,
            )
            == 0
        )


async def test_deleting_an_unknown_id_is_a_successful_zero(db_sessionmaker) -> None:
    async with client_for(db_sessionmaker) as client:
        await client.post(DOCS_URL, files=upload())
        removed = await client.request("DELETE", DOCS_URL, params={"document_id": "never-existed"})
        assert removed.status_code == 200
        assert removed.json()["chunks_deleted"] == 0
        # And it took nothing else with it.
        assert len((await client.get(DOCS_URL)).json()["items"]) == 1


async def test_another_tenants_documents_are_not_listed(db_sessionmaker) -> None:
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store) as client:
        await client.post(DOCS_URL, files=upload())
        collection_id = await collection_id_of(client)
        await plant_foreign_source(store, collection_id=collection_id)

        items = (await client.get(DOCS_URL)).json()["items"]
        assert [row["id"] for row in items] != []
        assert all(row["title"] != "Their private notes" for row in items)
        assert len(items) == 1


async def test_another_tenants_document_cannot_be_deleted(db_sessionmaker) -> None:
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store) as client:
        await client.post(DOCS_URL, files=upload())
        collection_id = await collection_id_of(client)
        await plant_foreign_source(store, collection_id=collection_id)

        removed = await client.request("DELETE", DOCS_URL, params={"document_id": "their-doc"})
        assert removed.status_code == 200
        assert removed.json()["chunks_deleted"] == 0
        # Their passages are still there, asked of the store under THEIR tenant.
        assert (
            await store.count(
                "kb",
                FakeEmbedder(dim=32).model,
                str(OTHER_TENANT.tenant_id),
                collection_id=collection_id,
            )
            > 0
        )


async def test_a_stranded_source_is_listed_as_unreachable_and_still_deletable(
    db_sessionmaker,
) -> None:
    # What an embedding-model change leaves behind: stored, occupying room, and
    # findable by no query. Hiding it would leave an operator unable to remove it.
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store) as client:
        await client.post(DOCS_URL, files=upload())
        collection_id = await collection_id_of(client)
        # The same tenant's document, written by a DIFFERENT embedder - which is
        # a different physical collection, exactly as a model change produces.
        await ingest_structured(
            source_type="records",
            tenant=TenantContext(
                tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin"
            ),
            items=[RagIngestItem(id="old-doc", title="Older notes", text="Written long ago.")],
            embedder=FakeEmbedder(dim=32, model="retired-embedder"),
            store=store,
            settings=get_settings(),
            collection_id=collection_id,
        )

        rows = {row["id"]: row for row in (await client.get(DOCS_URL)).json()["items"]}
        assert rows["old-doc"]["reachable"] is False
        assert rows["old-doc"]["embed_models"] == ["retired-embedder"]

        removed = await client.request("DELETE", DOCS_URL, params={"document_id": "old-doc"})
        assert removed.json()["chunks_deleted"] > 0
        assert await store.count("kb", "retired-embedder", str(DEFAULT_TENANT_ID)) == 0, (
            "the delete must reach spaces the active embedder cannot read"
        )


async def test_a_walk_that_hit_its_ceiling_says_the_list_is_partial(
    db_sessionmaker, monkeypatch
) -> None:
    # A corpus bigger than one walk can read does not omit the last few entries:
    # a source's chunks are spread through the store in point-id order, which is
    # a hash and has nothing to do with which document they belong to, so a
    # truncated walk under-counts everything. The flag is what stops the page
    # presenting a fraction as a total - the guard, and the reason the walk
    # reports its ceiling instead of trimming quietly.
    monkeypatch.setattr(qdrant_store, "_INVENTORY_SCROLL_LIMIT", 1)
    monkeypatch.setattr(qdrant_store, "_INVENTORY_MAX_PAGES", 1)
    async with client_for(db_sessionmaker) as client:
        ingested = await client.post(
            INGEST_URL,
            json={
                "items": [
                    {"id": f"rec-{n}", "title": f"Record {n}", "text": f"Stored record {n}."}
                    for n in range(4)
                ]
            },
        )
        assert ingested.status_code == 200

        page = (await client.get(DOCS_URL)).json()
        assert page["truncated"] is True
        assert len(page["items"]) < 4


async def test_a_complete_walk_says_the_list_is_whole(db_sessionmaker) -> None:
    # The other side of the flag, so "truncated" is not simply always true.
    async with client_for(db_sessionmaker) as client:
        await client.post(
            INGEST_URL,
            json={"items": [{"id": "rec-1", "title": "Record 1", "text": "Stored record."}]},
        )
        assert (await client.get(DOCS_URL)).json()["truncated"] is False


async def test_listing_survives_a_missing_embedder(db_sessionmaker) -> None:
    # A broken embedder configuration is exactly when someone needs to see what
    # is stored; every row then honestly reads unreachable.
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store) as seeding_client:
        await seeding_client.post(DOCS_URL, files=upload())
    async with client_for(db_sessionmaker, store=store, embedder=None) as client:
        page = (await client.get(DOCS_URL)).json()
        assert page["active_embedding_model"] is None
        assert page["items"][0]["reachable"] is False


async def test_upload_without_an_embedder_is_a_typed_503(db_sessionmaker) -> None:
    async with client_for(db_sessionmaker, embedder=None) as client:
        refused = await client.post(DOCS_URL, files=upload())
        assert refused.status_code == 503


# --- the admin gate --------------------------------------------------------
#
# The WHOLE router is admin-only, reads included. Curating the corpus every
# answer is grounded on is workspace administration, and so is knowing what is
# in it - a passage's title alone can say more than its workspace meant to
# publish. Enforced on the server; the rail hiding the page is a courtesy, never
# the boundary. Each test drives the real require_role dependency and overrides
# only the auth seam beneath it, exactly as the members API tests do.


async def test_a_member_cannot_list_the_knowledge_base(db_sessionmaker) -> None:
    async with client_for(db_sessionmaker) as admin:
        await admin.post(DOCS_URL, files=upload())
    async with client_for(db_sessionmaker, tenant=MEMBER) as client:
        refused = await client.get(DOCS_URL)
    assert refused.status_code == 403


async def test_a_member_cannot_add_to_the_knowledge_base(db_sessionmaker) -> None:
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store, tenant=MEMBER) as client:
        refused = await client.post(DOCS_URL, files=upload())
    assert refused.status_code == 403
    # Refused BEFORE anything was stored, not merely refused an answer.
    async with client_for(db_sessionmaker, store=store) as admin:
        assert (await admin.get(DOCS_URL)).json()["items"] == []


async def test_a_member_cannot_remove_from_the_knowledge_base(db_sessionmaker) -> None:
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store) as admin:
        added = (await admin.post(DOCS_URL, files=upload())).json()
    async with client_for(db_sessionmaker, store=store, tenant=MEMBER) as client:
        refused = await client.request("DELETE", DOCS_URL, params={"document_id": added["id"]})
    assert refused.status_code == 403
    # And the passages are still there: a refusal that deleted first would be
    # the only failure mode worth having this test for.
    async with client_for(db_sessionmaker, store=store) as admin:
        (row,) = (await admin.get(DOCS_URL)).json()["items"]
        assert row["chunks"] == added["chunks"]


async def test_the_admin_gate_is_what_refuses_them(db_sessionmaker) -> None:
    # The control for the three above: identical requests from a principal
    # differing ONLY in role succeed. Without this, a bug that refused everybody
    # would leave all three green.
    async with client_for(db_sessionmaker, tenant=ADMIN) as client:
        assert (await client.post(DOCS_URL, files=upload())).status_code == 200
        assert (await client.get(DOCS_URL)).status_code == 200


async def test_session_scoped_chunks_are_neither_listed_nor_deleted(db_sessionmaker) -> None:
    # The knowledge base is the PERSISTENT corpus, and a conversation's own
    # scratch must not surface on a workspace page. End-to-end evidence that it
    # does not.
    #
    # This is the WEAKER of the two guards behind that, and deliberately says so:
    # on this path the collection filter alone excludes session points (they
    # carry no collection_id), so swapping _kb_collections() for
    # _all_collections() leaves this test green. The kb/session split itself is
    # pinned at the seam, where the methods are called without a collection id -
    # see test_kb_inventory_never_sees_session_points and its delete sibling in
    # test_qdrant_store.py.
    store = make_memory_store()
    async with client_for(db_sessionmaker, store=store) as client:
        conversation = await client.post("/api/v1/conversations")
        assert conversation.status_code == 201
        session_id = conversation.json()["id"]
        planted = await client.post(
            INGEST_URL,
            json={
                "session_id": session_id,
                "items": [{"id": "scratch-1", "title": "Private note", "text": "Session only."}],
            },
        )
        assert planted.status_code == 200
        added = (await client.post(DOCS_URL, files=upload())).json()

        listed = (await client.get(DOCS_URL)).json()["items"]
        assert [row["id"] for row in listed] == [added["id"]]

        # Deleting the kb document must not reach into the session scope, and
        # naming the session document must not delete it either.
        await client.request("DELETE", DOCS_URL, params={"document_id": added["id"]})
        stray = await client.request("DELETE", DOCS_URL, params={"document_id": "scratch-1"})
        assert stray.json()["chunks_deleted"] == 0
        assert (
            await store.count(
                "session",
                FakeEmbedder(dim=32).model,
                str(DEFAULT_TENANT_ID),
                session_id=session_id,
                user_id=str(DEFAULT_USER_ID),
            )
            > 0
        ), "the console must not be able to touch a conversation's own scope"


async def test_mcp_records_still_cannot_reach_the_shared_knowledge_base() -> None:
    # The user-scope guard this surface must not weaken: records fetched under
    # one person's credential are per-conversation by construction, and a
    # knowledge base is shared by every member of the workspace.
    with pytest.raises(ValueError, match="session scope"):
        await ingest_structured(
            source_type="mcp_records",
            tenant=TenantContext(
                tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin"
            ),
            items=[RagIngestItem(id="rec-1", text="Their ticket.")],
            embedder=FakeEmbedder(dim=32),
            store=make_memory_store(),
            settings=get_settings(),
            collection_id="00000000-0000-0000-0000-0000000000aa",
        )
