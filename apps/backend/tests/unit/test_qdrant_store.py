"""QdrantVectorStore tests over the in-process local client (":memory:").

Local mode runs the real qdrant-client code paths (filters, counts,
delete-by-filter, query_points) without a server, so these tests exercise
the store's actual filter construction - the tenant chokepoint above all.
Payload-index creation is a local-mode no-op; index definitions are verified
against the real server during deployment checks instead.
"""

import uuid

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from app.providers.rag.base import ChunkPayload, ChunkPoint, VectorStoreUnavailableError
from app.providers.rag.embedder import FakeEmbedder
from app.providers.rag.qdrant_store import QdrantVectorStore, _collection_name

TENANT_A = "00000000-0000-0000-0000-00000000000a"
TENANT_B = "00000000-0000-0000-0000-00000000000b"
USER_A = "00000000-0000-0000-0000-0000000000a1"
USER_B = "00000000-0000-0000-0000-0000000000b1"
SPACE = "fake-embed"
DIM = 32

EMBEDDER = FakeEmbedder(dim=DIM, model=SPACE)


@pytest.fixture
async def store():
    s = QdrantVectorStore(client=AsyncQdrantClient(":memory:"))
    yield s
    await s.aclose()


def vec(text: str) -> list[float]:
    return EMBEDDER._hash_vector(text)


def make_point(
    text: str,
    *,
    doc_id: str = "doc-1",
    chunk_index: int = 0,
    tenant_id: str = TENANT_A,
    session_id: str | None = None,
    user_id: str | None = None,
    collection_id: str | None = None,
    ttl_at: float | None = None,
) -> ChunkPoint:
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant_id}:{doc_id}:{chunk_index}:{text}"))
    return ChunkPoint(
        id=pid,
        vector=vec(text),
        payload=ChunkPayload(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            collection_id=collection_id,
            doc_id=doc_id,
            title="T",
            text=text,
            chunk_index=chunk_index,
            created_at=1000.0,
            ttl_at=ttl_at,
            embed_model=SPACE,
        ),
    )


async def _upsert_without_user_key(
    store: QdrantVectorStore, text: str, *, session_id: str, tenant_id: str = TENANT_A
) -> None:
    """Write a session point whose payload has NO ``user_id`` key at all.

    Goes around :meth:`QdrantVectorStore.upsert` deliberately: its payload is a
    ``ChunkPayload``, and ``model_dump`` always emits the key (as null at worst),
    so the key-absent shape cannot be produced through the seam. It is what a
    point written by an older build - or by anything other than this app - looks
    like, and the legacy-tolerant filter exists for exactly that shape.
    """
    await store._client.upsert(
        collection_name=_collection_name("session", SPACE),
        points=[
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, text)),
                vector=vec(text),
                payload={
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    "doc_id": "legacy",
                    "text": text,
                    "chunk_index": 0,
                    "created_at": 1000.0,
                    "embed_model": SPACE,
                },
            )
        ],
        wait=True,
    )


async def test_upsert_query_roundtrip(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("refunds take 5 days")])
    hits = await store.query("kb", SPACE, TENANT_A, vec("refunds take 5 days"), top_k=3)
    assert len(hits) == 1
    assert hits[0].text == "refunds take 5 days"
    assert hits[0].doc_id == "doc-1"
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


async def test_query_on_missing_space_is_empty(store: QdrantVectorStore) -> None:
    assert await store.query("kb", "never-created", TENANT_A, vec("q"), top_k=3) == []
    assert await store.count("kb", "never-created", TENANT_A) == 0


async def test_tenant_isolation_on_query_and_count(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("alpha fact")])
    await store.upsert("kb", SPACE, TENANT_B, [make_point("beta fact", tenant_id=TENANT_B)])
    hits_a = await store.query("kb", SPACE, TENANT_A, vec("beta fact"), top_k=10)
    assert [h.text for h in hits_a] == ["alpha fact"]
    assert await store.count("kb", SPACE, TENANT_A) == 1
    assert await store.count("kb", SPACE, TENANT_B) == 1


async def test_upsert_stamps_bound_tenant_over_forged_payload(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    forged = make_point("smuggled", tenant_id=TENANT_B)
    await store.upsert("kb", SPACE, TENANT_A, [forged])
    assert await store.count("kb", SPACE, TENANT_B) == 0
    assert await store.count("kb", SPACE, TENANT_A) == 1


async def test_session_scope_filters_by_session_id(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.upsert(
        "session", SPACE, TENANT_A, [make_point("session one fact", session_id="s1")]
    )
    await store.upsert(
        "session", SPACE, TENANT_A, [make_point("session two fact", doc_id="d2", session_id="s2")]
    )
    hits = await store.query(
        "session", SPACE, TENANT_A, vec("session two fact"), top_k=10, session_id="s1"
    )
    assert [h.text for h in hits] == ["session one fact"]


async def test_session_scope_filters_by_user_id(store: QdrantVectorStore) -> None:
    # MUTATION GUARD: drop the user_id arm of _scope_filter and Bob's passage
    # comes back to Alice - which is exactly the cross-member leak this filter
    # is defence in depth against.
    await store.ensure_ready(SPACE, DIM)
    await store.upsert(
        "session",
        SPACE,
        TENANT_A,
        [make_point("alice private note", session_id="s1", user_id=USER_A)],
    )
    await store.upsert(
        "session",
        SPACE,
        TENANT_A,
        [make_point("bob private note", doc_id="d2", session_id="s1", user_id=USER_B)],
    )

    hits = await store.query(
        "session",
        SPACE,
        TENANT_A,
        vec("bob private note"),
        top_k=10,
        session_id="s1",
        user_id=USER_A,
    )
    assert [h.text for h in hits] == ["alice private note"]
    # The count applies the same filters, so it cannot promise candidates the
    # query would refuse to return.
    assert await store.count("session", SPACE, TENANT_A, session_id="s1") == 2
    assert await store.count("session", SPACE, TENANT_A, session_id="s1", user_id=USER_A) == 1


async def test_session_points_without_a_user_match_nobody(store: QdrantVectorStore) -> None:
    # The filter fails CLOSED: a point with no user_id belongs to no user, so a
    # user-scoped query returns none of them. No point this app writes is ever
    # unattributed (ingest stamps user_id on every one), so this is not a case
    # that occurs in practice - it pins the SAFE DIRECTION should one ever arise:
    # a stray point surfaces as its owner missing their own content, never as it
    # leaking to everyone. Both unattributed shapes are covered: an explicit null
    # and the key entirely absent.
    await store.ensure_ready(SPACE, DIM)
    await store.upsert(
        "session", SPACE, TENANT_A, [make_point("null user", session_id="s1", user_id=None)]
    )
    await _upsert_without_user_key(store, "absent user", session_id="s1")

    hits = await store.query(
        "session", SPACE, TENANT_A, vec("null user"), top_k=10, session_id="s1", user_id=USER_A
    )
    assert hits == []
    # The unfiltered count still sees both points; the user-scoped count sees
    # neither, exactly as the query does.
    assert await store.count("session", SPACE, TENANT_A, session_id="s1") == 2
    assert await store.count("session", SPACE, TENANT_A, session_id="s1", user_id=USER_A) == 0


async def test_kb_scope_ignores_the_user_filter_when_none(store: QdrantVectorStore) -> None:
    # The knowledge base is a workspace resource: whoever ingested a document,
    # every member searches the same corpus. Passing no user_id must therefore
    # leave the filter off entirely rather than default to the caller.
    await store.ensure_ready(SPACE, DIM)
    await store.upsert(
        "kb", SPACE, TENANT_A, [make_point("shared handbook", collection_id="c1", user_id=USER_B)]
    )
    hits = await store.query(
        "kb", SPACE, TENANT_A, vec("shared handbook"), top_k=10, collection_id="c1"
    )
    assert [h.text for h in hits] == ["shared handbook"]


async def test_kb_scope_filters_by_collection_id(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("in col-1", collection_id="col-1")])
    await store.upsert(
        "kb", SPACE, TENANT_A, [make_point("in col-2", doc_id="d2", collection_id="col-2")]
    )
    hits = await store.query(
        "kb", SPACE, TENANT_A, vec("in col-2"), top_k=10, collection_id="col-1"
    )
    assert [h.text for h in hits] == ["in col-1"]
    assert await store.count("kb", SPACE, TENANT_A) == 2
    assert await store.count("kb", SPACE, TENANT_A, collection_id="col-2") == 1


async def test_delete_doc_removes_stale_tail_chunks(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    points = [make_point(f"chunk {i}", chunk_index=i) for i in range(3)]
    await store.upsert("kb", SPACE, TENANT_A, points)
    await store.delete_doc("kb", SPACE, TENANT_A, "doc-1")
    assert await store.count("kb", SPACE, TENANT_A) == 0
    # Other docs survive.
    await store.upsert("kb", SPACE, TENANT_A, [make_point("keep me", doc_id="doc-2")])
    await store.delete_doc("kb", SPACE, TENANT_A, "doc-1")
    assert await store.count("kb", SPACE, TENANT_A) == 1


async def test_delete_doc_respects_tenant(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_B, [make_point("b doc", tenant_id=TENANT_B)])
    await store.delete_doc("kb", SPACE, TENANT_A, "doc-1")
    assert await store.count("kb", SPACE, TENANT_B) == 1


async def test_delete_session_spans_all_spaces(store: QdrantVectorStore) -> None:
    other_space = "old-model"
    await store.ensure_ready(SPACE, DIM)
    await store.ensure_ready(other_space, DIM)
    await store.upsert("session", SPACE, TENANT_A, [make_point("new vec", session_id="s1")])
    await store.upsert(
        "session", other_space, TENANT_A, [make_point("old vec", doc_id="d2", session_id="s1")]
    )
    await store.upsert(
        "session", SPACE, TENANT_A, [make_point("other session", doc_id="d3", session_id="s2")]
    )
    deleted = await store.delete_session(TENANT_A, "s1")
    assert deleted == 2
    assert await store.count("session", SPACE, TENANT_A, session_id="s2") == 1


async def test_sweep_expired_deletes_only_past_ttl(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.upsert(
        "session",
        SPACE,
        TENANT_A,
        [
            make_point("expired", session_id="s1", ttl_at=100.0),
            make_point("fresh", doc_id="d2", session_id="s1", ttl_at=10_000.0),
        ],
    )
    swept = await store.sweep_expired(now=5_000.0)
    assert swept == 1
    remaining = await store.query("session", SPACE, TENANT_A, vec("fresh"), top_k=10)
    assert [h.text for h in remaining] == ["fresh"]


async def test_sweep_ignores_kb_collections(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("kb never expires")])
    assert await store.sweep_expired(now=10**12) == 0
    assert await store.count("kb", SPACE, TENANT_A) == 1


async def test_spaces_do_not_mix(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.ensure_ready("other-space", DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("space one")])
    assert await store.count("kb", "other-space", TENANT_A) == 0


async def test_ensure_ready_is_idempotent(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("still fine")])
    assert await store.count("kb", SPACE, TENANT_A) == 1


async def test_unreachable_server_raises_seam_error() -> None:
    # An unroutable port answers instantly with a connection error.
    store = QdrantVectorStore(url="http://127.0.0.1:1", timeout=1)
    with pytest.raises(VectorStoreUnavailableError):
        await store.ensure_ready(SPACE, DIM)
    await store.aclose()


async def test_upsert_empty_points_is_noop(store: QdrantVectorStore) -> None:
    assert await store.upsert("kb", SPACE, TENANT_A, []) == 0


async def test_malformed_out_of_band_payload_raises_seam_error(store: QdrantVectorStore) -> None:
    # A point written around the store (wrong payload types) must surface as
    # the seam-level error retrieval degrades on - never a raw TypeError.
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("good point")])
    await store._client.upsert(
        collection_name=_collection_name("kb", SPACE),
        points=[
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec("rogue point"),
                payload={"tenant_id": TENANT_A, "chunk_index": {"bad": 1}},
            )
        ],
        wait=True,
    )
    with pytest.raises(VectorStoreUnavailableError):
        await store.query("kb", SPACE, TENANT_A, vec("rogue point"), top_k=10)


class RacyClient:
    """Wraps a real local client, denying each collection's existence once.

    ensure_ready then walks into create_collection on a collection that
    already exists - exactly what losing a concurrent create race looks like
    (local mode raises ValueError where a server returns 409).
    """

    def __init__(self, inner: AsyncQdrantClient) -> None:
        self._inner = inner
        self._denied: set[str] = set()

    async def collection_exists(self, collection_name: str) -> bool:
        if collection_name not in self._denied:
            self._denied.add(collection_name)
            return False
        return await self._inner.collection_exists(collection_name)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


async def test_ensure_ready_tolerates_lost_create_race() -> None:
    inner = AsyncQdrantClient(":memory:")
    # Pre-create both of the space's collections through a plain store.
    await QdrantVectorStore(client=inner).ensure_ready(SPACE, DIM)
    racy = QdrantVectorStore(client=RacyClient(inner))  # type: ignore[arg-type]
    # The create fails with already-exists; the re-check must treat that as
    # success (the collection IS there), not raise.
    await racy.ensure_ready(SPACE, DIM)
    await racy.upsert("kb", SPACE, TENANT_A, [make_point("still writable")])
    assert await racy.count("kb", SPACE, TENANT_A) == 1
    await racy.aclose()


# --- space enumeration (the stranded-vectors alarm's input) ---


async def test_list_spaces_on_empty_store_is_empty(store: QdrantVectorStore) -> None:
    # No collections at all is a real, healthy answer - distinct from the
    # "cannot ask" the unreachable case raises below.
    assert await store.list_spaces() == []


async def test_list_spaces_reports_kind_slug_count_and_provenance(
    store: QdrantVectorStore,
) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.upsert(
        "kb",
        SPACE,
        TENANT_A,
        [make_point("one"), make_point("two", chunk_index=1)],
    )
    spaces = {space.name: space for space in await store.list_spaces()}
    # ensure_ready creates BOTH collections of a space, so the empty session
    # one is listed too - a real collection that happens to hold nothing.
    assert set(spaces) == {_collection_name("kb", SPACE), _collection_name("session", SPACE)}
    kb = spaces[_collection_name("kb", SPACE)]
    assert kb.kind == "kb"
    assert kb.slug == SPACE
    assert kb.points == 2
    # Provenance is SAMPLED from the stored payloads, which is the only place
    # the true tag survives (the slug in the name is lossy).
    assert kb.embed_models == frozenset({SPACE})
    session = spaces[_collection_name("session", SPACE)]
    assert session.kind == "session"
    assert session.points == 0
    assert session.embed_models == frozenset()


async def test_list_spaces_samples_each_space_separately(store: QdrantVectorStore) -> None:
    other = "other-embed"
    await store.ensure_ready(SPACE, DIM)
    await store.ensure_ready(other, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("current")])
    # A point written under the OTHER space, stamped with that space's tag -
    # exactly what an embedder change leaves behind.
    stale = make_point("stale")
    stale = stale.model_copy(
        update={"payload": stale.payload.model_copy(update={"embed_model": other})}
    )
    await store.upsert("session", other, TENANT_A, [stale])
    by_name = {space.name: space for space in await store.list_spaces()}
    assert by_name[_collection_name("kb", SPACE)].embed_models == frozenset({SPACE})
    assert by_name[_collection_name("session", other)].embed_models == frozenset({other})
    assert by_name[_collection_name("session", other)].kind == "session"
    # Sorted by name, so two reports of the same store are diffable.
    names = [space.name for space in await store.list_spaces()]
    assert names == sorted(names)


async def test_list_spaces_ignores_foreign_collections(store: QdrantVectorStore) -> None:
    # A collection this store does not own must not appear: the report speaks
    # for the platform's spaces, not for whatever else shares the Qdrant.
    await store.ensure_ready(SPACE, DIM)
    await store._client.create_collection(
        collection_name="someone_elses_index",
        vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE),
    )
    assert all(space.name != "someone_elses_index" for space in await store.list_spaces())


async def test_list_spaces_on_unreachable_store_raises() -> None:
    # "Cannot tell" must never be delivered as "there is nothing stored": a
    # caller that could not distinguish them would report a healthy store
    # while every vector in it sat unreachable.
    store = QdrantVectorStore(url="http://127.0.0.1:1", timeout=1)
    with pytest.raises(VectorStoreUnavailableError):
        await store.list_spaces()
    await store.aclose()


# --- the knowledge-base inventory ------------------------------------------
#
# The console's two seam methods. Every test here calls them WITHOUT a
# collection id, which is the shape that actually exercises the kb/session
# split: passing one hides the question entirely, because session points carry
# no collection_id and that condition alone would exclude them.


async def test_kb_inventory_never_sees_session_points(store: QdrantVectorStore) -> None:
    # A conversation's uploaded files and fetched pages belong to one person and
    # are none of the workspace console's business. Restricting the walk to kb
    # collections is what keeps them out; without it the console would list
    # every conversation's private scratch as workspace knowledge.
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("shared", collection_id="c1")])
    await store.upsert(
        "session",
        SPACE,
        TENANT_A,
        [make_point("private", doc_id="scratch", session_id="s1", user_id=USER_A)],
    )
    inventory = await store.list_kb_sources(TENANT_A)
    assert [source.id for source in inventory.sources] == ["doc-1"]


async def test_deleting_a_kb_source_never_touches_session_points(
    store: QdrantVectorStore,
) -> None:
    # The other half, and the sharper one: a delete that swept session
    # collections would let this console destroy a conversation's own scratch -
    # somebody else's, once there is more than one person - while reporting it
    # as housekeeping on the shared corpus.
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("shared", collection_id="c1")])
    await store.upsert(
        "session",
        SPACE,
        TENANT_A,
        [make_point("private", doc_id="doc-1", session_id="s1", user_id=USER_A)],
    )
    # The SAME doc_id in both scopes on purpose: the id alone matches the
    # session point too, so only the collection split can save it.
    deleted = await store.delete_kb_source(TENANT_A, "doc-1")
    assert deleted == 1
    assert await store.count("kb", SPACE, TENANT_A) == 0
    assert await store.count("session", SPACE, TENANT_A, session_id="s1", user_id=USER_A) == 1


async def test_kb_inventory_and_delete_respect_the_tenant(store: QdrantVectorStore) -> None:
    await store.ensure_ready(SPACE, DIM)
    await store.upsert("kb", SPACE, TENANT_A, [make_point("mine", collection_id="c1")])
    await store.upsert(
        "kb", SPACE, TENANT_B, [make_point("theirs", doc_id="doc-1", collection_id="c1")]
    )
    assert [source.id for source in (await store.list_kb_sources(TENANT_B)).sources] == ["doc-1"]
    assert await store.delete_kb_source(TENANT_A, "doc-1") == 1
    # Their identically-named document survives.
    assert await store.count("kb", SPACE, TENANT_B) == 1
