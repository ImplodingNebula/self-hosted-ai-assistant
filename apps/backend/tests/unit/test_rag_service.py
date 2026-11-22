"""rag_service tests: ingest + retrieve over FakeEmbedder and the local store."""

import json
import pathlib
import time
from unittest import mock
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.providers.fake_provider import FakeProvider
from app.providers.rag.base import DocFingerprint, ScoredChunk
from app.providers.rag.chunker import TextChunker
from app.providers.rag.embedder import FakeEmbedder, ProviderEmbedder
from app.providers.rag.qdrant_store import QdrantVectorStore
from app.providers.rag.reranker import Reranker
from app.schemas.rag import RagIngestItem
from app.services.model_resolution import EmbeddingConfig
from app.services.rag_service import (
    content_fingerprint,
    has_content,
    ingest_structured,
    retrieve,
    retrieve_records,
)

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")
# Fixture collection row id for kb-scope ingests (the service requires exactly
# one of session_id/collection_id).
KB_COL = "22222222-2222-2222-2222-222222222222"
FOREIGN = TenantContext(
    tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
    user_id=DEFAULT_USER_ID,
    role="member",
)

CORPUS = json.loads(
    (pathlib.Path(__file__).parent.parent / "fixtures" / "rag_corpus.json").read_text()
)["records"]


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, model_provider="fake", **overrides)


def corpus_items() -> list[RagIngestItem]:
    return [RagIngestItem(**record) for record in CORPUS]


@pytest.fixture
async def store():
    s = QdrantVectorStore(client=AsyncQdrantClient(":memory:"))
    yield s
    await s.aclose()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=32)


async def test_ingest_then_retrieve_exact_text(store, embedder) -> None:
    settings = make_settings()
    documents, chunks = await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=corpus_items(),
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    assert documents == len(CORPUS)
    assert chunks >= len(CORPUS)
    # The hash embedder gives identical vectors for identical text, so
    # querying with a record's own text must rank it first with score ~1.
    hits = await retrieve(
        query=CORPUS[0]["text"],
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=settings,
    )
    assert hits
    assert hits[0].doc_id == CORPUS[0]["id"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


async def test_mcp_records_reject_a_persistent_collection(store, embedder) -> None:
    # The locked tripwire: records fetched through an MCP tool are per-conversation
    # (and, once per-user credentials land, per-user), so they must never reach the
    # tenant-shared knowledge base. ingest_structured RAISES rather than trusting
    # a caller to always pass a session - the leak is unrepresentable, not merely
    # avoided by convention.
    with pytest.raises(ValueError, match="session scope"):
        await ingest_structured(
            source_type="mcp_records",
            tenant=TENANT,
            items=[RagIngestItem(id="rec-1", text="a record body")],
            embedder=embedder,
            store=store,
            settings=make_settings(),
            collection_id=KB_COL,
        )


async def test_mcp_records_into_the_session_scope_are_accepted(store, embedder) -> None:
    # The allowed path: session-scoped mcp_records ingest and retrieve like any
    # other session source, stamped with their source_type.
    session = "00000000-0000-0000-0000-0000000000cc"
    documents, _chunks = await ingest_structured(
        source_type="mcp_records",
        tenant=TENANT,
        items=[RagIngestItem(id="rec-1", title="Printer", text="firmware rollback fixes it")],
        embedder=embedder,
        store=store,
        settings=make_settings(),
        session_id=session,
    )
    assert documents == 1
    hits = await retrieve(
        query="firmware rollback fixes it",
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=make_settings(),
        session_id=session,
        include_kb=False,
    )
    assert any(h.doc_id == "rec-1" and h.source_type == "mcp_records" for h in hits)


async def test_retrieve_applies_min_score_floor(store, embedder) -> None:
    settings = make_settings(rag_min_score=0.99)
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=corpus_items(),
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    # An unrelated query hashes far from every stored vector.
    hits = await retrieve(
        query="completely unrelated question about quantum chromodynamics",
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=settings,
    )
    assert hits == []


async def test_retrieve_relative_gap_drops_weak_neighbor(store, embedder, monkeypatch) -> None:
    # The real failure: a strong hit (0.69) next to a weak topical neighbor
    # (0.50). Both clear the absolute floor, but the neighbor is noise the
    # answer should not cite. Scores are fixed via a stub query so the gap logic
    # is tested independently of the hash embedder's arbitrary cosines.
    strong = ScoredChunk(id="p1", score=0.69, text="refunds take 5 days", doc_id="rec-strong")
    weak = ScoredChunk(id="p2", score=0.50, text="account deletion is permanent", doc_id="rec-weak")

    async def fake_query(kind, *args, **kwargs):
        return [strong, weak] if kind == "kb" else []

    monkeypatch.setattr(store, "query", fake_query)

    hits = await retrieve(
        query="how long do refunds take",
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=make_settings(rag_rel_margin=0.15),
    )
    assert [h.doc_id for h in hits] == ["rec-strong"]

    # Disabling the gap keeps both (subject only to the absolute floor).
    both = await retrieve(
        query="how long do refunds take",
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=make_settings(rag_rel_margin=0.0),
    )
    assert [h.doc_id for h in both] == ["rec-strong", "rec-weak"]


async def test_session_ingest_stamps_ttl_and_scopes_by_session(store, embedder) -> None:
    settings = make_settings(rag_session_ttl_minutes=1)
    session = "00000000-0000-0000-0000-0000000000aa"
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="s-doc", text="session scoped fact")],
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session,
    )
    hits = await retrieve(
        query="session scoped fact",
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session,
    )
    assert [h.doc_id for h in hits] == ["s-doc"]
    # Another session cannot see it, and neither can a kb-only query.
    other = await retrieve(
        query="session scoped fact",
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=settings,
        session_id="00000000-0000-0000-0000-0000000000bb",
    )
    assert other == []
    # The TTL is set, so a sweep past it removes the point.
    assert await store.sweep_expired(now=10**12) == 1


async def test_reingest_shrunken_document_leaves_no_stale_chunks(store, embedder) -> None:
    settings = make_settings(rag_chunk_tokens=10, rag_chunk_overlap_tokens=0)
    long_text = " ".join(f"Sentence number {i} here." for i in range(30))
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="doc-1", text=long_text)],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    before = await store.count("kb", embedder.model, str(TENANT.tenant_id))
    assert before > 1
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="doc-1", text="Now tiny.")],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    after = await store.count("kb", embedder.model, str(TENANT.tenant_id))
    assert after == 1


class CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that counts embed_texts calls (observes the ingest skip)."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.calls = 0

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return await super().embed_texts(texts)


async def test_reingest_unchanged_document_skips_embedding(store) -> None:
    # Identical content re-ingested is detected via its stored fingerprint and
    # skipped outright - no embed call, no rewrite - while still counting as a
    # present (retrievable) document. Changed content re-embeds normally.
    counting = CountingEmbedder(dim=32)
    settings = make_settings()
    item = RagIngestItem(id="doc-1", title="T", text="stable content")
    first = await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[item],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    assert first == (1, 1)
    calls_after_first = counting.calls
    second = await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[item],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    assert second == (1, 0)  # present, but nothing rewritten
    assert counting.calls == calls_after_first  # and nothing re-embedded
    third = await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="doc-1", title="T", text="updated content")],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    assert third == (1, 1)
    assert counting.calls > calls_after_first


async def test_reingest_skip_respects_chunking_config(store) -> None:
    # The fingerprint folds in the chunking knobs: identical text under
    # different chunk sizing must NOT skip, or the stored chunks would no
    # longer match what a fresh ingest of the active config produces.
    counting = CountingEmbedder(dim=32)
    text = " ".join(f"Sentence number {i} here." for i in range(30))
    item = RagIngestItem(id="doc-1", text=text)
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[item],
        embedder=counting,
        store=store,
        settings=make_settings(rag_chunk_tokens=10, rag_chunk_overlap_tokens=0),
        collection_id=KB_COL,
    )
    before = counting.calls
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[item],
        embedder=counting,
        store=store,
        settings=make_settings(rag_chunk_tokens=20, rag_chunk_overlap_tokens=0),
        collection_id=KB_COL,
    )
    assert counting.calls > before


async def test_reingest_with_changed_url_is_not_skipped(store) -> None:
    # The url is stamped on every chunk payload and rendered as the clickable
    # citation link, so a corrected link is a REAL change - it must re-ingest,
    # not be skipped as "unchanged".
    counting = CountingEmbedder(dim=32)
    settings = make_settings()
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[
            RagIngestItem(id="doc-1", title="T", text="same text", url="https://old.example/kb")
        ],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    before = counting.calls
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[
            RagIngestItem(id="doc-1", title="T", text="same text", url="https://new.example/kb")
        ],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    assert counting.calls > before
    hits = await retrieve(
        query="same text",
        tenant=TENANT,
        embedder=counting,
        store=store,
        settings=settings,
    )
    assert hits and hits[0].url == "https://new.example/kb"


async def test_fingerprint_field_boundary_shift_is_not_skipped(store) -> None:
    # Length-prefixed hashing: characters shifting across the title/text
    # boundary are a real change, never a hash collision that skips.
    counting = CountingEmbedder(dim=32)
    settings = make_settings()
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="doc-1", title="Q1", text=": revenue fell")],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    before = counting.calls
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="doc-1", title="Q1:", text=" revenue fell")],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    assert counting.calls > before


async def test_partial_write_does_not_satisfy_skip(store, monkeypatch) -> None:
    # A doc that lost chunks to a mid-write failure must not vouch for itself:
    # the skip also requires the stored chunk count to match a fresh chunking.
    counting = CountingEmbedder(dim=32)
    settings = make_settings()
    item = RagIngestItem(id="doc-1", title="T", text="stable content")
    digest = content_fingerprint(item, settings)

    async def partial_fingerprints(*args, **kwargs):
        # Hash matches, but the stored chunk count disagrees with what a fresh
        # ingest of this item would produce (1 chunk).
        return {
            "doc-1": DocFingerprint(
                doc_id="doc-1", content_hash=digest, created_at=time.time(), chunk_count=99
            )
        }

    monkeypatch.setattr(store, "doc_fingerprints", partial_fingerprints)
    result = await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[item],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    assert result == (1, 1)  # re-ingested, not skipped
    assert counting.calls == 1


async def test_session_skip_refreshes_ttl(store, embedder) -> None:
    # Re-searching an unchanged page extends its session lifetime exactly as
    # the rewrite it replaced would have - expiry must not stay anchored to
    # the FIRST ingest.
    settings = make_settings()  # 120-minute session TTL
    session = "00000000-0000-0000-0000-0000000000aa"
    item = RagIngestItem(id="w-1", text="stable page")
    now = time.time()
    with mock.patch("app.services.rag_service.time") as frozen:
        frozen.time.return_value = now - 110 * 60  # first ingest 110 min ago
        await ingest_structured(
            source_type="web",
            tenant=TENANT,
            items=[item],
            embedder=embedder,
            store=store,
            settings=settings,
            session_id=session,
        )
    # Original ttl_at is now+10min; without the refresh, the skip below would
    # leave it there and the sweep at now+20min would delete the doc.
    documents, chunks = await ingest_structured(
        source_type="web",
        tenant=TENANT,
        items=[item],
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session,
    )
    assert (documents, chunks) == (1, 0)  # the skip fired
    assert await store.sweep_expired(now=now + 20 * 60) == 0  # ...and refreshed


async def test_expired_unswept_doc_does_not_satisfy_skip(store) -> None:
    # An expired-but-unswept point must not vouch for its document: the
    # fingerprint lookup excludes it, so the re-ingest writes fresh points.
    counting = CountingEmbedder(dim=32)
    settings = make_settings()
    session = "00000000-0000-0000-0000-0000000000aa"
    item = RagIngestItem(id="w-1", text="stable page")
    with mock.patch("app.services.rag_service.time") as frozen:
        frozen.time.return_value = time.time() - 130 * 60  # ttl lapsed 10 min ago
        await ingest_structured(
            source_type="web",
            tenant=TENANT,
            items=[item],
            embedder=counting,
            store=store,
            settings=settings,
            session_id=session,
        )
    found = await store.doc_fingerprints(
        "session", counting.model, str(TENANT.tenant_id), ["w-1"], session_id=session
    )
    assert found == {}
    before = counting.calls
    documents, chunks = await ingest_structured(
        source_type="web",
        tenant=TENANT,
        items=[item],
        embedder=counting,
        store=store,
        settings=settings,
        session_id=session,
    )
    assert (documents, chunks) == (1, 1)  # rewritten with a fresh TTL
    assert counting.calls > before


async def test_kb_skip_is_scoped_per_collection(store) -> None:
    # One logical collection's fingerprint must not vouch for another: the
    # same doc id + text ingested into a second collection is a fresh write
    # there, retrievable under that collection's scope.
    counting = CountingEmbedder(dim=32)
    settings = make_settings()
    other_col = "33333333-3333-3333-3333-333333333333"
    item = RagIngestItem(id="faq-1", title="T", text="shared doc text")
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[item],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    before = counting.calls
    result = await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[item],
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=other_col,
    )
    assert result == (1, 1)
    assert counting.calls > before
    hits = await retrieve(
        query="shared doc text",
        tenant=TENANT,
        embedder=counting,
        store=store,
        settings=settings,
        collection_id=other_col,
    )
    assert hits


async def test_default_floors_match_prefixed_calibration(store, embedder, monkeypatch) -> None:
    # The floors and the query prefix are calibrated TOGETHER: a silent revert
    # to the old 0.50 floor would suppress almost all grounding under prefixed
    # scores (related passages measure 0.40-0.60). Pin the defaults and
    # exercise the discriminating band no other test touches.
    for key in ("RAG_MIN_SCORE", "RAG_REL_MARGIN"):
        monkeypatch.delenv(key, raising=False)
    settings = make_settings()
    assert settings.rag_min_score == pytest.approx(0.35)
    assert settings.rag_rel_margin == pytest.approx(0.10)
    related = ScoredChunk(id="p1", score=0.40, text="t", doc_id="rec-related")
    noise = ScoredChunk(id="p2", score=0.31, text="t", doc_id="rec-noise")

    async def fake_query(kind, *args, **kwargs):
        return [related, noise] if kind == "kb" else []

    monkeypatch.setattr(store, "query", fake_query)
    hits = await retrieve(
        query="q", tenant=TENANT, embedder=embedder, store=store, settings=settings
    )
    # 0.40 (measured related floor) survives; 0.31 (unrelated ceiling) drops.
    assert [h.doc_id for h in hits] == ["rec-related"]


class _EmbedRecordingProvider(FakeProvider):
    """Records every embed input; vectors stay FakeProvider's hash embeds."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_inputs: list[list[str]] = []

    async def embed(self, texts, *, model=None, options=None):  # type: ignore[override]
        self.embed_inputs.append(list(texts))
        return await super().embed(texts, model=model, options=options)


async def test_service_layer_prefixes_queries_and_embeds_documents_raw(store) -> None:
    # Pins the asymmetry invariant at the SERVICE call sites (the embedder
    # unit test alone would not catch retrieve() bypassing embed_query):
    # ingest embeds document text raw; retrieve embeds through embed_query,
    # which carries the catalog prefix.
    provider = _EmbedRecordingProvider()
    config = EmbeddingConfig(model="m", dim=16, query_prefix="Instruct: r\nQuery:")
    embedder = ProviderEmbedder(provider, config)
    settings = make_settings()
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="d", text="document text")],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    assert provider.embed_inputs == [["document text"]]
    provider.embed_inputs.clear()
    await retrieve(
        query="find it", tenant=TENANT, embedder=embedder, store=store, settings=settings
    )
    assert provider.embed_inputs == [["Instruct: r\nQuery:find it"]]


async def test_doc_fingerprints_scoped_lookup(store, embedder) -> None:
    # The fingerprint lookup honors the same tenant/session scoping as query();
    # unknown docs, foreign scopes, and a missing space all read as absent.
    settings = make_settings()
    session = "00000000-0000-0000-0000-0000000000aa"
    await ingest_structured(
        source_type="web",
        tenant=TENANT,
        items=[RagIngestItem(id="w-1", text="alpha")],
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session,
    )
    found = await store.doc_fingerprints(
        "session", embedder.model, str(TENANT.tenant_id), ["w-1", "w-2"], session_id=session
    )
    assert set(found) == {"w-1"}
    assert found["w-1"].content_hash
    assert found["w-1"].created_at > 0
    other_session = await store.doc_fingerprints(
        "session",
        embedder.model,
        str(TENANT.tenant_id),
        ["w-1"],
        session_id="00000000-0000-0000-0000-0000000000bb",
    )
    assert other_session == {}
    foreign = await store.doc_fingerprints(
        "session", embedder.model, str(FOREIGN.tenant_id), ["w-1"], session_id=session
    )
    assert foreign == {}
    missing_space = await store.doc_fingerprints(
        "session", "other-model", str(TENANT.tenant_id), ["w-1"], session_id=session
    )
    assert missing_space == {}


async def test_tenant_isolation_end_to_end(store, embedder) -> None:
    settings = make_settings()
    await ingest_structured(
        source_type="records",
        tenant=FOREIGN,
        items=[RagIngestItem(id="f-doc", text="foreign tenant secret")],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    hits = await retrieve(
        query="foreign tenant secret",
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=settings,
    )
    assert hits == []
    assert not await has_content(tenant=TENANT, embedder=embedder, store=store)


async def test_session_scope_is_isolated_between_two_members(store, embedder) -> None:
    # MUTATION GUARD: drop the user_id argument from the session query in
    # retrieve() and Bob's uploaded passage grounds Alice's answer. Two members
    # of ONE tenant, so the tenant filter cannot be what saves this - the
    # router's ownership check on the conversation is the boundary above, and
    # this is the defence in depth beneath it.
    settings = make_settings()
    session = "00000000-0000-0000-0000-0000000000bb"
    alice = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
    bob = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")

    await ingest_structured(
        source_type="upload",
        tenant=bob,
        items=[RagIngestItem(id="b-doc", text="bob salary review notes")],
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session,
    )

    # Bob still reads his own.
    assert [
        hit.text
        for hit in await retrieve(
            query="bob salary review notes",
            tenant=bob,
            embedder=embedder,
            store=store,
            settings=settings,
            session_id=session,
        )
    ] == ["bob salary review notes"]

    assert (
        await retrieve(
            query="bob salary review notes",
            tenant=alice,
            embedder=embedder,
            store=store,
            settings=settings,
            session_id=session,
        )
        == []
    )
    # And the cheap pre-check agrees, so Alice's turn does not embed a query for
    # candidates that were never hers.
    assert await has_content(tenant=bob, embedder=embedder, store=store, session_id=session)
    assert not await has_content(tenant=alice, embedder=embedder, store=store, session_id=session)


async def test_the_knowledge_base_stays_shared_between_members(store, embedder) -> None:
    # The other half of the rule: a workspace corpus is not private to whoever
    # ingested it, so the user filter must NOT reach the kb scope.
    settings = make_settings()
    alice = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
    bob = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")

    await ingest_structured(
        source_type="records",
        tenant=bob,
        items=[RagIngestItem(id="handbook", text="the shared refund policy")],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    hits = await retrieve(
        query="the shared refund policy",
        tenant=alice,
        embedder=embedder,
        store=store,
        settings=settings,
    )
    assert [hit.text for hit in hits] == ["the shared refund policy"]


async def test_named_attachment_retrieval_is_isolated_between_two_members(store, embedder) -> None:
    # MUTATION GUARD: drop `user_id=str(tenant.user_id)` from the retrieve_records
    # session query and Bob's attachment surfaces for Alice. This path is distinct
    # from retrieve() - it is how a NAMED source (an uploaded file the turn points
    # at) is read, and it deliberately ignores the relevance floors, so naming the
    # record is the only gate on the CONTENT and the user filter is the only gate
    # on WHOSE record may be named. The retrieve() test above does not cover it.
    settings = make_settings()
    session = "00000000-0000-0000-0000-0000000000cc"
    alice = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
    bob = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
    record = "attachment:bob-file:0:0"

    await ingest_structured(
        source_type="upload",
        tenant=bob,
        items=[RagIngestItem(id="b-doc", text="bob quarterly figures", record_id=record)],
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session,
        persist=True,
    )

    # Bob names his own record and gets it (one group, his passage).
    bob_groups = await retrieve_records(
        query="whatever the turn happens to ask",
        tenant=bob,
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session,
        record_ids=[record],
    )
    assert [chunk.text for group in bob_groups for chunk in group] == ["bob quarterly figures"]

    # Alice names the same record id in the same conversation and gets nothing -
    # the floors that would otherwise hide a weak match are OFF here, so only the
    # user filter stands between her and Bob's file.
    alice_groups = await retrieve_records(
        query="whatever the turn happens to ask",
        tenant=alice,
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session,
        record_ids=[record],
    )
    assert alice_groups == []


async def test_has_content_paths(store, embedder) -> None:
    settings = make_settings()
    session = "00000000-0000-0000-0000-0000000000aa"
    assert not await has_content(tenant=TENANT, embedder=embedder, store=store)
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="s1", text="ephemeral only")],
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session,
    )
    # Session content is visible only with the matching session scope.
    assert not await has_content(tenant=TENANT, embedder=embedder, store=store)
    assert await has_content(tenant=TENANT, embedder=embedder, store=store, session_id=session)


# Tiny chunks + batch size 1 force multiple embed batches for one doc.
MULTI_BATCH_TEXT = "First sentence here. Second sentence here. Third sentence here."


def multi_batch_settings(pause_seconds: float = 0.0) -> Settings:
    return make_settings(
        rag_embed_batch=1,
        rag_ingest_pause_seconds=pause_seconds,
        rag_chunk_tokens=5,
        rag_chunk_overlap_tokens=0,
    )


def record_sleeps(monkeypatch) -> list[float]:
    """Replace the sleep rag_service calls with a recorder (no real waiting)."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.services.rag_service.asyncio.sleep", fake_sleep)
    return sleeps


async def test_ingest_pauses_between_batches_while_chat_busy(store, embedder) -> None:
    settings = multi_batch_settings()
    calls = {"count": 0}

    def busy() -> bool:
        calls["count"] += 1
        return True

    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="d", text=MULTI_BATCH_TEXT)],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
        is_chat_busy=busy,
    )
    # With batch size 1 and multiple chunks, the busy signal is consulted
    # between batches (never before the first).
    assert calls["count"] >= 1


async def test_busy_ingest_sleeps_the_configured_pause(store, embedder, monkeypatch) -> None:
    settings = multi_batch_settings(pause_seconds=0.123)
    sleeps = record_sleeps(monkeypatch)
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="d", text=MULTI_BATCH_TEXT)],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
        is_chat_busy=lambda: True,
    )
    assert sleeps
    assert all(s == settings.rag_ingest_pause_seconds for s in sleeps)


async def test_idle_ingest_never_sleeps(store, embedder, monkeypatch) -> None:
    settings = multi_batch_settings(pause_seconds=0.123)
    sleeps = record_sleeps(monkeypatch)
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="d", text=MULTI_BATCH_TEXT)],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
        is_chat_busy=lambda: False,
    )
    assert sleeps == []


async def test_busy_check_never_pauses_before_first_batch(store, embedder, monkeypatch) -> None:
    # Even a permanently-busy chat signal must not starve ingestion: the
    # first batch embeds unconditionally, and pauses land only BETWEEN
    # batches - exactly one fewer sleep than batches.
    settings = multi_batch_settings(pause_seconds=0.123)
    n_chunks = len(
        TextChunker(
            chunk_tokens=settings.rag_chunk_tokens,
            overlap_tokens=settings.rag_chunk_overlap_tokens,
        ).chunk(MULTI_BATCH_TEXT, doc_id="d")
    )
    assert n_chunks > 1  # guard: the scenario must actually span batches
    sleeps = record_sleeps(monkeypatch)
    documents, chunks = await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=[RagIngestItem(id="d", text=MULTI_BATCH_TEXT)],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
        is_chat_busy=lambda: True,
    )
    # rag_embed_batch=1 makes batch count == chunk count.
    assert len(sleeps) == n_chunks - 1
    assert (documents, chunks) == (1, n_chunks)


class PromotingReranker(Reranker):
    """Test reranker that inverts the order (proves rerank scores win)."""

    async def rerank(self, query, chunks, top_k):  # type: ignore[override]
        reordered = [chunk.model_copy(update={"score": float(i)}) for i, chunk in enumerate(chunks)]
        reordered.sort(key=lambda c: c.score, reverse=True)
        return reordered[:top_k]


async def test_retrieve_uses_reranker_when_given(store, embedder) -> None:
    settings = make_settings(rag_min_score=0.0, rag_top_k=2, rag_fetch_k=5)
    await ingest_structured(
        source_type="records",
        tenant=TENANT,
        items=corpus_items()[:3],
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=KB_COL,
    )
    dense = await retrieve(
        query=CORPUS[0]["text"],
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=settings,
    )
    reranked = await retrieve(
        query=CORPUS[0]["text"],
        tenant=TENANT,
        embedder=embedder,
        store=store,
        settings=settings,
        reranker=PromotingReranker(),
    )
    assert len(reranked) == 2
    assert [c.id for c in reranked] != [c.id for c in dense[:2]]


def test_scored_chunk_defaults() -> None:
    chunk = ScoredChunk(id="p", score=0.5, text="t", doc_id="d")
    assert chunk.source_type == "records"
    assert chunk.chunk_index == 0
