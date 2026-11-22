"""RAG orchestration: the single ingestion entrypoint and retrieval.

``ingest_structured`` is the ONE way content enters the vector store - direct
API records today; uploaded files (Phase 4), MCP record results (Phase 5), and
web results (Phase 3b) all flow through it later, so there is never a second
ingestion code path to keep consistent.

GPU courtesy: embedding runs in batches, and between batches the ingest loop
yields while any chat stream is active so token generation keeps the
hardware. The dev tier additionally pins its small embedder to the CPU (see
EMBEDDING_DEVICE), so ingest cannot evict the chat model there at all.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.core.config import Settings
from app.core.tenancy import TenantContext
from app.providers.rag.base import (
    Chunk,
    ChunkPayload,
    ChunkPoint,
    CollectionKind,
    Embedder,
    Reranker,
    ScoredChunk,
    SourceType,
    VectorStore,
)
from app.providers.rag.chunker import TextChunker
from app.schemas.rag import RagIngestItem
from app.services.context_budget import APPROX_CHARS_PER_TOKEN

logger = logging.getLogger(__name__)

# Fixed namespace for deterministic chunk point ids. Qdrant accepts only
# unsigned-int or UUID point ids - uuid5 over the logical key gives stable,
# collision-resistant UUIDs (re-ingesting a doc rewrites the same ids).
_POINT_NAMESPACE = uuid.UUID("f3b7b6a8-5a58-4f2c-9d55-7c2e8e6d9b01")


def _point_id(tenant_id: str, kind: str, scope: str, doc_id: str, chunk_index: int) -> str:
    """Derive the deterministic point id for one chunk."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{tenant_id}:{kind}:{scope}:{doc_id}:{chunk_index}"))


def content_fingerprint(item: RagIngestItem, settings: Settings) -> str:
    """Hash the stored-content-relevant fields of one ingest item.

    Every field is length-prefixed so the key is unambiguous - characters
    shifting across a field boundary cannot collide. The chunking knobs are
    folded in because changed sizing re-chunks identical text differently, and
    ``url`` is folded in because it is stamped on every chunk payload and
    surfaces as the clickable citation link - a corrected link must not be
    skipped as "unchanged".
    """
    parts = (
        str(settings.rag_chunk_tokens),
        str(settings.rag_chunk_overlap_tokens),
        item.title or "",
        item.url or "",
        item.text,
    )
    key = "".join(f"{len(part)}:{part}" for part in parts)
    return hashlib.sha256(key.encode()).hexdigest()


async def ingest_structured(
    *,
    source_type: SourceType,
    tenant: TenantContext,
    items: list[RagIngestItem],
    embedder: Embedder,
    store: VectorStore,
    settings: Settings,
    session_id: str | None = None,
    collection_id: str | None = None,
    persist: bool = False,
    is_chat_busy: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    """Chunk, embed, and upsert structured records; returns (documents, chunks).

    Idempotent per document: each document's existing chunks are deleted
    before its new ones are written, so re-ingesting updated (even shorter)
    content leaves no stale chunks behind. A document whose content fingerprint
    AND chunk count match what the store already holds is SKIPPED outright -
    no embed, no rewrite (embedding is the dominant ingest cost) - but still
    counts in the returned ``documents`` because its content is present and
    retrievable; skipped session docs get a fresh TTL, exactly as the rewrite
    would have stamped. A ``session_id`` targets the ephemeral scope; otherwise
    ``collection_id`` scopes the persistent knowledge base.

    A session scope normally expires on a TTL - right for throwaway web results,
    which is where the TTL began. ``persist`` turns it off for THIS ingest, so
    the chunks live as long as their session (the conversation) does, cleaned up
    only when the conversation is deleted (which purges its session vectors by
    id, TTL or not). It is what an uploaded file needs: its bytes already live
    for the conversation's lifetime, so its searchable text must too, or the
    file silently stops grounding answers while the file itself remains.

    Args:
        source_type: Where the records came from (stamped on every chunk).
        tenant: The acting tenant; scopes every write.
        items: The records to ingest.
        embedder: Active embedder - defines the vector space and dimension.
        store: The vector store.
        settings: Application settings (chunking/batching/TTL knobs).
        session_id: Ephemeral scope (a conversation id), or None.
        collection_id: Persistent collection row id, or None for session scope.
        persist: Keep session chunks for the scope's lifetime (no TTL). Ignored
            for the knowledge base, which never carries a TTL anyway.
        is_chat_busy: Optional signal that a chat stream is active; ingestion
            pauses between embed batches while it reports True.

    Raises:
        ProviderUnavailableError: The embedding backend is unreachable.
        VectorStoreUnavailableError: The vector store is unreachable.
    """
    if (session_id is None) == (collection_id is None):
        # Programming-error guard, not client validation (the API layer
        # already 422s bad requests): a kb ingest without a collection_id
        # would delete-by-doc across EVERY collection of the tenant.
        raise ValueError("ingest_structured requires exactly one of session_id/collection_id")
    if source_type == "mcp_records" and session_id is None:
        # Records fetched through an MCP tool are per-conversation by
        # construction, and once per-user credentials land a user-scoped
        # server's records must never reach the tenant-shared knowledge base.
        # Until credential scope is plumbed through here, the safe form is
        # absolute: mcp_records ingests to the ephemeral SESSION scope only,
        # never a persistent collection. A raised exception, not a comment, so
        # the leak is unrepresentable rather than merely avoided by convention.
        raise ValueError("mcp_records may only be ingested into the session scope")
    kind: CollectionKind = "session" if session_id else "kb"
    scope = session_id or collection_id or ""
    space = embedder.model
    chunker = TextChunker(
        chunk_tokens=settings.rag_chunk_tokens,
        overlap_tokens=settings.rag_chunk_overlap_tokens,
    )
    now = time.time()
    ttl_at = now + settings.rag_session_ttl_minutes * 60 if (session_id and not persist) else None
    stored = await store.doc_fingerprints(
        kind,
        space,
        str(tenant.tenant_id),
        [item.id for item in items],
        session_id=session_id,
        collection_id=collection_id,
    )

    ready = False
    documents = 0
    total_chunks = 0
    skipped_session_docs: list[str] = []
    for item in items:
        chunks = chunker.chunk(item.text, doc_id=item.id, title=item.title)
        if not chunks:
            continue
        digest = content_fingerprint(item, settings)
        fingerprint = stored.get(item.id)
        if (
            fingerprint is not None
            and fingerprint.content_hash == digest
            and fingerprint.chunk_count == len(chunks)
        ):
            # Unchanged content, fully present: skip the embed and rewrite
            # entirely. The chunk-count check keeps a partially-written doc (a
            # mid-upsert failure) from vouching for itself - any mismatch
            # re-ingests. Session docs get their TTL refreshed below, exactly
            # as a full rewrite would have stamped.
            documents += 1
            if session_id is not None:
                skipped_session_docs.append(item.id)
            continue
        vectors = await _embed_chunks(chunks, embedder, settings, is_chat_busy)
        if not ready:
            dim = embedder.dim or len(vectors[0])
            await store.ensure_ready(space, dim)
            ready = True
        # Delete-then-upsert per doc: deterministic ids alone would leave
        # stale tail chunks alive when a document shrinks.
        await store.delete_doc(
            kind,
            space,
            str(tenant.tenant_id),
            item.id,
            session_id=session_id,
            collection_id=collection_id,
        )
        points = [
            ChunkPoint(
                id=_point_id(str(tenant.tenant_id), kind, scope, item.id, chunk.index),
                vector=vector,
                payload=ChunkPayload(
                    tenant_id=str(tenant.tenant_id),
                    user_id=str(tenant.user_id),
                    session_id=session_id,
                    collection_id=collection_id,
                    source_type=source_type,
                    doc_id=item.id,
                    record_id=item.record_id,
                    title=item.title,
                    text=chunk.text,
                    chunk_index=chunk.index,
                    created_at=now,
                    ttl_at=ttl_at,
                    embed_model=embedder.model,
                    url=item.url,
                    content_hash=digest,
                ),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        total_chunks += await store.upsert(kind, space, str(tenant.tenant_id), points)
        documents += 1
    if session_id is not None and ttl_at is not None and skipped_session_docs:
        # A skipped session doc is in active use: extend its expiry as the
        # rewrite it replaced would have, so a long conversation's repeatedly
        # searched pages do not expire anchored to their FIRST ingest.
        await store.refresh_doc_ttl(
            "session",
            space,
            str(tenant.tenant_id),
            skipped_session_docs,
            ttl_at=ttl_at,
            session_id=session_id,
        )
    return documents, total_chunks


async def _embed_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    settings: Settings,
    is_chat_busy: Callable[[], bool] | None,
) -> list[list[float]]:
    """Embed chunk texts batch-by-batch, yielding to active chat streams."""
    vectors: list[list[float]] = []
    batch_size = max(1, settings.rag_embed_batch)
    for start in range(0, len(chunks), batch_size):
        if start and is_chat_busy is not None and is_chat_busy():
            # Low priority: give the generation loop the hardware for a beat.
            await asyncio.sleep(settings.rag_ingest_pause_seconds)
        batch = chunks[start : start + batch_size]
        vectors.extend(await embedder.embed_texts([chunk.text for chunk in batch]))
    return vectors


async def retrieve(
    *,
    query: str,
    tenant: TenantContext,
    embedder: Embedder,
    store: VectorStore,
    settings: Settings,
    reranker: Reranker | None = None,
    session_id: str | None = None,
    collection_id: str | None = None,
    top_k: int | None = None,
    include_kb: bool = True,
    query_vector: list[float] | None = None,
) -> list[ScoredChunk]:
    """Dense-retrieve the most relevant chunks for ``query``.

    Searches the tenant's knowledge base (optionally narrowed to one
    collection) and, when ``session_id`` is given, that session's ephemeral
    chunks; merges by score, drops hits under the relevance floor, then
    optionally reranks. Hybrid (dense + sparse) retrieval can slot in behind
    this same signature later. ``include_kb`` gates the persistent knowledge
    base: set False to restrict retrieval to the session scope - a web-search
    turn when stored-record RAG is disabled, so the operator-disabled kb stays
    out while freshly-ingested web results still ground the answer.

    Raises:
        ProviderUnavailableError: The embedding backend is unreachable.
        VectorStoreUnavailableError: The vector store is unreachable.
    """
    limit = top_k or settings.rag_top_k
    fetch_k = max(limit, settings.rag_fetch_k) if reranker is not None else limit
    space = embedder.model
    tenant_id = str(tenant.tenant_id)
    vector = query_vector if query_vector is not None else await embedder.embed_query(query)
    # The kb and session scopes are independent collections - query them
    # concurrently instead of paying two sequential round trips.
    scope_queries = []
    if include_kb:
        # No user filter: the knowledge base is a WORKSPACE resource, and
        # narrowing it to whoever happened to ingest a document would make a
        # shared corpus private by accident.
        scope_queries.append(
            store.query("kb", space, tenant_id, vector, top_k=fetch_k, collection_id=collection_id)
        )
    if session_id is not None:
        scope_queries.append(
            store.query(
                "session",
                space,
                tenant_id,
                vector,
                top_k=fetch_k,
                session_id=session_id,
                user_id=str(tenant.user_id),
            )
        )
    hits: list[ScoredChunk] = [
        chunk for result in await asyncio.gather(*scope_queries) for chunk in result
    ]
    hits.sort(key=lambda chunk: chunk.score, reverse=True)
    relevant = [chunk for chunk in hits if chunk.score >= settings.rag_min_score]
    if reranker is not None and relevant:
        # The reranker does its own relevance discrimination and its score scale
        # is not cosine, so the cosine-calibrated gap below does not apply here.
        return await reranker.rerank(query, relevant[:fetch_k], limit)
    if relevant and settings.rag_rel_margin > 0:
        # Relative-relevance gap (dense-only path): once there is a clear best
        # match, drop weak topical neighbors that sit far below it (noise the
        # answer should not cite). relevant[0] is the top hit (hits are sorted
        # desc and the floor filter preserves order). Adaptive: a genuine
        # cluster near the top is kept intact; a lone weak hit is left alone
        # (nothing outscores it).
        floor = relevant[0].score - settings.rag_rel_margin
        relevant = [chunk for chunk in relevant if chunk.score >= floor]
    return relevant[:limit]


async def retrieve_records(
    *,
    query: str,
    tenant: TenantContext,
    embedder: Embedder,
    store: VectorStore,
    settings: Settings,
    session_id: str,
    record_ids: list[str],
    top_k: int | None = None,
    query_vector: list[float] | None = None,
) -> list[list[ScoredChunk]]:
    """Retrieve the best passages OF NAMED SOURCES, ignoring the relevance floors.

    Returns one list per named source, best passage first, in the order the
    sources were named. Grouped rather than flattened because each source has to
    be budgeted and merged on its own - a flat list lets one long file take every
    slot and leave the file beside it with nothing.

    The floors in :func:`retrieve` exist to stop a weak topical neighbour being
    cited alongside a strong match. They are exactly wrong here. When someone
    attaches a file and asks "summarise this", the question shares no vocabulary
    with the document, so every passage of it scores low - and a similarity floor
    would drop the very thing they pointed at. **Naming the source IS the
    relevance signal**; similarity only decides which PARTS of it to bring, not
    whether to bring any.

    One query per source, run concurrently: a single query over all of them would
    rank the sources against each other, which is precisely what must not happen
    when a person names two files and expects both to be read.

    Args:
        query: What to rank each source's own passages against.
        tenant: The acting tenant; scopes the search.
        embedder: Active embedder - defines the vector space.
        store: The vector store.
        settings: Application settings (``rag_attachment_top_k``).
        session_id: The conversation whose ephemeral scope holds the chunks.
        record_ids: The sources to draw from (attachment ids). Empty returns [].
        top_k: How many passages to bring back PER source.
        query_vector: A pre-computed embedding of ``query``, so a caller that
            also runs ordinary retrieval need not embed the same text twice.

    Raises:
        ProviderUnavailableError: The embedding backend is unreachable.
        VectorStoreUnavailableError: The vector store is unreachable.
    """
    if not record_ids:
        return []
    limit = top_k or settings.rag_attachment_top_k
    vector = query_vector if query_vector is not None else await embedder.embed_query(query)
    per_record = await asyncio.gather(
        *(
            store.query(
                "session",
                embedder.model,
                str(tenant.tenant_id),
                vector,
                top_k=limit,
                session_id=session_id,
                user_id=str(tenant.user_id),
                record_ids=[record_id],
            )
            for record_id in record_ids
        )
    )
    grouped: list[list[ScoredChunk]] = []
    for hits in per_record:
        if not hits:
            continue
        hits.sort(key=lambda chunk: chunk.score, reverse=True)
        grouped.append(hits[:limit])
    return grouped


# How the kept passages of one file are stitched into a single chunk. The
# budget below is measured against the joined length (this separator included),
# so what merge counts is exactly what trim_chunks_to_budget later measures.
_PASSAGE_JOIN = "\n\n"


def _passage_order_key(chunk: ScoredChunk) -> tuple[int, int, int]:
    """Order a file's passages for reading, ACROSS the documents it split into.

    ``chunk_index`` restarts at 0 for every RAG document, and one attachment
    becomes a document per parsed unit and per oversized part (its ``doc_id`` is
    ``attachment:<id>:<ordinal>:<part>``). So chunk_index alone interleaves the
    parts of a multi-document file; the file's true reading order is
    ``(ordinal, part, chunk_index)``, read numerically off the doc_id. A doc_id
    in any other shape (a source that is already one document) sorts on
    chunk_index alone, which is correct for it.
    """
    ordinal, part = 0, 0
    pieces = chunk.doc_id.rsplit(":", 2)
    if len(pieces) == 3:
        try:
            ordinal, part = int(pieces[1]), int(pieces[2])
        except ValueError:
            ordinal, part = 0, 0
    return (ordinal, part, chunk.chunk_index)


def merge_record_passages(
    grouped: list[list[ScoredChunk]], *, budget_tokens: int
) -> list[ScoredChunk]:
    """Fold each source's passages into ONE chunk, within a per-source budget.

    A file is many passages but ONE source. Left as separate chunks they would be
    cited once each - the duplicate-citation problem ``dedupe_chunks_by_source``
    exists to prevent - and the per-source dedupe would instead throw all but one
    of them away, which is worse: a long document would be summarised from a
    single paragraph of itself.

    So the passages are stitched back into one chunk in the file's READING order
    (see :func:`_passage_order_key`: ordinal, then part, then chunk_index - not
    score order, and not chunk_index alone, which would shuffle the parts of a
    multi-document file), and the result is one passage in the prompt and one
    citation for the file, keeping the [n] numbering aligned exactly as before.
    The merged chunk is anchored on the file's FIRST passage, so its citation
    (id, doc_id, title) points at where the file begins rather than at whichever
    part happened to score highest; it carries the best score so ranking still
    treats the file as the relevant hit it is.

    The per-source budget is what keeps this honest against
    :func:`~app.services.context_budget.trim_chunks_to_budget`, which drops a
    chunk WHOLE when it overflows: an unbounded merge would build one giant chunk
    that cannot fit and is therefore dropped entirely, so naming a big file would
    ground on nothing at all. Passages are taken best-first (so the most relevant
    survive the cut) and then re-ordered for reading. The budget is measured
    against the JOINED length - separators included - so a merged chunk that
    passes here is one trim will also keep, not one a few tokens over that trim
    then drops in full.

    Args:
        grouped: Passages per source, best first, as ``retrieve_records`` returns.
        budget_tokens: The token ceiling for EACH source's merged chunk.

    Returns:
        One chunk per source, in the order the sources arrived.
    """
    separator_chars = len(_PASSAGE_JOIN)
    merged: list[ScoredChunk] = []
    for passages in grouped:
        if not passages:
            continue
        kept: list[ScoredChunk] = []
        used_chars = 0
        for passage in passages:  # best-scoring first: those survive a tight budget
            addition = len(passage.text) + (separator_chars if kept else 0)
            candidate_tokens = (
                used_chars + addition + APPROX_CHARS_PER_TOKEN - 1
            ) // APPROX_CHARS_PER_TOKEN
            if kept and candidate_tokens > budget_tokens:
                continue
            kept.append(passage)
            used_chars += addition
        if not kept:
            continue
        # Read in the file's own order, whatever order relevance chose them in.
        kept.sort(key=_passage_order_key)
        anchor = kept[0]
        best_score = max(passage.score for passage in passages)
        merged.append(
            anchor.model_copy(
                update={
                    "text": _PASSAGE_JOIN.join(passage.text for passage in kept),
                    "score": best_score,
                }
            )
        )
    return merged


# Query params that identify a marketing/analytics campaign, not the page, so
# the same document arriving under different tracking URLs collapses to one.
# Kept to unambiguous tracking keys: a param like "ref" is sometimes content
# (a resource id), so dropping it could merge two genuinely distinct pages and
# lose a citation - it is deliberately excluded.
_TRACKING_PARAMS = frozenset(
    {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}
)


def _canonical_source_key(chunk: ScoredChunk) -> str:
    """Return a stable per-source key so chunks of one document group together.

    Web results key on their canonicalized URL - lowercased scheme/host, no
    fragment, tracking params dropped, trailing slash trimmed - so the same page
    under cosmetically different links counts once; everything else keys on its
    ``doc_id``. A malformed URL falls back to the raw string rather than raising.
    """
    url = chunk.url
    if not url:
        return f"doc:{chunk.doc_id}"
    try:
        parts = urlsplit(url)
    except ValueError:
        return f"url:{url}"
    path = parts.path.rstrip("/") or "/"
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_PARAMS
        ]
    )
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def dedupe_chunks_by_source(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Keep only the highest-ranked chunk per source, preserving order.

    Chunks arrive best-first (dense score, then any rerank), so the first chunk
    seen for a source is its strongest passage; later chunks of the same source
    are dropped. This collapses the many chunks a long web page or document splits
    into down to one grounded passage and one citation per source, which keeps the
    inline ``[n]`` numbering aligned across the grounded prompt and the citation
    list (both number the same deduplicated chunk list).
    """
    seen: set[str] = set()
    unique: list[ScoredChunk] = []
    for chunk in chunks:
        key = _canonical_source_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


async def has_content(
    *,
    tenant: TenantContext,
    embedder: Embedder,
    store: VectorStore,
    session_id: str | None = None,
    include_kb: bool = True,
) -> bool:
    """Cheap check whether retrieval could possibly return anything.

    Uses the exact scopes a chat-turn query would use, so a True here means a
    query has candidates. This is what lets chat skip loading the embedding
    model entirely for tenants that never ingested anything. ``include_kb``
    mirrors :func:`retrieve`: when False the kb scope is ignored, so the check
    reflects a session-only retrieval.
    """
    tenant_id = str(tenant.tenant_id)
    space = embedder.model
    if include_kb and await store.count("kb", space, tenant_id) > 0:
        return True
    if session_id is not None:
        # The same scopes :func:`retrieve` uses, user filter included - a check
        # that counted a colleague's passages would promise candidates that the
        # retrieval right after it cannot return.
        return (
            await store.count(
                "session",
                space,
                tenant_id,
                session_id=session_id,
                user_id=str(tenant.user_id),
            )
            > 0
        )
    return False
