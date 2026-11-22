"""Qdrant-backed VectorStore (a thin layer over qdrant-client, by design).

Collections are named per vector space - ``kb_chunks__{model}`` and
``session_chunks__{model}`` - so vectors from different embedding models (and
different dimensions) never share a collection; a tier or embedder change
lands in a fresh, empty space instead of colliding with stale vectors.

Tenant scoping is enforced here, at one chokepoint: every data-access call
builds its filter from the ``tenant_id`` argument, and every upsert stamps it
over whatever the payload carried - mirroring the repository layer's rule.
"""

from __future__ import annotations

import logging
import re
import time

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from qdrant_client.common.client_exceptions import QdrantException
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from app.providers.rag.base import (
    ChunkPoint,
    CollectionKind,
    DocFingerprint,
    ScoredChunk,
    StoredInventory,
    StoredSource,
    VectorSpace,
    VectorStore,
    VectorStoreUnavailableError,
)

logger = logging.getLogger(__name__)

_KB_PREFIX = "kb_chunks__"
_SESSION_PREFIX = "session_chunks__"
# Conservative collection-name alphabet; model tags carry ':' '/' '.'.
_SLUG_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]")
# Defensive server-side cap per upsert call; callers already flush per embed
# batch, this only guards a future caller passing one huge list.
_UPSERT_BATCH = 64
# One scroll page for fingerprint lookups. Deliberately a single page: the
# callers ask about a handful of docs (a search turn's results), and a doc
# whose chunks fall past the page simply reads as absent - it re-ingests,
# which is always safe.
_FINGERPRINT_SCROLL_LIMIT = 1024
# One scroll page of the knowledge-base inventory. Larger than the fingerprint
# page because this walk is exhaustive by nature - it has to see every chunk
# before it can say how many a document has - and a small page would only mean
# more round trips for the same bytes.
_INVENTORY_SCROLL_LIMIT = 512

# The ceiling on how many pages one inventory walk reads per collection: with
# the page size above, ~200k chunks apiece. A safety valve, not a tuning knob -
# past it the walk says so (StoredInventory.complete) instead of quietly
# reporting a fraction as a total.
#
# Sized against the largest thing the app will ACCEPT, which is the constraint
# that matters: attachment_max_bytes is 25 MB, the chunker's stride is 1200
# characters, so one maximum-size text file is ~21k chunks. A ceiling near that
# would let a single permitted upload put the workspace permanently past what
# any walk can describe - and since delete is by id, an id nobody can see is a
# document nobody can remove. Ten such files of headroom keeps the honest
# "partial" answer a warning rather than a dead end. The walk is payload-only
# and never reads the chunk TEXT, so the pages are small.
_INVENTORY_MAX_PAGES = 400

# How many points one space-enumeration reads to learn which embedding model
# wrote a collection. Deliberately tiny: provenance is near-uniform within a
# collection (the name already partitions by model), this runs per collection
# on a page load, and a bigger page would buy nothing but latency.
_PROVENANCE_SAMPLE_LIMIT = 8

# The exception surface qdrant-client actually leaks: its own two hierarchies
# (ApiException covers UnexpectedResponse; QdrantException covers
# ResourceExhaustedResponse, the 429 shape; ResponseHandlingException wraps
# httpx transport failures) plus the stdlib/local-mode shapes (ValueError for
# a vanished collection, Value/Type/KeyError from malformed out-of-band
# payloads). Anything here becomes the seam-level error.
_BACKEND_ERRORS = (
    ApiException,
    QdrantException,
    ResponseHandlingException,
    ConnectionError,
    TimeoutError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
)


def _space_slug(space: str) -> str:
    """Normalize an embedding model tag into a collection-name-safe slug."""
    return _SLUG_UNSAFE.sub("_", space)


def _collection_name(kind: CollectionKind, space: str) -> str:
    """Return the physical collection name for a (kind, space) pair."""
    prefix = _KB_PREFIX if kind == "kb" else _SESSION_PREFIX
    return f"{prefix}{_space_slug(space)}"


def _scope_filter(
    tenant_id: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    collection_id: str | None = None,
    doc_id: str | None = None,
    record_ids: list[str] | None = None,
    expired_before: float | None = None,
) -> qm.Filter:
    """Build the mandatory tenant filter plus any narrower scope conditions."""
    must: list[qm.Condition] = [
        qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))
    ]
    if session_id is not None:
        must.append(qm.FieldCondition(key="session_id", match=qm.MatchValue(value=session_id)))
    if user_id is not None:
        # Exact ownership, and NO tolerance for a missing user_id - the filter
        # fails closed. No backfill is needed to make that safe: ingest_structured
        # has stamped user_id on every session point since the store's first
        # commit (verified with git log -S), so no point this app has ever
        # written is unattributed, and there is no upgrade window in which one
        # could be. An earlier draft OR'd in an is_empty arm for hypothetical
        # unattributed points; it was dropped deliberately. This is a security
        # boundary (defence in depth beneath the router's ownership check), and
        # for a boundary the safe direction is closed: were a point ever to reach
        # the store without a user_id - a future ingest path that forgot to stamp
        # it, a manual row - matching nobody surfaces it as its owner reporting
        # "my file is not being found", a loud bug that gets fixed, rather than as
        # a silent cross-user leak. The keyword index on user_id serves this
        # exact match; the dropped is_empty arm could not have used it.
        must.append(qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)))
    if collection_id is not None:
        must.append(
            qm.FieldCondition(key="collection_id", match=qm.MatchValue(value=collection_id))
        )
    if doc_id is not None:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))
    if record_ids:
        # One source (an uploaded file) becomes many documents - a page, a part
        # of a page - so naming it means matching every chunk that came from it.
        must.append(qm.FieldCondition(key="record_id", match=qm.MatchAny(any=record_ids)))
    if expired_before is not None:
        must.append(qm.FieldCondition(key="ttl_at", range=qm.Range(lt=expired_before)))
    return qm.Filter(must=must)


def _expired_filter(expired_before: float) -> qm.Filter:
    """Filter for expired session points across every tenant (the sweep)."""
    return qm.Filter(must=[qm.FieldCondition(key="ttl_at", range=qm.Range(lt=expired_before))])


def _label_key(doc_id: str, payload: dict) -> tuple[int, int, int, str]:
    """Order one chunk within its source, for picking the source's own label.

    ``(ordinal, part, chunk_index)`` read NUMERICALLY off the trailing
    ``:<ordinal>:<part>`` of a multi-document source's doc id - the same reading
    ``rag_service._passage_order_key`` performs, and for the same reason: those
    segments are numbers, and comparing them as text puts unit 10 before unit 1.
    A doc id in any other shape (a source that is already one document) sorts on
    its chunk index alone, which is correct for it, with the id last so two
    sources can never compare equal.
    """
    ordinal, part = 0, 0
    pieces = doc_id.rsplit(":", 2)
    if len(pieces) == 3:
        try:
            ordinal, part = int(pieces[1]), int(pieces[2])
        except ValueError:
            ordinal, part = 0, 0
    return (ordinal, part, int(payload.get("chunk_index") or 0), doc_id)


def _source_filter(
    tenant_id: str, source_id: str, *, collection_id: str | None = None
) -> qm.Filter:
    """Match every chunk of one knowledge-base source, by either of its names.

    A source is addressed by ``record_id`` when its chunks carry one and by
    ``doc_id`` when they do not (see :class:`~app.providers.rag.base.StoredSource`),
    so the match is an either/or nested INSIDE the mandatory tenant condition.
    Nested rather than a sibling ``should``: a top-level ``should`` beside a
    ``must`` reads as an optimisation hint in some engines, and the one condition
    that must never be optional here is the tenant.

    The tenant condition is written out here rather than spread from
    :func:`_scope_filter`'s result. Spreading is correct today, but ``*(scope.must
    or [])`` is a shape that says "there may be nothing to add" - and the one
    thing this filter must never be missing is the tenant. Naming it makes a
    cross-tenant delete unrepresentable rather than merely true.
    """
    narrower = _scope_filter(tenant_id, collection_id=collection_id).must or []
    return qm.Filter(
        must=[
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
            # Everything _scope_filter added BEYOND the tenant (the collection),
            # which it always puts first.
            *narrower[1:],
            qm.Filter(
                should=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=source_id)),
                    qm.FieldCondition(key="record_id", match=qm.MatchValue(value=source_id)),
                ]
            ),
        ]
    )


class QdrantVectorStore(VectorStore):
    """VectorStore over qdrant-client's async API.

    Construction performs no network I/O (safe in offline app assembly and
    the OpenAPI export); connections happen lazily on first use. Tests inject
    ``AsyncQdrantClient(":memory:")`` - the in-process local mode - through
    the ``client`` parameter.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        client: AsyncQdrantClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Create the store against ``url``, or wrap an injected client.

        Args:
            url: Qdrant base URL; ignored when ``client`` is given.
            client: Optional externally-created client (tests use ":memory:").
            timeout: Per-request timeout in seconds for the owned client -
                explicit because large upsert batches outlive the default.
        """
        if client is not None:
            self._client = client
        else:
            self._client = AsyncQdrantClient(url=url, timeout=int(timeout))
        # Memoized (collection, dim) pairs so ensure_ready is one call per
        # space per process lifetime, not per request.
        self._ready: set[tuple[str, int]] = set()

    async def aclose(self) -> None:
        """Close the underlying client."""
        try:
            await self._client.close()
        except _BACKEND_ERRORS:  # pragma: no cover - best-effort shutdown
            logger.warning("Closing the Qdrant client failed", exc_info=True)

    async def ping(self) -> None:
        """List collection NAMES as the cheapest proof of life; see :meth:`VectorStore.ping`.

        Qdrant has no dedicated liveness verb that this client exposes, and
        listing names is the least work it will do: no counts, no scrolls, no
        per-collection round trips.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        try:
            await self._client.get_collections()
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc

    async def ensure_ready(self, space: str, dim: int) -> None:
        """Create both of the space's collections and their payload indexes.

        Safe under concurrent first ingests: losing the create race (the
        collection appearing between the exists-check and the create) is
        success, not an error. Index creation runs UNCONDITIONALLY (the calls
        are idempotent) so a crash between create and index on some earlier
        attempt cannot leave a collection permanently unindexed.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        for kind in ("kb", "session"):
            name = _collection_name(kind, space)  # type: ignore[arg-type]
            if (name, dim) in self._ready:
                continue
            try:
                if not await self._client.collection_exists(name):
                    try:
                        await self._client.create_collection(
                            collection_name=name,
                            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                        )
                    except _BACKEND_ERRORS:
                        # Lost a concurrent create race, or a real failure:
                        # re-check rather than guess from the error shape.
                        if not await self._client.collection_exists(name):
                            raise
                await self._create_indexes(name, kind == "session")
            except _BACKEND_ERRORS as exc:
                raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
            self._ready.add((name, dim))

    async def _create_indexes(self, name: str, is_session: bool) -> None:
        """Create the payload indexes every filter in this module relies on.

        All of them are created with ``wait=False``, which is load-bearing:
        adding an index to a collection that ALREADY holds points makes Qdrant
        build it over every one of them, and that build is slow enough on a real
        deployment to outrun the client timeout - a new index took 32 seconds
        against a populated collection here, so the first ingest after it shipped
        failed with a seam-level "store unavailable" and had to be retried. The
        wait buys nothing: a filter is CORRECT on an unindexed field (Qdrant just
        scans), so the only thing blocking on the build achieves is turning a
        transparent slow-down into an outage. The index lands moments later and
        every later query is fast.

        Local mode (":memory:") accepts these calls as no-ops, so index
        definitions are only exercised against a real server - the live
        deployment check asserts the collection's payload schema explicitly.
        """
        await self._client.create_payload_index(
            collection_name=name,
            field_name="tenant_id",
            # is_tenant marks the primary partitioning key so Qdrant
            # co-locates each tenant's points (its multitenancy guidance).
            field_schema=qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD, is_tenant=True),
            wait=False,
        )
        await self._client.create_payload_index(
            collection_name=name,
            field_name="doc_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
            wait=False,
        )
        # The source a document came from (an attachment id), so a turn that
        # names a file can address every chunk of it at once.
        await self._client.create_payload_index(
            collection_name=name,
            field_name="record_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
            wait=False,
        )
        if is_session:
            await self._client.create_payload_index(
                collection_name=name,
                field_name="session_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
                wait=False,
            )
            # Who owns an ephemeral point. Only the session scope narrows by it -
            # the knowledge base is shared workspace-wide - and every chat turn
            # with a conversation applies it, so it is worth an index rather than
            # a scan on the one collection that is read on the generation path.
            await self._client.create_payload_index(
                collection_name=name,
                field_name="user_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
                wait=False,
            )
            await self._client.create_payload_index(
                collection_name=name,
                field_name="ttl_at",
                field_schema=qm.PayloadSchemaType.FLOAT,
                wait=False,
            )
        else:
            await self._client.create_payload_index(
                collection_name=name,
                field_name="collection_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
                wait=False,
            )

    async def upsert(
        self, kind: CollectionKind, space: str, tenant_id: str, points: list[ChunkPoint]
    ) -> int:
        """Write points, stamping the bound tenant over every payload.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        if not points:
            return 0
        name = _collection_name(kind, space)
        structs = []
        for point in points:
            payload = point.payload.model_copy(update={"tenant_id": tenant_id})
            structs.append(
                qm.PointStruct(id=point.id, vector=point.vector, payload=payload.model_dump())
            )
        try:
            for start in range(0, len(structs), _UPSERT_BATCH):
                await self._client.upsert(
                    collection_name=name, points=structs[start : start + _UPSERT_BATCH], wait=True
                )
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        return len(structs)

    async def query(
        self,
        kind: CollectionKind,
        space: str,
        tenant_id: str,
        vector: list[float],
        *,
        top_k: int,
        session_id: str | None = None,
        user_id: str | None = None,
        collection_id: str | None = None,
        record_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        """Return the nearest chunks in scope; a missing space is just empty.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        name = _collection_name(kind, space)
        chunks: list[ScoredChunk] = []
        try:
            if not await self._client.collection_exists(name):
                return []
            response = await self._client.query_points(
                collection_name=name,
                query=vector,
                limit=top_k,
                query_filter=_scope_filter(
                    tenant_id,
                    session_id=session_id,
                    user_id=user_id,
                    collection_id=collection_id,
                    record_ids=record_ids,
                ),
                with_payload=True,
            )
            # Payload mapping stays INSIDE the try: a point written out of
            # band with a malformed payload must surface as the seam-level
            # error (which retrieval degrades on), never an untyped one.
            for point in response.points:
                payload = point.payload or {}
                chunks.append(
                    ScoredChunk(
                        id=str(point.id),
                        score=point.score,
                        text=str(payload.get("text") or ""),
                        doc_id=str(payload.get("doc_id") or ""),
                        record_id=payload.get("record_id"),
                        title=payload.get("title"),
                        source_type=str(payload.get("source_type") or "records"),
                        collection_id=payload.get("collection_id"),
                        session_id=payload.get("session_id"),
                        chunk_index=int(payload.get("chunk_index") or 0),
                        url=payload.get("url"),
                    )
                )
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        return chunks

    async def count(
        self,
        kind: CollectionKind,
        space: str,
        tenant_id: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        collection_id: str | None = None,
    ) -> int:
        """Count chunks under the exact filters :meth:`query` would apply.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        name = _collection_name(kind, space)
        try:
            if not await self._client.collection_exists(name):
                return 0
            result = await self._client.count(
                collection_name=name,
                count_filter=_scope_filter(
                    tenant_id,
                    session_id=session_id,
                    user_id=user_id,
                    collection_id=collection_id,
                ),
                exact=True,
            )
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        return result.count

    async def doc_fingerprints(
        self,
        kind: CollectionKind,
        space: str,
        tenant_id: str,
        doc_ids: list[str],
        *,
        session_id: str | None = None,
        collection_id: str | None = None,
    ) -> dict[str, DocFingerprint]:
        """Look up stored per-document fingerprints via one indexed scroll.

        Multiple chunks per document collapse to one fingerprint (they share
        ``content_hash`` and ``created_at`` by construction; the newest wins
        defensively) while ``chunk_count`` tallies them. TTL-expired points
        are excluded: an expired-but-unswept point must not vouch for its
        document when the sweep may delete it before this turn's retrieval
        runs. Best-effort single page - see _FINGERPRINT_SCROLL_LIMIT.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        if not doc_ids:
            return {}
        name = _collection_name(kind, space)
        found: dict[str, DocFingerprint] = {}
        try:
            if not await self._client.collection_exists(name):
                return {}
            scope = _scope_filter(tenant_id, session_id=session_id, collection_id=collection_id)
            must = [
                *(scope.must or []),
                qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)),
            ]
            expired = qm.FieldCondition(key="ttl_at", range=qm.Range(lt=time.time()))
            points, _next = await self._client.scroll(
                collection_name=name,
                scroll_filter=qm.Filter(must=must, must_not=[expired]),
                limit=_FINGERPRINT_SCROLL_LIMIT,
                with_payload=["doc_id", "content_hash", "created_at"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                doc_id = str(payload.get("doc_id") or "")
                if not doc_id:
                    continue
                created_at = float(payload.get("created_at") or 0.0)
                existing = found.get(doc_id)
                if existing is None:
                    found[doc_id] = DocFingerprint(
                        doc_id=doc_id,
                        content_hash=payload.get("content_hash"),
                        created_at=created_at,
                        chunk_count=1,
                    )
                    continue
                update: dict = {"chunk_count": existing.chunk_count + 1}
                if created_at > existing.created_at:
                    update["content_hash"] = payload.get("content_hash")
                    update["created_at"] = created_at
                found[doc_id] = existing.model_copy(update=update)
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        return found

    async def refresh_doc_ttl(
        self,
        kind: CollectionKind,
        space: str,
        tenant_id: str,
        doc_ids: list[str],
        *,
        ttl_at: float,
        session_id: str | None = None,
        collection_id: str | None = None,
    ) -> None:
        """Stamp a fresh ``ttl_at`` on the given documents' points by filter.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        if not doc_ids:
            return
        name = _collection_name(kind, space)
        try:
            if not await self._client.collection_exists(name):
                return
            scope = _scope_filter(tenant_id, session_id=session_id, collection_id=collection_id)
            must = [
                *(scope.must or []),
                qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)),
            ]
            await self._client.set_payload(
                collection_name=name,
                payload={"ttl_at": ttl_at},
                points=qm.Filter(must=must),
                wait=True,
            )
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc

    async def delete_doc(
        self,
        kind: CollectionKind,
        space: str,
        tenant_id: str,
        doc_id: str,
        *,
        session_id: str | None = None,
        collection_id: str | None = None,
    ) -> None:
        """Drop one document's chunks (clean-slate step before re-ingest).

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        name = _collection_name(kind, space)
        try:
            if not await self._client.collection_exists(name):
                return
            await self._client.delete(
                collection_name=name,
                points_selector=qm.FilterSelector(
                    filter=_scope_filter(
                        tenant_id,
                        session_id=session_id,
                        collection_id=collection_id,
                        doc_id=doc_id,
                    )
                ),
                wait=True,
            )
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc

    async def delete_record(
        self, tenant_id: str, record_id: str, *, session_id: str | None = None
    ) -> int:
        """Purge one source record's chunks across every space, session and kb.

        Filters on ``record_id``, which every chunk of a file carries, because a
        file spans many ``doc_id``s (one per page or part). ``session_id``
        narrows it to the conversation the file belongs to when the caller knows
        it - defence in depth beside the mandatory tenant filter. Note that
        passing it excludes knowledge-base points, which carry no session.

        Both kinds are swept because one ``ingest_structured`` entrypoint serves
        both: a caller deleting a record must not be told "nothing found" merely
        because it happened to be ingested into the other kind.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        deleted = 0
        try:
            for name in await self._all_collections():
                scope = _scope_filter(tenant_id, session_id=session_id, record_ids=[record_id])
                counted = await self._client.count(
                    collection_name=name, count_filter=scope, exact=True
                )
                if counted.count == 0:
                    continue
                await self._client.delete(
                    collection_name=name,
                    points_selector=qm.FilterSelector(filter=scope),
                    wait=True,
                )
                deleted += counted.count
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        return deleted

    async def list_kb_sources(
        self, tenant_id: str, *, collection_id: str | None = None
    ) -> StoredInventory:
        """Walk every kb space, folding this tenant's chunks back into sources.

        Payload-only scrolls (no vectors, and never the chunk TEXT, which is by
        far the largest field) paged across every knowledge-base collection, so a
        source stranded in an old embedding space is listed beside the readable
        ones. Newest ingest first.

        The walk runs to the END of each collection, and the one bound on it is
        :data:`_INVENTORY_MAX_PAGES` pages apiece - a safety valve against a
        corpus far larger than a console page should try to describe, not a
        page-size knob. Stopping sooner is not an option that trades size for
        speed: a source's points are spread through the collection in point-id
        order, which is a hash of the logical key and therefore unrelated to
        which document they belong to, so an early stop does not omit the last
        few sources - it under-counts ALL of them. Reaching the ceiling is
        reported through ``complete`` rather than silently trimmed.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        found: dict[str, StoredSource] = {}
        # (source id) -> the doc ids it spans, and the label of its first
        # document. Kept beside the model because neither is a running total.
        doc_ids: dict[str, set[str]] = {}
        labels: dict[str, tuple[tuple[int, int, int, str], str | None]] = {}
        complete = True
        try:
            for name in await self._kb_collections():
                offset = None
                for _page in range(_INVENTORY_MAX_PAGES):
                    points, offset = await self._client.scroll(
                        collection_name=name,
                        scroll_filter=_scope_filter(tenant_id, collection_id=collection_id),
                        limit=_INVENTORY_SCROLL_LIMIT,
                        offset=offset,
                        with_payload=[
                            "doc_id",
                            "record_id",
                            "title",
                            "source_type",
                            "url",
                            "collection_id",
                            "created_at",
                            "content_hash",
                            "embed_model",
                            "chunk_index",
                        ],
                        with_vectors=False,
                    )
                    for point in points:
                        self._fold_point(point.payload or {}, found, doc_ids, labels)
                    if offset is None:
                        break
                else:
                    # The page ceiling ran out with the collection still going.
                    # Errs toward "partial": a collection holding exactly a
                    # whole number of pages hands back a non-None offset on its
                    # last one, so it reports incomplete having in fact been
                    # read whole. The wrong direction to be wrong in is the
                    # other one.
                    complete = False
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        sources = [
            source.model_copy(
                update={
                    "documents": len(doc_ids[source_id]),
                    "title": labels[source_id][1],
                }
            )
            for source_id, source in found.items()
        ]
        # Newest first, and the id breaks ties so two sources ingested in the
        # same second do not come back in whatever order the walk happened to
        # visit their collections in.
        sources.sort(key=lambda source: (-source.created_at, source.id))
        return StoredInventory(sources=sources, complete=complete)

    @staticmethod
    def _fold_point(
        payload: dict,
        found: dict[str, StoredSource],
        doc_ids: dict[str, set[str]],
        labels: dict[str, tuple[tuple[int, int, int, str], str | None]],
    ) -> None:
        """Fold one scrolled chunk payload into its source's running description.

        The NEWEST chunk decides the mutable facts (content hash, link, type):
        a re-ingest rewrites a source's chunks, and what it currently says is
        what a reader is asking about. The label is decided separately, by the
        FIRST chunk of the FIRST document, because that is the source's opening -
        the newest chunk could be the last page of a long file.

        "First" is read NUMERICALLY out of the doc id, never lexicographically,
        for the reason ``rag_service._passage_order_key`` already documents:
        ``...:10:0`` sorts before ``...:1:0`` as text. Ordinal 0 normally hides
        the bug (``0`` sorts before every other digit), so it surfaces only when
        the first parsed unit produced no text and was dropped - a scanned cover
        page - and the file has ten or more units, which is exactly when a
        mislabelled row is hardest to explain.
        """
        doc_id = str(payload.get("doc_id") or "")
        if not doc_id:
            return
        record_id = payload.get("record_id")
        source_id = str(record_id) if record_id else doc_id
        created_at = float(payload.get("created_at") or 0.0)
        tag = payload.get("embed_model")
        existing = found.get(source_id)
        if existing is None:
            found[source_id] = StoredSource(
                id=source_id,
                source_type=str(payload.get("source_type") or "records"),
                url=payload.get("url"),
                collection_id=payload.get("collection_id"),
                created_at=created_at,
                content_hash=payload.get("content_hash"),
                chunks=1,
                embed_models=frozenset({str(tag)} if tag else set()),
            )
            doc_ids[source_id] = {doc_id}
            labels[source_id] = (
                _label_key(doc_id, payload),
                payload.get("title"),
            )
            return
        update: dict = {
            "chunks": existing.chunks + 1,
            "embed_models": existing.embed_models | ({str(tag)} if tag else set()),
        }
        if created_at > existing.created_at:
            update["created_at"] = created_at
            update["content_hash"] = payload.get("content_hash")
            update["source_type"] = str(payload.get("source_type") or "records")
            update["url"] = payload.get("url")
            update["collection_id"] = payload.get("collection_id")
        found[source_id] = existing.model_copy(update=update)
        doc_ids[source_id].add(doc_id)
        key = _label_key(doc_id, payload)
        if key < labels[source_id][0]:
            labels[source_id] = (key, payload.get("title"))

    async def delete_kb_source(
        self, tenant_id: str, source_id: str, *, collection_id: str | None = None
    ) -> int:
        """Purge one source's chunks from every kb space; see the seam's contract.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        deleted = 0
        try:
            for name in await self._kb_collections():
                scope = _source_filter(tenant_id, source_id, collection_id=collection_id)
                counted = await self._client.count(
                    collection_name=name, count_filter=scope, exact=True
                )
                if counted.count == 0:
                    continue
                await self._client.delete(
                    collection_name=name,
                    points_selector=qm.FilterSelector(filter=scope),
                    wait=True,
                )
                deleted += counted.count
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        return deleted

    async def delete_session(self, tenant_id: str, session_id: str) -> int:
        """Purge one session's points across every session space.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        deleted = 0
        try:
            for name in await self._session_collections():
                scope = _scope_filter(tenant_id, session_id=session_id)
                counted = await self._client.count(
                    collection_name=name, count_filter=scope, exact=True
                )
                if counted.count == 0:
                    continue
                await self._client.delete(
                    collection_name=name,
                    points_selector=qm.FilterSelector(filter=scope),
                    wait=True,
                )
                deleted += counted.count
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        return deleted

    async def sweep_expired(self, now: float) -> int:
        """Delete expired session points across every space; return the count.

        The count is taken before the delete (Qdrant's delete reports only an
        operation status, never a count), so it can over-report under a
        concurrent write - acceptable for a maintenance log line.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        swept = 0
        try:
            for name in await self._session_collections():
                expired = _expired_filter(now)
                counted = await self._client.count(
                    collection_name=name, count_filter=expired, exact=True
                )
                if counted.count == 0:
                    continue
                await self._client.delete(
                    collection_name=name,
                    points_selector=qm.FilterSelector(filter=expired),
                    wait=True,
                )
                swept += counted.count
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        return swept

    async def list_spaces(self) -> list[VectorSpace]:
        """Enumerate every owned collection, sampling each one's provenance.

        Two bounded calls per collection - an exact count and one small
        payload-only scroll - because the slug in the name is lossy and cannot
        be turned back into the embedding model tag that wrote the points.
        Ordered by name so repeated reports are diffable.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached. Never an
                empty list on failure: a caller must be able to tell "nothing
                stored" from "could not ask".
        """
        spaces: list[VectorSpace] = []
        try:
            for name, kind in await self._owned_collections():
                counted = await self._client.count(collection_name=name, exact=True)
                points, _next = await self._client.scroll(
                    collection_name=name,
                    limit=_PROVENANCE_SAMPLE_LIMIT,
                    with_payload=["embed_model"],
                    with_vectors=False,
                )
                sampled = {
                    str(tag)
                    for point in points
                    if (tag := (point.payload or {}).get("embed_model"))
                }
                prefix = _SESSION_PREFIX if kind == "session" else _KB_PREFIX
                spaces.append(
                    VectorSpace(
                        name=name,
                        kind=kind,
                        slug=name[len(prefix) :],
                        points=counted.count,
                        embed_models=frozenset(sampled),
                    )
                )
        except _BACKEND_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        return sorted(spaces, key=lambda space: space.name)

    async def _owned_collections(self) -> list[tuple[str, CollectionKind]]:
        """List every collection this store owns, each tagged with its kind.

        The single enumeration point: the sweep, the cross-space deletes and
        the space report all read from here, so "which collections are ours"
        is decided by the prefixes in exactly one place.
        """
        collections = await self._client.get_collections()
        owned: list[tuple[str, CollectionKind]] = []
        for candidate in collections.collections:
            if candidate.name.startswith(_SESSION_PREFIX):
                owned.append((candidate.name, "session"))
            elif candidate.name.startswith(_KB_PREFIX):
                owned.append((candidate.name, "kb"))
        return owned

    async def _session_collections(self) -> list[str]:
        """List every session collection, across all vector spaces."""
        return [name for name, kind in await self._owned_collections() if kind == "session"]

    async def _kb_collections(self) -> list[str]:
        """List every knowledge-base collection, across all vector spaces."""
        return [name for name, kind in await self._owned_collections() if kind == "kb"]

    async def _all_collections(self) -> list[str]:
        """List every collection this store owns, session and knowledge-base alike."""
        return [name for name, _kind in await self._owned_collections()]
