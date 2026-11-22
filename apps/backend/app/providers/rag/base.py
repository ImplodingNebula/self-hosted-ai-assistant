"""The RAG seam: the interfaces every retrieval backend implements.

Services depend only on the four small protocols in this module - Embedder,
Chunker, VectorStore, Reranker - and their shared value types, never on
qdrant-client or a concrete embedding backend. This is a thin custom layer by
design (no retrieval framework), so the pipeline stays small, debuggable, and
swappable. See docs/ARCHITECTURE.md section 4.2.

Tenant scoping is mandatory at this seam exactly as it is at the repository
layer: every data method takes the tenant id explicitly, and implementations
must scope reads and stamp writes with it - callers cannot opt out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Which physical collection a point lives in: the persistent per-tenant
# knowledge base, or the ephemeral session scratch space (TTL-swept).
CollectionKind = Literal["kb", "session"]

# Where ingested content came from. "records" is direct API ingestion; the
# other sources arrive with later phases but are part of the frozen payload
# vocabulary now so nothing needs re-stamping.
SourceType = Literal["records", "upload", "mcp_records", "web"]


class VectorStoreUnavailableError(Exception):
    """Raised when the vector store cannot be reached or answers abnormally.

    Seam-level type: services catch this instead of any backend-specific
    exception (qdrant-client, httpx, ...), keeping the boundary agnostic.
    """


class Chunk(BaseModel):
    """One chunker output unit, ready to embed."""

    text: str
    index: int
    doc_id: str
    title: str | None = None


class ChunkPayload(BaseModel):
    """The frozen per-point payload schema (docs/ARCHITECTURE.md section 6).

    ``created_at`` and ``ttl_at`` are UNIX epoch seconds - numeric so the
    store can range-filter them (the TTL sweep) without any datetime parsing.
    ``embed_model`` records PROVENANCE only: vector-space isolation comes from
    the per-model collection names (see :class:`VectorStore`), no query
    filters on this field.
    """

    tenant_id: str
    user_id: str | None = None
    session_id: str | None = None
    collection_id: str | None = None
    source_type: SourceType = "records"
    doc_id: str
    # The SOURCE this chunk came from, when one source produces many documents:
    # an uploaded file becomes a document per page and per 50k-char part, and
    # this is what lets a turn that names the file address all of them at once.
    # None for sources that are already one document (a web page, a record).
    record_id: str | None = None
    title: str | None = None
    text: str
    chunk_index: int = 0
    created_at: float
    ttl_at: float | None = None
    embed_model: str
    # Optional source link (e.g. a web result's page). Surfaces on the citation
    # so the UI can render a clickable source; None for records without a URL.
    url: str | None = None
    # Fingerprint of the source content this chunk came from (hash over the
    # chunking config + document text). Lets a re-ingest of the same document
    # skip the embed-and-rewrite entirely; None on points written before the
    # field existed (those simply never match, so they re-ingest normally).
    content_hash: str | None = None


class ChunkPoint(BaseModel):
    """A vector plus its payload, as written to the store."""

    id: str
    vector: list[float]
    payload: ChunkPayload


class DocFingerprint(BaseModel):
    """Identity of one stored document: enough to detect an unchanged re-ingest.

    ``chunk_count`` guards completeness: a document that lost part of its
    chunks to a mid-write failure must not satisfy the ingest skip.
    """

    doc_id: str
    content_hash: str | None = None
    created_at: float = 0.0
    chunk_count: int = 0


class StoredSource(BaseModel):
    """One SOURCE the knowledge base holds, as the store itself can describe it.

    The knowledge base has no row anywhere in the database - ``ingest_structured``
    writes chunks and nothing else - so the only description of what is stored is
    the one assembled from the chunk payloads themselves. This is that
    description, and it is what the knowledge console lists.

    The unit is a SOURCE, not a RAG document: one uploaded file becomes a
    document per parsed unit and per oversized part, and listing those separately
    would show a person six rows for one file they sent. ``id`` is therefore
    ``record_id`` where the chunks carry one and ``doc_id`` where they do not
    (a directly-ingested record IS its own source), which is exactly the identity
    :meth:`VectorStore.delete_kb_source` deletes by.

    ``embed_models`` can hold more than one tag, and that is the point rather
    than an edge case: a source ingested before an embedding-model change and
    again after it exists in two vector spaces at once, one of which no query
    can reach. Everything here is aggregated ACROSS those spaces, so the counts
    describe every copy of the source that is stored.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    title: str | None = None
    source_type: str = "records"
    # The link a citation would open, for sources that have one (a web page).
    url: str | None = None
    collection_id: str | None = None
    # The MOST RECENT ingest of this source, as UNIX epoch seconds: a re-ingest
    # rewrites the chunks, and when it was last written is the useful fact.
    created_at: float = 0.0
    content_hash: str | None = None
    # How many RAG documents the source became, and how many chunks in total.
    documents: int = 0
    chunks: int = 0
    # The embedding tags that actually wrote the stored points. Empty means the
    # points carry no provenance, which reads as "cannot tell" - never as
    # "written by something else" (the same rule :class:`VectorSpace` follows).
    embed_models: frozenset[str] = frozenset()


class StoredInventory(BaseModel):
    """Everything one knowledge-base walk found, and whether it saw it all.

    ``complete`` is not a nicety. A source's chunks are scattered ARBITRARILY
    through the store - point ids are hashes of the logical key, so scroll order
    has nothing to do with which document a point belongs to - which means a
    walk that stops early does not stop "after" some documents and "before"
    others. It stops in the middle of all of them at once, and every count it
    reports is then a floor rather than a total.

    So the flag qualifies the WHOLE answer, not its tail: False means neither
    the list of sources nor any figure on it is final, and a caller showing
    those figures has to say so.
    """

    model_config = ConfigDict(frozen=True)

    sources: list[StoredSource] = []
    complete: bool = True


class VectorSpace(BaseModel):
    """One physical collection the store holds, and whose vectors are in it.

    ``embed_models`` is SAMPLED from stored points rather than parsed out of
    ``slug``: the slug is a lossy normalization of the model tag (every
    character a collection name cannot carry collapses to one placeholder), so
    it cannot be reversed into the tag that wrote the points. The sample is
    therefore the only trustworthy statement of provenance - and it is a
    sample, so a collection holding a mixture may report only the tags that
    happened to fall in the page. ``points`` counts the WHOLE collection, so
    the two must not be multiplied together into a per-tag figure.

    An empty ``embed_models`` on a non-empty collection means the points carry
    no provenance at all, which readers must treat as "cannot tell", never as
    "written by someone else".
    """

    model_config = ConfigDict(frozen=True)

    name: str
    kind: CollectionKind
    # The collection name with its kind prefix removed - the store's own,
    # lossy rendering of the embedding model tag. Display only.
    slug: str
    points: int = 0
    embed_models: frozenset[str] = frozenset()


class ScoredChunk(BaseModel):
    """One retrieval hit: the stored chunk plus its similarity score."""

    id: str
    score: float
    text: str
    doc_id: str
    # The source this chunk came from, when one source produced many documents
    # (see ChunkPayload.record_id). Carried through retrieval so a caller can
    # group a file's passages back together.
    record_id: str | None = None
    title: str | None = None
    source_type: str = "records"
    collection_id: str | None = None
    session_id: str | None = None
    chunk_index: int = 0
    url: str | None = None


class Embedder(ABC):
    """Turns text into vectors; one embedder defines one vector space."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The embedding model tag - stamped on every point it produces."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int | None:
        """The vector dimension, or None when nothing states it."""
        raise NotImplementedError

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts (implementations batch internally)."""
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        vectors = await self.embed_texts([text])
        return vectors[0]


class Chunker(ABC):
    """Splits document text into embed-ready chunks."""

    @abstractmethod
    def chunk(self, text: str, *, doc_id: str, title: str | None = None) -> list[Chunk]:
        """Split ``text`` into ordered chunks; empty input yields no chunks."""
        raise NotImplementedError


class VectorStore(ABC):
    """The vector-store seam; implemented first by QdrantVectorStore.

    Physical collections are scoped per vector ``space`` - the embedding
    model tag - so vectors from different embedders (and different
    dimensions) can never mix: a tier or embedder change simply lands in a
    fresh, empty space. Old spaces linger until manually purged.

    Every DATA-ACCESS method (upsert/query/count/list_kb_sources/delete_doc/
    delete_kb_source/delete_session) scopes to the ``tenant_id`` it is given -
    reads filter by it, writes stamp it over whatever the payload carried -
    mirroring the repository chokepoint so a service cannot touch another
    tenant's vectors through this interface. ``ensure_ready``/``list_spaces``/
    ``sweep_expired``/``ping``/``aclose`` are the sanctioned tenant-free
    maintenance surface (the TTL sweep is cross-tenant by nature, and so is
    asking which spaces exist).
    """

    @abstractmethod
    async def ping(self) -> None:
        """Check that the store is reachable, and do nothing else.

        Deliberately narrower than ``list_spaces``: the readiness probe wants
        one cheap round trip, where the space report counts and samples every
        collection it finds.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        raise NotImplementedError

    @abstractmethod
    async def ensure_ready(self, space: str, dim: int) -> None:
        """Create the space's collections and payload indexes if needed.

        Idempotent and memoized by implementations; called lazily on the
        first write into a space, never at application startup.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self, kind: CollectionKind, space: str, tenant_id: str, points: list[ChunkPoint]
    ) -> int:
        """Write points (idempotent by point id); return how many were written."""
        raise NotImplementedError

    @abstractmethod
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

        ``record_ids`` narrows the search to chunks whose ``record_id`` is one of
        them - the chunks of a named source (an uploaded file), wherever they
        landed among that source's many documents.

        ``user_id`` narrows to one PERSON inside the tenant, and is what the
        ephemeral session scope is searched with: a conversation's uploaded
        files and fetched pages belong to whoever opened it, not to the
        workspace. It is an EXACT match that fails closed - a point carrying no
        ``user_id`` matches nobody (see the reasoning in the Qdrant
        implementation) - and it must NOT be applied when it is None, which is
        how the shared knowledge base stays shared.
        """
        raise NotImplementedError

    @abstractmethod
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
        """Count stored chunks in scope (cheap has-anything-to-search check).

        Applies the exact same filters as :meth:`query` so a nonzero count
        guarantees a query has candidates; a missing space counts as zero. That
        equivalence is why ``user_id`` belongs here too: a count that ignored it
        would report a colleague's passages as candidates and send the turn off
        to embed a query that could only come back empty.
        """
        raise NotImplementedError

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
        """Return the stored fingerprint of each already-present document.

        Powers the unchanged-content ingest skip and the web-result reuse
        window: a document whose stored ``content_hash`` matches the incoming
        one (or whose ``created_at`` is recent enough) is not fetched or
        re-embedded again. Best-effort by contract - a missing document is
        simply absent from the result, and this concrete default finds
        nothing, which callers must treat as "unknown: ingest normally".

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        return {}

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
        """Extend the TTL of documents whose re-ingest or reuse was skipped.

        The unchanged-content skip and the web-result reuse window leave
        stored points untouched, which would otherwise pin a session
        document's expiry to its FIRST ingest; refreshing keeps documents in
        active use alive, exactly as the rewrite they replaced would have.
        Default: no-op (stores without TTL semantics need nothing).

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        return None

    @abstractmethod
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
        """Drop every chunk of one document before its re-ingest.

        Point ids are deterministic per (doc, chunk index), but a shrunk
        document would leave stale tail chunks behind - deleting by filter
        first is what makes re-ingest actually clean.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_record(
        self, tenant_id: str, record_id: str, *, session_id: str | None = None
    ) -> int:
        """Drop every chunk that came from one source record, across ALL spaces.

        A source (an uploaded file) becomes many documents - a page, a part of a
        page - so it cannot be addressed by ``doc_id``; ``record_id`` is the one
        payload field that names the whole of it. This backs deleting a single
        attachment: its text must stop being retrievable the moment its owner
        removes it, and a delete that left the vectors behind would still answer
        from a file the user believes is gone.

        Space-agnostic for the same reason as :meth:`delete_session`: points may
        have been ingested under an older embedding model. Returns a best-effort
        count.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_kb_sources(
        self, tenant_id: str, *, collection_id: str | None = None
    ) -> StoredInventory:
        """Describe what this tenant's knowledge base holds, source by source.

        The knowledge base is stored ONLY here - no table records its contents -
        so this is the whole inventory, and the reason the seam grows a read that
        is neither a similarity search nor a count. Chunks are grouped into
        sources exactly as :class:`StoredSource` defines, and the walk spans
        every ``kb`` space rather than only the active embedder's: a source left
        behind by an embedding-model change is still stored, still occupying
        room, and still something an operator has to be able to see and remove.

        Deliberately takes no ``limit``. Grouping chunks into sources is only
        correct once every chunk has been SEEN - a source's points are scattered
        through the store, so a walk cut short under-reports every source at
        once rather than simply omitting the last few (see
        :class:`StoredInventory`). Implementations bound the work by their own
        ceiling and report reaching it through ``complete``; a caller wanting
        fewer rows takes them from the answer, after the grouping is done.

        Tenant-scoped like every other data access here, and NOT user-scoped:
        the knowledge base is a workspace resource (``retrieve`` searches it with
        no user filter for that reason), so narrowing it to whoever ingested a
        document would make a shared corpus look private.

        Args:
            tenant_id: The acting tenant; the mandatory filter.
            collection_id: Narrow to one named corpus, or None for all of them.

        Returns:
            The inventory, sources most recently ingested first, and whether the
            walk saw the whole corpus.

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_kb_source(
        self, tenant_id: str, source_id: str, *, collection_id: str | None = None
    ) -> int:
        """Drop one knowledge-base source's chunks across every vector space.

        Addressed by :attr:`StoredSource.id`, which is a ``record_id`` for a
        source that split into several documents and a ``doc_id`` for one that
        did not, so the match is on EITHER field - the caller lists sources and
        deletes one of them, and should not have to know which shape it was
        stored in.

        Space-agnostic for the same reason :meth:`delete_record` is: a source may
        have copies in a space written by an older embedding model, and a delete
        that removed only the reachable copy would report success while leaving
        stored text behind - the exact orphan this method exists to prevent.

        Session points are never touched: this is the persistent knowledge base,
        and one conversation's scratch is not the workspace's to remove.

        Args:
            tenant_id: The acting tenant; the mandatory filter.
            source_id: The source to remove, as :meth:`list_kb_sources` named it.
            collection_id: Narrow to one named corpus, or None for all of them.

        Returns:
            How many chunks were deleted, counted before the delete (so a
            concurrent write can make it an over-report - it is an answer to
            "what went", not a transactional total).

        Raises:
            VectorStoreUnavailableError: The store cannot be reached.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_session(self, tenant_id: str, session_id: str) -> int:
        """Drop one session's ephemeral points across ALL spaces.

        Space-agnostic on purpose: this backs the privacy purge on
        conversation delete, which must not miss points ingested under an
        older embedding model. Returns a best-effort count.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_spaces(self) -> list[VectorSpace]:
        """Enumerate every collection this store owns, with sampled provenance.

        Tenant-free maintenance, like the sweep: vector spaces are a property
        of the DEPLOYMENT (which embedding model wrote what), not of any one
        tenant, and the question being asked - "is anything stored here still
        reachable by the embedder in use?" - has no tenant to scope it to.

        Each row carries a point count plus the ``embed_model`` values read off
        a SMALL sample of real points, which is the only trustworthy source of
        the tag that wrote them (see :class:`VectorSpace`). Implementations
        keep the sample bounded and payload-only - this backs a status report,
        not a scan. The count is a snapshot rather than a transactional read:
        a concurrent write moves it, which no caller here depends on.

        Returns:
            One :class:`VectorSpace` per physical collection, empty when the
            store holds none.

        Raises:
            VectorStoreUnavailableError: The store cannot be enumerated. An
                implementation that genuinely cannot enumerate must RAISE
                rather than return an empty list: "there are no spaces" and
                "I cannot tell you" are different answers, and a caller that
                cannot distinguish them would report a healthy store while
                every stored vector sat unreachable.
        """
        raise NotImplementedError

    @abstractmethod
    async def sweep_expired(self, now: float) -> int:
        """Delete session points expired at ``now`` across ALL spaces.

        Returns a best-effort count (counted before deletion - the two steps
        are not atomic, which is fine for a log line).
        """
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """Release the underlying client/connections."""
        raise NotImplementedError


class Reranker(ABC):
    """Re-scores retrieval hits so only the sharpest context reaches the model."""

    @abstractmethod
    async def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return the ``top_k`` chunks re-ordered by relevance to ``query``."""
        raise NotImplementedError
