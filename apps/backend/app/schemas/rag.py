"""HTTP schemas for the RAG API (/rag/ingest, /rag/query).

Also home to the stranded-vectors report, which is a retrieval fact rather
than a model-manager one - it is merely SURFACED on the management view (and,
as a standing notice, on a chat turn) rather than owned by either.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.providers.rag.base import CollectionKind, SourceType


def _validate_session_id(value: str | None) -> str | None:
    """Require session ids to be UUIDs, normalized to the canonical spelling.

    Chat retrieval scopes ephemeral vectors by the canonical string form of
    the conversation id, so a non-UUID session id - or a UUID stored in an
    alternate spelling (uppercase, braced, unhyphenated) - could never be
    seen by a chat turn. Rejecting the former and canonicalizing the latter
    turns silent black holes into a 422 or a match.
    """
    if value is None:
        return value
    try:
        return str(UUID(value))
    except ValueError:
        raise ValueError("session_id must be a UUID (use the conversation id)") from None


class RagIngestItem(BaseModel):
    """One structured record to ingest."""

    id: str = Field(min_length=1, max_length=256)
    title: str | None = Field(default=None, max_length=500)
    text: str = Field(min_length=1, max_length=50_000)
    # Optional source link, carried onto the citation so a web result can be
    # opened; None for records (their id/title already identify them).
    url: str | None = Field(default=None, max_length=2000)
    # The source this item belongs to, when one source splits into several
    # items: an uploaded file becomes an item per page and per oversized part,
    # and this id is what later lets a chat turn ask for "that file" as a whole.
    # None when the item IS the source.
    record_id: str | None = Field(default=None, max_length=256)


class RagIngestRequest(BaseModel):
    """Ingest structured records into RAG storage.

    Exactly one destination applies: ``session_id`` targets the ephemeral
    session scope (vectors expire on a TTL; use the conversation id so chat
    turns can retrieve them), otherwise the named persistent collection
    (default name "default") is used. Setting both is rejected - silently
    ignoring one would hide a caller bug.
    """

    collection: str | None = Field(default=None, min_length=1, max_length=200)
    session_id: str | None = None
    source_type: SourceType = "records"
    items: list[RagIngestItem] = Field(min_length=1, max_length=100)

    _session_uuid = field_validator("session_id")(_validate_session_id)

    @model_validator(mode="after")
    def _one_destination(self) -> RagIngestRequest:
        """Reject requests that name both a collection and a session."""
        if self.collection is not None and self.session_id is not None:
            raise ValueError("set either collection or session_id, not both")
        return self

    @model_validator(mode="after")
    def _unique_item_ids(self) -> RagIngestRequest:
        """Reject duplicate item ids - they would silently overwrite each other."""
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise ValueError(f"duplicate item id: {item.id!r}")
            seen.add(item.id)
        return self


class RagIngestResponse(BaseModel):
    """Result of an ingest call (200: upserts are idempotent re-runs)."""

    collection_id: str | None = None
    documents: int
    chunks: int


class RagQueryRequest(BaseModel):
    """Similarity-search the stored chunks."""

    query: str = Field(min_length=1, max_length=32_000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: str | None = Field(default=None, min_length=1, max_length=200)
    session_id: str | None = None

    _session_uuid = field_validator("session_id")(_validate_session_id)


class RagQueryResult(BaseModel):
    """One retrieval hit."""

    id: str
    doc_id: str
    source_type: str
    title: str | None = None
    text: str
    score: float
    chunk_index: int


class RagQueryResponse(BaseModel):
    """Retrieval hits, best first."""

    results: list[RagQueryResult]


class VectorSpaceRow(BaseModel):
    """One physical vector collection as the stranded-vectors report sees it.

    Attributes:
        name: The collection name, exactly as the store holds it.
        kind: ``kb`` for the persistent knowledge base, ``session`` for the
            TTL-swept conversation scratch space.
        points: How many points it holds. Exact as of the moment it was read,
            but not transactional with the rest of the report - a concurrent
            write moves it, which nothing here depends on.
        embed_models: The embedding model tags SAMPLED from real stored
            points. Empty means the sample found no provenance, which reads as
            "cannot tell", never as "written by another model".
        readable: The active embedding model wrote these points, so a query
            issued today can still reach them.
    """

    name: str
    # The seam's own literal, not a copy of it: the two would drift the moment
    # a third collection kind existed.
    kind: CollectionKind
    points: int = 0
    embed_models: list[str] = []
    readable: bool = False


class StrandedVectorsReport(BaseModel):
    """Whether stored vectors are still reachable by the active embedder.

    The embedding model DEFINES the vector space, so changing it leaves every
    previously stored vector in a space nothing queries any more: retrieval
    simply returns nothing and the deployment looks as though it forgot what
    it was taught. This report is what turns that silence into a sentence.

    Out-of-reach points come in two shapes, and only one of them is a problem.
    With NOTHING readable left, retrieval genuinely returns nothing - the alarm.
    With a working corpus beside them, they are leftovers from a completed
    embedding-model change (old collections linger until removed by hand), and
    calling that an emergency would be false and would never clear. The two are
    told apart by ``readable_points``; only the first sets ``stranded``.

    Attributes:
        known: The store could be enumerated. False means the report says
            nothing at all - never that everything is fine.
        active_model: The embedding model the deployment is configured to use,
            or null when nothing resolves one.
        stranded: The alarm, and the ONLY field a user-facing surface may act
            on. True when a persistent (``kb``) collection holds points the
            active embedder cannot reach AND nothing readable remains beside
            them.
        stranded_points: How many persistent points are out of reach. Summed
            across the WHOLE deployment - the enumeration behind it is
            tenant-free - so this belongs on an operator's page and never in a
            sentence shown to one user.
        stranded_models: The model tags those points were written with, as
            sampled - what a reader needs in order to restore the old embedder
            or re-ingest against the new one.
        stranded_session_points: The same count for the ephemeral session
            scope, reported but deliberately NOT alarmed: those points carry a
            TTL and the sweep removes them within hours on their own.
        readable_points: How many persistent points the ACTIVE embedder wrote,
            and can therefore still reach. Zero beside a nonzero
            ``stranded_points`` is what makes the difference an emergency.
        spaces: Every collection found, readable and stranded alike.
        message: A plain-language summary, or null when there is nothing to
            say. Two shapes: the alarm wording when ``stranded`` is true, and
            an operator-facing cleanup note when it is false. Only the alarm
            shape is ever shown to a user, which is why only it is count-free.
    """

    known: bool = True
    active_model: str | None = None
    stranded: bool = False
    stranded_points: int = 0
    stranded_models: list[str] = []
    stranded_session_points: int = 0
    readable_points: int = 0
    spaces: list[VectorSpaceRow] = []
    message: str | None = None
