"""HTTP schemas for the knowledge console (/knowledge/documents).

The knowledge base has no table behind it - ``ingest_structured`` writes chunks
into the vector store and nothing else - so everything here is assembled from
what the store itself can say about the points it holds. That is why the shapes
below carry provenance rather than identity: there is no row to have an id, a
size or an owner, and inventing those fields would be describing a record that
does not exist.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.providers.rag.base import StoredSource


class KnowledgeDocumentOut(BaseModel):
    """One source in the knowledge base, with everything known about it.

    Attributes:
        id: What addresses this source on the delete endpoint. Opaque: it is
            whichever payload field names the whole source (see
            :class:`~app.providers.rag.base.StoredSource`), not a database id.
        title: The label a citation shows for the source's first document,
            which for an uploaded file is its filename. Null when the ingest
            supplied none.
        source_type: Where the content came from - ``records`` for the direct
            ingest API, ``upload`` for a file sent to this console, ``web`` for
            a page ingested under that label. ``mcp_records`` cannot appear:
            ``ingest_structured`` refuses to put a tool server's results
            anywhere but a conversation's own scope, because they may have been
            fetched under one person's credential.
        url: The link a citation would open, for sources that have one.
        documents: How many RAG documents the source became. An uploaded file
            becomes one per parsed unit (a page, a sheet) and per oversized
            part, so this is usually more than one for a large file.
        chunks: How many embedded passages are stored, summed across every
            vector space the source has a copy in.
        embed_models: The embedding tags that actually wrote those passages.
            More than one means the source exists in more than one vector
            space, which is what an embedding-model change leaves behind. Empty
            means the points carry no provenance, which is "cannot tell".
        reachable: Whether a query issued TODAY can find this source - that is,
            whether the active embedding model is among the tags above. False
            means the passages are stored but no answer can ever cite them.
        ingested_at: When the source was last written, UTC.
        content_hash: The fingerprint the ingest skip compares against, so a
            re-ingest of unchanged content costs nothing. It is ONE document's
            fingerprint - the hash is computed per document, and a source that
            split into several has one apiece - so it identifies the whole
            source only when ``documents`` is 1, which is why the client shows
            it only then. Null on points written before the field existed.
        collection_id: The named corpus holding it.
    """

    id: str
    title: str | None = None
    source_type: str
    url: str | None = None
    documents: int
    chunks: int
    embed_models: list[str] = []
    reachable: bool
    ingested_at: datetime | None = None
    content_hash: str | None = None
    collection_id: str | None = None

    @classmethod
    def from_source(cls, source: StoredSource, *, active_model: str | None) -> KnowledgeDocumentOut:
        """Render one stored source, judged against the active embedding model.

        ``reachable`` is decided the same way the stranded-vectors report decides
        it: on the provenance SAMPLED from real points, never on the collection
        name (which carries a lossy slug of the tag and cannot be reversed). With
        no embedder configured nothing is reachable, which describes the missing
        configuration rather than the passages.
        """
        return cls(
            id=source.id,
            title=source.title,
            source_type=source.source_type,
            url=source.url,
            documents=source.documents,
            chunks=source.chunks,
            embed_models=sorted(source.embed_models),
            reachable=active_model is not None and active_model in source.embed_models,
            ingested_at=(
                datetime.fromtimestamp(source.created_at, UTC) if source.created_at else None
            ),
            content_hash=source.content_hash,
            collection_id=source.collection_id,
        )


class KnowledgeListResponse(BaseModel):
    """The knowledge base's contents, newest first.

    Attributes:
        items: The stored sources.
        collection: The named corpus these came from.
        active_embedding_model: The tag every query is issued against, or null
            when nothing resolves one. It is what ``reachable``
            on each item was judged against, so a client can say WHY a source
            is out of reach rather than only that it is.
        truncated: The store held more sources than the walk returned. A client
            must not present the list as complete when this is true.
    """

    items: list[KnowledgeDocumentOut]
    collection: str
    active_embedding_model: str | None = None
    truncated: bool = False


class KnowledgeUploadResponse(BaseModel):
    """What one uploaded file added to the knowledge base.

    Attributes:
        id: The stored source's id, as the list and delete endpoints know it.
        title: The label its first document carries.
        documents: How many RAG documents the file became.
        chunks: How many passages were embedded and stored. Zero alongside a
            nonzero ``documents`` means every document was skipped as unchanged
            - a re-upload of a file already stored, which is a success.
        collection: The named corpus it landed in.
    """

    id: str
    title: str
    documents: int
    chunks: int
    collection: str


class KnowledgeDeleteResponse(BaseModel):
    """What a delete actually removed.

    Deliberately not a 204. A source can have copies in more than one vector
    space, and "nothing matched" is a real outcome worth telling apart from
    "3,000 passages went" - an empty answer usually means the id was already
    deleted, and a person who just pressed a button deserves to know which
    happened.

    Attributes:
        id: The source that was addressed.
        chunks_deleted: How many passages went, across every vector space.
    """

    id: str
    chunks_deleted: int = Field(ge=0)
