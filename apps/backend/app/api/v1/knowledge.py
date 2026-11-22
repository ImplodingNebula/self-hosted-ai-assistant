"""Knowledge-base endpoints: list what is stored, add a file, remove one.

The operational surface over the workspace's persistent corpus. Until now the
only way into it was ``POST /rag/ingest``, which takes structured JSON records -
fine for a program, useless for a person holding a PDF - and nothing at all read
it back or removed anything from it.

Routers stay HTTP-only: the multipart form and the response shapes live here,
:mod:`app.services.knowledge` decides what a file becomes and where it goes, and
every read and write is scoped to the acting tenant.

Two shapes here differ from their neighbours, both deliberately:

* **Adding a file answers when the work is done**, rather than 202-ing a job.
  The knowledge base has no table behind it, so there is nowhere to record
  "parsing" for a client to poll - see the service module's docstring. Every
  failure a file can produce is therefore answered in this response.
* **Deleting takes the id as a QUERY parameter.** A source id is opaque text
  chosen by whoever ingested it (the direct ingest API accepts any string), and
  one containing a slash cannot survive a path segment: the ASGI scope carries an
  already-decoded path, so ``%2F`` arrives as a separator and the route simply
  does not match. A query parameter carries any id there is.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_embedder,
    get_parser_registry,
    get_session,
    get_stream_limiter,
    get_vector_store,
    require_role,
)
from app.core.concurrency import StreamLimiter
from app.core.config import Settings, get_settings
from app.core.errors import BadRequestError, PayloadTooLargeError, ServiceUnavailableError
from app.core.tenancy import TenantContext
from app.parsing.registry import ParserRegistry
from app.providers.base import ProviderUnavailableError
from app.providers.rag.base import Embedder, VectorStore, VectorStoreUnavailableError
from app.repositories.collections import CollectionRepository
from app.schemas.knowledge import (
    KnowledgeDeleteResponse,
    KnowledgeDocumentOut,
    KnowledgeListResponse,
    KnowledgeUploadResponse,
)
from app.services import knowledge

router = APIRouter(tags=["knowledge"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
# Admin-gated like every other console surface (models, integrations, members).
# Curating the corpus every answer is grounded on is workspace administration,
# not something one member does to everyone else's results.
AdminDep = Annotated[TenantContext, Depends(require_role("admin"))]

# What a browser sends when it cannot guess a type. Treated as "unknown" so the
# parser registry falls back to sniffing the bytes.
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


def _require_embedder(embedder: Embedder | None) -> Embedder:
    """Fail typed-503 when nothing resolves an embedding model."""
    if embedder is None:
        raise ServiceUnavailableError(
            "No embedding model is configured for this workspace, so the knowledge "
            "base cannot be read or added to. Choose an embedding model in the "
            "workspace defaults."
        )
    return embedder


@router.get("/knowledge/documents", response_model=KnowledgeListResponse)
async def list_knowledge_documents(
    session: SessionDep,
    tenant: AdminDep,
    store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    collection: Annotated[str, Query(min_length=1, max_length=200)] = knowledge.DEFAULT_COLLECTION,
) -> KnowledgeListResponse:
    """List the workspace's stored knowledge, newest ingest first.

    Every source is reported, including any left in a vector space an earlier
    embedding model wrote: those are still stored and still occupy room, and
    hiding them would leave an operator unable to see - let alone remove - the
    thing a model change stranded. Each row says whether the ACTIVE embedder can
    reach it, which is the only sense in which a stored document is currently
    part of what answers are grounded on.

    A workspace that has never ingested anything answers with an empty list
    rather than a 404: an empty knowledge base is a state, not a missing
    resource.

    The embedder is not required here, and that is deliberate - a deployment
    whose embedder configuration is broken is exactly when someone needs to see
    what is stored. Every row then reads unreachable, which is true.
    """
    inventory = await knowledge.list_sources(
        tenant=tenant,
        collection_repo=CollectionRepository(session, tenant),
        store=store,
        collection=collection,
    )
    active_model = embedder.model if embedder is not None else None
    return KnowledgeListResponse(
        items=[
            KnowledgeDocumentOut.from_source(source, active_model=active_model)
            for source in inventory.sources
        ],
        collection=inventory.collection,
        active_embedding_model=active_model,
        truncated=inventory.truncated,
    )


@router.post("/knowledge/documents", response_model=KnowledgeUploadResponse)
async def add_knowledge_document(
    session: SessionDep,
    tenant: AdminDep,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    registry: Annotated[ParserRegistry, Depends(get_parser_registry)],
    limiter: Annotated[StreamLimiter, Depends(get_stream_limiter)],
    file: Annotated[UploadFile, File()],
    collection: Annotated[str, Query(min_length=1, max_length=200)] = knowledge.DEFAULT_COLLECTION,
) -> KnowledgeUploadResponse:
    """Read one uploaded file and store its text in the knowledge base.

    Returns 200, not 201: ingestion is idempotent per document id, so uploading
    the same file twice updates what is stored rather than creating a second
    copy of it, and callers are meant to re-run freely after an interruption.

    Synchronous by design - the response arrives when the text is genuinely
    searchable. See :mod:`app.services.knowledge` for why there is no background
    job: without a table there is nowhere to record a status a client could
    poll, and every failure this file can produce is answered right here
    instead.

    The bytes are parsed and discarded. Nothing stores the original file, so
    what lands in the knowledge base is its text and nothing else.
    """
    active_embedder = _require_embedder(embedder)
    filename = (file.filename or "").strip()
    if not filename:
        raise BadRequestError("The uploaded file has no name.")
    # The exact per-file limit. The multipart body is already parsed by the time
    # a handler runs, so this is not what stops a disk-filling upload -
    # MaxBodySizeMiddleware refuses an over-large body before it is read. This is
    # the precise check on the file part itself; file.size can be None, so the
    # real byte length is re-checked below as the authoritative backstop.
    if file.size is not None and file.size > settings.attachment_max_bytes:
        raise PayloadTooLargeError(
            f"The file is larger than the {settings.attachment_max_bytes} byte limit."
        )
    data = await file.read()
    if not data:
        raise BadRequestError("The uploaded file is empty.")
    if len(data) > settings.attachment_max_bytes:
        raise PayloadTooLargeError(
            f"The file is larger than the {settings.attachment_max_bytes} byte limit."
        )

    try:
        result = await knowledge.add_file(
            data=data,
            filename=filename,
            content_type=file.content_type or _DEFAULT_CONTENT_TYPE,
            tenant=tenant,
            collection_repo=CollectionRepository(session, tenant),
            store=store,
            embedder=active_embedder,
            registry=registry,
            settings=settings,
            collection=collection,
            is_chat_busy=limiter.any_active,
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    except VectorStoreUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    return KnowledgeUploadResponse(
        id=result.id,
        title=result.title,
        documents=result.documents,
        chunks=result.chunks,
        collection=result.collection,
    )


@router.delete("/knowledge/documents", response_model=KnowledgeDeleteResponse)
async def remove_knowledge_document(
    session: SessionDep,
    tenant: AdminDep,
    store: Annotated[VectorStore, Depends(get_vector_store)],
    document_id: Annotated[str, Query(min_length=1, max_length=512)],
    collection: Annotated[str, Query(min_length=1, max_length=200)] = knowledge.DEFAULT_COLLECTION,
) -> KnowledgeDeleteResponse:
    """Remove one source's passages from the knowledge base.

    Nothing in a knowledge base is undeletable - it is a corpus somebody
    curates - so this refuses nothing, and answers with the count instead of a
    bare 204. An id that matches nothing is a successful zero rather than a 404:
    there is no row whose absence could be checked, and "already gone" is an
    honest answer to "remove this". The count is what tells the two apart.

    Copies in every vector space go, not only the one the active embedder
    reads, so a source stranded by an embedding-model change is genuinely
    removable rather than merely invisible.
    """
    try:
        deleted = await knowledge.delete_source(
            source_id=document_id,
            tenant=tenant,
            collection_repo=CollectionRepository(session, tenant),
            store=store,
            collection=collection,
        )
    except VectorStoreUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    return KnowledgeDeleteResponse(id=document_id, chunks_deleted=deleted)
