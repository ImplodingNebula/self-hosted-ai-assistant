"""RAG endpoints: structured-record ingestion and similarity queries."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_tenant,
    get_embedder,
    get_reranker,
    get_session,
    get_stream_limiter,
    get_vector_store,
)
from app.core.concurrency import StreamLimiter
from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError, ServiceUnavailableError
from app.core.tenancy import TenantContext
from app.providers.base import ProviderUnavailableError
from app.providers.rag.base import Embedder, Reranker, VectorStore, VectorStoreUnavailableError
from app.repositories.collections import CollectionRepository
from app.repositories.conversations import ConversationRepository
from app.schemas.rag import (
    RagIngestRequest,
    RagIngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    RagQueryResult,
)
from app.services import rag_service
from app.services.embedding_resolution import UNCONFIGURED_DETAIL
from app.services.workspace_defaults import apply_reranker_default, load_defaults

router = APIRouter(tags=["rag"])


def _require_embedder(embedder: Embedder | None) -> Embedder:
    """Fail typed-503 when nothing resolves an embedder for this workspace."""
    if embedder is None:
        raise ServiceUnavailableError(UNCONFIGURED_DETAIL)
    return embedder


async def _require_own_session(
    session_id: str | None, session: AsyncSession, tenant: TenantContext
) -> str | None:
    """Resolve a session id as one of the caller's OWN conversations.

    The session scope is a conversation's private scratch: uploaded file text,
    fetched pages, and records a tool returned under that user's credential. The
    id is the conversation id, which is not a secret - it sits in the URL a
    client shows - so validating it as a UUID (all the schema can do) is not a
    permission check. Resolving it through the user-scoped
    :class:`ConversationRepository` is: an id belonging to someone else is simply
    not found, so neither reading that scope nor planting chunks into it (which
    the owner's next turn would retrieve and cite as her own material) is
    expressible.

    A no-op today with one seeded user, and the whole boundary the moment login
    lands. Answers exactly as the conversations router does, so an id that exists
    for another principal is indistinguishable from one that never existed.
    """
    if session_id is None:
        return None
    # The schema already rejected non-UUIDs and canonicalized the spelling.
    conversation = await ConversationRepository(session, tenant).get(UUID(session_id))
    if conversation is None:
        raise NotFoundError("Conversation not found.")
    return session_id


@router.post("/rag/ingest", response_model=RagIngestResponse)
async def rag_ingest(
    req: RagIngestRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    limiter: Annotated[StreamLimiter, Depends(get_stream_limiter)],
) -> RagIngestResponse:
    """Ingest structured records: chunk, embed, and store them for retrieval.

    Synchronous by design for structured records (file uploads get a
    background pipeline in a later phase): a maximum-size request (100 items
    of 50k characters) embeds a few thousand chunks and can take minutes on
    a CPU-pinned embedder. Ingestion is idempotent per document id, so a
    timed-out or aborted call heals by simply re-running it.

    Returns 200 (not 201) because re-ingesting existing documents is an
    update, not a creation, and callers are encouraged to re-run freely.

    A ``session_id`` must name one of the caller's own conversations (404
    otherwise) - see :func:`_require_own_session`. Without that, an id lifted
    from a URL would let anything be planted in another user's session scope,
    where their next turn retrieves and cites it as their own stored material.
    """
    active_embedder = _require_embedder(embedder)
    session_id = await _require_own_session(req.session_id, session, tenant)
    collection_id: str | None = None
    if req.session_id is None:
        repo = CollectionRepository(session, tenant)
        collection = await repo.get_or_create(req.collection or "default", user_id=tenant.user_id)
        # Commit before the (potentially long) embedding work so the source
        # row is durable even if ingestion is interrupted mid-way.
        await repo.commit()
        collection_id = str(collection.id)
    try:
        documents, chunks = await rag_service.ingest_structured(
            source_type=req.source_type,
            tenant=tenant,
            items=req.items,
            embedder=active_embedder,
            store=store,
            settings=settings,
            session_id=session_id,
            collection_id=collection_id,
            is_chat_busy=limiter.any_active,
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    except VectorStoreUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    return RagIngestResponse(collection_id=collection_id, documents=documents, chunks=chunks)


@router.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(
    req: RagQueryRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    reranker: Annotated[Reranker | None, Depends(get_reranker)],
) -> RagQueryResponse:
    """Similarity-search stored chunks (best hits first).

    Searches the tenant's whole knowledge base unless ``collection`` narrows
    it to one named corpus; ``session_id`` additionally searches that
    session's ephemeral chunks. Scores are cosine similarities from the dense
    search (or reranker relevance where a reranker is wired).

    A ``session_id`` must name one of the caller's own conversations (404
    otherwise) - see :func:`_require_own_session`. That scope holds uploaded
    file text, fetched pages and records fetched under a per-user credential,
    so reading it is a permission question, not a validation one.
    """
    active_embedder = _require_embedder(embedder)
    session_id = await _require_own_session(req.session_id, session, tenant)
    collection_id: str | None = None
    if req.collection is not None:
        repo = CollectionRepository(session, tenant)
        collection = await repo.get_by_name(req.collection)
        if collection is None:
            raise NotFoundError(f"No collection named '{req.collection}' exists.")
        collection_id = str(collection.id)
    try:
        hits = await rag_service.retrieve(
            query=req.query,
            tenant=tenant,
            embedder=active_embedder,
            store=store,
            settings=settings,
            # The workspace's reranker toggle, applied on the session this
            # handler already holds. A search endpoint that re-scored what a
            # chat turn would not have would rank the same corpus two ways for
            # one workspace, which is the disagreement this clamp exists to
            # prevent - not an optimization.
            reranker=apply_reranker_default(reranker, await load_defaults(session, tenant)),
            session_id=session_id,
            collection_id=collection_id,
            top_k=req.top_k,
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    except VectorStoreUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    return RagQueryResponse(
        results=[
            RagQueryResult(
                id=hit.id,
                doc_id=hit.doc_id,
                source_type=hit.source_type,
                title=hit.title,
                text=hit.text,
                score=hit.score,
                chunk_index=hit.chunk_index,
            )
            for hit in hits
        ]
    )
