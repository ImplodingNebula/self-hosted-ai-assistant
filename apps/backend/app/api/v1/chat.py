"""Chat endpoint: streams a model response as Server-Sent Events."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import (
    get_current_tenant,
    get_embedder,
    get_job_queue,
    get_mcp_sources,
    get_model_provider,
    get_page_fetcher,
    get_reranker,
    get_search_provider,
    get_sessionmaker,
    get_storage_backend,
    get_stream_limiter,
    get_tool_vector_cache,
    get_vector_store,
)
from app.core.concurrency import StreamLimiter
from app.core.config import Settings, get_settings
from app.core.errors import TooManyRequestsError
from app.core.tenancy import TenantContext
from app.providers.base import ChatChunk, ModelProvider
from app.providers.rag.base import Embedder, Reranker, VectorStore
from app.providers.search.base import SearchProvider
from app.providers.search.fetcher import PageFetcher
from app.providers.tools.base import ToolSource
from app.schemas.chat import (
    ChatRequest,
    CitationEvent,
    ContextUsageEvent,
    DoneEvent,
    ErrorEvent,
    FileEvent,
    MetaEvent,
    StatusEvent,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolConfirmationEvent,
    ToolResultEvent,
    TurnUsageEvent,
)
from app.services import chat_service
from app.services.tools.selection import ToolVectorCache
from app.sse import sse_event, with_heartbeat
from app.storage.base import StorageBackend
from app.workers.queue import JobQueue

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


async def hold_stream_slot(
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    limiter: Annotated[StreamLimiter, Depends(get_stream_limiter)],
) -> AsyncIterator[None]:
    """Hold this user's stream slot for the request's full lifetime.

    A yield dependency releases in the request exit stack, which unwinds only
    after the response body finishes or aborts - the one hook that covers a
    client disconnecting before the response even starts. Releasing inside the
    SSE generator instead would leak the slot on such disconnects: the body is
    never iterated, and a never-started generator skips its ``finally``
    forever (only garbage collection would ever reclaim it).

    Raises:
        TooManyRequestsError: This user is already at their concurrent-stream
            cap; reaches the client as a problem+json 429 before any stream.
    """
    if not await limiter.acquire(tenant.tenant_id, tenant.user_id):
        raise TooManyRequestsError(
            "This account already has as many chat streams running at once as this "
            "deployment allows; retry in a moment."
        )
    try:
        yield
    finally:
        await limiter.release(tenant.tenant_id, tenant.user_id)


async def _stream(chunks: AsyncIterator[ChatChunk]) -> AsyncIterator[str]:
    """Map the chat service's ChatChunks onto SSE frames; no orchestration here.

    The finally closes the service generator on any early exit, so its cleanup
    (partial-reply persistence) runs promptly on a mid-stream client
    disconnect instead of waiting for garbage collection.
    """
    try:
        async for chunk in chunks:
            if chunk.type == "meta" and chunk.conversation_id:
                yield sse_event(
                    "meta",
                    MetaEvent(
                        conversation_id=chunk.conversation_id,
                        user_message_id=chunk.user_message_id,
                    ),
                )
            elif chunk.type == "token" and chunk.delta:
                yield sse_event("token", TokenEvent(delta=chunk.delta))
            elif chunk.type == "thinking" and chunk.delta:
                yield sse_event("thinking", ThinkingEvent(delta=chunk.delta))
            elif chunk.type == "status" and chunk.delta:
                yield sse_event(
                    "status",
                    StatusEvent(message=chunk.delta, code=chunk.status_code or "working"),
                )
            elif chunk.type == "citation" and chunk.citation is not None:
                citation = chunk.citation
                yield sse_event(
                    "citation",
                    CitationEvent(
                        id=citation.id,
                        source=citation.source,
                        snippet=citation.snippet,
                        score=citation.score,
                        index=citation.index,
                        title=citation.title,
                        doc_id=citation.doc_id,
                        source_type=citation.source_type,
                        url=citation.url,
                    ),
                )
            elif chunk.type == "tool_call" and chunk.tool_call is not None:
                call = chunk.tool_call
                yield sse_event(
                    "tool_call",
                    ToolCallEvent(id=call.id, name=call.name, arguments=call.arguments),
                )
            elif chunk.type == "tool_result" and chunk.tool_result is not None:
                tool_result = chunk.tool_result
                yield sse_event(
                    "tool_result",
                    ToolResultEvent(
                        id=tool_result.id,
                        name=tool_result.name,
                        result=tool_result.content,
                        is_error=tool_result.is_error,
                        denied=tool_result.denied,
                        count=tool_result.count,
                    ),
                )
            elif chunk.type == "file" and chunk.file is not None:
                file = chunk.file
                yield sse_event(
                    "file",
                    FileEvent(
                        attachment_id=file.attachment_id,
                        filename=file.filename,
                        content_type=file.content_type,
                        size_bytes=file.size_bytes,
                        origin=file.origin,
                    ),
                )
            elif chunk.type == "tool_confirmation" and chunk.confirmation is not None:
                confirmation = chunk.confirmation
                yield sse_event(
                    "tool_confirmation",
                    ToolConfirmationEvent(
                        id=confirmation.id,
                        conversation_id=chunk.conversation_id or "",
                        calls=[
                            ToolCallEvent(id=call.id, name=call.name, arguments=call.arguments)
                            for call in confirmation.calls
                        ],
                    ),
                )
                # Terminal for this turn: the turn is suspended, not finished, so
                # no done follows. The generator stops after this chunk; return
                # keeps a stray later frame from slipping onto a closed turn.
                return
            elif chunk.type == "error":
                yield sse_event(
                    "error",
                    ErrorEvent(
                        message=chunk.error or "unknown error",
                        code=chunk.error_code or "internal_error",
                    ),
                )
                return
            elif chunk.type == "done":
                yield sse_event(
                    "done",
                    DoneEvent(
                        # An empty generation persists nothing; a synthetic id
                        # (canonical dashed form, like every persisted id) keeps
                        # the wire contract (message_id always present).
                        message_id=chunk.message_id or str(uuid.uuid4()),
                        conversation_id=chunk.conversation_id,
                        usage=chunk.usage,
                        content=chunk.content,
                        context_usage=(
                            ContextUsageEvent(
                                prompt_tokens=chunk.context_usage.prompt_tokens,
                                num_ctx=chunk.context_usage.num_ctx,
                                model=chunk.context_usage.model,
                            )
                            if chunk.context_usage is not None
                            else None
                        ),
                        turn_usage=(
                            TurnUsageEvent(
                                completion_tokens=chunk.turn_usage.completion_tokens,
                                generation_ms=chunk.turn_usage.generation_ms,
                            )
                            if chunk.turn_usage is not None
                            else None
                        ),
                    ),
                )
    except Exception:
        # Log the real failure server-side; never leak exception text to clients.
        logger.exception("chat stream failed")
        yield sse_event(
            "error",
            ErrorEvent(message="The request failed unexpectedly.", code="internal_error"),
        )
    finally:
        aclose = getattr(chunks, "aclose", None)
        if aclose is not None:
            await aclose()


@router.post("/chat/stream", dependencies=[Depends(hold_stream_slot)])
async def chat_stream(
    req: ChatRequest,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    reranker: Annotated[Reranker | None, Depends(get_reranker)],
    search_provider: Annotated[SearchProvider | None, Depends(get_search_provider)],
    page_fetcher: Annotated[PageFetcher | None, Depends(get_page_fetcher)],
    mcp_sources: Annotated[list[ToolSource], Depends(get_mcp_sources)],
    tool_vector_cache: Annotated[ToolVectorCache, Depends(get_tool_vector_cache)],
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
    queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> StreamingResponse:
    """Stream a chat completion as SSE (meta, token, thinking, status, citation, error, done).

    The stream slot is claimed (or a 429 raised) by :func:`hold_stream_slot`
    before any work and released when the request unwinds, on every outcome.
    Conversation resolution is awaited before the response starts, so lookup
    (400/404) failures reach clients as problem+json instead of a broken event
    stream. Failures after streaming begins stay in-band SSE error events -
    except retrieval failures, which degrade to an ungrounded answer.
    """
    chunks = await chat_service.stream_chat(
        req,
        provider,
        settings,
        sessionmaker,
        tenant,
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        search_provider=search_provider,
        page_fetcher=page_fetcher,
        mcp_sources=mcp_sources,
        tool_vector_cache=tool_vector_cache,
        storage=storage,
        queue=queue,
    )
    return StreamingResponse(
        with_heartbeat(_stream(chunks), settings.sse_heartbeat_seconds),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
