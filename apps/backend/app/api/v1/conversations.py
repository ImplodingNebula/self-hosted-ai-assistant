"""Conversations endpoints: list, detail, update, delete.

Routers stay HTTP-only: every data access goes through the tenant-scoped
``ConversationRepository``, so nothing here filters by tenant itself and a
foreign-tenant id is indistinguishable from a missing one (404 either way).

A syntactically malformed conversation id also maps to 404, not 400: a value
that cannot be a UUID can never address a resource, and answering exactly as
for an absent id keeps "does this id exist" unanswerable in every case.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_tenant,
    get_job_queue,
    get_model_provider,
    get_session,
    get_vector_store,
)
from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError
from app.core.tenancy import TenantContext
from app.models.entities import ATTACHMENT_ORIGIN_GENERATED, TITLE_ORIGIN_MANUAL, utcnow
from app.providers.base import ModelProvider
from app.providers.rag.base import VectorStore, VectorStoreUnavailableError
from app.repositories import ConversationRepository
from app.repositories.attachments import AttachmentRepository
from app.schemas.conversations import (
    ConversationContextEstimate,
    ConversationDetail,
    ConversationListResponse,
    ConversationSummary,
    ConversationUpdate,
)
from app.services.chat_service import estimate_context_usage
from app.workers.main import JOB_PURGE_ATTACHMENTS
from app.workers.queue import JobQueue, JobQueueError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversations"])

_NOT_FOUND_DETAIL = "Conversation not found."

SessionDep = Annotated[AsyncSession, Depends(get_session)]
TenantDep = Annotated[TenantContext, Depends(get_current_tenant)]


def _parse_conversation_id(raw: str) -> UUID:
    """Parse a path id, treating malformed values as absent resources (404)."""
    try:
        return UUID(raw)
    except ValueError as exc:
        raise NotFoundError(_NOT_FOUND_DETAIL) from exc


@router.post("/conversations", status_code=201)
async def create_conversation(
    response: Response,
    session: SessionDep,
    tenant: TenantDep,
) -> ConversationSummary:
    """Create an empty conversation and return its summary.

    A conversation normally comes into being with its first chat turn. This
    mints one up front so a client can attach a file before that turn exists -
    an upload must name the conversation it belongs to, and there is nothing to
    name until the first message otherwise. The new conversation has no title
    yet; the first message derives one, exactly as an implicitly created
    conversation does.
    """
    repo = ConversationRepository(session, tenant)
    conversation = await repo.create()
    # Commit before responding: the returned id is used immediately (an upload
    # names it) and must survive a follow-up request on another connection.
    await repo.commit()
    response.headers["Location"] = f"/api/v1/conversations/{conversation.id}"
    return ConversationSummary.from_entity(conversation)


@router.get("/conversations")
async def list_conversations(
    session: SessionDep,
    tenant: TenantDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationListResponse:
    """Return one page of the tenant's conversations, pinned first then newest activity.

    Pinned threads lead the page so a pin survives paging: a client that reads
    only the first page still sees every pinned thread, however much newer
    activity sits behind it.
    """
    repo = ConversationRepository(session, tenant)
    rows, total = await repo.list_summaries(limit=limit, offset=offset)
    return ConversationListResponse(
        items=[ConversationSummary.from_entity(c) for c in rows],
        total=total,
    )


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    session: SessionDep,
    tenant: TenantDep,
) -> ConversationDetail:
    """Return one conversation with its messages in chronological order, plus its files.

    The top-level attachments list exists to restore the COMPOSER. A client puts
    back the files no message carries: those were uploaded and ingested but never
    sent, so a reopened conversation shows them again rather than losing a draft
    the server is still holding.

    That is why files the assistant GENERATED are excluded from it. A generated
    file is never the user's unsent draft - nobody chose it, and the client tells
    a draft from a sent file only by "no message carries it", which a generated
    file can satisfy honestly (its turn saved no reply, or is still paused at a
    confirmation). Left in, the assistant's own document would appear as a chip
    in the composer and ride out attached to the user's next question.

    The exclusion is deliberately HERE and not in the repository: the delete path
    below reads the same list to collect blobs to reclaim, and it must see every
    row. Generated files are still reported per MESSAGE, on the reply that made
    them (``meta["attachments"]``), which is what a transcript renders them from.
    """
    parsed_id = _parse_conversation_id(conversation_id)
    repo = ConversationRepository(session, tenant)
    found = await repo.get_with_messages(parsed_id)
    if found is None:
        raise NotFoundError(_NOT_FOUND_DETAIL)
    conversation, messages = found
    stored = await AttachmentRepository(session, tenant).list_for_conversation(parsed_id)
    attachments = [row for row in stored if row.origin != ATTACHMENT_ORIGIN_GENERATED]
    return ConversationDetail.from_entities(conversation, messages, attachments)


@router.get("/conversations/{conversation_id}/context")
async def get_conversation_context(
    conversation_id: str,
    session: SessionDep,
    tenant: TenantDep,
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    model: Annotated[str | None, Query(max_length=200)] = None,
) -> ConversationContextEstimate:
    """Estimate the window's fill for this conversation before any turn runs.

    The eager reading behind the composer's context meter: on pane load or a
    model switch, the client asks what the next prompt already opens with
    instead of waiting for a turn to report it. ``model`` is the composer's
    pin, honoured verbatim; absent, the default resolves through the same
    stored chain a turn would use. Character arithmetic over stored rows -
    no generation, no model load.
    """
    parsed_id = _parse_conversation_id(conversation_id)
    estimate = await estimate_context_usage(
        session,
        tenant,
        parsed_id,
        model=model,
        provider=provider,
        settings=settings,
    )
    if estimate is None:
        raise NotFoundError(_NOT_FOUND_DETAIL)
    return ConversationContextEstimate(
        prompt_tokens=estimate.prompt_tokens,
        num_ctx=estimate.num_ctx,
        # The service always names the tag it measured; "" can only mean a
        # future refactor stopped setting it, and an empty echo is more honest
        # than a crash on the read path.
        model=estimate.model or "",
    )


@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    session: SessionDep,
    tenant: TenantDep,
) -> ConversationSummary:
    """Apply the provided fields; ``settings`` merges per field, never wholesale.

    An update that changes nothing - an empty body, a title already at its
    requested value, a settings patch the stored values already satisfy - does
    not bump ``updated_at``, so a spurious PATCH cannot reorder the sidebar.
    (Re-submitting the same title does still commit when it is the first time
    the name is recorded as a human choice; see below.) A real settings change commits but
    does not bump ``updated_at`` either: settings are metadata, not
    activity, and flipping a composer control must not resurface an old
    thread. Only a title change counts as activity.

    A title written here is stamped as MANUAL, which is what stops the
    background namer - now or after a later edit of the opening question -
    from replacing a name its owner chose.

    ``pinned`` is intent, not a time: the server stamps ``pinned_at`` itself, so
    a client clock can never decide where a row sits in the pinned group.
    Pinning is metadata like settings, so it commits without bumping
    ``updated_at`` - a pin must not resurface an old thread in the recent list
    it just left.

    The row is read FOR UPDATE because the settings merge is a
    read-merge-write: without the lock, two overlapping single-field saves
    could both read the same base and the later commit would silently drop
    the earlier one's axis.
    """
    repo = ConversationRepository(session, tenant)
    conversation = await repo.get_for_update(_parse_conversation_id(conversation_id))
    if conversation is None:
        raise NotFoundError(_NOT_FOUND_DETAIL)

    provided = payload.model_dump(exclude_unset=True)
    commit_needed = False
    if provided.get("title") is not None:
        # A name a person typed, recorded as such: nothing generated may
        # overwrite it afterwards. Stamped whenever a title is SUBMITTED, not
        # only when the string moves - confirming the name a thread already has
        # is still choosing it, and a provenance that skipped that case would let
        # the namer overwrite a title the user had just deliberately kept.
        if conversation.title_origin != TITLE_ORIGIN_MANUAL:
            conversation.title_origin = TITLE_ORIGIN_MANUAL
            commit_needed = True
        if provided["title"] != conversation.title:
            conversation.title = provided["title"]
            # Only a real change counts as activity: re-submitting the same name
            # must not reorder the sidebar.
            conversation.updated_at = utcnow()
            commit_needed = True
    if provided.get("pinned") is not None:
        # Already pinned and pinned again is a no-op, deliberately: restamping
        # would jump the thread to the head of the pinned group for a request
        # that asked for no change.
        already_pinned = conversation.pinned_at is not None
        if provided["pinned"] != already_pinned:
            conversation.pinned_at = utcnow() if provided["pinned"] else None
            commit_needed = True
    if payload.settings is not None:
        merged = payload.settings.merged_into(conversation.settings)
        if merged != conversation.settings:
            # Reassignment is load-bearing: the JSON column has no mutable
            # wrapper, so an in-place merge would never reach the database.
            conversation.settings = merged
            commit_needed = True
    if commit_needed:
        # Commit before responding (read-after-write consistency); a no-op
        # PATCH skips it entirely, releasing the row lock without a write.
        await repo.commit()
    return ConversationSummary.from_entity(conversation)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    session: SessionDep,
    tenant: TenantDep,
    store: Annotated[VectorStore, Depends(get_vector_store)],
    queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> Response:
    """Delete a conversation, its messages, its ephemeral vectors and its files.

    Everything downstream of the row is best-effort privacy hygiene with a
    sweep behind it, because the conversation delete has already committed and
    must not be reported as failed once it has:

    - Session-scoped vectors belong to the conversation, so they go with it
      rather than waiting for the TTL sweep. A store failure only logs, and the
      TTL sweep is genuinely the backstop - for the web pages and other ephemeral
      chunks that carry one.
    - Uploaded files outlive the row on purpose (the FK is ON DELETE SET NULL),
      leaving a row that still knows where the bytes are. The attachment ids are
      read BEFORE the delete - afterwards nothing links them to this
      conversation - and handed to a job that removes the file's indexed text,
      its bytes and its row. If the queue is down, the rows now have no
      conversation, which is exactly what the periodic attachment sweep collects.

      A file's chunks are session-scoped, so the drop above normally takes them
      too - but only normally, and that is why the job removes them again. The
      drop is best-effort (it may have just failed), and an ingest still in flight
      can write its chunks in behind it. Nothing else would catch either case:
      attachment chunks carry NO TTL (they have to outlive the session, or an
      uploaded file would stop grounding two hours after it arrived), so the TTL
      sweep is not a backstop for them, and once the row is gone nothing left in
      the system can even name them.
    """
    parsed_id = _parse_conversation_id(conversation_id)
    repo = ConversationRepository(session, tenant)
    attachments = await AttachmentRepository(session, tenant).list_for_conversation(parsed_id)
    attachment_ids = [str(attachment.id) for attachment in attachments]

    deleted = await repo.delete_with_messages(parsed_id)
    if not deleted:
        raise NotFoundError(_NOT_FOUND_DETAIL)
    # Commit before responding: the 204 must not race a follow-up list read.
    await repo.commit()

    try:
        await store.delete_session(str(tenant.tenant_id), str(parsed_id))
    except VectorStoreUnavailableError:
        logger.warning(
            "Could not purge session vectors for deleted conversation %s; the TTL sweep "
            "removes the ephemeral ones, and the file purge below owns the rest",
            parsed_id,
        )

    if attachment_ids:
        try:
            await queue.enqueue(
                JOB_PURGE_ATTACHMENTS,
                attachment_ids,
                str(tenant.tenant_id),
                str(tenant.user_id),
                job_id=f"attachment-purge:{parsed_id}",
            )
        except JobQueueError:
            logger.warning(
                "Could not queue the file purge for deleted conversation %s; "
                "the attachment sweep will reclaim them",
                parsed_id,
            )
    return Response(status_code=204)
