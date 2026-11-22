"""An uploaded file's lifecycle: accept it, and later remove it for good.

The orchestration behind ``POST /attachments`` and ``DELETE /attachments/{id}``.
The router stays HTTP-only; the interesting invariant in both directions is the
ORDER of the three places a file lives - object storage, the database row, and
the vector store - because a crash between any two of them must leave a state
the system can recover from rather than a lie.

Accepting an upload:

1. Refuse anything the app cannot handle (unreadable type, oversized) BEFORE a
   single byte is written, so a rejected upload leaves nothing behind.
2. Write the bytes, THEN the row. If the write fails, no row promises a blob
   that is not there; if the commit fails, the orphaned blob is never referenced
   by a row and is reclaimed by
   :func:`app.services.attachment_gc.sweep_orphan_blobs`.
3. Enqueue only after the row is committed, because the worker looks the row up
   by id - a job that raced the commit would find nothing.

Terminal content problems that need the bytes to detect (an unreadable type) are
caught here for an immediate, honest 4xx. Everything a parse can still fail on
later (a corrupt or password-protected file) is the background job's business
and lands on the row as ``failed``.

Removing one runs the same reasoning backwards - indexed text, then bytes, then
row - and is stricter about it, because the failure it must never produce is a
file that is invisible to its owner yet still answerable by the assistant. See
:func:`delete_attachment`.
"""

from __future__ import annotations

import hashlib
import logging
from uuid import UUID

from app.core.config import Settings
from app.core.errors import (
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    UnsupportedMediaTypeError,
)
from app.core.tenancy import TenantContext
from app.models import Attachment
from app.parsing.base import UnsupportedFileError
from app.parsing.registry import ParserRegistry
from app.providers.rag.base import VectorStore, VectorStoreUnavailableError
from app.repositories.attachments import AttachmentRepository
from app.repositories.conversations import ConversationRepository
from app.storage.base import StorageBackend, StorageError
from app.workers.main import JOB_PARSE_AND_INGEST, ingest_job_id
from app.workers.queue import JobQueue, JobQueueUnavailableError

logger = logging.getLogger(__name__)


async def delete_attachment(
    *,
    attachment: Attachment,
    tenant: TenantContext,
    attachment_repo: AttachmentRepository,
    storage: StorageBackend,
    vector_store: VectorStore,
) -> None:
    """Delete one attachment: its indexed text, then its bytes, then its row.

    The order is what makes "remove" mean remove. Vectors go FIRST, and a
    failure there fails the whole request: an attachment whose row is gone but
    whose chunks survive is the worst of both worlds - the file is invisible to
    its owner yet still retrievable, so the assistant would go on quoting a
    document the user believes they deleted. Bytes go next (the row is the only
    thing that knows the storage key, so it must outlive them), and the row
    last.

    Every step is idempotent, so a caller that retries after a failure resumes
    rather than double-deleting: purging already-purged vectors matches nothing,
    and deleting a missing object is a no-op. A failure of the row delete itself
    (a database outage) surfaces as a 500, having already made the file
    unretrievable - the same request simply finishes the job when repeated.

    Args:
        attachment: The row to delete; already resolved and tenant-scoped.
        tenant: The acting tenant.
        attachment_repo: Tenant-scoped attachment repository.
        storage: Where the bytes live.
        vector_store: Where the file's chunks live.

    Raises:
        ServiceUnavailableError: The vector store or storage is unreachable. The
            file is left addressable rather than half-deleted.
    """
    # An attachment whose conversation was deleted has a NULL conversation_id;
    # the tenant + record filter still names its chunks exactly.
    session_id = str(attachment.conversation_id) if attachment.conversation_id else None
    record_id = str(attachment.id)
    try:
        await vector_store.delete_record(str(tenant.tenant_id), record_id, session_id=session_id)
    except VectorStoreUnavailableError as exc:
        raise ServiceUnavailableError(
            "Could not remove the file's indexed text, so nothing was deleted. Try again."
        ) from exc

    try:
        await storage.delete(attachment.storage_key)
    except StorageError as exc:
        raise ServiceUnavailableError(
            "Could not remove the stored file. Its text is no longer searchable; try again."
        ) from exc

    await attachment_repo.delete(attachment)
    await attachment_repo.commit()

    # A second purge, AFTER the row is gone. The ingest job may still be running:
    # if it wrote this file's chunks between the purge above and this commit, and
    # checked the row before the commit landed, neither side would have cleaned
    # them up, and they would answer questions forever with no row left to
    # address them by. Once the row is gone the job's own post-ingest check takes
    # over, so between the two the window is closed. Best-effort: the delete has
    # already committed and must not be reported as failed now.
    try:
        orphaned = await vector_store.delete_record(
            str(tenant.tenant_id), record_id, session_id=session_id
        )
        if orphaned:
            logger.info(
                "Purged %d chunk(s) written by an in-flight ingest of deleted attachment %s",
                orphaned,
                record_id,
            )
    except VectorStoreUnavailableError:
        logger.warning(
            "Could not re-check attachment %s for chunks written by an in-flight ingest; "
            "the ingest job purges them when it finds the row gone",
            record_id,
        )


def _storage_key(tenant: TenantContext, conversation_id: UUID, attachment_id: UUID) -> str:
    """Mint the object-storage key for an attachment's bytes.

    Scoped by tenant then conversation so the layout mirrors ownership and a
    conversation's blobs are a single prefix to sweep. Every segment is a UUID,
    which the storage key validator accepts as-is.
    """
    return f"{tenant.tenant_id}/{conversation_id}/{attachment_id}"


async def _enqueue_ingest(queue: JobQueue, attachment: Attachment, tenant: TenantContext) -> None:
    """Queue the parse-and-ingest job for one attachment.

    The job id is derived from the attachment, so a re-enqueue (a resumed upload,
    a client retry) collapses onto the same job rather than parsing twice.

    Raises:
        ServiceUnavailableError: The job queue is unreachable. The bytes and row
            are already safe, so this reports the hand-off failure honestly
            rather than pretending the file was accepted for processing.
    """
    try:
        await queue.enqueue(
            JOB_PARSE_AND_INGEST,
            str(attachment.id),
            str(tenant.tenant_id),
            str(tenant.user_id),
            job_id=ingest_job_id(attachment.id),
        )
    except JobQueueUnavailableError as exc:
        logger.warning("Queued no ingestion job for attachment %s: %s", attachment.id, exc)
        raise ServiceUnavailableError(
            "The file was received but could not be queued for processing. Try again."
        ) from exc


async def create_attachment(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    origin: str,
    conversation_id: UUID,
    tenant: TenantContext,
    attachment_repo: AttachmentRepository,
    conversation_repo: ConversationRepository,
    storage: StorageBackend,
    registry: ParserRegistry,
    queue: JobQueue,
    settings: Settings,
) -> Attachment:
    """Store an uploaded file and queue it for parsing and ingestion.

    Args:
        data: The file's bytes, already read.
        filename: The client's filename (a label; never a path).
        content_type: The declared media type.
        origin: ``"upload"`` or ``"paste"`` (validated by the repository).
        conversation_id: The conversation to attach to; must exist for the tenant.
        tenant: The acting tenant.
        attachment_repo: Tenant-scoped attachment repository.
        conversation_repo: Tenant-scoped conversation repository (existence check).
        storage: Where the bytes are written.
        registry: Chooses a parser; used here only to reject unreadable files early.
        queue: The background job queue.
        settings: Application settings (the size cap).

    Returns:
        The persisted attachment, ``pending`` and enqueued.

    Raises:
        PayloadTooLargeError: The file exceeds ``attachment_max_bytes``.
        UnsupportedMediaTypeError: No parser can read the file.
        NotFoundError: The conversation does not exist for this tenant.
        ServiceUnavailableError: Storage or the job queue is unreachable.
    """
    if len(data) > settings.attachment_max_bytes:
        raise PayloadTooLargeError(
            f"The file is larger than the {settings.attachment_max_bytes} byte limit."
        )

    # Fail an unreadable type now, with the bytes in hand for the sniff, rather
    # than storing it and letting the job report `failed` seconds later.
    try:
        registry.resolve(filename=filename, content_type=content_type, data=data)
    except UnsupportedFileError as exc:
        raise UnsupportedMediaTypeError(str(exc)) from exc

    if await conversation_repo.get(conversation_id) is None:
        # A foreign or absent conversation is indistinguishable, by design.
        raise NotFoundError("The requested conversation does not exist.")

    content_hash = hashlib.sha256(data).hexdigest()
    existing = await attachment_repo.get_by_content_hash(
        content_hash, conversation_id=conversation_id
    )
    if existing is not None:
        # The same bytes are already attached to this conversation, in a row that
        # is still worth something. Reuse it instead of storing a second copy:
        # this dedupes a double-submitted upload, and lets a client that lost the
        # response to a queue outage resume the same attachment (re-enqueuing is
        # idempotent on the job id). A FAILED row is not a match at all - the
        # repository will not return one - so re-uploading a file that failed
        # gets a genuine second attempt rather than the corpse of the first.
        await _enqueue_ingest(queue, existing, tenant)
        return existing

    attachment = Attachment(
        tenant_id=tenant.tenant_id,
        user_id=tenant.user_id,
        conversation_id=conversation_id,
        filename=filename,
        content_type=content_type,
        size_bytes=len(data),
        storage_key="",  # set once the id is known, just below
        content_hash=content_hash,
        origin=origin,
    )
    attachment.storage_key = _storage_key(tenant, conversation_id, attachment.id)

    try:
        await storage.put(attachment.storage_key, data, content_type=content_type)
    except StorageError as exc:
        raise ServiceUnavailableError("Could not store the uploaded file.") from exc

    # Row after bytes: a committed row always points at a blob that exists.
    attachment_repo.add(attachment)
    await attachment_repo.commit()

    await _enqueue_ingest(queue, attachment, tenant)
    return attachment
