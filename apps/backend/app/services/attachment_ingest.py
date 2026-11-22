"""Turn a stored attachment into retrievable, citable chunks.

This is the whole attachment pipeline in one function: read the bytes from the
storage seam, hand them to whichever parser the registry picks, and feed the
extracted text into ``ingest_structured`` - the same entrypoint that serves
direct API records, MCP results and web pages. Attachments never grow a second
ingestion path.

Nothing here knows about arq or FastAPI. The job in ``app/workers/main.py`` is
a thin adapter; the tests drive this function directly.

**Failure semantics.** Two kinds of failure are told apart, because they call
for opposite handling:

* *Terminal* - the file itself is the problem (unreadable, unsupported, empty,
  its blob or conversation is gone). Retrying accomplishes nothing. The row is
  marked ``failed`` with a reason worth showing a person, and the function
  returns normally so the job does not retry.
* *Transient* - something the file depends on is down (storage, the embedding
  backend, the vector store), or the machine ran out of memory while reading it.
  These propagate untouched so arq retries with backoff. The row stays in
  ``parsing``, which is exactly what ``updated_at`` makes visible: a row that has
  sat in ``parsing`` far longer than a parse takes is a job that died, not a new
  upload.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.models import (
    ATTACHMENT_EXPIRED,
    ATTACHMENT_FAILED,
    ATTACHMENT_ORIGIN_PASTE,
    ATTACHMENT_PARSING,
    ATTACHMENT_READY,
    Attachment,
)
from app.parsing.base import ParsedDocument, ParsingError
from app.parsing.image import ImageParser
from app.parsing.registry import ParserRegistry, normalize_media_type
from app.providers.rag.base import Embedder, VectorStore
from app.repositories.attachments import AttachmentRepository
from app.schemas.rag import RagIngestItem
from app.services.rag_service import ingest_structured
from app.storage.base import ObjectNotFoundError, StorageBackend

logger = logging.getLogger(__name__)

# One ingest item's text cap. MUST equal RagIngestItem.text's max_length - a
# parsed unit longer than this is split across several items rather than
# truncated, because silently dropping the tail of a document is the kind of
# bug nobody notices until an answer is confidently incomplete. Pinned by
# test_split_cap_matches_the_ingest_schema.
MAX_ITEM_CHARS = 50_000

# When splitting, back up this far looking for whitespace so a split lands
# between words rather than through one.
_SPLIT_BACKTRACK_CHARS = 500

# RagIngestItem.title's max_length.
_MAX_TITLE_CHARS = 500

# A stored `error` is shown to a person, not parsed; a parser that returns an
# essay gets cut off rather than filling the column.
_MAX_ERROR_CHARS = 500

# What a citation calls a pasted block. A paste is an ordinary attachment on the
# identical pipeline - only its label differs, because no one chose a filename.
PASTED_DISPLAY_NAME = "Pasted content"

_NO_TEXT_MESSAGE = "No readable text could be extracted from this file."
_BLOB_GONE_MESSAGE = "The uploaded file is no longer available."
_CONVERSATION_GONE_MESSAGE = "The conversation this file belonged to was deleted."


# Not a row status - by the time it is reported there is no row. It says the
# attachment was deleted while this job was ingesting it, so the job undid its
# own write instead of leaving orphaned chunks behind.
_DELETED_MID_INGEST = "deleted"


@dataclass(frozen=True)
class AttachmentIngestResult:
    """The outcome of one ingestion attempt.

    Attributes:
        status: The lifecycle status the row now holds (``ready``, ``failed``, or
            ``expired`` for a row the retention sweep emptied before this job
            ran), or ``deleted`` when the row was removed mid-ingest and there is
            no longer a row to hold one.
        documents: How many units of the file were ingested.
        chunks: How many chunks were written to the vector store.
        error: Why it failed, when it failed.
    """

    status: str
    documents: int = 0
    chunks: int = 0
    error: str | None = None


def display_name(attachment: Attachment) -> str:
    """Return the name a citation should show for this attachment."""
    if attachment.origin == ATTACHMENT_ORIGIN_PASTE:
        return PASTED_DISPLAY_NAME
    return attachment.filename


def _doc_id(attachment: Attachment, ordinal: int, part: int) -> str:
    """Derive the stable RAG document id of one part of one parsed unit.

    Stability is what makes a retry idempotent: ``ingest_structured`` deletes a
    document's existing chunks before writing its new ones, so re-running this
    job rewrites the same documents rather than accumulating duplicates.
    """
    return f"attachment:{attachment.id}:{ordinal}:{part}"


def _title(name: str, unit_title: str | None, part: int, parts: int) -> str:
    """Compose the human label a citation shows for one ingested document."""
    label = f"{name} - {unit_title}" if unit_title else name
    if parts > 1:
        label = f"{label} (part {part + 1})"
    return label[:_MAX_TITLE_CHARS]


def split_text(text: str, limit: int = MAX_ITEM_CHARS) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` characters.

    Splits land on whitespace when any is within reach of the cap, so a word is
    not cut in half at the seam. Text with no whitespace at all (a minified
    bundle, a base64 blob) is cut at the cap - a hard limit has to hold.

    Content is preserved: concatenating the pieces reproduces the input, with the
    one exception of an all-whitespace piece, which is dropped (nothing worth
    embedding is lost). A caller reconstructing the original must not rely on the
    join for such input.
    """
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + limit
        if end >= len(text):
            pieces.append(text[start:])
            break
        # Never search back past `start`, or a limit smaller than the backtrack
        # window could produce an empty piece and loop forever.
        window_start = max(start + 1, end - _SPLIT_BACKTRACK_CHARS)
        window = text[window_start:end]
        offset = max(window.rfind("\n"), window.rfind(" "))
        if offset != -1:
            end = window_start + offset + 1
        pieces.append(text[start:end])
        start = end
    return [piece for piece in pieces if piece.strip()]


def build_ingest_items(
    attachment: Attachment, documents: list[ParsedDocument]
) -> list[RagIngestItem]:
    """Turn parsed units into ingest items, splitting any that exceed the cap.

    Units holding no text are dropped rather than embedded: a blank PDF page or
    an empty spreadsheet sheet is not a passage anyone should retrieve.
    """
    name = display_name(attachment)
    items: list[RagIngestItem] = []
    for document in documents:
        if not document.text.strip():
            continue
        pieces = split_text(document.text)
        for part, piece in enumerate(pieces):
            items.append(
                RagIngestItem(
                    id=_doc_id(attachment, document.ordinal, part),
                    title=_title(name, document.title, part, len(pieces)),
                    text=piece,
                    # Every piece of this file carries the file's id, so a chat
                    # turn that names the attachment can pull back all of them -
                    # a document id alone cannot, because how many documents a
                    # file became is only known here.
                    record_id=str(attachment.id),
                )
            )
    return items


async def ingest_attachment(
    *,
    attachment: Attachment,
    repo: AttachmentRepository,
    storage: StorageBackend,
    registry: ParserRegistry,
    embedder: Embedder,
    store: VectorStore,
    settings: Settings,
) -> AttachmentIngestResult:
    """Parse ``attachment``'s stored bytes and ingest them into its conversation.

    The attachment's text lands in the ephemeral session scope keyed by its
    conversation id - the same scope web results use, but persisted (no TTL), so
    it lives as long as the conversation and is reclaimed only when the
    conversation is deleted. A file's bytes already outlive any TTL; its
    searchable text has to as well, or the file would silently stop grounding
    answers two hours after upload while the file itself remained.

    Each lifecycle transition is committed as it happens. The row is a progress
    indicator a client polls, and a status that only becomes visible once the
    job ends indicates nothing.

    Args:
        attachment: The row to ingest, loaded through ``repo`` (so already
            scoped to the acting tenant).
        repo: Tenant-scoped attachment repository; also supplies the tenant.
        storage: Where the bytes live.
        registry: Chooses the parser for this file.
        embedder: Defines the vector space.
        store: The vector store.
        settings: Chunking, batching and TTL knobs.

    Returns:
        The outcome, including a terminal failure (which is recorded on the row,
        not raised - retrying an unreadable file only wastes a worker).

    Raises:
        StorageUnavailableError: Object storage is unreachable.
        ProviderUnavailableError: The embedding backend is unreachable.
        VectorStoreUnavailableError: The vector store is unreachable.
        ParserOutOfMemoryError: The machine ran out of memory reading the file.
            Transient like the three above, and deliberately not caught with the
            parsing errors below: it is not a verdict on the file.
    """
    if attachment.status == ATTACHMENT_READY:
        # A retry of a job that already succeeded: the worker died between the
        # write and the acknowledgement. Embedding is the expensive half of
        # ingestion, so do not pay for it twice.
        logger.info("Attachment %s is already ingested; nothing to do", attachment.id)
        return AttachmentIngestResult(status=ATTACHMENT_READY)

    if attachment.status == ATTACHMENT_EXPIRED:
        # The retention sweep emptied this row. Left to fall through, the job
        # would stamp it `parsing`, find nothing in the store, and settle it
        # `failed` with "no longer available" - replacing an honest "this aged
        # out" with a fault that never happened, on the one row that exists
        # precisely to say the file was here. Reachability is low (the dedupe
        # check no longer hands an expired row back to an upload, so a job would
        # have to outlive the whole retention window) but the state is real now.
        logger.info("Attachment %s expired before this job ran; leaving it", attachment.id)
        return AttachmentIngestResult(status=ATTACHMENT_EXPIRED)

    if attachment.conversation_id is None:
        # The conversation was deleted while the file waited in the queue (the
        # FK is ON DELETE SET NULL). There is no session scope left to ingest
        # into, and nobody left to read an answer grounded on it.
        return await _fail(repo, attachment, _CONVERSATION_GONE_MESSAGE)

    await repo.set_status(attachment, ATTACHMENT_PARSING)
    await repo.commit()

    try:
        data = await storage.get(attachment.storage_key)
    except ObjectNotFoundError:
        # Terminal: the blob is gone. Any OTHER storage error (unavailable) is
        # transient and propagates, so the worker retries. A bad-key error cannot
        # reach here - the app minted this key - so it is not caught as terminal.
        return await _fail(repo, attachment, _BLOB_GONE_MESSAGE)

    if attachment.content_hash is None:
        # Hashing here rather than at upload keeps the endpoint's job to storing
        # bytes. The value is persisted by the next status write.
        attachment.content_hash = hashlib.sha256(data).hexdigest()

    try:
        parser = registry.resolve(
            filename=attachment.filename, content_type=attachment.content_type, data=data
        )
        documents = await parser.parse(
            data,
            filename=attachment.filename,
            media_type=normalize_media_type(attachment.content_type),
        )
    except ParsingError as exc:
        return await _fail(repo, attachment, str(exc))

    items = build_ingest_items(attachment, documents)
    if not items:
        if isinstance(parser, ImageParser):
            # An image is the one file whose value is not only its text: a vision
            # model can still see it. So an image that OCR found no text in is
            # ACCEPTED, not failed - stored and ready, contributing nothing to
            # retrieval but available to be looked at. (No chunks were written, so
            # there is nothing for the delete-mid-ingest guard below to reclaim.)
            await repo.set_status(attachment, ATTACHMENT_READY)
            await repo.commit()
            logger.info("Ingested image attachment %s with no readable text", attachment.id)
            return AttachmentIngestResult(status=ATTACHMENT_READY)
        # Parsing succeeded and produced nothing. Marking this `ready` would be
        # a quiet lie: the file would look answerable and ground no answer.
        return await _fail(repo, attachment, _NO_TEXT_MESSAGE)

    # No is_chat_busy signal: that flag lives in the API process and this runs in
    # the worker, so there is nothing here to ask. It costs nothing on the dev
    # deployment where EMBEDDING_DEVICE pins the embedder to the CPU, where a
    # retrieval cannot evict the chat model at all. Where it embeds on the GPU,
    # a large upload can slow a concurrent chat; sharing the signal would take a
    # Redis flag, and it is not worth one until a deployment actually feels it.
    session_id = str(attachment.conversation_id)
    ingested, chunks = await ingest_structured(
        source_type="upload",
        tenant=repo.tenant,
        items=items,
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session_id,
        # A file lives with its conversation, not on the web-results TTL.
        persist=True,
    )

    # Parsing and embedding take seconds, and the owner can delete the file
    # during them. The delete purged whatever chunks existed AT THAT MOMENT -
    # which was nothing - and the write above then landed behind it. Left alone
    # those chunks are unreachable garbage that still answers questions: the row
    # was the only thing that could address them, attachment chunks carry no TTL,
    # and nothing but a conversation delete would ever sweep them. So the job
    # that wrote them is the one that has to take them back.
    # Read the id BEFORE the rollback below: rolling back expires the instance,
    # and every attribute read after that would try to reload a row that is gone.
    attachment_id = attachment.id
    if not await repo.still_exists(attachment_id):
        # Drop the pending edits to a row that no longer exists (the content hash
        # set above), or the next flush would raise instead of finishing here.
        await repo.rollback()
        purged = await store.delete_record(
            str(repo.tenant.tenant_id), str(attachment_id), session_id=session_id
        )
        logger.info(
            "Attachment %s was deleted while it was being ingested; purged %d chunk(s)",
            attachment_id,
            purged,
        )
        return AttachmentIngestResult(status=_DELETED_MID_INGEST)

    await repo.set_status(attachment, ATTACHMENT_READY)
    await repo.commit()
    logger.info("Ingested attachment %s: %d documents, %d chunks", attachment.id, ingested, chunks)
    return AttachmentIngestResult(status=ATTACHMENT_READY, documents=ingested, chunks=chunks)


async def _fail(
    repo: AttachmentRepository, attachment: Attachment, reason: str
) -> AttachmentIngestResult:
    """Record a terminal failure on the row and return it as the outcome."""
    message = reason.strip()[:_MAX_ERROR_CHARS] or _NO_TEXT_MESSAGE
    await repo.set_status(attachment, ATTACHMENT_FAILED, error=message)
    await repo.commit()
    logger.info("Attachment %s failed to ingest: %s", attachment.id, message)
    return AttachmentIngestResult(status=ATTACHMENT_FAILED, error=message)
