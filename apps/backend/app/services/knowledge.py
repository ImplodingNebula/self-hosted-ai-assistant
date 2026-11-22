"""The workspace knowledge base as an operator manages it: list, add, remove.

Composition, not new machinery. Adding a file here runs the SAME two steps an
uploaded attachment runs - the parser registry reads the bytes, and
``ingest_structured`` chunks, embeds and stores the text - with one difference
that is the whole point of this module: the destination is the tenant's
PERSISTENT collection rather than a conversation's ephemeral scratch. Nothing
about chunking, embedding, the unchanged-content skip or the delete-then-upsert
per document is re-implemented here; only the destination changes.

**There is no table behind any of this**, and it shapes every function below.
``ingest_structured`` writes chunks into the vector store and records nothing
anywhere else, so the knowledge base's inventory is assembled by reading the
stored chunks back (:meth:`~app.providers.rag.base.VectorStore.list_kb_sources`)
rather than by querying rows. Two consequences worth stating out loud:

* **Adding a file is synchronous.** A background job would need somewhere to
  record "parsing" and "failed" for a client to poll, and there is no such
  place. The direct ingest API (``POST /rag/ingest``) is synchronous for the
  same reason and documents the same trade: ingestion is idempotent per
  document, so a request that times out heals by simply being repeated. Every
  failure a file can produce - unreadable type, corrupt bytes, no extractable
  text - is therefore answered in the response, rather than landing minutes
  later on a row nobody is watching.
* **The original bytes are not kept.** They are parsed and discarded, so there
  is nothing to re-ingest FROM: re-running the pipeline against a stored file is
  a capability that needs somewhere to put the file first.

Tenancy: the knowledge base is a WORKSPACE resource. Every call here is scoped
to the acting tenant and deliberately NOT to the acting user - retrieval
searches the kb with no user filter for exactly that reason, and narrowing the
management view to whoever happened to ingest a document would present a shared
corpus as a private one.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import UnprocessableEntityError, UnsupportedMediaTypeError
from app.core.tenancy import TenantContext
from app.parsing.base import ParsedDocument, ParsingError, UnsupportedFileError
from app.parsing.registry import ParserRegistry, normalize_media_type
from app.providers.rag.base import Embedder, StoredSource, VectorStore
from app.repositories.collections import CollectionRepository
from app.schemas.rag import RagIngestItem
from app.services.attachment_ingest import MAX_ITEM_CHARS, split_text
from app.services.rag_service import ingest_structured

logger = logging.getLogger(__name__)

# The corpus a workspace gets without asking for one. The same name the direct
# ingest API defaults to, so a document added through either path lands in the
# same place and the console lists what that API stored.
DEFAULT_COLLECTION = "default"

# The most sources one listing returns. Generous enough that no hand-curated
# corpus meets it before the console grows paging, and it trims the ANSWER only -
# the store's own ceiling is what bounds the work.
MAX_LISTED_SOURCES = 500

# RagIngestItem.title's max_length.
_MAX_TITLE_CHARS = 500

_NO_TEXT_MESSAGE = (
    "No readable text could be extracted from this file, so there is nothing to "
    "search. Only files the platform can read into text belong in the knowledge base."
)


@dataclass(frozen=True)
class KnowledgeInventory:
    """What one listing found.

    Attributes:
        sources: The stored sources, newest ingest first.
        collection: The corpus they were read from.
        truncated: The store held more than the listing returned.
    """

    sources: list[StoredSource]
    collection: str
    truncated: bool = False


@dataclass(frozen=True)
class UploadResult:
    """What adding one file to the knowledge base produced.

    Attributes:
        id: The stored source's id - what the list and delete calls address.
        title: The label its first document carries.
        documents: How many RAG documents the file became.
        chunks: How many passages were embedded and stored. Zero beside a
            nonzero ``documents`` is the unchanged-content skip: the file was
            already stored, byte for byte, and re-embedding it was skipped.
        collection: The corpus it landed in.
    """

    id: str
    title: str
    documents: int
    chunks: int
    collection: str


def _source_id() -> str:
    """Mint the id that names one uploaded file as a whole.

    Prefixed rather than a bare UUID so an id read off the API, or out of a
    chunk payload, says what it is. The shape mirrors an attachment's
    ``attachment:<id>`` for the same reason: one source, many documents.
    """
    return f"knowledge:{uuid.uuid4()}"


def _document_id(source_id: str, ordinal: int, part: int) -> str:
    """Derive the stable RAG document id of one part of one parsed unit.

    Stability is what makes a repeated upload of the same file idempotent:
    ``ingest_structured`` deletes a document's chunks before writing its new
    ones, so re-running against the same ids rewrites rather than accumulates.
    """
    return f"{source_id}:{ordinal}:{part}"


def _document_title(name: str, unit_title: str | None, part: int, parts: int) -> str:
    """Compose the label a citation shows for one ingested document."""
    label = f"{name} - {unit_title}" if unit_title else name
    if parts > 1:
        label = f"{label} (part {part + 1})"
    return label[:_MAX_TITLE_CHARS]


def build_ingest_items(
    source_id: str, filename: str, documents: list[ParsedDocument]
) -> list[RagIngestItem]:
    """Turn one file's parsed units into ingest items, splitting oversized ones.

    Units holding no text are dropped rather than embedded - a blank PDF page is
    not a passage anyone should retrieve - and a unit longer than the ingest
    schema's cap is SPLIT rather than truncated, because silently losing the tail
    of a document is the kind of bug nobody notices until an answer is
    confidently incomplete.

    Every item carries the same ``record_id``, which is what lets the console
    list one file as one row and delete it in one call, however many documents it
    became.

    Args:
        source_id: The id naming the whole file.
        filename: The name a citation should show.
        documents: The parser's output, in file order.

    Returns:
        The items to ingest, empty when the file yielded no text at all.
    """
    items: list[RagIngestItem] = []
    for document in documents:
        if not document.text.strip():
            continue
        pieces = split_text(document.text, MAX_ITEM_CHARS)
        for part, piece in enumerate(pieces):
            items.append(
                RagIngestItem(
                    id=_document_id(source_id, document.ordinal, part),
                    title=_document_title(filename, document.title, part, len(pieces)),
                    text=piece,
                    record_id=source_id,
                )
            )
    return items


async def list_sources(
    *,
    tenant: TenantContext,
    collection_repo: CollectionRepository,
    store: VectorStore,
    collection: str = DEFAULT_COLLECTION,
    limit: int = MAX_LISTED_SOURCES,
) -> KnowledgeInventory:
    """Describe what the tenant's knowledge base currently holds.

    A workspace that has never ingested anything has no collection row, which is
    an empty knowledge base rather than a missing resource - so it answers with
    an empty inventory instead of a 404. There is nothing to create here either:
    the row is minted by the first ingest, and creating one on a READ would have
    a page load write to the database.

    ``limit`` trims the ANSWER, never the walk. The store has to see every chunk
    before it can say how many a document has, so cutting the walk short would
    under-report every figure at once rather than omit a few rows (see
    :class:`~app.providers.rag.base.StoredInventory`). Both ways the list can be
    incomplete - more sources than were asked for, or a corpus past the store's
    own ceiling - collapse into ``truncated``, because a reader needs the same
    thing from either: do not read this as the whole of it.

    Args:
        tenant: The acting tenant; scopes the read.
        collection_repo: Tenant-scoped collection repository.
        store: The vector store, where the knowledge base actually lives.
        collection: Which named corpus to describe.
        limit: The most sources to return.

    Returns:
        The inventory, newest ingest first.

    Raises:
        VectorStoreUnavailableError: The store cannot be reached.
    """
    row = await collection_repo.get_by_name(collection)
    if row is None:
        return KnowledgeInventory(sources=[], collection=collection)
    inventory = await store.list_kb_sources(str(tenant.tenant_id), collection_id=str(row.id))
    return KnowledgeInventory(
        sources=inventory.sources[:limit],
        collection=collection,
        truncated=not inventory.complete or len(inventory.sources) > limit,
    )


async def add_file(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    tenant: TenantContext,
    collection_repo: CollectionRepository,
    store: VectorStore,
    embedder: Embedder,
    registry: ParserRegistry,
    settings: Settings,
    collection: str = DEFAULT_COLLECTION,
    is_chat_busy: Callable[[], bool] | None = None,
) -> UploadResult:
    """Read one file and store its text in the persistent knowledge base.

    The order is: refuse what cannot be read, parse, then ingest. Nothing is
    written anywhere until the file has actually produced text, so a rejected
    upload leaves the knowledge base exactly as it was - there is no half-added
    document to clean up, which is the one advantage a synchronous path has over
    the queued one an attachment takes.

    The collection row is committed BEFORE the embedding work, exactly as the
    direct ingest API does it, so the collection row is durable even if the
    (potentially long) ingest is interrupted.

    Args:
        data: The file's bytes, already read.
        filename: The client's filename - a label, never a path.
        content_type: The declared media type; used for dispatch only.
        tenant: The acting tenant; scopes every write.
        collection_repo: Tenant-scoped collection repository.
        store: The vector store.
        embedder: The active embedder; it defines the vector space.
        registry: Chooses the parser for this file.
        settings: Chunking, batching and size knobs.
        collection: The named corpus to add it to.
        is_chat_busy: Optional signal that a chat stream is active; the ingest
            pauses between embed batches while it reports True.

    Returns:
        What was stored.

    Raises:
        UnsupportedMediaTypeError: No parser can read this file.
        UnprocessableEntityError: The file was read but yielded no text worth
            storing, or the parse failed on the file's own contents. Both are
            terminal - the same request repeated fails the same way - which is
            why neither is a 503.
        ProviderUnavailableError: The embedding backend is unreachable.
        VectorStoreUnavailableError: The vector store is unreachable.
    """
    try:
        parser = registry.resolve(filename=filename, content_type=content_type, data=data)
    except UnsupportedFileError as exc:
        raise UnsupportedMediaTypeError(str(exc)) from exc

    try:
        parsed = await parser.parse(
            data, filename=filename, media_type=normalize_media_type(content_type)
        )
    except ParsingError as exc:
        # The file's own problem (corrupt, password-protected, malformed), told
        # in the parser's words. Retrying changes nothing, so it is a 422 rather
        # than the 503 a transient failure earns.
        raise UnprocessableEntityError(str(exc)) from exc

    source_id = _source_id()
    items = build_ingest_items(source_id, filename, parsed)
    if not items:
        # Parsing succeeded and produced nothing. Storing it would be a quiet
        # lie: the file would be listed as knowledge and ground no answer. An
        # image whose OCR found no text lands here too, and belongs here - the
        # knowledge base is searched by text, and there is no conversation for a
        # vision model to look at it in.
        raise UnprocessableEntityError(_NO_TEXT_MESSAGE)

    row = await collection_repo.get_or_create(collection, user_id=tenant.user_id)
    await collection_repo.commit()

    documents, chunks = await ingest_structured(
        source_type="upload",
        tenant=tenant,
        items=items,
        embedder=embedder,
        store=store,
        settings=settings,
        collection_id=str(row.id),
        is_chat_busy=is_chat_busy,
    )
    logger.info(
        "Added %s to the %s knowledge base as %s: %d document(s), %d chunk(s)",
        filename,
        collection,
        source_id,
        documents,
        chunks,
    )
    return UploadResult(
        id=source_id,
        title=items[0].title or filename,
        documents=documents,
        chunks=chunks,
        collection=collection,
    )


async def delete_source(
    *,
    source_id: str,
    tenant: TenantContext,
    collection_repo: CollectionRepository,
    store: VectorStore,
    collection: str = DEFAULT_COLLECTION,
) -> int:
    """Remove one source's passages from the knowledge base; returns how many.

    Nothing here is undeletable - a knowledge base is a corpus somebody curates,
    and every document in it is theirs to remove - so the only refusal possible
    is the tenant filter, which is applied inside the store and makes another
    workspace's source unaddressable rather than forbidden.

    Deleting an id that matches nothing is a successful zero, not a 404: the two
    are indistinguishable from here (there is no row whose absence could be
    checked), and the honest answer to "remove this" when it is already gone is
    that nothing was there to remove. The count is what says so.

    Args:
        source_id: The source to remove, as the listing named it.
        tenant: The acting tenant; scopes the delete.
        collection_repo: Tenant-scoped collection repository.
        store: The vector store.
        collection: The named corpus to remove it from.

    Returns:
        How many chunks were deleted, across every vector space that held a copy.

    Raises:
        VectorStoreUnavailableError: The store cannot be reached. Nothing is
            reported as deleted, so the same call can simply be repeated.
    """
    row = await collection_repo.get_by_name(collection)
    if row is None:
        return 0
    deleted = await store.delete_kb_source(
        str(tenant.tenant_id), source_id, collection_id=str(row.id)
    )
    logger.info("Removed knowledge source %s: %d chunk(s)", source_id, deleted)
    return deleted
