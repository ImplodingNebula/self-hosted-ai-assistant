"""Attachments endpoints: upload a file, poll its status, download it, remove it.

Routers stay HTTP-only: parsing the multipart form and shaping the response
live here, while :mod:`app.services.attachment_service` decides the order that
keeps storage and the database consistent, and every data access goes through
the tenant-scoped ``AttachmentRepository``.

A pasted block of text is an ordinary upload: the client sends it as a file part
with ``origin=paste``, and it travels the identical pipeline. Only the label a
citation shows for it differs.

The download route's status-to-status-code mapping lives here rather than in a
service on purpose: every branch of it is a statement about HTTP (which code,
which problem body, which headers), and the underlying question - "may these
bytes be served" - is answered by two columns the router already holds.

That route has a cheap twin. HEAD answers the same question from the row alone,
so a client can find out whether a file is still there without either side
reading the object; the two share their gate and their headers so they cannot
come to disagree about one file.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_tenant,
    get_job_queue,
    get_parser_registry,
    get_session,
    get_storage_backend,
    get_vector_store,
)
from app.core.config import Settings, get_settings
from app.core.errors import (
    AttachmentExpiredError,
    AttachmentNotReadyError,
    AttachmentUnreadableError,
    BadRequestError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
)
from app.core.tenancy import TenantContext
from app.models import (
    ATTACHMENT_EXPIRED,
    ATTACHMENT_FAILED,
    ATTACHMENT_ORIGIN_GENERATED,
    ATTACHMENT_ORIGIN_PASTE,
    ATTACHMENT_ORIGIN_UPLOAD,
    ATTACHMENT_PARSING,
    ATTACHMENT_PENDING,
    ATTACHMENT_READY,
    Attachment,
)
from app.parsing.registry import ParserRegistry
from app.providers.rag.base import VectorStore
from app.repositories.attachments import AttachmentRepository
from app.repositories.conversations import ConversationRepository
from app.schemas.attachments import AttachmentOut
from app.services.attachment_service import create_attachment, delete_attachment
from app.services.tool_artifacts import safe_artifact_content_type
from app.storage.base import (
    InvalidStorageKeyError,
    ObjectNotFoundError,
    StorageBackend,
    StorageError,
)
from app.workers.queue import JobQueue

logger = logging.getLogger(__name__)

router = APIRouter(tags=["attachments"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
TenantDep = Annotated[TenantContext, Depends(get_current_tenant)]

# The default a browser sends when it cannot guess a type. Treated as "unknown"
# so the parser registry falls back to sniffing the bytes.
_DEFAULT_CONTENT_TYPE = "application/octet-stream"

# The origins a PERSON can perform, which is deliberately narrower than the full
# set the column accepts. "generated" describes a file the assistant's own tool
# produced; a client that could claim it would be asserting a provenance that
# never happened, and the UI reads exactly that field to decide whether a file is
# offered back as a composer draft.
_UPLOADABLE_ORIGINS = frozenset({ATTACHMENT_ORIGIN_UPLOAD, ATTACHMENT_ORIGIN_PASTE})


def _parse_uuid(raw: str, *, field: str) -> UUID:
    """Parse a form-supplied id, mapping a malformed value to a 400."""
    try:
        return UUID(raw)
    except ValueError as exc:
        raise BadRequestError(f"{field} is not a valid UUID.") from exc


async def _require_attachment(
    attachment_id: str, session: AsyncSession, tenant: TenantContext
) -> Attachment:
    """Resolve a path id to this tenant's attachment, or raise 404.

    A malformed id is a 404 like an absent one: a value that cannot address a
    resource, or that addresses another tenant's, must not be distinguishable
    from one that never existed.
    """
    try:
        parsed_id = UUID(attachment_id)
    except ValueError as exc:
        raise NotFoundError("Attachment not found.") from exc
    attachment = await AttachmentRepository(session, tenant).get(parsed_id)
    if attachment is None:
        raise NotFoundError("Attachment not found.")
    return attachment


# What a Content-Disposition filename may hold literally: printable US-ASCII and
# nothing else. A control character, a bare CR or LF, any accented letter - all
# replaced in the ASCII fallback, which is why the RFC 6266 ``filename*`` form
# always rides beside it carrying the real name as percent-encoded UTF-8.
_NON_ASCII_PRINTABLE = re.compile(r"[^\x20-\x7e]")

# A quoted-string cannot hold a bare quote or backslash, and an escape sequence
# is one more thing to get wrong in a header assembled from row data. Both are
# replaced rather than escaped, so the quoted form cannot be terminated early.
_UNQUOTABLE = re.compile(r'["\\]')

# The longest Content-Disposition value this route will emit, in BYTES - which
# is the unit the server's header budget is spent in, and the reason a character
# count is not a bound here: percent-encoding costs up to 12 bytes per code
# point, so 180 emoji are a 2 KB header. A stored filename has no length limit of
# its own, so something has to be. Shortening a download's SUGGESTED name costs
# nothing (the row keeps the whole thing); a header long enough to exhaust the
# budget costs the entire response.
_MAX_DISPOSITION_BYTES = 512

# A suffix plain and short enough to be worth carrying through a truncation: the
# operating system picks an application from it, so a clamp that cuts it off
# turns a named PDF into a blob nothing will open. Same shape as the one the
# generated-file path validates, but searched at the END of a whole filename
# rather than matched against an extension already on its own - hence its own
# pattern rather than an import.
_TRAILING_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,12}\Z")

# The placeholder for a name that leaves nothing quotable behind (one written
# entirely in a non-Latin script, say). Only the clients that cannot read
# ``filename*`` ever see it; the rest still get the real name.
_FALLBACK_DOWNLOAD_NAME = "download"


def _disposition_value(stem: str, extension: str) -> str:
    """Render one candidate disposition header from a name's two halves."""
    name = f"{stem}{extension}"
    fallback = _UNQUOTABLE.sub("_", _NON_ASCII_PRINTABLE.sub("_", name)).strip()
    return (
        f'attachment; filename="{fallback or _FALLBACK_DOWNLOAD_NAME}"; '
        f"filename*=UTF-8''{quote(name, safe='')}"
    )


def _content_disposition(filename: str) -> str:
    """Build an RFC 6266 ``attachment`` disposition for an untrusted filename.

    The name is row data someone else chose - a person's upload, or a remote tool
    server's suggestion - so it is treated as hostile: it may decide how the
    header READS and never what the header IS. Two forms go out together, exactly
    as RFC 6266 prescribes: a conservative quoted-ASCII ``filename`` every client
    understands, and ``filename*`` carrying the real name for those that do.

    An over-long name is shortened by rebuilding until the rendered header fits
    :data:`_MAX_DISPOSITION_BYTES`, rather than by clamping the input: the two
    encodings inflate at wildly different rates, so only the finished header
    knows how big it is. The extension is held back from the truncation.

    Args:
        filename: The stored filename, unsanitized.

    Returns:
        A complete ``Content-Disposition`` header value.
    """
    match = _TRAILING_EXTENSION.search(filename)
    extension = match.group(0) if match else ""
    stem = filename[: len(filename) - len(extension)]
    header = _disposition_value(stem, extension)
    # Halved rather than stepped: the exact length of a truncated SUGGESTION
    # means nothing as long as it fits, and one hostile name must not cost
    # hundreds of rebuilds. Terminates - an empty stem leaves only the bounded
    # extension, which always fits.
    while stem and len(header.encode("utf-8")) > _MAX_DISPOSITION_BYTES:
        stem = stem[: len(stem) // 2]
        header = _disposition_value(stem, extension)
    return header


def _servable_content_type(declared: str) -> str:
    """Return a media type safe to echo into a response header.

    An upload's stored type is the whole media-type header its multipart part
    carried, chosen by the client, and this route is the first thing that hands
    it back out - so it goes through the same bounded ``type/subtype`` gate a
    generated file's type already passes, rather than a second, weaker copy of
    that check.

    Parameters are dropped BEFORE the check rather than failing it, because they
    are the common honest case: a browser that sends ``text/plain;
    charset=utf-8`` should get its file back as text, not as the neutral
    fallback. The charset itself buys nothing on a response the
    ``Content-Disposition`` already sends to disk.

    Args:
        declared: The media type recorded on the row.

    Returns:
        The bare type/subtype when it is well-formed, else
        ``application/octet-stream``.
    """
    return safe_artifact_content_type(declared.split(";", 1)[0])


def _download_headers(attachment: Attachment) -> dict[str, str]:
    """Build the headers that turn a stored object into a download.

    Shared by the GET and the HEAD rather than written twice, and that is not
    tidiness: a HEAD that named the file differently from the GET beside it
    would be worse than no HEAD at all, and the header-injection and
    content-sniffing guards must have exactly one implementation between them.

    Args:
        attachment: The resolved, tenant-scoped row.

    Returns:
        Response headers, without the media type (which rides on the response).
    """
    return {
        "Content-Disposition": _content_disposition(attachment.filename),
        # The bytes are untrusted and the declared type may be wrong or neutral;
        # a browser that sniffs its way to text/html over a file someone else
        # uploaded is the one outcome worth spending a header on. Harmless
        # alongside the attachment disposition, and the belt to its braces.
        "X-Content-Type-Options": "nosniff",
        # One tenant's private file behind a shared same-origin path: no
        # intermediary, and no shared browser cache, keeps a copy.
        "Cache-Control": "private, no-store",
    }


def _refuse_unless_servable(attachment: Attachment) -> None:
    """Raise the honest problem for a file this route will not serve.

    The gate is BYTE availability, qualified by provenance - not ingestion
    state. An upload's bytes are written before its row is committed and are
    never removed by a failed parse, so a ``pending``, ``parsing`` or ``failed``
    row is sitting on its blob; only ``expired`` has genuinely lost its bytes.

    That is why a ``failed`` upload or paste is SERVED rather than refused. What
    failed is INGESTION - the platform's reading of the file - and the file
    itself is untouched and still its owner's, so withholding it would be
    refusing someone their own document over a parser's opinion of it. The
    ``pending`` and ``parsing`` refusals stand: those files have not settled
    yet, and a moment's wait may make them ordinary.

    A ``failed`` file the assistant GENERATED is still refused, and the
    asymmetry is the whole point - there is no original behind it. A generated
    file exists only because the platform finished producing it, so "the
    platform did not finish" leaves nothing trustworthy to hand back. No code
    path reaches that combination today (a generated row is born ``ready`` and
    nothing parses it), which is precisely why the gate is written around what
    makes serving SAFE instead of around what happens to be reachable.

    One branch per lifecycle value, because the refusals mean different things
    to whoever clicked: come back in a moment, this aged out, this one can be
    asked for again. Collapsing them into a single code would leave a client
    with a dead affordance and no way to say why.

    Args:
        attachment: The resolved, tenant-scoped row.

    Raises:
        AttachmentExpiredError: Retention removed the bytes (410).
        AttachmentUnreadableError: A generated file never got finished, or the
            status is one this build does not recognise (410).
        AttachmentNotReadyError: Ingestion is still in flight (409).
    """
    if attachment.status == ATTACHMENT_READY:
        return
    if attachment.status == ATTACHMENT_EXPIRED:
        detail = "This file aged out of retention, so its contents are no longer stored."
        if attachment.origin == ATTACHMENT_ORIGIN_GENERATED:
            # The one refusal with a way forward: nothing was lost that the
            # assistant cannot make again.
            detail += " Asking for it again produces a fresh copy."
        raise AttachmentExpiredError(detail)
    if attachment.status == ATTACHMENT_FAILED:
        if attachment.origin in _UPLOADABLE_ORIGINS:
            # A person's own file, unchanged since they handed it over.
            return
        raise AttachmentUnreadableError(
            "This file was never finished when it was produced, so there is no "
            "original to hand back."
        )
    if attachment.status in {ATTACHMENT_PENDING, ATTACHMENT_PARSING}:
        raise AttachmentNotReadyError(
            "This file is still being processed; its status says when it is ready."
        )
    # Unreachable for every value in ATTACHMENT_STATUSES today. A status added
    # without a branch here must not be served on the strength of not being
    # recognised - the closed door is the safe default, and it is loud enough in
    # the log to be found.
    logger.warning(
        "Refusing to serve attachment %s: unhandled status %r",
        attachment.id,
        attachment.status,
    )
    raise AttachmentUnreadableError("This file is not available to download.")


@router.post("/attachments", response_model=AttachmentOut, status_code=201)
async def upload_attachment(
    response: Response,
    session: SessionDep,
    tenant: TenantDep,
    settings: Annotated[Settings, Depends(get_settings)],
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
    registry: Annotated[ParserRegistry, Depends(get_parser_registry)],
    queue: Annotated[JobQueue, Depends(get_job_queue)],
    file: Annotated[UploadFile, File()],
    conversation_id: Annotated[str, Form()],
    origin: Annotated[str, Form()] = ATTACHMENT_ORIGIN_UPLOAD,
) -> AttachmentOut:
    """Store an uploaded file and queue it for parsing into the conversation.

    The response is the attachment in its ``pending`` state; the client polls
    :func:`get_attachment` until ``status`` reaches ``ready`` or ``failed``.
    Returns 201: a new resource is created even though its content is not yet
    retrievable.
    """
    if origin not in _UPLOADABLE_ORIGINS:
        raise BadRequestError(f"origin must be one of {sorted(_UPLOADABLE_ORIGINS)}.")
    parsed_conversation_id = _parse_uuid(conversation_id, field="conversation_id")

    filename = (file.filename or "").strip()
    if not filename:
        raise BadRequestError("The uploaded file has no name.")
    # The exact per-file limit. The multipart body is already parsed by the time
    # a handler runs, so this is not what stops a disk-filling upload -
    # MaxBodySizeMiddleware refuses an over-large body before it is read. This is
    # the precise check on the file part itself (file.size can be None, so the
    # service re-checks the real byte length as the authoritative backstop).
    if file.size is not None and file.size > settings.attachment_max_bytes:
        raise PayloadTooLargeError(
            f"The file is larger than the {settings.attachment_max_bytes} byte limit."
        )
    data = await file.read()
    if not data:
        raise BadRequestError("The uploaded file is empty.")

    attachment = await create_attachment(
        data=data,
        filename=filename,
        content_type=file.content_type or _DEFAULT_CONTENT_TYPE,
        origin=origin,
        conversation_id=parsed_conversation_id,
        tenant=tenant,
        attachment_repo=AttachmentRepository(session, tenant),
        conversation_repo=ConversationRepository(session, tenant),
        storage=storage,
        registry=registry,
        queue=queue,
        settings=settings,
    )
    response.headers["Location"] = f"/api/v1/attachments/{attachment.id}"
    return AttachmentOut.from_entity(attachment)


@router.get("/attachments/{attachment_id}", response_model=AttachmentOut)
async def get_attachment(
    attachment_id: str,
    session: SessionDep,
    tenant: TenantDep,
) -> AttachmentOut:
    """Return one attachment, for polling its ingestion status.

    A malformed or foreign id is a 404, the same as an absent one: an id that
    cannot address a resource, or addresses another tenant's, must not be
    distinguishable from one that never existed.
    """
    attachment = await _require_attachment(attachment_id, session, tenant)
    return AttachmentOut.from_entity(attachment)


@router.get(
    "/attachments/{attachment_id}/content",
    response_class=Response,
    responses={200: {"content": {"application/octet-stream": {}}, "description": "The file."}},
)
async def download_attachment(
    attachment_id: str,
    session: SessionDep,
    tenant: TenantDep,
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
) -> Response:
    """Return one attachment's stored bytes as a download.

    This is what makes a file the assistant produced actually reachable: the
    ``file`` stream event announces it and the reply's chip names it, and both
    are decoration until the bytes can be fetched.

    The whole object is read and answered in one response rather than streamed.
    That is a deliberate bound, not an oversight: uploads are capped by
    ``attachment_max_bytes`` and generated files far lower, so the ceiling is
    known and small, and a streaming variant would need a new seam method on
    every storage backend to earn nothing at these sizes.

    Which files are served is :func:`_refuse_unless_servable`'s decision - a
    ``ready`` file, and a person's own upload whose INGESTION failed - and every
    refusal carries its own reason. A row that claims to hold bytes that are not
    there answers 404 rather than 500: the client asked an answerable question
    and deserves an answer, while the inconsistency is the server's to log.

    A malformed or foreign id is a 404, exactly as for the metadata route: an id
    that addresses another user's file must not be distinguishable from one that
    never existed.
    """
    attachment = await _require_attachment(attachment_id, session, tenant)
    _refuse_unless_servable(attachment)
    try:
        data = await storage.get(attachment.storage_key)
    except ObjectNotFoundError as exc:
        # The row promised a blob that is not there. Ordering makes this rare
        # (bytes are written before the row is committed), so it means something
        # removed the object out from under a live row - worth a log line, not a
        # 500 for the person who clicked.
        logger.warning(
            "Attachment %s is %s but its object %r is missing from storage",
            attachment.id,
            attachment.status,
            attachment.storage_key,
        )
        raise NotFoundError("This file's contents are no longer stored.") from exc
    except InvalidStorageKeyError as exc:
        # A key the storage seam refuses is a broken invariant, not weather: the
        # app mints every key itself, so this means a row is carrying something
        # no version of that code could have written. It must NOT fall into the
        # 503 below - that tells the client to retry a request whose outcome can
        # never change, forever. Logged at error level because it is a bug
        # somebody has to see; answered 404 because for the person asking, the
        # file is simply unreachable.
        logger.error(
            "Attachment %s has an unusable storage key %r; its bytes cannot be addressed",
            attachment.id,
            attachment.storage_key,
        )
        raise NotFoundError("This file's contents cannot be located.") from exc
    except StorageError as exc:
        raise ServiceUnavailableError(
            "Could not read the stored file. Try again in a moment."
        ) from exc

    return Response(
        content=data,
        # The row's declared type, but only when it is a plain bounded
        # type/subtype - the same gate a generated file's type already passed
        # through, reused rather than re-implemented more weakly. An upload's
        # type is whatever the client's multipart part claimed, so this is the
        # first place it is checked before being echoed back into a header.
        media_type=_servable_content_type(attachment.content_type),
        headers=_download_headers(attachment),
    )


@router.head(
    "/attachments/{attachment_id}/content",
    response_class=Response,
    responses={200: {"description": "The file is there, and these headers describe it."}},
)
async def head_attachment_content(
    attachment_id: str,
    session: SessionDep,
    tenant: TenantDep,
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
) -> Response:
    """Answer whether the file can be downloaded, without moving its bytes.

    The same question :func:`download_attachment` answers, reaching the same
    codes (200, 409, 410, 404, 503) without reading the object - the row and one
    ``exists`` call on the storage seam are the whole cost.

    That is what earns it a place. A client that CHECKS before downloading -
    so a file that aged out explains itself instead of saving to disk as an
    error document - would otherwise spend two whole-object reads per saved
    file, and the download route is rate-limited into a small budget precisely
    because it buffers one. With this, checking is ordinary traffic and only the
    file itself is expensive.

    Nothing is being replaced here: FastAPI overwrites the HEAD that Starlette
    adds beside a GET (``fastapi.routing`` assigns the declared methods
    verbatim), so without this route the method is a plain 405.

    ``Content-Length`` is the row's recorded size, which is the length of the
    object as it was written. It is second-hand by construction - the GET, which
    reads the blob, remains the authority - but quoting it is closer to RFC 9110
    ("send the same header fields a GET would") than the ``0`` a bodiless
    response would otherwise advertise, which is a plain lie about the file.
    """
    attachment = await _require_attachment(attachment_id, session, tenant)
    _refuse_unless_servable(attachment)
    try:
        present = await storage.exists(attachment.storage_key)
    except ObjectNotFoundError as exc:
        # The seam promises ``exists`` returns a bool rather than raising this,
        # and no shipped backend does. Mapped anyway, and to the GET's code
        # rather than the outage code below, because the alternative is the two
        # methods disagreeing about one file - the exact thing sharing a gate
        # was meant to make impossible.
        raise NotFoundError("This file's contents are no longer stored.") from exc
    except InvalidStorageKeyError as exc:
        # Same reading as on the GET: the app mints every key itself, so a key
        # the seam refuses is a broken invariant rather than weather, and it
        # must not fall into the retry-forever 503 below.
        logger.error(
            "Attachment %s has an unusable storage key %r; its bytes cannot be addressed",
            attachment.id,
            attachment.storage_key,
        )
        raise NotFoundError("This file's contents cannot be located.") from exc
    except StorageError as exc:
        raise ServiceUnavailableError(
            "Could not read the stored file. Try again in a moment."
        ) from exc
    if not present:
        logger.warning(
            "Attachment %s is %s but its object %r is missing from storage",
            attachment.id,
            attachment.status,
            attachment.storage_key,
        )
        raise NotFoundError("This file's contents are no longer stored.")

    return Response(
        status_code=200,
        media_type=_servable_content_type(attachment.content_type),
        headers={
            **_download_headers(attachment),
            # Set explicitly so the framework does not fill in the length of
            # the empty body it is about to send.
            "Content-Length": str(attachment.size_bytes),
        },
    )


@router.delete("/attachments/{attachment_id}", status_code=204)
async def remove_attachment(
    attachment_id: str,
    session: SessionDep,
    tenant: TenantDep,
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> Response:
    """Delete an uploaded file: its indexed text, its stored bytes, and its row.

    Taking a file off a turn has to actually remove it. An upload is ingested
    the moment it lands, so a file that was merely hidden from the composer
    would still be retrievable, and the assistant would go on answering from a
    document its owner believes is gone.

    Deleting an already-deleted file is a 404, exactly as for an id that never
    existed; a partial failure leaves the file addressable so the same request
    can simply be retried.
    """
    attachment = await _require_attachment(attachment_id, session, tenant)
    await delete_attachment(
        attachment=attachment,
        tenant=tenant,
        attachment_repo=AttachmentRepository(session, tenant),
        storage=storage,
        vector_store=vector_store,
    )
    return Response(status_code=204)
