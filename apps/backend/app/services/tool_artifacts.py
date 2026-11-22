"""Turning a file a tool returned into a stored, announceable attachment.

A tool that generates a document hands its bytes back on
:class:`~app.providers.tools.base.ToolArtifact`, deliberately outside the text
the model reads. This module is where those bytes become a real
:class:`~app.models.Attachment`: written to object storage, committed as a row,
and described by a :class:`~app.providers.base.StoredFile` the generation loop
announces on the wire.

Three properties are load-bearing and are why this is a module rather than a few
lines in the loop:

- **Bytes before row.** Storage is written first and the row committed second,
  the same ordering the upload path uses, so a committed row always points at a
  blob that exists. A crash between the two leaves an orphaned blob, which the
  attachment garbage collector already reclaims - it sweeps exactly the
  ``{tenant}/{conversation}/{attachment}`` keys this mints.
- **The name and the media type are untrusted.** Both arrived from a remote
  server. The filename is re-validated against the same shape a storage key
  segment must have, and the media type against the plain ``type/subtype``
  grammar; either one that does not fit is REPLACED, not repaired. Both end up
  in response headers the day a download route exists (no route serves an
  attachment's bytes yet), which would be a late moment to discover that a
  server put a newline in one.
- **Nothing is dropped silently.** A file over the size cap, or past the turn's
  count, produces a one-line notice the caller folds into the result text and
  the tool card, because a file that quietly failed to appear is
  indistinguishable from one the tool never made.
"""

from __future__ import annotations

import logging
import mimetypes
import re
from collections.abc import Sequence
from urllib.parse import unquote
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.tenancy import TenantContext
from app.models import ATTACHMENT_ORIGIN_GENERATED, ATTACHMENT_READY, Attachment
from app.providers.base import StoredFile
from app.providers.tools.base import ToolArtifact
from app.repositories.attachments import AttachmentRepository
from app.storage.base import InvalidStorageKeyError, StorageBackend, validate_key

logger = logging.getLogger(__name__)

# Long enough for any honest document name and short enough to sit in a UI
# column and a Content-Disposition header without special handling.
_MAX_FILENAME_CHARS = 128

# A generated name's extension has to be as conservative as the name itself:
# mimetypes is consulted for a suggestion, never trusted with the result.
_SAFE_EXTENSION = re.compile(r"^\.[A-Za-z0-9]{1,12}$")
_FALLBACK_EXTENSION = ".bin"

# Whitespace is the ONE thing normalized rather than rejected. A real document is
# very often called "Q3 report.xlsx", and failing that name into "artifact-1.xlsx"
# would lose information for no security gain - a dash is not a separator, an
# escape, or a header terminator. Everything else that fails validation is
# replaced outright, so the rule stays "the name is a valid key segment" with a
# single, stated normalization in front of it.
_WHITESPACE = re.compile(r"\s+")

# The media type is untrusted too, and it is the value a download response would
# echo into a Content-Type header - so a CR/LF or a stray quote in it is header
# injection waiting for the endpoint that serves these files. RFC 9110's type
# and subtype grammar, bounded, and no parameters (a charset the server invented
# buys nothing here and is one more thing to escape).
# Ends in \Z, not $, for the reason app/storage/base.py spells out about storage
# keys: in Python '$' also matches just before a trailing newline, so '$' would
# accept "text/plain\n" - a value that becomes a header, on the one field whose
# whole job is to be unable to carry a line break. Only the caller's strip() has
# been standing between that and the wire.
_MEDIA_TYPE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,62}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,62}\Z"
)
DEFAULT_ARTIFACT_CONTENT_TYPE = "application/octet-stream"


def safe_artifact_content_type(declared: str) -> str:
    """Return a media type safe to store and to serve, or the neutral default.

    Args:
        declared: The media type the tool's server claimed.

    Returns:
        ``declared`` when it is a plain, bounded ``type/subtype``; otherwise
        ``application/octet-stream``, which is the honest answer for bytes whose
        type the host is not willing to vouch for.
    """
    candidate = declared.strip()
    return candidate if _MEDIA_TYPE.match(candidate) else DEFAULT_ARTIFACT_CONTENT_TYPE


def _fallback_filename(content_type: str, index: int) -> str:
    """Build the name used when the tool's own is unusable.

    ``index`` counts KEPT files in the turn, so two generated names never
    collide. Counting artifacts SEEN instead would: a call that drops one file
    and stores another advances the seen count by two and the kept count by one,
    so the next call's first file would reuse a number already handed out.
    """
    guessed = mimetypes.guess_extension(content_type) or ""
    extension = guessed if _SAFE_EXTENSION.match(guessed) else _FALLBACK_EXTENSION
    return f"artifact-{index}{extension}"


def safe_artifact_filename(hint: str, *, content_type: str, index: int) -> str:
    """Return a filename safe to store and serve, from an untrusted hint.

    The hint is whatever a remote server suggested, so it is accepted only if it
    is a single valid storage-key segment: ASCII letters, digits, dot, dash and
    underscore, starting with a letter or digit. Anything else gets a generated
    name rather than a repaired one - silently editing a filename until it
    passes is how a name that means something becomes a name that means
    something else.

    Two normalizations run first, and only two. The hint is percent-DECODED,
    because a uri is where it came from and ``Q3%20report.pdf`` is the correct
    spelling of a name with a space in it; then whitespace is folded to dashes.
    Both are lossless enough to keep an honest name honest, and neither can
    smuggle anything past the check that follows - a decoded ``%2F`` is still a
    separator, and is still refused.

    The rule is deliberately narrower than a filesystem's, which does cost real
    names: ``rapport-ete.pdf`` survives, ``rapport-été.pdf`` and ``report
    (final).pdf`` do not. That is the accepted price of one rule - "the name is
    a valid key segment" - covering path traversal, device names, control
    characters and header terminators together, rather than a blocklist per
    hazard.

    Args:
        hint: The server-supplied name, possibly empty or hostile.
        content_type: The declared media type, used to pick a fallback suffix.
        index: How many files the turn has already kept, for the fallback name.

    Returns:
        A filename that is a valid key segment and no longer than
        :data:`_MAX_FILENAME_CHARS`.
    """
    return _usable_hint(hint) or _fallback_filename(content_type, index)


def _usable_hint(hint: str) -> str | None:
    """The server's own name if it survives validation, else None.

    Split out from :func:`safe_artifact_filename` because the two callers want
    opposite things from a name that does not survive. Storing a file needs SOME
    name, so it takes the numbered fallback; TALKING about a file that was
    dropped must not - a fallback number belongs to a file that was actually
    kept, and handing one to a notice would let it name a document the user can
    see while meaning one they cannot.
    """
    candidate = _WHITESPACE.sub("-", unquote(hint).strip())
    if not candidate or len(candidate) > _MAX_FILENAME_CHARS or "/" in candidate:
        return None
    try:
        # A filename is ONE segment; validate_key would happily accept "a/b" as
        # two, which is why the separator is refused above before this runs.
        return validate_key(candidate)
    except InvalidStorageKeyError:
        return None


def _dropped_label(hint: str, content_type: str) -> str:
    """Name a file that was NOT kept, in a way no kept file can answer to.

    A dropped file is described by the server's own name when that name is
    usable, and by its type alone when it is not. It never gets a numbered
    fallback: those number the files a turn KEPT, and a call that drops one
    artifact and stores the next would otherwise hand both the same
    ``artifact-N.pdf`` - the notice apologising for a file whose name the
    user can see attached to the answer.
    """
    usable = _usable_hint(hint)
    if usable:
        return f"the file {usable!r}"
    if content_type == DEFAULT_ARTIFACT_CONTENT_TYPE:
        return "an unnamed file"
    return f"an unnamed {content_type.split('/')[-1]} file"


def _storage_key(tenant: TenantContext, conversation_id: UUID, attachment_id: UUID) -> str:
    """Mint the object-storage key for a generated file.

    Deliberately the SAME ``{tenant}/{conversation}/{attachment}`` triple the
    upload path mints. The blob sweep only considers keys of that shape, so a
    generated file that loses its row is reclaimed by the machinery that already
    exists rather than leaking forever under a shape nothing recognises.
    """
    return f"{tenant.tenant_id}/{conversation_id}/{attachment_id}"


def format_bytes(size_bytes: int) -> str:
    """A compact human size ("512 B", "12.3 KB", "1.4 MB").

    Scales the unit rather than fixing it at MB, because both readers of this
    can legitimately be handed a small number: a deployment may lower the
    artifact cap, and a notice reading "0.0 MB, over the 0.0 MB limit" explains
    nothing to the person whose file went missing.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1_048_576:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1_048_576:.1f} MB"


def _too_large_notice(label: str, size_bytes: int, limit: int) -> str:
    """Say plainly that a file was produced and then not kept."""
    return (
        f"[{label} this tool produced was {format_bytes(size_bytes)}, "
        f"over the {format_bytes(limit)} limit, and was not kept]"
    )


async def store_tool_artifacts(
    artifacts: Sequence[ToolArtifact],
    *,
    tenant: TenantContext,
    conversation_id: UUID,
    sessionmaker: async_sessionmaker[AsyncSession],
    storage: StorageBackend | None,
    settings: Settings,
    remaining: int,
    first_index: int,
) -> tuple[list[StoredFile], list[str]]:
    """Store the files one tool call produced; return what was kept and why not.

    Never raises. A tool call that produced a file the host could not keep still
    answers the user's question, so every failure here degrades to a notice
    rather than to a failed turn.

    Each artifact is written to storage and committed as a ``ready`` attachment
    with ``origin="generated"`` and NO content hash. The missing hash is not an
    omission: :meth:`AttachmentRepository.get_by_content_hash` is the upload
    path's dedupe, and a NULL hash can never match it, so an upload of the same
    bytes can never be handed a generated row instead of storing its own.
    Generated files are born ``ready`` because there is no parse job and no RAG
    ingest for them in this slice - the file is offered to the USER, not folded
    back into what the model can retrieve.

    Args:
        artifacts: The files this one call returned.
        tenant: The acting principal; the sole source of the row's scoping.
        conversation_id: The thread the files belong to.
        sessionmaker: Session factory; a fresh session is opened and committed
            here so an announced file is durable before it is announced.
        storage: The object store, or None when the turn has no backend wired.
        settings: Supplies the per-file size cap.
        remaining: How many more files this TURN may keep, across all its calls.
        first_index: How many files the turn has already kept, plus one, so a
            generated fallback name is unique across the whole turn.

    Returns:
        A ``(stored, notices)`` pair: the files kept, in order, and one
        human-readable line per file that was not.
    """
    if not artifacts:
        return [], []
    if storage is None:
        logger.warning(
            "No storage backend on this turn; dropping %d generated file(s)", len(artifacts)
        )
        return [], ["[this deployment has no file storage, so files this tool produced were lost]"]

    stored: list[StoredFile] = []
    notices: list[str] = []
    for artifact in artifacts:
        if len(stored) >= remaining:
            notices.append(
                f"[this turn already kept {settings.tool_artifact_max_per_turn} files, "
                "so the rest were not kept]"
            )
            break
        content_type = safe_artifact_content_type(artifact.content_type)
        if len(artifact.data) > settings.tool_artifact_max_bytes:
            # Described, not NAMED: a dropped file never spends a fallback
            # number, so its notice cannot end up quoting the name of the file
            # stored right after it. See _dropped_label.
            notices.append(
                _too_large_notice(
                    _dropped_label(artifact.filename, content_type),
                    len(artifact.data),
                    settings.tool_artifact_max_bytes,
                )
            )
            continue
        # Numbered by files KEPT, not files seen, and that has to hold WITHIN a
        # call as well as across them - see _fallback_filename. Computed only
        # once the size check has passed, so a dropped artifact consumes nothing.
        filename = safe_artifact_filename(
            artifact.filename, content_type=content_type, index=first_index + len(stored)
        )
        attachment = Attachment(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            conversation_id=conversation_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(artifact.data),
            storage_key="",  # set once the id is known, just below
            # Born ready: nothing parses or ingests a generated file, so any
            # other status would be a state it could never leave.
            status=ATTACHMENT_READY,
            # Never a hash - see the docstring: this is what keeps a generated
            # row out of the upload path's dedupe.
            content_hash=None,
            origin=ATTACHMENT_ORIGIN_GENERATED,
        )
        attachment.storage_key = _storage_key(tenant, conversation_id, attachment.id)
        try:
            await storage.put(attachment.storage_key, artifact.data, content_type=content_type)
        except Exception:  # noqa: BLE001 - see below; a lost file must not fail the answer
            # Broader than StorageError on purpose. The seam PROMISES to raise
            # only its own types, but a backend that breaks that promise (an
            # OSError from a full disk, a vendor SDK error) would otherwise take
            # down a turn the user is waiting on, over an attachment. "Never
            # raises" has to be true for every backend, not just conforming ones.
            logger.exception(
                "Could not store a generated file for conversation %s", conversation_id
            )
            # Described rather than named, for the same reason the size drop is:
            # this artifact never reached `stored`, so the fallback number it was
            # tentatively given is still free and the next file will take it.
            notices.append(
                f"[{_dropped_label(artifact.filename, content_type)} this tool produced "
                "could not be stored]"
            )
            continue
        try:
            async with sessionmaker() as session:
                AttachmentRepository(session, tenant).add(attachment)
                await session.commit()
        except Exception:  # noqa: BLE001 - a lost file must not fail the answer
            # The blob is now an orphan under a minted key, which is exactly what
            # the attachment blob sweep collects; there is nothing to undo here
            # beyond saying so.
            logger.exception(
                "Stored the bytes of a generated file but could not commit its row "
                "(conversation %s); the blob sweep will reclaim %s",
                conversation_id,
                attachment.storage_key,
            )
            notices.append(
                f"[{_dropped_label(artifact.filename, content_type)} this tool produced "
                "could not be saved]"
            )
            continue
        stored.append(
            StoredFile(
                attachment_id=str(attachment.id),
                filename=filename,
                content_type=attachment.content_type,
                size_bytes=attachment.size_bytes,
                origin=ATTACHMENT_ORIGIN_GENERATED,
            )
        )
    return stored, notices
