"""HTTP schemas for the attachments endpoints.

A DTO separate from the ``Attachment`` table model: the wire shape is a
deliberate subset. ``storage_key`` and ``tenant_id`` never leave the server -
the key is an internal handle a client has no use for and no business knowing,
and the tenant is derived server-side, never echoed.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from app.models import Attachment
from app.schemas.common import UtcDatetime


class AttachmentOut(BaseModel):
    """One attachment as returned to a client.

    ``status`` is the field a client polls after upload: it moves from
    ``pending`` to ``parsing`` to ``ready`` (retrievable) or ``failed``, and
    ``error`` carries a human-readable reason when it is ``failed``.

    A ``ready`` file may later become ``expired``, which no poll will ever see -
    it is written by the retention sweep days or weeks afterwards. The file's
    bytes and indexed text are gone while this row survives, so the name, size
    and type here are still accurate and still worth showing: the row exists so
    an interface can say the file was here and has aged out, rather than losing
    the record. ``error`` stays null, because nothing failed. A client that
    treats the status as a closed set of four values will meet a fifth.

    ``created_at`` is load-bearing beyond bookkeeping: a client tells an unsent
    draft file from one an earlier turn already carried by comparing it against
    that conversation's newest message, so it must speak the same timestamp
    format the messages do (see :data:`app.schemas.common.UtcDatetime`).
    """

    id: UUID
    conversation_id: UUID | None
    filename: str
    content_type: str
    size_bytes: int
    status: str
    origin: str
    error: str | None
    created_at: UtcDatetime
    updated_at: UtcDatetime

    @classmethod
    def from_entity(cls, attachment: Attachment) -> AttachmentOut:
        """Project a stored attachment onto its public shape."""
        return cls(
            id=attachment.id,
            conversation_id=attachment.conversation_id,
            filename=attachment.filename,
            content_type=attachment.content_type,
            size_bytes=attachment.size_bytes,
            status=attachment.status,
            origin=attachment.origin,
            error=attachment.error,
            created_at=attachment.created_at,
            updated_at=attachment.updated_at,
        )
