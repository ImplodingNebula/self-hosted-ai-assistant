"""Document-parsing seam: the DocumentParser abstraction and its value types.

Mirrors the storage, RAG and search seams: an ABC with one abstract async
method, a pydantic value type, and plain-Exception seam errors that services
catch instead of a backend-specific one (UnicodeDecodeError, a PDF library's
error type, ...).

Everything an uploaded file "is" collapses to text here. A parser turns raw
bytes into a list of :class:`ParsedDocument`, and that text flows into the one
existing ``ingest_structured()`` entrypoint, so attachments never grow a second
ingestion path. A file yields a LIST because its natural units differ by format:
one text file is one document, while a spreadsheet is one per sheet and a PDF
one per page. Keeping those units separate gives retrieval a real citation
target ("page 4") instead of one undifferentiated blob.

Adding a file type means adding a parser module and one registry line. It never
means changing the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field


class ParsingError(Exception):
    """Base class for document-parsing seam failures."""


class UnsupportedFileError(ParsingError):
    """Raised when no registered parser handles a file.

    An honest refusal, not a crash: the upload is rejected with a message naming
    what the app can read.
    """


class ParseFailedError(ParsingError):
    """Raised when a parser recognised the file but could not read it.

    Seam-level type: services catch this rather than a decoding error or a
    third-party library's own exception, so the boundary stays format-agnostic.
    Covers a truncated, corrupt, or password-protected file.
    """


class ParserOutOfMemoryError(Exception):
    """Raised when the reader process was killed from outside, for want of memory.

    Deliberately NOT a :class:`ParsingError`. Every error in that family is a
    terminal statement about the FILE - it is corrupt, it is encrypted, nothing
    here can read it - and services treat them as final because retrying an
    unreadable file only wastes a worker. This one says nothing about the file at
    all: something on the machine (a concurrent ingest, a model loading beside it)
    took the memory, and the very same bytes may read perfectly a minute later. So
    it stays outside that hierarchy, propagates past the services that would
    record it as a permanent failure, and reaches the worker's retry.
    """


class ParsedDocument(BaseModel):
    """One addressable unit of text extracted from an uploaded file.

    Attributes:
        ordinal: 0-based position within the file, so a caller can order and
            label units without knowing the format.
        text: The extracted text. May be empty when a unit genuinely holds none
            (a blank page, an empty sheet); callers skip those rather than
            embedding whitespace.
        title: A human label for this unit ("page 3", "Sheet1"), or None when
            the file has exactly one unit and the filename already names it.
    """

    ordinal: int = Field(ge=0)
    text: str
    title: str | None = None


class DocumentParser(ABC):
    """The document-parsing seam. First impl: :class:`PlainTextParser`.

    A parser declares which media types and filename extensions it handles, and
    the registry dispatches on those. Subclasses override :meth:`handles` only
    when the declaration cannot be a plain set (a whole ``text/*`` family, say).
    """

    #: Exact media types this parser claims, lowercase and without parameters.
    media_types: ClassVar[frozenset[str]] = frozenset()
    #: Filename extensions this parser claims, lowercase and dot-prefixed.
    extensions: ClassVar[frozenset[str]] = frozenset()

    def handles(self, media_type: str | None, extension: str) -> bool:
        """Return whether this parser claims a file with this type or extension.

        Args:
            media_type: The declared media type, lowercased and stripped of
                parameters, or None when the client sent none.
            extension: The lowercase, dot-prefixed filename extension, or "".
        """
        if media_type is not None and media_type in self.media_types:
            return True
        return bool(extension) and extension in self.extensions

    def sniff(self, data: bytes) -> bool:
        """Return whether this parser claims a file by its content signature.

        The registry consults this AFTER the declared media type and extension
        and BEFORE the plain-text fallback: a parser that recognises its format's
        magic bytes claims a file even when the client mislabelled it, closing
        the gap where a binary format decodes to garbage text. The default claims
        nothing, so a parser opts in only when it has a reliable signature.

        Args:
            data: The file's raw bytes.
        """
        return False

    @abstractmethod
    async def parse(
        self, data: bytes, *, filename: str, media_type: str | None = None
    ) -> list[ParsedDocument]:
        """Extract text from ``data``.

        Args:
            data: The file's raw bytes.
            filename: The client's filename, for labelling only. It is never a
                path and never decides where anything is read or written.
            media_type: The declared media type, when the client sent one.

        Returns:
            One entry per natural unit of the file, in reading order. A file
            with no extractable text returns an empty list rather than raising.

        Raises:
            ParseFailedError: The file is recognised but unreadable.
        """
        raise NotImplementedError
