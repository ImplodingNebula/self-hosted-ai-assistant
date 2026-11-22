"""Object-storage seam: the StorageBackend abstraction and its value types.

Mirrors the RAG and search seams (app/providers/rag/base.py,
app/providers/search/base.py): an ABC with a few abstract async methods, a
pydantic value type, and seam-level exceptions that services catch instead of
any backend-specific error (OSError, botocore, ...). The first impl writes to a
local directory; MinIO/S3 drops in behind the same boundary for the cloud.

Uploaded bytes are addressed by an opaque ``key`` chosen by the caller, never by
the name the client supplied. A hostile filename must not be able to decide
where bytes land, so :func:`validate_key` enforces a deliberately narrow shape
and every backend is required to reject anything else.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime

from pydantic import BaseModel

# A key is a '/'-joined path of conservative segments. Backslashes, drive
# letters, absolute paths, '.', '..' and empty segments are all excluded by
# construction rather than by a blocklist, because a blocklist of traversal
# spellings is exactly the thing that gets outsmarted.
#
# The pattern ends in \Z, not $: in Python '$' also matches just before a
# trailing newline, so '^...$' would accept "attachment\n" - a validation bypass
# in the one function whose entire job is to be the hard gate.
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\Z")
MAX_KEY_LENGTH = 512

# Windows resolves these names to devices no matter which directory they sit in,
# and silently strips a trailing dot or space from a filename (so "a." and "a"
# collide). Neither can escape the root, but a storage key must address exactly
# one object on every platform the app runs on.
_RESERVED_WINDOWS_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)


class StorageError(Exception):
    """Base class for storage seam failures."""


class StorageUnavailableError(StorageError):
    """Raised when the storage backend cannot be reached or answers abnormally.

    Seam-level type: services catch this (not OSError or a vendor SDK error),
    keeping the boundary backend-agnostic.
    """


class ObjectNotFoundError(StorageError):
    """Raised when a key has no stored object.

    Deleting a missing key is NOT an error (delete is idempotent); only reading
    one is.
    """


class InvalidStorageKeyError(StorageError):
    """Raised when a key is not a safe, relative, backend-neutral path."""


def validate_key(key: str) -> str:
    """Return ``key`` unchanged if it is a safe storage key, else raise.

    A valid key is a non-empty, '/'-joined sequence of segments, each starting
    with an alphanumeric and otherwise limited to letters, digits, dot, dash and
    underscore, ending in neither a dot nor a Windows device name. This admits
    the keys the app actually mints
    (``{tenant_id}/{conversation_id}/{attachment_id}``) and excludes absolute
    paths, parent traversal, empty segments, and Windows separators.

    Args:
        key: The candidate storage key.

    Returns:
        The same key, once validated.

    Raises:
        InvalidStorageKeyError: The key could address something other than a
            plain object beneath the backend's root.
    """
    if not key or len(key) > MAX_KEY_LENGTH:
        raise InvalidStorageKeyError(f"Storage key must be 1-{MAX_KEY_LENGTH} characters.")
    for segment in key.split("/"):
        if not _SEGMENT.match(segment):
            raise InvalidStorageKeyError(f"Storage key has an unsafe segment: {key!r}")
        if segment.endswith("."):
            raise InvalidStorageKeyError(f"Storage key segment ends in a dot: {key!r}")
        if segment.split(".")[0].upper() in _RESERVED_WINDOWS_NAMES:
            raise InvalidStorageKeyError(f"Storage key uses a reserved device name: {key!r}")
    return key


def validate_prefix(prefix: str) -> str:
    """Return ``prefix`` unchanged if it is a safe listing prefix, else raise.

    A prefix is either empty (list everything) or a key-shaped path of whole
    segments. Partial segments are deliberately NOT a thing: a prefix names a
    subtree ("this tenant", "this conversation"), never a fragment of a name, so
    listing ``"abc"`` cannot reach into ``"abcd/..."``.

    Args:
        prefix: The candidate listing prefix, possibly empty.

    Returns:
        The same prefix, once validated.

    Raises:
        InvalidStorageKeyError: The prefix could address something other than a
            subtree beneath the backend's root.
    """
    if prefix == "":
        return prefix
    return validate_key(prefix)


class StoredObject(BaseModel):
    """Metadata describing one object that was written to storage."""

    key: str
    size_bytes: int
    # Recorded by the caller, not sniffed by the backend: the local-fs impl has
    # nowhere to persist it, and the Attachment row is the source of truth.
    content_type: str | None = None


class StoredObjectInfo(BaseModel):
    """One object as a listing sees it: what it is called, how big, how old.

    ``content_type`` is absent on purpose - the local-fs backend never persisted
    it, so a listing cannot report it honestly. ``modified_at`` is what the
    garbage collector needs and every backend can supply (a file's mtime, S3's
    LastModified): it is how an orphaned blob is told apart from one whose row
    is a few milliseconds from being committed.
    """

    key: str
    size_bytes: int
    modified_at: datetime


class StorageBackend(ABC):
    """The object-storage seam. First impl: :class:`LocalFsStorage`."""

    @abstractmethod
    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> StoredObject:
        """Write ``data`` at ``key``, replacing any existing object.

        Args:
            key: A key accepted by :func:`validate_key`.
            data: The bytes to store.
            content_type: The caller's declared media type, echoed back on the
                returned metadata.

        Returns:
            Metadata for the object just written.

        Raises:
            InvalidStorageKeyError: ``key`` is unsafe.
            StorageUnavailableError: The backend could not be written to.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Return the bytes stored at ``key``.

        Raises:
            InvalidStorageKeyError: ``key`` is unsafe.
            ObjectNotFoundError: Nothing is stored at ``key``.
            StorageUnavailableError: The backend could not be read.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove the object at ``key``; deleting a missing key is a no-op.

        Idempotent on purpose: cleanup paths (conversation delete, an orphan
        reaper) must be safe to retry after a partial failure.

        Raises:
            InvalidStorageKeyError: ``key`` is unsafe.
            StorageUnavailableError: The backend could not be written to.
        """
        raise NotImplementedError

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return whether an object is stored at ``key``.

        Raises:
            InvalidStorageKeyError: ``key`` is unsafe.
            StorageUnavailableError: The backend could not be read.
        """
        raise NotImplementedError

    @abstractmethod
    def list_objects(self, prefix: str = "") -> AsyncIterator[StoredObjectInfo]:
        """Yield every object beneath ``prefix`` (empty lists the whole store).

        An iterator rather than a list because the caller is a sweep over the
        entire store: S3 pages its listings, and a backend must be free to hand
        results back a page at a time instead of pulling the whole inventory
        into memory first.

        Order is unspecified. So is the behaviour of a concurrent write during
        the walk - the one caller (the blob sweep) tolerates both by only ever
        deleting objects that are old AND unreferenced.

        Args:
            prefix: A subtree to list, as whole key segments; ``""`` for all.

        Yields:
            Metadata for each object found.

        Raises:
            InvalidStorageKeyError: ``prefix`` is unsafe.
            StorageUnavailableError: The backend could not be listed.
        """
        raise NotImplementedError
