"""Local-filesystem StorageBackend, and an in-memory fake for tests.

This is the development and single-node default: a named volume holds a plain
directory tree, so uploaded bytes survive a container recreate with no extra
service to run. MinIO/S3 replaces it in the cloud behind the same seam.

Filesystem calls block, and this is the first seam in the app that touches disk,
so every one of them is handed to a worker thread. A blocking read inside a
request handler would stall the event loop and, with it, every concurrent SSE
stream.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from app.storage.base import (
    InvalidStorageKeyError,
    ObjectNotFoundError,
    StorageBackend,
    StorageUnavailableError,
    StoredObject,
    StoredObjectInfo,
    validate_key,
    validate_prefix,
)


class LocalFsStorage(StorageBackend):
    """Stores objects as files beneath a root directory."""

    def __init__(self, root: str | Path) -> None:
        """Create a backend rooted at ``root``, creating the directory if absent.

        Args:
            root: Directory that will contain every stored object.
        """
        self._root = Path(root).resolve()

    def _safe_path(self, key: str) -> Path:
        """Map an already-validated key to a path proven to sit beneath the root.

        ``validate_key`` excludes traversal at the string level, so this second
        check is defence in depth: resolving the path also follows symlinks, and
        so catches a symlinked ancestor redirecting the write outside the root -
        something no amount of string validation could see.

        ``Path.resolve`` issues real syscalls per component, so this runs inside
        the worker thread with the I/O it guards, never on the event loop.
        """
        path = (self._root / key).resolve()
        if path != self._root and self._root not in path.parents:
            raise InvalidStorageKeyError(f"Storage key escapes the storage root: {key!r}")
        return path

    def _put_sync(self, key: str, data: bytes) -> None:
        """Write ``data`` at ``key`` atomically.

        A reader must never observe a half-written object, so the bytes land in
        a temporary sibling first and are then moved into place; ``os.replace``
        is atomic on both POSIX and Windows.
        """
        path = self._safe_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _get_sync(self, key: str) -> bytes:
        """Read the object at ``key``, treating a directory as a missing object."""
        path = self._safe_path(key)
        # Reading a directory raises IsADirectoryError on POSIX but
        # PermissionError on Windows, so the check is explicit rather than an
        # except clause that would mean different things on different hosts.
        if path.is_dir():
            raise ObjectNotFoundError(f"No object stored at {key!r}")
        return path.read_bytes()

    def _delete_sync(self, key: str) -> None:
        """Remove the object at ``key`` if it is there."""
        self._safe_path(key).unlink(missing_ok=True)

    def _exists_sync(self, key: str) -> bool:
        """Return whether ``key`` names a regular file."""
        return self._safe_path(key).is_file()

    def _list_sync(self, prefix: str) -> list[StoredObjectInfo]:
        """Walk the tree beneath ``prefix`` and describe every object in it.

        Symlinked directories are not followed (``os.walk`` defaults to that),
        so a link planted in the root cannot make the sweep enumerate - and
        therefore consider deleting - files outside it.
        """
        base = self._safe_path(prefix) if prefix else self._root
        if not base.is_dir():
            return []
        found: list[StoredObjectInfo] = []
        for dirpath, dirnames, filenames in os.walk(base):
            # A half-written object is a dot-prefixed temporary sibling (see
            # _put_sync) and no valid key may start with a dot, so nothing
            # hidden is an object. Pruning dirnames in place also stops the walk
            # from descending into hidden directories.
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for name in filenames:
                if name.startswith("."):
                    continue
                path = Path(dirpath) / name
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    # Deleted between the walk and the stat. A listing that
                    # omits it reports the truth, so this is not an error.
                    continue
                found.append(
                    StoredObjectInfo(
                        key=path.relative_to(self._root).as_posix(),
                        size_bytes=stat.st_size,
                        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                    )
                )
        return found

    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> StoredObject:
        """Write ``data`` at ``key``, replacing any existing object."""
        validate_key(key)
        try:
            await asyncio.to_thread(self._put_sync, key, data)
        except OSError as exc:
            raise StorageUnavailableError(f"Could not write to storage: {exc}") from exc
        return StoredObject(key=key, size_bytes=len(data), content_type=content_type)

    async def get(self, key: str) -> bytes:
        """Return the bytes stored at ``key``."""
        validate_key(key)
        try:
            return await asyncio.to_thread(self._get_sync, key)
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(f"No object stored at {key!r}") from exc
        except OSError as exc:
            raise StorageUnavailableError(f"Could not read from storage: {exc}") from exc

    async def delete(self, key: str) -> None:
        """Remove the object at ``key``; deleting a missing key is a no-op."""
        validate_key(key)
        try:
            await asyncio.to_thread(self._delete_sync, key)
        except OSError as exc:
            raise StorageUnavailableError(f"Could not delete from storage: {exc}") from exc

    async def exists(self, key: str) -> bool:
        """Return whether an object is stored at ``key``."""
        validate_key(key)
        try:
            return await asyncio.to_thread(self._exists_sync, key)
        except OSError as exc:
            raise StorageUnavailableError(f"Could not stat storage: {exc}") from exc

    async def list_objects(self, prefix: str = "") -> AsyncIterator[StoredObjectInfo]:
        """Yield every object beneath ``prefix``.

        The walk happens in one shot inside the worker thread and is then
        yielded from memory: a directory tree cannot be paged the way S3 lists
        can, and holding one small record per file is affordable at the scale a
        single-node local disk implies. An S3 backend overrides this with a real
        paginated listing.

        Being an async generator, the body (and so ``validate_prefix``) runs on
        first iteration, not at the call - callers always iterate, and a test
        asserting the rejection must do the same.
        """
        validate_prefix(prefix)
        try:
            found = await asyncio.to_thread(self._list_sync, prefix)
        except OSError as exc:
            raise StorageUnavailableError(f"Could not list storage: {exc}") from exc
        for info in found:
            yield info


class FakeStorageBackend(StorageBackend):
    """In-memory StorageBackend for tests: deterministic, no disk, no cleanup.

    Enforces the same key rules as the real backends so a test cannot pass with
    a key that production would reject.
    """

    def __init__(self) -> None:
        """Create an empty in-memory store."""
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str | None] = {}
        # Public so a test can age an object into the past. The blob sweep only
        # deletes what has sat unreferenced for a while, and a fake that could
        # only ever report "just now" would make that rule untestable.
        self.modified_at: dict[str, datetime] = {}

    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> StoredObject:
        """Record ``data`` under ``key``, replacing any existing object."""
        validate_key(key)
        self.objects[key] = data
        self.content_types[key] = content_type
        self.modified_at[key] = datetime.now(UTC)
        return StoredObject(key=key, size_bytes=len(data), content_type=content_type)

    async def get(self, key: str) -> bytes:
        """Return the bytes recorded under ``key``."""
        validate_key(key)
        try:
            return self.objects[key]
        except KeyError as exc:
            raise ObjectNotFoundError(f"No object stored at {key!r}") from exc

    async def delete(self, key: str) -> None:
        """Forget ``key``; deleting a missing key is a no-op."""
        validate_key(key)
        self.objects.pop(key, None)
        self.content_types.pop(key, None)
        self.modified_at.pop(key, None)

    async def exists(self, key: str) -> bool:
        """Return whether ``key`` was recorded."""
        validate_key(key)
        return key in self.objects

    async def list_objects(self, prefix: str = "") -> AsyncIterator[StoredObjectInfo]:
        """Yield every recorded object beneath ``prefix``, key order.

        Matching is by whole segments, exactly as a directory walk behaves, so
        the fake cannot pass a test that the real backend would fail. A prefix
        naming an object EXACTLY yields nothing, for the same reason: on disk
        that prefix is a file rather than a directory, and on S3 the request
        goes out as ``"key/"``. The fake used to yield it, which was the one
        place its listing disagreed with both real backends.
        """
        validate_prefix(prefix)
        for key in sorted(self.objects):
            if prefix and not key.startswith(f"{prefix}/"):
                continue
            yield StoredObjectInfo(
                key=key,
                size_bytes=len(self.objects[key]),
                modified_at=self.modified_at[key],
            )
