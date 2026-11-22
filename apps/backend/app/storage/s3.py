"""S3-compatible StorageBackend: AWS S3, MinIO, or anything speaking the API.

The cloud counterpart to :class:`app.storage.local.LocalFsStorage`. A single
node can keep its uploads on a named volume; several backend replicas cannot,
because the replica that answers a download is rarely the one that took the
upload. Object storage is the seam's reason for existing, and this is the impl
that makes it real.

aiobotocore rather than boto3: boto3's client blocks, and a blocking network
call inside a request handler stalls the event loop and every concurrent SSE
stream with it. aiobotocore drives the same botocore model layer over aiohttp,
so the request shapes and the error types are botocore's familiar ones while
the I/O is awaited.

Nothing botocore-typed leaves this module. Services above the seam catch
:class:`StorageUnavailableError` and :class:`ObjectNotFoundError`; a
``ClientError`` escaping to them would make the boundary a lie and would couple
the garbage collector to a vendor SDK.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC
from typing import Any

import aiohttp
from aiobotocore.config import AioConfig
from aiobotocore.session import AioSession, get_session
from botocore.exceptions import BotoCoreError, ClientError

from app.storage.base import (
    ObjectNotFoundError,
    StorageBackend,
    StorageUnavailableError,
    StoredObject,
    StoredObjectInfo,
    validate_key,
    validate_prefix,
)

# Everything a call can fail with that is not the service answering properly.
# aiohttp is in here because aiobotocore drives it and does not wrap every one
# of its errors on the way out: a connection dropped part-way through a response
# body surfaces as aiohttp's own ClientPayloadError, which is neither a
# BotoCoreError nor an OSError. Without it, a MinIO restart during a download
# would send a raw transport exception straight through the seam and turn an
# honest "the file could not be read" into a 500.
_TRANSPORT_ERRORS = (BotoCoreError, aiohttp.ClientError, OSError)

# S3 reports a missing object or a missing bucket through several spellings: the
# GET operations name what was missing, while the HEAD operations have no
# response body and so can only report the status line. All of them mean the
# same thing to this seam - there is nothing stored there.
_MISSING_CODES = frozenset({"NoSuchKey", "NoSuchBucket", "404", "NotFound"})

# HeadBucket needs s3:ListBucket, which a least-privilege policy granting only
# object operations does not include. A refusal is not "no bucket": the bucket
# is there, this credential simply may not inspect it, and creating it would be
# refused too. Carrying on lets the operation the caller actually wanted give
# the real answer, instead of every upload failing forever on the bootstrap.
_FORBIDDEN_CODES = frozenset({"AccessDenied", "403", "Forbidden", "AllAccessDisabled"})

# Two processes (the API and the worker) can reach for the bucket at the same
# instant on a fresh deployment, and both get as far as CreateBucket. The loser
# is told the bucket is already theirs, which is not a failure - it is the
# outcome it wanted.
_BUCKET_EXISTS_CODES = frozenset({"BucketAlreadyOwnedByYou", "OperationAborted"})

# S3's own ceiling on one ListObjectsV2 page. Kept as the default so a sweep of
# a large store makes as few round trips as the service allows.
MAX_LIST_PAGE_SIZE = 1000

# How long to wait for the service, and how hard to try. botocore's defaults are
# five attempts on an exponential backoff, which turns a refused connection into
# a NINETEEN-SECOND wait (measured). An attachment read happens on the chat
# request path, so a store that is simply down has to be reported quickly rather
# than absorbed - the user gets an honest failure instead of a stalled turn.
DEFAULT_MAX_ATTEMPTS = 3
_CONNECT_TIMEOUT_SECONDS = 10.0
# Generous by comparison, because this one is paid per BYTE: the upload cap is
# 25 MB, and a slow link moving a large attachment is not a failing link.
_READ_TIMEOUT_SECONDS = 60.0


def _error_code(exc: ClientError) -> str:
    """Return the S3 error code carried by a ``ClientError``, or an empty string."""
    error = exc.response.get("Error") or {}
    return str(error.get("Code", ""))


def _unavailable(action: str, exc: Exception) -> StorageUnavailableError:
    """Wrap any transport or service failure as the seam's own exception.

    The cause message is kept terse and is chained rather than reformatted: a
    botocore error string names the endpoint and the operation, which is what an
    operator reading the log needs, and inventing a friendlier wording would
    only lose that.

    Args:
        action: What was being attempted, for the message.
        exc: The underlying botocore or transport error.

    Returns:
        A seam-level exception ready to raise.
    """
    return StorageUnavailableError(f"Could not {action}: {type(exc).__name__}: {exc}")


class S3Storage(StorageBackend):
    """Stores objects in one bucket of an S3-compatible service."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        page_size: int = MAX_LIST_PAGE_SIZE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        session: AioSession | None = None,
    ) -> None:
        """Create a backend addressing ``bucket``, opening no connection yet.

        Construction performs no I/O, matching every other factory-built
        singleton: a storage service that is still booting must not be able to
        take the API process down with it (see app.core.factories).

        Args:
            bucket: The bucket every object lives in.
            endpoint_url: Base URL of the service; ``None`` uses AWS's own
                endpoint for ``region``. MinIO sets http://minio:9000.
            region: The region to sign for, and the location a created bucket
                is placed in.
            access_key_id: Access key, or ``None`` to let botocore's own
                credential chain (instance role, shared config) supply one.
            secret_access_key: The matching secret.
            page_size: How many keys one listing page carries. The service caps
                this at 1000; lowering it only costs round trips.
            max_attempts: Total tries per operation, retries included. 1
                disables retrying, which is what a deployment wants when it
                would rather fail a turn than delay it.
            session: An aiobotocore session to build clients from. Injected by
                tests; otherwise this instance owns one.
        """
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._region = region
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._page_size = max(1, min(page_size, MAX_LIST_PAGE_SIZE))
        self._config = AioConfig(
            connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            read_timeout=_READ_TIMEOUT_SECONDS,
            retries={"max_attempts": max(1, max_attempts), "mode": "standard"},
            # Path style ("host/bucket/key") rather than virtual-hosted
            # ("bucket.host/key"). MinIO serves one hostname and no wildcard
            # DNS, so a virtual-hosted URL simply does not resolve. botocore
            # already picks path style for a custom endpoint, but that is a
            # default rather than a promise, and this is the deployment it
            # would break.
            s3={"addressing_style": "path"} if endpoint_url else {},
        )
        self._session = session or get_session()
        self._client_ctx: Any = None
        self._client: Any = None
        self._client_lock = asyncio.Lock()
        self._bucket_ready = False
        self._bucket_lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        """Return this instance's client, creating it on first use.

        One client per backend instance, not one per call: the client owns a
        connection pool, and rebuilding it per operation would pay a TLS
        handshake for every attachment byte the app moves. The lock makes the
        first concurrent burst build exactly one.

        Building it is where botocore resolves credentials and the endpoint, so
        a bad AWS profile or an unresolvable region raises HERE rather than on
        the call - and has to be translated like any other failure, or the seam
        would leak a vendor exception through the one path that never touches
        the network.

        Raises:
            StorageUnavailableError: The client could not be constructed.
        """
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                try:
                    # aiobotocore hands back an async context manager rather
                    # than a client, so the enter is held and released by
                    # aclose().
                    ctx = self._session.create_client(
                        "s3",
                        endpoint_url=self._endpoint_url,
                        region_name=self._region,
                        aws_access_key_id=self._access_key_id,
                        aws_secret_access_key=self._secret_access_key,
                        config=self._config,
                    )
                    self._client_ctx = ctx
                    self._client = await ctx.__aenter__()
                except (ClientError, ValueError, *_TRANSPORT_ERRORS) as exc:
                    self._client_ctx = None
                    raise _unavailable("open a storage client", exc) from exc
        return self._client

    def _create_bucket_kwargs(self) -> dict[str, Any]:
        """Return the region argument CreateBucket needs, if any.

        us-east-1 is the one region that must NOT be named: S3 rejects a
        LocationConstraint that repeats its default.
        """
        if self._region and self._region != "us-east-1":
            return {"CreateBucketConfiguration": {"LocationConstraint": self._region}}
        return {}

    async def _ensure_bucket(self, client: Any) -> None:
        """Create the bucket if it is not there, once per instance.

        Mirrors the local backend, which makes its root directory on the first
        write rather than at construction. A deployment whose bucket is
        pre-provisioned pays one HeadBucket for the life of the process.
        """
        if self._bucket_ready:
            return
        async with self._bucket_lock:
            if self._bucket_ready:
                return
            try:
                await client.head_bucket(Bucket=self._bucket)
            except ClientError as exc:
                code = _error_code(exc)
                if code in _MISSING_CODES:
                    await self._create_bucket(client)
                elif code not in _FORBIDDEN_CODES:
                    raise _unavailable(f"reach bucket {self._bucket!r}", exc) from exc
            except _TRANSPORT_ERRORS as exc:
                raise _unavailable(f"reach bucket {self._bucket!r}", exc) from exc
            self._bucket_ready = True

    async def _create_bucket(self, client: Any) -> None:
        """Create the bucket, tolerating a concurrent creator that got there first."""
        try:
            await client.create_bucket(Bucket=self._bucket, **self._create_bucket_kwargs())
        except ClientError as exc:
            # BucketAlreadyExists (as opposed to ...OwnedByYou) means the name
            # belongs to someone else, which no retry fixes - it is
            # configuration, and it surfaces rather than hides.
            if _error_code(exc) not in _BUCKET_EXISTS_CODES:
                raise _unavailable(f"create bucket {self._bucket!r}", exc) from exc
        except _TRANSPORT_ERRORS as exc:
            raise _unavailable(f"create bucket {self._bucket!r}", exc) from exc

    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> StoredObject:
        """Write ``data`` at ``key``, replacing any existing object."""
        validate_key(key)
        client = await self._get_client()
        await self._ensure_bucket(client)
        extra: dict[str, Any] = {"ContentType": content_type} if content_type else {}
        try:
            await client.put_object(Bucket=self._bucket, Key=key, Body=data, **extra)
        except (ClientError, *_TRANSPORT_ERRORS) as exc:
            raise _unavailable("write to storage", exc) from exc
        return StoredObject(key=key, size_bytes=len(data), content_type=content_type)

    async def get(self, key: str) -> bytes:
        """Return the bytes stored at ``key``."""
        validate_key(key)
        client = await self._get_client()
        try:
            response = await client.get_object(Bucket=self._bucket, Key=key)
            # NOT `async with response["Body"]`: aiobotocore's body is a proxy
            # whose __aenter__ hands back the RAW aiohttp response, so `as body`
            # would silently unwrap it - reading through the wrapper is what
            # verifies the content length and the checksum, and what translates
            # a mid-body transport failure into a botocore error. close() is the
            # sync call that returns the connection to the pool.
            body = response["Body"]
            try:
                return await body.read()
            finally:
                body.close()
        except ClientError as exc:
            if _error_code(exc) in _MISSING_CODES:
                raise ObjectNotFoundError(f"No object stored at {key!r}") from exc
            raise _unavailable("read from storage", exc) from exc
        except _TRANSPORT_ERRORS as exc:
            raise _unavailable("read from storage", exc) from exc

    async def delete(self, key: str) -> None:
        """Remove the object at ``key``; deleting a missing key is a no-op.

        S3 answers DeleteObject with success whether or not the key was there,
        so idempotence is mostly free; the missing-code branch covers the case
        where the bucket itself has yet to be created.
        """
        validate_key(key)
        client = await self._get_client()
        try:
            await client.delete_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _error_code(exc) in _MISSING_CODES:
                return
            raise _unavailable("delete from storage", exc) from exc
        except _TRANSPORT_ERRORS as exc:
            raise _unavailable("delete from storage", exc) from exc

    async def exists(self, key: str) -> bool:
        """Return whether an object is stored at ``key``."""
        validate_key(key)
        client = await self._get_client()
        try:
            await client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _error_code(exc) in _MISSING_CODES:
                return False
            raise _unavailable("stat storage", exc) from exc
        except _TRANSPORT_ERRORS as exc:
            raise _unavailable("stat storage", exc) from exc
        return True

    async def list_objects(self, prefix: str = "") -> AsyncIterator[StoredObjectInfo]:
        """Yield every object beneath ``prefix``, a page at a time.

        The whole store can be far larger than memory, and the one caller is a
        sweep over all of it, so pages are yielded as they arrive rather than
        collected first.

        A prefix names whole segments, so it is sent to S3 with a trailing
        slash: listing ``"abc"`` asks for ``"abc/"`` and therefore cannot reach
        ``"abcd/x"``. That is the local backend's behaviour (a directory walk
        rooted at ``abc/``) expressed in S3's flat namespace.

        A bucket that does not exist yet lists as empty rather than failing,
        matching the local backend's missing-root behaviour: only a write
        creates the store.

        Being an async generator, the body (and so ``validate_prefix``) runs on
        first iteration, not at the call.
        """
        validate_prefix(prefix)
        client = await self._get_client()
        s3_prefix = f"{prefix}/" if prefix else ""
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self._bucket,
            Prefix=s3_prefix,
            PaginationConfig={"PageSize": self._page_size},
        )
        try:
            async for page in pages:
                for obj in page.get("Contents") or []:
                    yield StoredObjectInfo(
                        key=obj["Key"],
                        size_bytes=obj["Size"],
                        # botocore parses LastModified into a tz-aware datetime,
                        # but not necessarily in UTC. The blob sweep compares it
                        # against a UTC cutoff, so normalise here rather than
                        # trusting the service's choice of offset.
                        modified_at=obj["LastModified"].astimezone(UTC),
                    )
        except ClientError as exc:
            if _error_code(exc) in _MISSING_CODES:
                return
            raise _unavailable("list storage", exc) from exc
        except _TRANSPORT_ERRORS as exc:
            raise _unavailable("list storage", exc) from exc

    async def aclose(self) -> None:
        """Release the connection pool; called from the API and worker shutdowns.

        Both shutdown paths look for an ``aclose`` on whatever they built, so
        this is the hook that keeps an S3-backed deployment from leaking sockets
        on every reload. The local backend has no client and so has none.

        Takes the same lock the client is built under: a shutdown that lands
        while a first request is mid-build would otherwise find nothing to
        close, and the client arriving a moment later would never be released.
        The bucket check resets with it, so a reopened backend bootstraps again
        rather than assuming a bucket it has not looked at since.
        """
        async with self._client_lock:
            ctx, self._client_ctx = self._client_ctx, None
            self._client = None
            self._bucket_ready = False
            if ctx is not None:
                await ctx.__aexit__(None, None, None)
