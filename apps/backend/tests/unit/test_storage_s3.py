"""S3-specific behaviour: error mapping, bucket bootstrap, and the wire details.

The shared contract lives in test_storage_local.py, which runs every one of its
tests against this backend too. What is here is the half that has no analogue in
a directory tree: what happens when the service answers badly, when the bucket is
not there yet, and what exactly goes out on the wire.

Several tests synthesise a service response through botocore's event system.
That is not a stand-in for the adapter - the real ``S3Storage`` methods run, and
they see exactly the exception botocore would raise for that response. It is the
only way to produce answers (a lost CreateBucket race, an AccessDenied) that a
local moto server will not.
"""

from __future__ import annotations

import asyncio
import base64
import socket
import zlib
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiobotocore.session import AioSession, get_session
from botocore.exceptions import ClientError, EndpointConnectionError

from app.storage.base import (
    ObjectNotFoundError,
    StorageUnavailableError,
)
from app.storage.s3 import S3Storage
from tests.s3_server import ACCESS_KEY, SECRET_KEY, make_s3_storage, s3_endpoint  # noqa: F401

KEY = "11111111-1111-1111-1111-111111111111/conv/att.bin"

# Every operation the adapter issues on behalf of one seam method, so a failure
# can be injected into each in turn.
_OPERATIONS = ["PutObject", "GetObject", "DeleteObject", "HeadObject", "ListObjectsV2"]


def _client_error(code: str, operation: str) -> ClientError:
    """Build the exception botocore raises for an S3 error response."""
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


def _raising(exc: Exception):  # type: ignore[no-untyped-def]
    """Return a botocore event handler that raises ``exc`` instead of calling out."""

    def handler(**_: object) -> None:
        raise exc

    return handler


async def _invoke(backend: S3Storage, operation: str) -> None:
    """Drive the seam method that issues ``operation``."""
    if operation == "PutObject":
        await backend.put(KEY, b"x")
    elif operation == "GetObject":
        await backend.get(KEY)
    elif operation == "DeleteObject":
        await backend.delete(KEY)
    elif operation == "HeadObject":
        await backend.exists(KEY)
    else:
        [info async for info in backend.list_objects()]


def _closed_port() -> int:
    """Bind and release a port, so nothing is listening on it."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
async def storage(s3_endpoint: str) -> AsyncIterator[S3Storage]:  # noqa: F811
    """An S3 backend with a bucket of its own, released afterwards."""
    backend = make_s3_storage(s3_endpoint)
    try:
        yield backend
    finally:
        await backend.aclose()


@asynccontextmanager
async def _serving(response: bytes) -> AsyncIterator[str]:
    """Answer every request with ``response`` verbatim, and yield the base URL.

    moto answers correctly or not at all, so a malformed reply - the shape a
    store restarting mid-download or a corrupted object actually takes - can
    only be produced by speaking HTTP badly on purpose.
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(response)
            await writer.drain()
        except (OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            # The client hung up first, which is fine: the point was made.
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
async def truncating_server() -> AsyncIterator[str]:
    """A server that promises 100 bytes, sends 5, and hangs up."""
    async with _serving(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort") as url:
        yield url


@pytest.fixture
async def corrupting_server() -> AsyncIterator[str]:
    """A server whose object does not match the checksum it advertises."""
    body = b"hello"
    wrong = base64.b64encode(zlib.crc32(b"different bytes entirely").to_bytes(4, "big")).decode()
    response = (
        "HTTP/1.1 200 OK\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"x-amz-checksum-crc32: {wrong}\r\n"
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + body
    async with _serving(response) as url:
        yield url


@pytest.fixture
def recording_session() -> Iterator[tuple[AioSession, list[str]]]:
    """A session that records every S3 operation the adapter issues."""
    session = get_session()
    calls: list[str] = []

    def record(model, **_: object) -> None:
        calls.append(model.name)

    session.register("before-call.s3", record)
    yield session, calls


# ------------------------------------------------------------- error mapping


async def test_a_genuinely_unreachable_service_reports_as_unavailable() -> None:
    """The end-to-end proof: a real socket, a real refusal, a seam exception.

    Services above the seam catch StorageUnavailableError. If a botocore
    EndpointConnectionError escaped instead, the attachment purge would abort on
    an exception it does not handle and the chat path would answer 500 rather
    than saying plainly that the file could not be read.

    One method only, because a refused connection to a closed loopback port
    costs seconds on some hosts. The per-operation coverage is the injected
    test below, which exercises the same mapping without the wait.
    """
    backend = S3Storage(
        "attachments",
        endpoint_url=f"http://127.0.0.1:{_closed_port()}",
        access_key_id=ACCESS_KEY,
        secret_access_key=SECRET_KEY,
        # No retries: this asserts the mapping, and the backoff schedule would
        # only make it slower.
        max_attempts=1,
    )
    try:
        with pytest.raises(StorageUnavailableError):
            await backend.exists(KEY)
    finally:
        await backend.aclose()


@pytest.mark.parametrize("operation", _OPERATIONS)
async def test_a_transport_failure_never_escapes_as_a_botocore_error(
    s3_endpoint: str,  # noqa: F811
    operation: str,
) -> None:
    """Every seam method must translate a dead connection, not just the first.

    A partially mapped adapter is the dangerous shape: the paths a developer
    happened to try look clean, and the one that was missed takes down a
    background sweep months later.
    """
    session = get_session()
    session.register(
        f"before-call.s3.{operation}",
        _raising(EndpointConnectionError(endpoint_url=s3_endpoint)),
    )
    backend = make_s3_storage(s3_endpoint, session=session)
    try:
        with pytest.raises(StorageUnavailableError):
            await _invoke(backend, operation)
    finally:
        await backend.aclose()


@pytest.mark.parametrize("operation", _OPERATIONS)
async def test_a_service_error_never_escapes_as_a_client_error(
    s3_endpoint: str,  # noqa: F811
    operation: str,
) -> None:
    """AccessDenied is a real answer, and it must not leak botocore's type.

    A misconfigured bucket policy answers every call this way. The seam promises
    callers two exception types; a ClientError arriving instead would couple the
    garbage collector and the attachment service to a vendor SDK, which is the
    thing the seam exists to prevent.
    """
    session = get_session()
    session.register(
        f"before-call.s3.{operation}",
        _raising(_client_error("AccessDenied", operation)),
    )
    backend = make_s3_storage(s3_endpoint, session=session)
    try:
        with pytest.raises(StorageUnavailableError):
            await _invoke(backend, operation)
    finally:
        await backend.aclose()


async def test_a_client_that_cannot_even_be_built_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    s3_endpoint: str,  # noqa: F811
) -> None:
    """Credentials and endpoints resolve when the client is built, not when it is used.

    A bad AWS_PROFILE raises there, on the one path that never touches the
    network - and it sits BEFORE every per-operation try block, so it is the
    easiest place in the module for a botocore exception to escape unnoticed.
    """
    monkeypatch.setenv("AWS_PROFILE", "definitely-not-a-real-profile")
    # No explicit credentials, so botocore consults the profile and fails.
    backend = make_s3_storage(s3_endpoint, access_key_id=None, secret_access_key=None)
    try:
        with pytest.raises(StorageUnavailableError, match="open a storage client"):
            await backend.exists(KEY)
    finally:
        await backend.aclose()


async def test_a_missing_key_reads_as_not_found_not_unavailable(storage: S3Storage) -> None:
    # The two are handled completely differently upstream: not-found is a 404 to
    # the user, unavailable is a retry. Collapsing them would make a deleted
    # attachment look like an outage.
    await storage.put("11111111-1111-1111-1111-111111111111/conv/other.bin", b"x")

    with pytest.raises(ObjectNotFoundError):
        await storage.get(KEY)


async def test_a_bucket_that_does_not_exist_yet_reads_as_empty(s3_endpoint: str) -> None:  # noqa: F811
    """Only a write creates the store, exactly as the local backend behaves.

    A read before the first upload must not look like an outage: on the local
    backend the root directory simply is not there yet, and the seam reports
    nothing stored. The S3 adapter has to say the same thing about a bucket it
    has not created.
    """
    backend = make_s3_storage(s3_endpoint)
    try:
        assert await backend.exists(KEY) is False
        assert [info async for info in backend.list_objects()] == []
        await backend.delete(KEY)  # idempotent, even with no bucket to delete from
        with pytest.raises(ObjectNotFoundError):
            await backend.get(KEY)
    finally:
        await backend.aclose()


# ----------------------------------------------------------- bucket bootstrap


async def test_the_bucket_is_created_once_and_only_by_a_write(
    s3_endpoint: str,  # noqa: F811
    recording_session: tuple[AioSession, list[str]],
) -> None:
    """Construction opens nothing, and the check does not repeat per upload.

    A HeadBucket on every single upload is a round trip bought for nothing, and
    the factory contract says building the backend performs no I/O at all - a
    storage service still booting must not be able to take the API down with it.
    """
    session, calls = recording_session
    backend = make_s3_storage(s3_endpoint, session=session)
    assert calls == []  # construction is free

    try:
        await backend.put(KEY, b"one")
        await backend.put(KEY, b"two")
        await backend.put(f"{KEY}.2", b"three")
    finally:
        await backend.aclose()

    assert calls.count("HeadBucket") == 1
    assert calls.count("CreateBucket") == 1
    assert calls.count("PutObject") == 3


@pytest.mark.parametrize("code", ["BucketAlreadyOwnedByYou", "OperationAborted"])
async def test_losing_the_create_bucket_race_is_not_a_failure(
    s3_endpoint: str,  # noqa: F811
    code: str,
) -> None:
    """Two processes bootstrap the same fresh bucket; the loser must not care.

    The API and the worker both build a backend and both may take their first
    upload at the same instant. BucketAlreadyOwnedByYou is the outcome the loser
    wanted, so treating it as an error would fail a perfectly good upload on
    every fresh deployment.

    The error is raised AFTER the call, which is what makes this the real race
    rather than a broken create: the bucket genuinely exists once the adapter
    moves on, exactly as it would if the other process had won by a millisecond.
    OperationAborted is the same race reported by a service that noticed the
    conflicting create rather than the finished one.
    """
    session = get_session()
    session.register(
        "after-call.s3.CreateBucket",
        _raising(_client_error(code, "CreateBucket")),
    )
    backend = make_s3_storage(s3_endpoint, session=session)
    try:
        await backend.put(KEY, b"loser")

        assert await backend.get(KEY) == b"loser"
    finally:
        await backend.aclose()


async def test_a_bucket_owned_by_somebody_else_is_reported_not_swallowed(
    s3_endpoint: str,  # noqa: F811
) -> None:
    """BucketAlreadyExists means the global name belongs to another account.

    No retry fixes that and no upload will ever succeed against it, so it has to
    surface rather than be filed under "the race we tolerate".

    The message matters as much as the type here: swallowed, the upload fails a
    breath later on NoSuchBucket, which is the SAME seam exception saying
    something quite different and sends an operator hunting a network fault
    instead of renaming a bucket.
    """
    session = get_session()
    session.register(
        "before-call.s3.CreateBucket",
        _raising(_client_error("BucketAlreadyExists", "CreateBucket")),
    )
    backend = make_s3_storage(s3_endpoint, session=session)
    try:
        with pytest.raises(StorageUnavailableError, match="create bucket"):
            await backend.put(KEY, b"x")
    finally:
        await backend.aclose()


# ------------------------------------------------------------ the wire itself


async def test_a_listing_prefix_goes_out_as_a_whole_segment(
    s3_endpoint: str,  # noqa: F811
) -> None:
    """S3 matches a listing prefix as a raw string, so the slash is the fence.

    Sent as "abc" rather than "abc/", the request would return every key under
    "abcd/" as well - and the one caller is a sweep that DELETES what it finds
    unreferenced. This is the assertion that the fence is on the wire, not only
    in the result of a test whose fixture happened to have no sibling.
    """
    session = get_session()
    sent: list[str | None] = []
    session.register(
        "before-parameter-build.s3.ListObjectsV2",
        lambda params, **_: sent.append(params.get("Prefix")),
    )
    backend = make_s3_storage(s3_endpoint, session=session)
    try:
        [info async for info in backend.list_objects("abc")]
        [info async for info in backend.list_objects()]
    finally:
        await backend.aclose()

    assert sent == ["abc/", ""]


async def test_content_type_is_recorded_on_the_stored_object(
    storage: S3Storage,
    s3_endpoint: str,  # noqa: F811
) -> None:
    """Unlike the local backend, S3 can hold the media type - so it should.

    Nothing reads it back through the seam (StoredObjectInfo deliberately has no
    content_type, because the local backend could never report one honestly),
    but an object browsed in a console or served through a signed URL is only
    usable if it carries the right type.
    """
    stored = await storage.put(KEY, b"%PDF-1.4", content_type="application/pdf")

    assert stored.content_type == "application/pdf"

    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=s3_endpoint,
        region_name="us-east-1",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    ) as client:
        head = await client.head_object(Bucket=storage._bucket, Key=KEY)

    assert head["ContentType"] == "application/pdf"


async def test_one_client_serves_every_call_and_aclose_can_be_repeated(
    s3_endpoint: str,  # noqa: F811
) -> None:
    """The pool is built once and released once.

    A client per operation would pay a fresh connection for every attachment
    byte the app moves, and an aclose that only works the first time would raise
    during shutdown - after the API has already stopped serving, where nobody
    would see why.
    """
    backend = make_s3_storage(s3_endpoint)
    await backend.put(KEY, b"x")
    first = backend._client
    await backend.get(KEY)

    assert backend._client is first
    assert first is not None

    await backend.aclose()
    await backend.aclose()

    assert backend._client is None


async def test_a_reopened_backend_checks_its_bucket_again(
    s3_endpoint: str,  # noqa: F811
    recording_session: tuple[AioSession, list[str]],
) -> None:
    """Closing the client throws away what it learned, including the bucket check.

    The flag records "this CLIENT has seen the bucket". Left set across an
    aclose, a reopened backend would write to a bucket it has not looked at
    since - and on a fresh store that write is the one that was supposed to
    create it.
    """
    session, calls = recording_session
    backend = make_s3_storage(s3_endpoint, session=session)
    try:
        await backend.put(KEY, b"one")
        await backend.aclose()
        await backend.put(KEY, b"two")
    finally:
        await backend.aclose()

    assert calls.count("HeadBucket") == 2


async def test_concurrent_first_calls_build_a_single_client(
    s3_endpoint: str,  # noqa: F811
) -> None:
    """Every upload on a cold process arrives at once during a boot burst.

    Without the lock each would build its own client and all but one would be
    dropped unclosed, leaking a connection pool per request. Counting the builds
    is the only assertion that can see it - the puts all succeed either way.

    The delay is not decoration. Measured on this runtime, client creation with
    static credentials never yields to the event loop, so a plain gather runs
    each build to completion and the race cannot happen; a credential chain that
    reaches for an instance role or a config file DOES await, which is the
    deployment the lock is for. One sleep reproduces that window.
    """
    session = get_session()
    real_create = session.create_client
    built: list[object] = []

    class _SlowCreate:
        """The real client context manager, entered after a turn of the loop."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        async def __aenter__(self) -> Any:
            await asyncio.sleep(0.01)
            return await self._inner.__aenter__()

        async def __aexit__(self, *exc_info: Any) -> Any:
            return await self._inner.__aexit__(*exc_info)

    def counting_create(*args: object, **kwargs: object) -> _SlowCreate:
        built.append(object())
        return _SlowCreate(real_create(*args, **kwargs))

    session.create_client = counting_create  # type: ignore[method-assign]
    backend = make_s3_storage(s3_endpoint, session=session)
    try:
        await asyncio.gather(*(backend.put(f"{KEY}.{n}", b"x") for n in range(5)))

        assert len(built) == 1
        assert backend._client is not None
        assert len([info.key async for info in backend.list_objects()]) == 5
    finally:
        await backend.aclose()


async def test_a_bucket_this_credential_may_not_inspect_is_still_usable(
    s3_endpoint: str,  # noqa: F811
) -> None:
    """HeadBucket needs s3:ListBucket, and a least-privilege policy may not grant it.

    A refusal is not "no bucket": the bucket is there, this credential simply
    cannot look at it, and CreateBucket would be refused too. Treated as a
    failure, the bootstrap would fail EVERY upload forever with a message about
    reaching the bucket - a permissions problem dressed as an outage.
    """
    session = get_session()
    session.register(
        "before-call.s3.HeadBucket",
        _raising(_client_error("403", "HeadBucket")),
    )
    # The bucket really exists; this backend just cannot ask whether it does.
    opener = make_s3_storage(s3_endpoint)
    bucket = opener._bucket
    await opener.put(KEY, b"seeded")
    await opener.aclose()

    blinkered = S3Storage(
        bucket,
        endpoint_url=s3_endpoint,
        access_key_id=ACCESS_KEY,
        secret_access_key=SECRET_KEY,
        session=session,
    )
    try:
        await blinkered.put(f"{KEY}.2", b"written anyway")

        assert await blinkered.get(f"{KEY}.2") == b"written anyway"
    finally:
        await blinkered.aclose()


async def test_a_dropped_connection_mid_body_is_a_seam_error(
    truncating_server: str,
) -> None:
    """A response that stops short of its Content-Length must not leak aiohttp.

    This is what a MinIO restart, or a proxy reaping an idle connection, looks
    like part-way through a download. aiobotocore does not wrap the failure -
    it arrives as aiohttp's own ClientPayloadError - so without the transport
    tuple it would reach app.services.chat_service, which catches only
    StorageError, and take the whole SSE turn down with a 500 instead of saying
    the file could not be read.
    """
    backend = S3Storage(
        "attachments",
        endpoint_url=truncating_server,
        access_key_id=ACCESS_KEY,
        secret_access_key=SECRET_KEY,
        max_attempts=1,
    )
    try:
        with pytest.raises(StorageUnavailableError):
            await backend.get(KEY)
    finally:
        await backend.aclose()


async def test_an_object_that_fails_its_checksum_is_never_returned(
    corrupting_server: str,
) -> None:
    """Corrupt bytes must raise, not come back looking like a file.

    The integrity check lives in the streaming body OBJECT, not in the socket:
    reading through it verifies the checksum the service advertised, and reading
    the raw response underneath it does not. The two are one keyword apart -
    aiobotocore's body is a proxy whose ``async with ... as body`` hands back
    the bare aiohttp response - and the difference is silent. A parser would
    then index a corrupted document and the model would answer from it.
    """
    backend = S3Storage(
        "attachments",
        endpoint_url=corrupting_server,
        access_key_id=ACCESS_KEY,
        secret_access_key=SECRET_KEY,
        max_attempts=1,
    )
    try:
        with pytest.raises(StorageUnavailableError):
            await backend.get(KEY)
    finally:
        await backend.aclose()
