"""The orphan-blob sweep, run against a real S3 service rather than the fake.

The sweep is the one consumer of ``list_objects``, and it is the job in this
codebase with the most to lose from a backend disagreeing with the fake: it
DELETES what it finds. So the three decisions that keep it safe - referenced
keys survive, aged unreferenced keys go, a store that cannot be listed aborts
the walk - are asserted again here against the S3 adapter and its paged
listing.

Age is expressed by moving the sweep's clock forward (``now=``) rather than by
backdating an object. A real service will not let a client rewrite an object's
LastModified, and a test that could would be testing something the deployment
cannot do.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from aiobotocore.session import get_session
from botocore.exceptions import EndpointConnectionError
from sqlalchemy import select

from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID
from app.models import Attachment
from app.services.attachment_gc import sweep_orphan_blobs
from app.storage.base import StorageUnavailableError
from app.storage.s3 import S3Storage
from tests.s3_server import make_s3_storage, s3_endpoint  # noqa: F401  - fixture

DATA = b"the bytes of an uploaded file"
GRACE = timedelta(minutes=60)
# Far enough past the grace window that anything already stored counts as
# garbage, without touching the objects themselves.
LATER = datetime.now(UTC) + timedelta(minutes=90)


@pytest.fixture
async def storage(s3_endpoint: str) -> AsyncIterator[S3Storage]:  # noqa: F811
    """An S3 backend with a bucket of its own, released afterwards."""
    backend = make_s3_storage(s3_endpoint)
    try:
        yield backend
    finally:
        await backend.aclose()


def _raise(exc: Exception):  # type: ignore[no-untyped-def]
    """Return a botocore event handler that raises ``exc`` instead of calling out."""

    def handler(**_: object) -> None:
        raise exc

    return handler


def _minted_key(conversation_id: UUID | None = None) -> str:
    """A key of the exact {uuid}/{uuid}/{uuid} shape the app mints."""
    return f"{DEFAULT_TENANT_ID}/{conversation_id or uuid4()}/{uuid4()}"


async def _row_for(db_sessionmaker, storage_key: str) -> None:
    """Commit the attachment row that claims ``storage_key``."""
    async with db_sessionmaker() as session:
        session.add(
            Attachment(
                tenant_id=DEFAULT_TENANT_ID,
                user_id=DEFAULT_USER_ID,
                conversation_id=None,
                filename="incident.txt",
                content_type="text/plain",
                size_bytes=len(DATA),
                storage_key=storage_key,
            )
        )
        await session.commit()


async def _keys(storage: S3Storage) -> set[str]:
    return {info.key async for info in storage.list_objects()}


async def test_an_aged_unreferenced_blob_is_deleted(db_sessionmaker, storage: S3Storage) -> None:
    key = _minted_key()
    await storage.put(key, DATA)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE, now=LATER)

    assert deleted == 1
    assert await _keys(storage) == set()


async def test_a_referenced_blob_survives(db_sessionmaker, storage: S3Storage) -> None:
    key = _minted_key()
    await storage.put(key, DATA)
    await _row_for(db_sessionmaker, key)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE, now=LATER)

    assert deleted == 0
    assert await _keys(storage) == {key}


async def test_an_upload_in_flight_is_protected_by_the_grace_window(
    db_sessionmaker, storage: S3Storage
) -> None:
    # An upload writes its bytes and commits the row moments later. Between the
    # two it looks exactly like garbage - and here the timestamp being compared
    # is the service's own LastModified, not one the test wrote.
    key = _minted_key()
    await storage.put(key, DATA)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE)

    assert deleted == 0
    assert await _keys(storage) == {key}


async def test_the_sweep_spans_every_listing_page(db_sessionmaker, storage: S3Storage) -> None:
    """More objects than one page holds, half of them claimed by a row.

    S3 hands back a bounded page at a time. A sweep that stopped after the first
    would leave every orphan behind it uncollected, silently and forever - and a
    sweep that mishandled the page boundary could just as easily delete a
    referenced object. The backend under test pages deliberately small.
    """
    kept = [_minted_key() for _ in range(3)]
    doomed = [_minted_key() for _ in range(4)]
    for key in kept + doomed:
        await storage.put(key, DATA)
    for key in kept:
        await _row_for(db_sessionmaker, key)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE, now=LATER)

    assert deleted == len(doomed)
    assert await _keys(storage) == set(kept)


async def test_deleting_while_the_listing_is_still_paging_loses_nothing(
    db_sessionmaker,
    storage: S3Storage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At scale the sweep deletes DURING the walk, not after it.

    It flushes a batch of keys every 500, so on any real store the deletions and
    the paging interleave - and S3's continuation token was handed out before
    those objects disappeared. Shrinking the batch is the only way to reach that
    regime in a test; leaving it at 500 means every assertion here is really
    about a sweep that finished listing first, which is the easy case.
    """
    monkeypatch.setattr("app.services.attachment_gc._KEY_BATCH", 2)
    kept = [_minted_key() for _ in range(2)]
    doomed = [_minted_key() for _ in range(6)]
    for key in kept + doomed:
        await storage.put(key, DATA)
    for key in kept:
        await _row_for(db_sessionmaker, key)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE, now=LATER)

    assert deleted == len(doomed)
    assert await _keys(storage) == set(kept)


async def test_a_key_this_app_did_not_mint_is_left_alone(
    db_sessionmaker, storage: S3Storage
) -> None:
    # Deleting files purely because a table does not mention them is a dangerous
    # shape, so the sweep only touches the exact key shape the app writes. The
    # fence has to hold over a real listing, where every key is a flat string
    # with no directory structure to lean on.
    strays = ["notes.txt", f"{DEFAULT_TENANT_ID}/loose.bin", f"{DEFAULT_TENANT_ID}/nope/{uuid4()}"]
    for key in strays:
        await storage.put(key, DATA)

    async with db_sessionmaker() as session:
        deleted = await sweep_orphan_blobs(session, storage, grace=GRACE, now=LATER)

    assert deleted == 0
    assert await _keys(storage) == set(strays)


async def test_a_store_that_cannot_be_listed_aborts_the_sweep(
    db_sessionmaker,
    s3_endpoint: str,  # noqa: F811
) -> None:
    """A half-read listing has seen an arbitrary subset, so the caller decides.

    Swallowing this would be the worst possible reading of the failure: every
    key the listing never reached looks unreferenced by omission. The sweep
    re-raises, and the adapter has to give it the seam exception it expects.
    """
    session_with_dead_listing = get_session()
    session_with_dead_listing.register(
        "before-call.s3.ListObjectsV2",
        _raise(EndpointConnectionError(endpoint_url=s3_endpoint)),
    )
    backend = make_s3_storage(s3_endpoint, session=session_with_dead_listing)
    try:
        async with db_sessionmaker() as db:
            with pytest.raises(StorageUnavailableError):
                await sweep_orphan_blobs(db, backend, grace=GRACE, now=LATER)
    finally:
        await backend.aclose()


async def test_no_attachment_row_leaks_between_tests(db_sessionmaker) -> None:
    # The sweep's verdict is "no row claims this key", so a row surviving from a
    # neighbouring test would quietly turn a deletion assertion into a survival
    # assertion. The in-memory database is per test; this pins that.
    async with db_sessionmaker() as session:
        assert (await session.execute(select(Attachment))).scalars().all() == []
