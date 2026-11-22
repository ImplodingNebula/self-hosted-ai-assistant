"""The StorageBackend contract, asserted once against every implementation.

Three backends answer the same seam - a local directory, an in-memory fake, and
an S3-compatible service - and the tests below do not know which one they are
talking to. That is the point: a fake that quietly disagrees with the real thing
is worse than no fake, and an S3 adapter is only a drop-in replacement if it
behaves identically down to the awkward cases (a prefix that is a name fragment,
a delete of something that was never there, a listing that spans pages).

Tests further down name a specific backend where the behaviour is genuinely
specific to it - atomic temp files on disk, for instance, have no S3 analogue.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.storage.base import (
    InvalidStorageKeyError,
    ObjectNotFoundError,
    StorageBackend,
    validate_key,
)
from app.storage.local import FakeStorageBackend, LocalFsStorage
from tests.s3_server import make_s3_storage, s3_endpoint  # noqa: F401  - fixture

KEY = "11111111-1111-1111-1111-111111111111/conv/att.bin"


@pytest.fixture(params=["local", "fake", "s3"])
async def storage(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[StorageBackend]:
    """A StorageBackend under test, one per implementation.

    The moto server is requested lazily rather than declared as a parameter:
    named up front it would start for the local and fake runs too, and then the
    local-fs contract could not be checked on a machine without moto installed.
    """
    if request.param == "local":
        yield LocalFsStorage(tmp_path)
    elif request.param == "fake":
        yield FakeStorageBackend()
    else:
        backend = make_s3_storage(request.getfixturevalue("s3_endpoint"))
        try:
            yield backend
        finally:
            # The adapter owns a connection pool; leaving it open leaks a socket
            # per test and buries the run in unclosed-session warnings.
            await backend.aclose()


async def test_put_get_round_trip(storage: StorageBackend) -> None:
    stored = await storage.put(KEY, b"hello bytes", content_type="application/octet-stream")

    assert stored.key == KEY
    assert stored.size_bytes == 11
    assert stored.content_type == "application/octet-stream"
    assert await storage.get(KEY) == b"hello bytes"


async def test_put_overwrites_an_existing_object(storage: StorageBackend) -> None:
    await storage.put(KEY, b"first")
    await storage.put(KEY, b"second is longer")

    assert await storage.get(KEY) == b"second is longer"


async def test_exists_tracks_put_and_delete(storage: StorageBackend) -> None:
    assert await storage.exists(KEY) is False

    await storage.put(KEY, b"x")
    assert await storage.exists(KEY) is True

    await storage.delete(KEY)
    assert await storage.exists(KEY) is False


async def test_delete_of_a_missing_key_is_idempotent(storage: StorageBackend) -> None:
    await storage.delete(KEY)
    await storage.delete(KEY)  # a retried cleanup must not explode


async def test_get_of_a_missing_key_raises_not_found(storage: StorageBackend) -> None:
    with pytest.raises(ObjectNotFoundError):
        await storage.get(KEY)


async def test_empty_payload_round_trips(storage: StorageBackend) -> None:
    await storage.put(KEY, b"")

    assert await storage.exists(KEY) is True
    assert await storage.get(KEY) == b""


async def test_listing_describes_every_stored_object(storage: StorageBackend) -> None:
    await storage.put(KEY, b"hello bytes")

    listed = [info async for info in storage.list_objects()]

    assert [info.key for info in listed] == [KEY]
    assert listed[0].size_bytes == 11
    # The blob sweep decides what is garbage by age, so a listing that cannot
    # date an object is useless to it.
    assert listed[0].modified_at.tzinfo is not None


async def test_listing_an_empty_store_yields_nothing(storage: StorageBackend) -> None:
    assert [info async for info in storage.list_objects()] == []


async def test_a_deleted_object_leaves_the_listing(storage: StorageBackend) -> None:
    await storage.put(KEY, b"x")
    await storage.delete(KEY)

    assert [info async for info in storage.list_objects()] == []


# A prefix names a subtree, never a fragment of a name: "aaa...1/conv" must not
# reach into a sibling directory that merely starts with the same characters.
# Both impls have to agree, or the fake would pass a sweep the real one fails.
async def test_a_prefix_matches_whole_segments_only(storage: StorageBackend) -> None:
    tenant = "11111111-1111-1111-1111-111111111111"
    mine = f"{tenant}/conv/att.bin"
    sibling = f"{tenant}/conv2/att.bin"
    await storage.put(mine, b"mine")
    await storage.put(sibling, b"theirs")

    listed = {info.key async for info in storage.list_objects(f"{tenant}/conv")}

    assert listed == {mine}


async def test_a_prefix_lists_a_whole_subtree(storage: StorageBackend) -> None:
    tenant = "11111111-1111-1111-1111-111111111111"
    other = "22222222-2222-2222-2222-222222222222"
    await storage.put(f"{tenant}/a/one.bin", b"1")
    await storage.put(f"{tenant}/b/two.bin", b"2")
    await storage.put(f"{other}/c/three.bin", b"3")

    listed = {info.key async for info in storage.list_objects(tenant)}

    assert listed == {f"{tenant}/a/one.bin", f"{tenant}/b/two.bin"}


# The whole-segment rule spelled out on the case it exists for: two sibling
# subtrees whose names share a prefix. S3 has no directories - a listing there
# is a string prefix match - so this is the assertion that keeps the adapter
# from handing the blob sweep a neighbouring tenant's objects.
async def test_a_prefix_never_reaches_a_longer_sibling_name(storage: StorageBackend) -> None:
    await storage.put("abc/one.bin", b"mine")
    await storage.put("abcd/two.bin", b"not mine")

    listed = {info.key async for info in storage.list_objects("abc")}

    assert listed == {"abc/one.bin"}


async def test_a_prefix_naming_an_object_exactly_lists_nothing(storage: StorageBackend) -> None:
    # A prefix names a SUBTREE. On disk the prefix here is a file, not a
    # directory; on S3 the request goes out as "abc/". Both therefore find
    # nothing, and the fake has to agree or it would pass a sweep the real
    # backends fail - the one case where it used to diverge.
    await storage.put("abc", b"an object, not a subtree")

    assert [info async for info in storage.list_objects("abc")] == []


async def test_a_listing_returns_every_object_across_page_boundaries(
    storage: StorageBackend,
) -> None:
    # S3 hands back a bounded page at a time, and the one caller sweeps the
    # whole store: a backend that returns only the first page would leave every
    # orphan behind it uncollected, forever, with nothing to see. The S3 backend
    # under test pages deliberately small so this crosses several boundaries.
    tenant = "11111111-1111-1111-1111-111111111111"
    expected = {f"{tenant}/conv/file{n}.bin" for n in range(5)}
    for key in sorted(expected):
        await storage.put(key, b"x")

    assert {info.key async for info in storage.list_objects()} == expected
    assert {info.key async for info in storage.list_objects(tenant)} == expected


async def test_a_listing_dates_objects_in_utc(storage: StorageBackend) -> None:
    # The blob sweep subtracts a grace window from an aware UTC "now" and
    # compares. A naive timestamp raises on that comparison and a wrongly
    # offset one silently ages objects by hours, which is how a sweep starts
    # deleting uploads that are still in flight.
    before = datetime.now(UTC)
    await storage.put(KEY, b"x")

    (info,) = [i async for i in storage.list_objects()]

    # `is UTC`, not merely a zero offset: botocore hands back its own tzutc(),
    # which passes an offset check while printing and comparing differently
    # from the other two backends. One spelling across all three is what makes
    # a timestamp read the same in a log wherever it came from.
    assert info.modified_at.tzinfo is UTC
    assert info.modified_at.utcoffset() == timedelta(0)
    assert (
        before - timedelta(minutes=5)
        <= info.modified_at
        <= datetime.now(UTC) + timedelta(minutes=5)
    )


async def test_deleting_a_key_that_never_existed_is_silent(storage: StorageBackend) -> None:
    # Not the same test as deleting an already-deleted key: this one has never
    # had a parent directory, a bucket entry, or anything else the backend might
    # be relying on. Cleanup paths retry, so both spellings must be no-ops.
    await storage.delete("22222222-2222-2222-2222-222222222222/never/seen.bin")
    await storage.delete("standalone.bin")


async def test_listing_rejects_an_unsafe_prefix(storage: StorageBackend) -> None:
    # The body of an async generator runs on first iteration, so the rejection
    # only surfaces when the caller actually starts consuming it.
    with pytest.raises(InvalidStorageKeyError):
        [info async for info in storage.list_objects("../etc")]


# A hostile filename must never decide where bytes land. These are the spellings
# an attacker reaches for; each must be refused before any path is touched.
@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "..",
        "../etc/passwd",
        "a/../../b",
        "/absolute/path",
        "a//b",  # empty segment
        "a/",  # trailing empty segment
        "./a",
        r"a\..\..\b",  # Windows separator
        "C:/Windows/System32",
        "-leading-dash",  # segment must start alphanumeric
        "a/" + "x" * 600,  # over MAX_KEY_LENGTH
        # Python's '$' also matches before a trailing newline, so an anchored
        # '^...$' pattern would wave this through. Regression guard.
        "att\n",
        "tenant/att\n",
        "a.",  # Windows strips a trailing dot, colliding with "a"
        "CON",  # Windows device names address a device, not a file
        "nul.txt",
        "аbc",  # Cyrillic lookalike, not [A-Za-z0-9]
    ],
)
async def test_unsafe_keys_are_rejected_by_every_operation(bad_key: str, tmp_path: Path) -> None:
    # The remote backend points at a port with nothing behind it, deliberately.
    # Every call below must be refused on the key alone, so the endpoint is
    # never dialled and its deadness never matters; an adapter that reached the
    # network first would fail this test with the wrong exception AND would have
    # handed a hostile key to the service to interpret.
    remote = make_s3_storage("http://127.0.0.1:1", max_attempts=1)
    for storage in (LocalFsStorage(tmp_path), FakeStorageBackend(), remote):
        with pytest.raises(InvalidStorageKeyError):
            await storage.put(bad_key, b"x")
        with pytest.raises(InvalidStorageKeyError):
            await storage.get(bad_key)
        with pytest.raises(InvalidStorageKeyError):
            await storage.delete(bad_key)
        with pytest.raises(InvalidStorageKeyError):
            await storage.exists(bad_key)

    # Nothing was ever opened, which is also why there is no aclose to run.
    assert remote._client is None


async def test_traversal_key_writes_nothing_outside_the_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"
    storage = LocalFsStorage(root)

    with pytest.raises(InvalidStorageKeyError):
        await storage.put("../outside.txt", b"escaped")

    assert not outside.exists()


def test_validate_key_accepts_the_keys_the_app_mints() -> None:
    assert validate_key(KEY) == KEY
    assert validate_key("tenant/conv/file.pdf") == "tenant/conv/file.pdf"
    assert validate_key("a") == "a"


async def test_symlinked_ancestor_cannot_redirect_a_write_outside_the_root(
    tmp_path: Path,
) -> None:
    """A symlink escape passes string validation, so the path must be resolved.

    Without the containment check in `_safe_path` this test writes to `outside/`
    while every other traversal test still passes, because `validate_key` never
    sees anything suspicious in the key `link/loot.bin`.
    """
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):  # unprivileged Windows
        pytest.skip("symlink creation is not permitted on this host")

    storage = LocalFsStorage(root)

    with pytest.raises(InvalidStorageKeyError):
        await storage.put("link/loot.bin", b"escaped")

    assert not (outside / "loot.bin").exists()


async def test_local_storage_writes_under_its_root(tmp_path: Path) -> None:
    storage = LocalFsStorage(tmp_path)

    await storage.put(KEY, b"payload")

    assert (tmp_path / KEY).read_bytes() == b"payload"


async def test_local_storage_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    storage = LocalFsStorage(tmp_path)

    await storage.put(KEY, b"payload")

    leftovers = [p.name for p in (tmp_path / KEY).parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


async def test_local_storage_never_blocks_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every object operation must reach disk only inside a worker thread.

    Path resolution issues syscalls too, so it lives inside the offloaded
    function rather than beside it. Asserting the offload directly is the only
    way to keep a future refactor from quietly reintroducing a blocking read
    inside a request handler.
    """
    calls: list[str] = []
    real_to_thread = asyncio.to_thread

    async def counting_to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(getattr(func, "__name__", repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", counting_to_thread)
    storage = LocalFsStorage(tmp_path)

    await storage.put(KEY, b"x")
    await storage.get(KEY)
    await storage.exists(KEY)
    await storage.delete(KEY)

    assert len(calls) == 4


async def test_a_key_naming_a_directory_reads_as_a_missing_object(tmp_path: Path) -> None:
    """Reading a directory raises different OSErrors per platform.

    POSIX gives IsADirectoryError and Windows gives PermissionError, so the impl
    checks explicitly and the seam's contract holds identically on both.
    """
    storage = LocalFsStorage(tmp_path)
    (tmp_path / "a").mkdir()

    with pytest.raises(ObjectNotFoundError):
        await storage.get("a")

    assert await storage.exists("a") is False


async def test_listing_ignores_a_half_written_object(tmp_path: Path) -> None:
    """A put in flight leaves a dot-prefixed temporary sibling.

    No valid key may start with a dot, so such a file is not an object - and the
    blob sweep must never see one, or it would delete the bytes of an upload
    that is still being written.
    """
    storage = LocalFsStorage(tmp_path)
    await storage.put(KEY, b"real")
    (
        tmp_path / "11111111-1111-1111-1111-111111111111" / "conv" / ".att.bin.abc123.tmp"
    ).write_bytes(b"partial")

    listed = [info async for info in storage.list_objects()]

    assert [info.key for info in listed] == [KEY]


async def test_listing_runs_off_the_event_loop(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Walking a directory tree is the most blocking call in this module; doing it
    # inline would stall every concurrent SSE stream for the length of the walk.
    calls: list[str] = []
    real_to_thread = asyncio.to_thread

    async def counting_to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(getattr(func, "__name__", repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", counting_to_thread)
    storage = LocalFsStorage(tmp_path)

    assert [info async for info in storage.list_objects()] == []

    assert calls == ["_list_sync"]


async def test_fake_records_content_type(tmp_path: Path) -> None:
    storage = FakeStorageBackend()

    await storage.put(KEY, b"x", content_type="text/plain")

    assert storage.objects[KEY] == b"x"
    assert storage.content_types[KEY] == "text/plain"
