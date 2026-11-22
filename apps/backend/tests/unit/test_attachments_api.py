"""Attachments endpoints: upload validation/storage/enqueue, polling, download, delete."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_tenant
from app.api.v1.attachments import _refuse_unless_servable
from app.core.config import Settings, get_settings
from app.core.errors import AttachmentUnreadableError
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
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
from app.parsing.base import ParsedDocument
from app.providers.rag.base import VectorStoreUnavailableError
from app.providers.rag.embedder import FakeEmbedder
from app.repositories.attachments import AttachmentRepository
from app.repositories.conversations import ConversationRepository
from app.services import rag_service
from app.services.attachment_gc import expire_aged_attachments
from app.services.attachment_ingest import build_ingest_items
from app.storage.base import (
    InvalidStorageKeyError,
    ObjectNotFoundError,
    StorageUnavailableError,
)
from app.storage.local import FakeStorageBackend
from app.workers.main import JOB_PARSE_AND_INGEST
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store

URL = "/api/v1/attachments"
TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, model_provider="fake", **overrides)


@asynccontextmanager
async def attachments_client(db_sessionmaker, *, storage, queue, settings=None, store=None):
    """App-wired client for the attachments endpoints.

    ``store`` is injected for the delete path, which purges the file's chunks;
    without it the app would build a real Qdrant client and reach for a network.
    """
    app = create_app()
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings
    app.state.db_sessionmaker = db_sessionmaker
    app.state.storage = storage
    app.state.job_queue = queue
    app.state.vector_store = store if store is not None else make_memory_store()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
    app.dependency_overrides.clear()


async def _make_conversation(db_sessionmaker) -> str:
    async with db_sessionmaker() as session:
        repo = ConversationRepository(session, TENANT)
        conversation = await repo.create(title="thread")
        await repo.commit()
        return str(conversation.id)


def _upload(**over):
    """Build httpx multipart kwargs for one upload."""
    files = {
        "file": (
            over.get("filename", "notes.txt"),
            over.get("data", b"hello world"),
            over.get("content_type", "text/plain"),
        )
    }
    data = {"conversation_id": over["conversation_id"]}
    if "origin" in over:
        data["origin"] = over["origin"]
    return {"files": files, "data": data}


async def test_upload_stores_bytes_creates_row_and_enqueues(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(URL, **_upload(conversation_id=conversation_id))

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == ATTACHMENT_PENDING
    assert body["origin"] == "upload"
    assert body["filename"] == "notes.txt"
    assert body["conversation_id"] == conversation_id
    # The bytes reached storage and the ingestion job was queued exactly once.
    assert len(storage.objects) == 1
    assert len(queue.calls) == 1
    name, args, _ = queue.calls[0]
    assert name == JOB_PARSE_AND_INGEST
    assert args == (body["id"], str(TENANT.tenant_id), str(TENANT.user_id))
    # A per-attachment job id makes a double-submitted upload dedupe to one job.
    assert f"attachment-ingest:{body['id']}" in queue.job_ids
    # The response never leaks the internal storage key.
    assert "storage_key" not in body


async def test_a_pasted_block_is_stored_with_paste_origin(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(
            URL,
            **_upload(
                conversation_id=conversation_id,
                filename="pasted-text.txt",
                data=b"A long pasted traceback...",
                origin=ATTACHMENT_ORIGIN_PASTE,
            ),
        )

    assert resp.status_code == 201
    assert resp.json()["origin"] == "paste"


async def test_upload_to_a_missing_conversation_is_404(db_sessionmaker) -> None:
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(URL, **_upload(conversation_id=str(uuid4())))

    assert resp.status_code == 404
    # Nothing was stored or queued for a conversation that does not exist.
    assert storage.objects == {}
    assert queue.calls == []


async def test_a_malformed_conversation_id_is_400(db_sessionmaker) -> None:
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(URL, **_upload(conversation_id="not-a-uuid"))

    assert resp.status_code == 400


async def test_an_empty_file_is_rejected(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(URL, **_upload(conversation_id=conversation_id, data=b""))

    assert resp.status_code == 400
    assert queue.calls == []


async def test_an_unknown_origin_is_400(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(URL, **_upload(conversation_id=conversation_id, origin="smuggled"))

    assert resp.status_code == 400


async def test_an_unsupported_file_is_415_and_stores_nothing(db_sessionmaker) -> None:
    # A binary blob no parser claims is refused up front, before it is stored,
    # so a rejected upload leaves no orphan bytes behind.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(
            URL,
            **_upload(
                conversation_id=conversation_id,
                filename="clip.heic",
                data=b"\x00\x01\x02\x00binary",
                content_type="image/heic",
            ),
        )

    assert resp.status_code == 415
    assert storage.objects == {}
    assert queue.calls == []


async def test_a_file_over_the_cap_is_413(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    settings = make_settings(attachment_max_bytes=8)
    async with attachments_client(
        db_sessionmaker, storage=storage, queue=queue, settings=settings
    ) as client:
        resp = await client.post(
            URL, **_upload(conversation_id=conversation_id, data=b"way past eight bytes")
        )

    assert resp.status_code == 413
    assert storage.objects == {}
    assert queue.calls == []


async def test_an_unreachable_queue_is_503_after_storing(db_sessionmaker) -> None:
    # The bytes are safe and the row exists; only the hand-off failed, so the
    # honest answer is 503, not a false success.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue(unavailable=True)
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(URL, **_upload(conversation_id=conversation_id))

    assert resp.status_code == 503
    assert len(storage.objects) == 1


class _DownStorage(FakeStorageBackend):
    """Object storage that refuses to write."""

    async def put(self, key, data, *, content_type=None):
        raise StorageUnavailableError("Object storage is unavailable.")


async def test_unreachable_storage_is_503_and_creates_no_row(db_sessionmaker) -> None:
    # When the bytes cannot be stored, nothing is committed and nothing is
    # queued: there is no row that could ever point at a blob that is not there.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = _DownStorage(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(URL, **_upload(conversation_id=conversation_id))

    assert resp.status_code == 503
    assert queue.calls == []


async def test_the_same_bytes_uploaded_twice_dedupe_to_one_row(db_sessionmaker) -> None:
    # Re-uploading identical bytes into one conversation must not store a second
    # copy or create a duplicate row - it reuses the first, so a client that
    # retried a lost request lands back on the same attachment.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        first = await client.post(
            URL, **_upload(conversation_id=conversation_id, data=b"identical bytes")
        )
        second = await client.post(
            URL, **_upload(conversation_id=conversation_id, data=b"identical bytes")
        )

    assert first.json()["id"] == second.json()["id"]
    # One blob, and one job: the re-enqueue collapses on the shared job id.
    assert len(storage.objects) == 1
    assert len(queue.job_ids) == 1


async def test_re_uploading_a_file_after_it_expired_stores_it_fresh(db_sessionmaker) -> None:
    # The retention sweep empties a row and keeps it as a tombstone, so from the
    # dedupe query's point of view an expired row looks like a perfectly healthy
    # match: right hash, right conversation, a filename and a size. Reusing it
    # would hand this second upload a row whose blob was deleted - the response
    # would say 201 over a file that does not exist, and the ingest it re-enqueues
    # would find an empty store and mark the row `failed`, replacing the honest
    # "this aged out" with a fault that never happened. A fresh row, a fresh blob.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        first = await client.post(
            URL, **_upload(conversation_id=conversation_id, data=b"identical bytes")
        )
        # Exactly what the sweep leaves behind: bytes gone, row kept, status flipped.
        async with db_sessionmaker() as session:
            repo = AttachmentRepository(session, TENANT)
            row = await repo.get(UUID(first.json()["id"]))
            assert row is not None
            await storage.delete(row.storage_key)
            await repo.set_status(row, ATTACHMENT_EXPIRED)
            await repo.commit()

        second = await client.post(
            URL, **_upload(conversation_id=conversation_id, data=b"identical bytes")
        )

    assert second.status_code == 201
    assert second.json()["id"] != first.json()["id"]
    assert second.json()["status"] == ATTACHMENT_PENDING
    # A blob for the new row only: the expired one was never resurrected.
    assert len(storage.objects) == 1
    async with db_sessionmaker() as session:
        fresh = await AttachmentRepository(session, TENANT).get(UUID(second.json()["id"]))
        assert fresh is not None
        assert fresh.storage_key in storage.objects


async def test_a_retry_after_a_queue_outage_resumes_the_same_attachment(db_sessionmaker) -> None:
    # The real payoff of dedup: the first upload stored the row but the queue was
    # down (503, no job), so the file would sit unprocessed forever. Re-uploading
    # the same bytes finds that pending row and finally queues its ingestion,
    # rather than orphaning a second blob no one can reach.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue(unavailable=True)
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        first = await client.post(URL, **_upload(conversation_id=conversation_id, data=b"same"))
        assert first.status_code == 503

        queue.unavailable = False
        second = await client.post(URL, **_upload(conversation_id=conversation_id, data=b"same"))

    assert second.status_code == 201
    # Still one blob, and now exactly one queued job for that same row.
    assert len(storage.objects) == 1
    assert len(queue.calls) == 1
    assert queue.calls[0][1][0] == second.json()["id"]


async def test_upload_sets_a_location_header(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.post(URL, **_upload(conversation_id=conversation_id))

    assert resp.headers["location"] == f"{URL}/{resp.json()['id']}"


async def test_cannot_attach_to_another_tenants_conversation(db_sessionmaker) -> None:
    # The conversation exists, but for a different tenant: from this tenant it is
    # indistinguishable from one that never existed, so the upload is a 404 and
    # stores nothing - no cross-tenant write.
    conversation_id = await _make_conversation(db_sessionmaker)
    other = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    app = create_app()
    app.dependency_overrides[get_current_tenant] = lambda: other
    app.state.db_sessionmaker = db_sessionmaker
    app.state.storage = storage
    app.state.job_queue = queue
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(URL, **_upload(conversation_id=conversation_id))
    app.dependency_overrides.clear()

    assert resp.status_code == 404
    assert storage.objects == {}


async def test_get_returns_the_attachment_for_polling(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        created = await client.post(URL, **_upload(conversation_id=conversation_id))
        attachment_id = created.json()["id"]

        got = await client.get(f"{URL}/{attachment_id}")

    assert got.status_code == 200
    assert got.json()["id"] == attachment_id
    assert got.json()["status"] == ATTACHMENT_PENDING


async def test_get_of_a_missing_or_malformed_id_is_404(db_sessionmaker) -> None:
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        assert (await client.get(f"{URL}/{uuid4()}")).status_code == 404
        assert (await client.get(f"{URL}/not-a-uuid")).status_code == 404


async def test_another_tenant_cannot_read_the_attachment(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    other = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        created = await client.post(URL, **_upload(conversation_id=conversation_id))
        attachment_id = created.json()["id"]

    # Re-open the app as a different tenant: the row must be invisible.
    app = create_app()
    app.dependency_overrides[get_current_tenant] = lambda: other
    app.state.db_sessionmaker = db_sessionmaker
    app.state.storage = storage
    app.state.job_queue = queue
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"{URL}/{attachment_id}")
    app.dependency_overrides.clear()

    assert resp.status_code == 404


# -------------------------------------------------------------------- download

PDF_BYTES = b"%PDF-1.7 not really a pdf, but bytes all the same"


async def _seed_file(
    db_sessionmaker,
    storage,
    *,
    conversation_id: str,
    filename: str = "quarterly-report.pdf",
    content_type: str = "application/pdf",
    status: str = ATTACHMENT_READY,
    origin: str = ATTACHMENT_ORIGIN_GENERATED,
    data: bytes = PDF_BYTES,
    store_bytes: bool = True,
    tenant: TenantContext = TENANT,
    error: str | None = None,
    age: timedelta | None = None,
) -> str:
    """Store bytes and a row over them, the way the generated-file path does.

    Written directly rather than through the upload endpoint because the download
    route's input is a ROW, and the states worth testing (expired, failed, a
    filename carrying a CRLF) are ones no upload request can produce - httpx
    escapes a hostile filename out of its own multipart header, and the retention
    sweep that writes ``expired`` lives elsewhere.
    """
    attachment_id = uuid4()
    key = f"{tenant.tenant_id}/{conversation_id}/{attachment_id}"
    if store_bytes:
        await storage.put(key, data, content_type=content_type)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, tenant)
        row = Attachment(
            id=attachment_id,
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            conversation_id=UUID(conversation_id),
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            storage_key=key,
            status=status,
            origin=origin,
            error=error,
        )
        if age is not None:
            # Backdated so the retention sweep has something to act on; the
            # column defaults to now, which no window can ever reach.
            row.created_at = datetime.now(UTC) - age
        repo.add(row)
        await repo.commit()
    return str(attachment_id)


def _content_url(attachment_id: str) -> str:
    return f"{URL}/{attachment_id}/content"


async def test_download_returns_the_bytes_with_their_name_type_and_length(
    db_sessionmaker,
) -> None:
    # The whole point of the route: a file the assistant produced is reachable,
    # arrives as a download rather than a page, and keeps its name.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(db_sessionmaker, storage, conversation_id=conversation_id)
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 200
    assert resp.content == PDF_BYTES
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.headers["content-length"] == str(len(PDF_BYTES))
    disposition = resp.headers["content-disposition"]
    assert disposition.startswith("attachment; ")
    assert 'filename="quarterly-report.pdf"' in disposition
    assert "filename*=UTF-8''quarterly-report.pdf" in disposition


async def test_download_says_do_not_sniff_and_do_not_cache(db_sessionmaker) -> None:
    # One tenant's private file on a shared same-origin path, whose bytes and
    # declared type both came from outside: neither a cache nor a content sniffer
    # gets to make its own decision about them.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(db_sessionmaker, storage, conversation_id=conversation_id)
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["cache-control"] == "private, no-store"


async def test_a_hostile_filename_cannot_shape_the_header(db_sessionmaker) -> None:
    # The filename is row data someone else chose. A quote would close the
    # quoted-string early and a CRLF would start a header of the attacker's
    # choosing, so both are replaced rather than escaped - and the real name
    # still travels, percent-encoded, in the filename* form.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    hostile = 'ev"il\r\nX-Injected: yes\r\n\r\nreporté.pdf'
    file_id = await _seed_file(
        db_sessionmaker, storage, conversation_id=conversation_id, filename=hostile
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 200
    disposition = resp.headers["content-disposition"]
    # Exact, because every character of it is the guard: the quoted form is the
    # neutered one, and the real name rides percent-encoded on filename*.
    assert disposition == (
        'attachment; filename="ev_il__X-Injected: yes____report_.pdf"; '
        "filename*=UTF-8''ev%22il%0D%0AX-Injected%3A%20yes%0D%0A%0D%0Areport%C3%A9.pdf"
    )
    assert "\r" not in disposition and "\n" not in disposition
    assert "x-injected" not in resp.headers


async def test_a_name_outside_ascii_survives_only_in_the_encoded_form(
    db_sessionmaker,
) -> None:
    # A name written in another script leaves the quoted fallback almost nothing
    # to say. It must still be A name - the clients that can read filename* get
    # the real one, and the rest get something they can save to disk.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker, storage, conversation_id=conversation_id, filename="季度.pdf"
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    disposition = resp.headers["content-disposition"]
    assert 'filename="__.pdf"' in disposition
    assert "filename*=UTF-8''%E5%AD%A3%E5%BA%A6.pdf" in disposition


async def test_a_filename_with_nothing_left_to_quote_gets_a_placeholder(
    db_sessionmaker,
) -> None:
    # Defensive: the upload route refuses a blank name, so only a row written by
    # something else can be blank here. An empty filename="" is not a filename,
    # and some clients save it as the URL's last segment - which for this route
    # is the word "content".
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker, storage, conversation_id=conversation_id, filename="   "
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert 'filename="download"' in resp.headers["content-disposition"]


async def test_an_enormous_filename_is_clamped_before_it_becomes_a_header(
    db_sessionmaker,
) -> None:
    # Nothing caps the stored filename, and a header long enough to exhaust the
    # server's header budget would cost the whole response rather than the name.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        filename=f"{'a' * 4000}.pdf",
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 200
    assert len(resp.headers["content-disposition"].encode()) <= 512
    # The extension is held back from the truncation: an operating system picks
    # an application from it, and a clamped name without one opens in nothing.
    assert '.pdf";' in resp.headers["content-disposition"]


async def test_the_clamp_is_measured_in_bytes_not_characters(db_sessionmaker) -> None:
    # Percent-encoding costs up to twelve bytes per code point, so a character
    # count is not a bound on the header at all - a name of astral characters
    # clears two kilobytes while looking short.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        filename=f"{'\U0001f4c4' * 400}.pdf",
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 200
    assert len(resp.headers["content-disposition"].encode()) <= 512


async def test_a_declared_charset_does_not_cost_the_file_its_type(db_sessionmaker) -> None:
    # A browser that sends "text/plain; charset=utf-8" is being ordinary, not
    # hostile. The type it qualified survives, and the parameter that reaches the
    # wire is the SERVER's own - the client's is dropped before the check, never
    # echoed - so a smuggled one cannot ride along behind a legitimate type.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        filename="notes.txt",
        content_type='text/plain; charset="><script>',
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.headers["content-type"] == "text/plain; charset=utf-8"


async def test_a_content_type_the_host_will_not_vouch_for_becomes_octet_stream(
    db_sessionmaker,
) -> None:
    # An upload's content type is whatever the client's multipart part claimed,
    # and this route is the first place it is echoed back into a header. A value
    # that is not a media type at all gets the neutral one.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        content_type='"><script>alert(1)</script>',
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.headers["content-type"] == "application/octet-stream"


async def test_an_expired_generated_file_is_410_and_offers_a_way_back(
    db_sessionmaker,
) -> None:
    # Retention took the bytes. The row is still there, so 404 would be a lie -
    # and a file the assistant made can simply be asked for again.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        status=ATTACHMENT_EXPIRED,
        store_bytes=False,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 410
    body = resp.json()
    assert body["code"] == "attachment_expired"
    assert "aged out of retention" in body["detail"]
    assert "Asking for it again" in body["detail"]


async def test_an_expired_upload_does_not_promise_a_regeneration(db_sessionmaker) -> None:
    # Nothing can reproduce a file a person uploaded, so the copy must not
    # suggest asking for it again would.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        status=ATTACHMENT_EXPIRED,
        origin=ATTACHMENT_ORIGIN_UPLOAD,
        store_bytes=False,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 410
    assert resp.json()["code"] == "attachment_expired"
    assert "Asking for it again" not in resp.json()["detail"]


@pytest.mark.parametrize("status", [ATTACHMENT_PENDING, ATTACHMENT_PARSING])
async def test_a_file_still_being_processed_is_409_not_a_download(
    db_sessionmaker, status: str
) -> None:
    # The bytes are there, but the file is mid-ingest: "come back in a moment"
    # has to look different from "this never worked", because only one is worth
    # retrying.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        status=status,
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 409
    assert resp.json()["code"] == "attachment_not_ready"


@pytest.mark.parametrize("origin", [ATTACHMENT_ORIGIN_UPLOAD, ATTACHMENT_ORIGIN_PASTE])
async def test_a_persons_file_is_served_even_though_ingestion_failed(
    db_sessionmaker, origin: str
) -> None:
    # What failed is the platform's READING of the file, not the file. The bytes
    # were written before the row was committed and no parse failure removes
    # them, so they are still exactly what their owner handed over - and
    # refusing someone their own document over a parser's opinion of it would be
    # the platform keeping a file it could not even use.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        filename="quarterly-report.pdf",
        status=ATTACHMENT_FAILED,
        origin=origin,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 200
    assert resp.content == PDF_BYTES
    # Through the SAME guards the ready path uses - the spoof and header
    # injection defences must not fork per lifecycle state.
    assert resp.headers["content-type"] == "application/pdf"
    assert 'filename="quarterly-report.pdf"' in resp.headers["content-disposition"]
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["cache-control"] == "private, no-store"


async def test_a_failed_upload_over_a_missing_blob_is_still_404(db_sessionmaker) -> None:
    # Serving a failed row rests entirely on its bytes still being there. When
    # they are not, it answers exactly as the ready path does rather than 500ing
    # on the way to a file that is not there.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = _MissingBlobStorage(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        status=ATTACHMENT_FAILED,
        origin=ATTACHMENT_ORIGIN_UPLOAD,
        store_bytes=False,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


async def test_a_failed_upload_that_aged_out_says_so_rather_than_why_it_failed(
    db_sessionmaker,
) -> None:
    # End to end, through the real sweep: a file the platform could not read is
    # downloadable right up until retention takes it, and from that moment it is
    # indistinguishable from any other tombstone. The parse verdict described
    # bytes that no longer exist, so the answer is "it aged out" - and, being an
    # upload, without the "ask for it again" a generated file's 410 offers.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        status=ATTACHMENT_FAILED,
        origin=ATTACHMENT_ORIGIN_UPLOAD,
        error="The file is not a readable spreadsheet.",
        age=timedelta(days=120),
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        assert (await client.get(_content_url(file_id))).status_code == 200

    async with db_sessionmaker() as session:
        swept = await expire_aged_attachments(
            session,
            storage,
            make_memory_store(),
            generated_retention=None,
            upload_retention=timedelta(days=90),
        )
    assert swept == 1

    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 410
    body = resp.json()
    assert body["code"] == "attachment_expired"
    assert "aged out of retention" in body["detail"]
    assert "Asking for it again" not in body["detail"]
    # The reason it failed went with the bytes it described.
    assert "readable spreadsheet" not in body["detail"]


async def test_a_generated_file_that_never_finished_has_no_original_to_serve(
    db_sessionmaker,
) -> None:
    # The asymmetry with the upload above, and the reason for it: a generated
    # file exists only because the platform finished making it, so "it did not
    # finish" leaves nothing trustworthy behind. No code path writes this
    # combination today - a generated row is born ready and nothing parses it -
    # so the row is written directly, which is the point: the guard keys on what
    # makes serving safe, not on what happens to be reachable.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        status=ATTACHMENT_FAILED,
        origin=ATTACHMENT_ORIGIN_GENERATED,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 410
    assert resp.json()["code"] == "attachment_unreadable"
    assert resp.content != PDF_BYTES


def test_a_status_this_build_does_not_know_is_refused_rather_than_served() -> None:
    # The guard is exhaustive over today's five values, so this can only be
    # reached by adding a sixth. The safe default when that happens is a closed
    # door, not a served file - and no repository would let such a row be
    # written, which is why the guard is called directly here.
    unknown = Attachment(
        tenant_id=DEFAULT_TENANT_ID,
        user_id=DEFAULT_USER_ID,
        filename="x.pdf",
        content_type="application/pdf",
        size_bytes=1,
        storage_key="t/c/a",
        status="quarantined",
    )
    with pytest.raises(AttachmentUnreadableError):
        _refuse_unless_servable(unknown)


class _MissingBlobStorage(FakeStorageBackend):
    """Storage that has forgotten an object a live row still points at."""

    async def get(self, key):
        raise ObjectNotFoundError(f"No object stored at {key!r}")


async def test_a_ready_row_over_a_missing_blob_is_404_not_500(db_sessionmaker) -> None:
    # The row promised bytes that are not there. That is the server's problem to
    # log, but the client asked an answerable question and gets an answer.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = _MissingBlobStorage(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker, storage, conversation_id=conversation_id, store_bytes=False
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


class _UnreadableStorage(FakeStorageBackend):
    """Object storage that cannot be read at all.

    Both questions fail, because a store that cannot be read cannot answer
    "is it there" either - a double that only broke ``get`` would let the HEAD
    route look healthier than the backend it is standing in for.
    """

    async def get(self, key):
        raise StorageUnavailableError("Object storage is unavailable.")

    async def exists(self, key):
        raise StorageUnavailableError("Object storage is unavailable.")


class _RefusingKeyStorage(FakeStorageBackend):
    """Storage that rejects the key the row is carrying, whichever way it is asked."""

    async def get(self, key):
        raise InvalidStorageKeyError(f"Storage key has an unsafe segment: {key!r}")

    async def exists(self, key):
        raise InvalidStorageKeyError(f"Storage key has an unsafe segment: {key!r}")


async def test_an_unusable_storage_key_is_404_not_a_forever_retry(db_sessionmaker) -> None:
    # The app mints every storage key itself, so one the seam refuses means a row
    # is carrying something no version of that code could have written. That is a
    # permanent fault: answering 503 "try again in a moment" would invite a retry
    # whose outcome can never change, so it must not share the outage branch.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = _RefusingKeyStorage(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker, storage, conversation_id=conversation_id, store_bytes=False
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


async def test_unreachable_storage_is_503_on_download(db_sessionmaker) -> None:
    # The file is fine; the store is down. A retry is the right advice, and a
    # 404 would tell the user their file is gone.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = _UnreadableStorage(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker, storage, conversation_id=conversation_id, store_bytes=False
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.get(_content_url(file_id))

    assert resp.status_code == 503
    assert resp.json()["code"] == "service_unavailable"


async def test_download_of_a_missing_or_malformed_id_is_404(db_sessionmaker) -> None:
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        assert (await client.get(_content_url(str(uuid4())))).status_code == 404
        assert (await client.get(_content_url("not-a-uuid"))).status_code == 404


async def test_another_user_in_the_same_tenant_cannot_download_the_file(
    db_sessionmaker,
) -> None:
    # The row is user-scoped, and this is the route that would hand over its
    # actual CONTENTS - the one place a tenant-only filter would not be enough.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(db_sessionmaker, storage, conversation_id=conversation_id)
    intruder = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")

    app = create_app()
    app.dependency_overrides[get_current_tenant] = lambda: intruder
    app.state.db_sessionmaker = db_sessionmaker
    app.state.storage = storage
    app.state.job_queue = queue
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(_content_url(file_id))
    app.dependency_overrides.clear()

    assert resp.status_code == 404
    assert resp.content != PDF_BYTES


async def test_another_tenant_cannot_download_the_file(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(db_sessionmaker, storage, conversation_id=conversation_id)
    other = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")

    app = create_app()
    app.dependency_overrides[get_current_tenant] = lambda: other
    app.state.db_sessionmaker = db_sessionmaker
    app.state.storage = storage
    app.state.job_queue = queue
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(_content_url(file_id))
    app.dependency_overrides.clear()

    assert resp.status_code == 404


# ------------------------------------------------------------------------ head


class _CountingStorage(FakeStorageBackend):
    """Records which question was asked of it, so a cheap route can be proven cheap."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] = []

    async def get(self, key):
        self.reads.append(key)
        return await super().get(key)


async def test_head_answers_the_download_question_without_reading_the_file(
    db_sessionmaker,
) -> None:
    # The whole reason the method exists. A client that checks before it
    # downloads would otherwise pull the object twice per saved file, on the one
    # route whose small rate budget exists precisely because it buffers a whole
    # object - so a check that reads nothing is what keeps a real download to a
    # single expensive request.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = _CountingStorage(), FakeJobQueue()
    file_id = await _seed_file(db_sessionmaker, storage, conversation_id=conversation_id)
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.head(_content_url(file_id))

    assert resp.status_code == 200
    assert resp.content == b""
    assert storage.reads == []


async def test_head_describes_the_file_exactly_as_the_download_would(db_sessionmaker) -> None:
    # A HEAD that named the file differently from the GET beside it would be
    # worse than no HEAD at all, so the headers are asserted against the GET's
    # own rather than against a copy of the expected strings.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        filename='ev"il\r\nX-Injected: yes\r\nreporté.pdf',
        content_type='text/plain; charset="><script>',
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        head = await client.head(_content_url(file_id))
        get = await client.get(_content_url(file_id))

    for header in (
        "content-type",
        "content-disposition",
        "content-length",
        "cache-control",
        "x-content-type-options",
    ):
        assert head.headers[header] == get.headers[header], header
    assert "x-injected" not in head.headers


async def test_head_does_not_advertise_an_empty_file(db_sessionmaker) -> None:
    # A bodiless response defaults to Content-Length: 0, which is a plain lie
    # about the file - and the one number a client asks HEAD for.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(db_sessionmaker, storage, conversation_id=conversation_id)
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.head(_content_url(file_id))

    assert resp.headers["content-length"] == str(len(PDF_BYTES))


async def test_head_serves_a_persons_file_whose_ingestion_failed(db_sessionmaker) -> None:
    # The cheap check and the download must agree about WHICH files are
    # servable; a HEAD that refused what the GET serves would have the UI hide
    # an affordance that works.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        status=ATTACHMENT_FAILED,
        origin=ATTACHMENT_ORIGIN_UPLOAD,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.head(_content_url(file_id))

    assert resp.status_code == 200


@pytest.mark.parametrize(
    ("status", "origin", "expected"),
    [
        (ATTACHMENT_EXPIRED, ATTACHMENT_ORIGIN_UPLOAD, 410),
        (ATTACHMENT_FAILED, ATTACHMENT_ORIGIN_GENERATED, 410),
        (ATTACHMENT_PENDING, ATTACHMENT_ORIGIN_UPLOAD, 409),
        (ATTACHMENT_PARSING, ATTACHMENT_ORIGIN_UPLOAD, 409),
    ],
)
async def test_head_refuses_exactly_what_the_download_refuses(
    db_sessionmaker, status: str, origin: str, expected: int
) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker,
        storage,
        conversation_id=conversation_id,
        status=status,
        origin=origin,
        store_bytes=status != ATTACHMENT_EXPIRED,
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        head = await client.head(_content_url(file_id))
        get = await client.get(_content_url(file_id))

    assert head.status_code == expected
    assert head.status_code == get.status_code


async def test_head_over_a_missing_blob_is_404(db_sessionmaker) -> None:
    # The row promises bytes that are not there. Answering 200 would send the
    # client on to a download that cannot work, which is the one thing a check
    # exists to prevent.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker, storage, conversation_id=conversation_id, store_bytes=False
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.head(_content_url(file_id))

    assert resp.status_code == 404


async def test_head_of_an_unusable_storage_key_is_404_not_a_forever_retry(
    db_sessionmaker,
) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = _RefusingKeyStorage(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker, storage, conversation_id=conversation_id, store_bytes=False
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.head(_content_url(file_id))

    assert resp.status_code == 404


async def test_head_of_an_unreachable_store_is_503(db_sessionmaker) -> None:
    # The file is fine; the store is down. "Try again" is the right advice, and
    # a 404 would tell the user their file is gone.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = _UnreadableStorage(), FakeJobQueue()
    file_id = await _seed_file(
        db_sessionmaker, storage, conversation_id=conversation_id, store_bytes=False
    )
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        resp = await client.head(_content_url(file_id))

    assert resp.status_code == 503


async def test_head_of_a_missing_malformed_or_foreign_id_is_404(db_sessionmaker) -> None:
    # The cheap route must not become the one that answers "does this id exist"
    # for somebody else's files.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    file_id = await _seed_file(db_sessionmaker, storage, conversation_id=conversation_id)
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        assert (await client.head(_content_url(str(uuid4())))).status_code == 404
        assert (await client.head(_content_url("not-a-uuid"))).status_code == 404

    intruder = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
    app = create_app()
    app.dependency_overrides[get_current_tenant] = lambda: intruder
    app.state.db_sessionmaker = db_sessionmaker
    app.state.storage = storage
    app.state.job_queue = queue
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.head(_content_url(file_id))
    app.dependency_overrides.clear()

    assert resp.status_code == 404


# ---------------------------------------------------------------------- delete


class DeadStore:
    """A vector store whose purge always fails, to prove a half-delete cannot happen."""

    async def delete_record(self, tenant_id, record_id, *, session_id=None) -> int:
        raise VectorStoreUnavailableError("The vector store is unavailable.")


FILE_TEXT = "Zarquon Industries reported a decline in quarterly turbine shipments."


async def _ingest_the_file(
    db_sessionmaker, store, attachment_id: str, conversation_id: str
) -> None:
    """Put the file's text where the ingest worker would have put it."""
    async with db_sessionmaker() as session:
        attachment = await AttachmentRepository(session, TENANT).get(UUID(attachment_id))
        assert attachment is not None
        await rag_service.ingest_structured(
            source_type="upload",
            tenant=TENANT,
            items=build_ingest_items(attachment, [ParsedDocument(ordinal=0, text=FILE_TEXT)]),
            embedder=FakeEmbedder(dim=32),
            store=store,
            settings=make_settings(),
            session_id=conversation_id,
        )


async def _chunk_count(store, conversation_id: str) -> int:
    """How many of the conversation's chunks are still retrievable."""
    hits = await store.query(
        "session",
        "fake-embed",
        str(DEFAULT_TENANT_ID),
        await FakeEmbedder(dim=32).embed_query("turbine shipments"),
        top_k=50,
        session_id=conversation_id,
    )
    return len(hits)


async def test_delete_removes_the_row_the_bytes_and_the_indexed_text(db_sessionmaker) -> None:
    # Removing a file from a turn has to actually remove it. An upload is
    # ingested the moment it lands, so a row-only delete would leave the text
    # retrievable and the assistant would go on answering from a file its owner
    # believes is gone.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue, store = FakeStorageBackend(), FakeJobQueue(), make_memory_store()
    async with attachments_client(
        db_sessionmaker, storage=storage, queue=queue, store=store
    ) as client:
        created = await client.post(URL, **_upload(conversation_id=conversation_id))
        attachment_id = created.json()["id"]
        await _ingest_the_file(db_sessionmaker, store, attachment_id, conversation_id)
        assert await _chunk_count(store, conversation_id) >= 1
        assert storage.objects  # bytes are there

        resp = await client.delete(f"{URL}/{attachment_id}")

        assert resp.status_code == 204
        assert resp.content == b""
        # Gone from all three places, not just the one the client can see.
        assert (await client.get(f"{URL}/{attachment_id}")).status_code == 404
        assert storage.objects == {}
        assert await _chunk_count(store, conversation_id) == 0


async def test_delete_leaves_the_file_intact_when_its_text_cannot_be_purged(
    db_sessionmaker,
) -> None:
    # A file whose row is gone but whose chunks survive is the worst outcome:
    # invisible to its owner, still quotable by the assistant. So a failed purge
    # deletes NOTHING and the whole request can simply be retried.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(
        db_sessionmaker, storage=storage, queue=queue, store=DeadStore()
    ) as client:
        created = await client.post(URL, **_upload(conversation_id=conversation_id))
        attachment_id = created.json()["id"]

        resp = await client.delete(f"{URL}/{attachment_id}")

        assert resp.status_code == 503
        assert resp.json()["code"] == "service_unavailable"
        # Still fully addressable: row AND bytes.
        assert (await client.get(f"{URL}/{attachment_id}")).status_code == 200
        assert storage.objects


async def test_delete_of_a_missing_or_malformed_id_is_404(db_sessionmaker) -> None:
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        assert (await client.delete(f"{URL}/{uuid4()}")).status_code == 404
        assert (await client.delete(f"{URL}/not-a-uuid")).status_code == 404


async def test_delete_is_idempotent_from_the_clients_point_of_view(db_sessionmaker) -> None:
    # A double-clicked remove must not 500; the second call finds nothing, which
    # is exactly the state the caller asked for.
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        created = await client.post(URL, **_upload(conversation_id=conversation_id))
        attachment_id = created.json()["id"]

        assert (await client.delete(f"{URL}/{attachment_id}")).status_code == 204
        assert (await client.delete(f"{URL}/{attachment_id}")).status_code == 404


async def test_another_tenant_cannot_delete_the_attachment(db_sessionmaker) -> None:
    conversation_id = await _make_conversation(db_sessionmaker)
    storage, queue = FakeStorageBackend(), FakeJobQueue()
    other = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="member")
    async with attachments_client(db_sessionmaker, storage=storage, queue=queue) as client:
        created = await client.post(URL, **_upload(conversation_id=conversation_id))
        attachment_id = created.json()["id"]

    app = create_app()
    app.dependency_overrides[get_current_tenant] = lambda: other
    app.state.db_sessionmaker = db_sessionmaker
    app.state.storage = storage
    app.state.job_queue = queue
    app.state.vector_store = make_memory_store()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete(f"{URL}/{attachment_id}")
    app.dependency_overrides.clear()

    assert resp.status_code == 404
    # And the owner's file is untouched by the attempt.
    assert storage.objects
