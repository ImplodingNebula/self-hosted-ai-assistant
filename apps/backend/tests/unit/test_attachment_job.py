"""The arq adapter around attachment ingestion, and the job-queue seam."""

from uuid import uuid4

import pytest
from arq import Retry
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.config import get_settings
from app.models import ATTACHMENT_FAILED, ATTACHMENT_PARSING, ATTACHMENT_READY, Conversation
from app.parsing.base import DocumentParser, ParsedDocument, ParserOutOfMemoryError
from app.parsing.registry import default_registry
from app.providers.base import ModelCapabilities, ModelInfo, ProviderUnavailableError
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.repositories.attachments import AttachmentRepository
from app.schemas.rag import RagIngestItem
from app.schemas.settings import WorkspaceDefaultsPatch
from app.services.rag_service import ingest_structured
from app.services.workspace_defaults import apply_defaults
from app.storage.base import StorageUnavailableError
from app.storage.local import FakeStorageBackend
from app.workers import queue as queue_module
from app.workers.main import (
    JOB_PARSE_AND_INGEST,
    MAX_TRIES,
    _defer_seconds,
    parse_and_ingest_attachment,
)
from app.workers.queue import ArqJobQueue, FakeJobQueue, JobQueueUnavailableError
from tests.conftest import make_memory_store
from tests.unit.test_attachment_ingest import TENANT, TEXT, _attachment

# arq registers the job by its function name; the shared constant is derived
# from it, so a real enqueue always targets a job the worker registered.
JOB_NAME = JOB_PARSE_AND_INGEST


async def _ctx(db_sessionmaker, storage, *, job_try: int = 1, embedder=True, store=None) -> dict:
    return {
        "settings": get_settings(),
        "db_sessionmaker": db_sessionmaker,
        "storage": storage,
        "parser_registry": default_registry(),
        "embedder": FakeEmbedder(dim=32) if embedder else None,
        "vector_store": store if store is not None else make_memory_store(),
        "job_try": job_try,
    }


async def _stored_attachment(db_sessionmaker, storage, **over):
    async with db_sessionmaker() as session:
        conversation = Conversation(tenant_id=TENANT.tenant_id, user_id=TENANT.user_id)
        session.add(conversation)
        await session.commit()
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation.id, **over))
        await repo.commit()
    await storage.put(attachment.storage_key, TEXT.encode())
    return attachment


async def _reload(db_sessionmaker, attachment_id):
    async with db_sessionmaker() as session:
        return await AttachmentRepository(session, TENANT).get(attachment_id)


async def test_the_job_ingests_and_reports_its_chunk_count(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    attachment = await _stored_attachment(db_sessionmaker, storage)

    chunks = await parse_and_ingest_attachment(
        await _ctx(db_sessionmaker, storage),
        str(attachment.id),
        str(TENANT.tenant_id),
        str(TENANT.user_id),
    )

    assert chunks >= 1
    reloaded = await _reload(db_sessionmaker, attachment.id)
    assert reloaded.status == ATTACHMENT_READY


async def test_a_vanished_attachment_is_skipped_not_retried(db_sessionmaker) -> None:
    # Deleted between enqueue and run. Raising would burn every retry on a row
    # that will never come back.
    chunks = await parse_and_ingest_attachment(
        await _ctx(db_sessionmaker, FakeStorageBackend()),
        str(uuid4()),
        str(TENANT.tenant_id),
        str(TENANT.user_id),
    )

    assert chunks == 0


async def test_a_vanished_attachment_takes_any_chunks_it_left_with_it(db_sessionmaker) -> None:
    """The orphan the retry path could otherwise resurrect, pinned.

    The file was deleted mid-ingest, so the job purged its own write - but that
    purge hit an unavailable vector store and the job was requeued. On the retry
    the row is gone, and a bare skip here would abandon the chunks the FIRST
    attempt wrote: nothing else knows the record id once the row is deleted, they
    carry no TTL, and they would go on answering questions forever. So the retry
    finishes the cleanup its predecessor could not.
    """
    storage = FakeStorageBackend()
    store = make_memory_store()
    conversation_id = uuid4()
    orphan_id = uuid4()

    # Stand in for the chunks a previous attempt wrote just before the row went.
    await ingest_structured(
        source_type="upload",
        tenant=TENANT,
        items=[RagIngestItem(id=str(orphan_id), record_id=str(orphan_id), text=TEXT)],
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=get_settings(),
        session_id=str(conversation_id),
        persist=True,
    )

    chunks = await parse_and_ingest_attachment(
        await _ctx(db_sessionmaker, storage, store=store),
        str(orphan_id),
        str(TENANT.tenant_id),
        str(TENANT.user_id),
    )

    assert chunks == 0
    hits = await store.query(
        "session",
        "fake-embed",
        str(TENANT.tenant_id),
        await FakeEmbedder(dim=32).embed_query("deployment"),
        top_k=50,
        session_id=str(conversation_id),
    )
    assert hits == []


async def test_another_tenants_job_cannot_touch_the_row(db_sessionmaker) -> None:
    # The tenant comes from the job arguments and the row is loaded through a
    # repository bound to it, so a mismatched tenant finds nothing at all.
    storage = FakeStorageBackend()
    attachment = await _stored_attachment(db_sessionmaker, storage)

    chunks = await parse_and_ingest_attachment(
        await _ctx(db_sessionmaker, storage),
        str(attachment.id),
        str(uuid4()),
        str(uuid4()),
    )

    assert chunks == 0
    reloaded = await _reload(db_sessionmaker, attachment.id)
    assert reloaded.status != ATTACHMENT_READY


class _DeadStorage(FakeStorageBackend):
    async def get(self, key: str) -> bytes:
        raise StorageUnavailableError("Object storage is unavailable.")


async def test_a_transient_failure_asks_arq_to_retry(db_sessionmaker) -> None:
    # arq requeues a job ONLY when it raises Retry (or CancelledError); a bare
    # exception is finished, not retried. So a transient failure with attempts
    # left must raise Retry, or the row would silently wedge in `parsing`.
    storage = _DeadStorage()
    attachment = await _stored_attachment(db_sessionmaker, storage)

    with pytest.raises(Retry):
        await parse_and_ingest_attachment(
            await _ctx(db_sessionmaker, storage, job_try=1),
            str(attachment.id),
            str(TENANT.tenant_id),
            str(TENANT.user_id),
        )

    # Still processing: the row waits in `parsing` for the requeued attempt.
    reloaded = await _reload(db_sessionmaker, attachment.id)
    assert reloaded.status == ATTACHMENT_PARSING
    assert reloaded.error is None


async def test_the_last_attempt_fails_the_row_and_stops_retrying(db_sessionmaker) -> None:
    # On the final attempt there is nothing left to retry, so the job must NOT
    # raise Retry (which would requeue past the limit) - it re-raises the real
    # error and leaves a row that explains itself, rather than one stuck in
    # `parsing` forever.
    storage = _DeadStorage()
    attachment = await _stored_attachment(db_sessionmaker, storage)

    with pytest.raises(StorageUnavailableError):
        await parse_and_ingest_attachment(
            await _ctx(db_sessionmaker, storage, job_try=MAX_TRIES),
            str(attachment.id),
            str(TENANT.tenant_id),
            str(TENANT.user_id),
        )

    reloaded = await _reload(db_sessionmaker, attachment.id)
    assert reloaded.status == ATTACHMENT_FAILED
    assert "may be unavailable" in reloaded.error


class _StarvedParser(DocumentParser):
    """A parser whose reader gets killed for memory, as the kernel would kill it."""

    async def parse(self, data, *, filename, media_type=None) -> list[ParsedDocument]:
        raise ParserOutOfMemoryError("The PDF could not be read (it ran out of memory).")


class _StarvedRegistry:
    """A registry that hands back the starved parser whatever the file is."""

    def resolve(self, *, filename: str, content_type=None, data=None) -> DocumentParser:
        return _StarvedParser()


async def _starved_ctx(db_sessionmaker, storage, *, job_try: int) -> dict:
    ctx = await _ctx(db_sessionmaker, storage, job_try=job_try)
    ctx["parser_registry"] = _StarvedRegistry()
    return ctx


async def test_a_reader_killed_for_memory_is_retried_rather_than_blamed_on_the_file(
    db_sessionmaker,
) -> None:
    # An out-of-memory kill is the ONE parse failure that says nothing about the
    # file: something else on the machine took the memory, and the same bytes may
    # read perfectly a moment later. Treated like the other parsing errors it
    # would be recorded as permanent on the first try, and a perfectly good file
    # would be dead on arrival because a model happened to be loading beside it.
    storage = FakeStorageBackend()
    attachment = await _stored_attachment(db_sessionmaker, storage)

    with pytest.raises(Retry):
        await parse_and_ingest_attachment(
            await _starved_ctx(db_sessionmaker, storage, job_try=1),
            str(attachment.id),
            str(TENANT.tenant_id),
            str(TENANT.user_id),
        )

    reloaded = await _reload(db_sessionmaker, attachment.id)
    assert reloaded.status == ATTACHMENT_PARSING
    assert reloaded.error is None


async def test_a_file_that_never_fits_says_so_on_the_last_attempt(db_sessionmaker) -> None:
    # Retried to the end and still killed: the file really is too big for this
    # machine. The row must say THAT, not the generic "a service may be
    # unavailable" - the way out is a smaller file, not waiting for a service.
    storage = FakeStorageBackend()
    attachment = await _stored_attachment(db_sessionmaker, storage)

    with pytest.raises(ParserOutOfMemoryError):
        await parse_and_ingest_attachment(
            await _starved_ctx(db_sessionmaker, storage, job_try=MAX_TRIES),
            str(attachment.id),
            str(TENANT.tenant_id),
            str(TENANT.user_id),
        )

    reloaded = await _reload(db_sessionmaker, attachment.id)
    assert reloaded.status == ATTACHMENT_FAILED
    assert "memory" in reloaded.error


class _DeadEmbedder(FakeEmbedder):
    """An embedding backend that is down - the commonest transient failure there is."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise ProviderUnavailableError("The embedding backend is unavailable.")


async def test_a_last_attempt_that_fails_mid_ingest_still_explains_itself(
    db_sessionmaker,
) -> None:
    """The row must end `failed`, not `parsing`, however far into the job it got.

    The ingest stamps the file's content hash on the row before it parses, so by
    the time anything downstream fails - the embedder here, the vector store, a
    reader killed for memory - the row is carrying an uncommitted change. Rolling
    that back leaves the job holding an expired object, and reading so much as the
    tenant off one is a database round trip that a plain attribute access cannot
    await. The handler used to swallow that and log; the row stayed in `parsing`,
    and the person who uploaded the file watched a spinner that had already given
    up on them. Which is the one thing the final attempt exists to prevent.
    """
    storage = FakeStorageBackend()
    attachment = await _stored_attachment(db_sessionmaker, storage)
    ctx = await _ctx(db_sessionmaker, storage, job_try=MAX_TRIES)
    ctx["embedder"] = _DeadEmbedder(dim=32)

    with pytest.raises(ProviderUnavailableError):
        await parse_and_ingest_attachment(
            ctx, str(attachment.id), str(TENANT.tenant_id), str(TENANT.user_id)
        )

    reloaded = await _reload(db_sessionmaker, attachment.id)
    assert reloaded.status == ATTACHMENT_FAILED
    assert "may be unavailable" in reloaded.error


def test_a_starved_reader_waits_out_the_thing_that_starved_it() -> None:
    # The cause this error names - a model loading beside the worker, a big ingest
    # next to it - holds its memory for a minute or more. Retrying inside the
    # ordinary ten-second backoff just gets the reader killed again, three times,
    # and then the file is blamed for being too large.
    memory = _defer_seconds(ParserOutOfMemoryError("killed"), 1)
    ordinary = _defer_seconds(StorageUnavailableError("down"), 1)

    assert memory > ordinary
    assert memory >= 60


async def test_a_deployment_without_an_embedder_fails_loudly(db_sessionmaker) -> None:
    # Not the file's fault: nothing configures an embedding model. A silent
    # `failed` row would blame the upload for a deployment mistake.
    storage = FakeStorageBackend()
    attachment = await _stored_attachment(db_sessionmaker, storage)

    with pytest.raises(RuntimeError, match="No embedding model"):
        await parse_and_ingest_attachment(
            await _ctx(db_sessionmaker, storage, embedder=False),
            str(attachment.id),
            str(TENANT.tenant_id),
            str(TENANT.user_id),
        )


async def test_the_job_embeds_with_its_own_workspace_choice(db_sessionmaker) -> None:
    # Resolved PER JOB, not once at worker startup. A process that resolved the
    # embedder at boot would keep writing into the old vector space after the
    # choice changed - and into a DIFFERENT one from the one the backend's
    # queries read, which raises no error anywhere: a query against an empty
    # collection simply returns nothing.
    #
    # The server holds exactly one embedding-capable tag, which is the rung
    # BELOW the workspace choice in the resolution chain and would answer on
    # its own. So the assertion has something to fail on: pin the startup
    # resolution back into ctx, or drop the stored choice, and the space comes
    # out named after the installed tag instead.
    async with db_sessionmaker() as session:
        await apply_defaults(session, TENANT, WorkspaceDefaultsPatch(embedding_model="chosen:2b"))
        await session.commit()
    storage = FakeStorageBackend()
    attachment = await _stored_attachment(db_sessionmaker, storage)
    ctx = await _ctx(db_sessionmaker, storage)
    # The KEY absent is what production looks like; its presence is the test
    # injection point.
    del ctx["embedder"]
    ctx["settings"] = get_settings().model_copy(update={"model_provider": "fake"})
    ctx["model_provider"] = FakeProvider(
        models=[ModelInfo(name="installed-embed:1b")],
        capabilities_by_model={
            "installed-embed:1b": ModelCapabilities(embedding=True, completion=False)
        },
    )

    await parse_and_ingest_attachment(
        ctx, str(attachment.id), str(TENANT.tenant_id), str(TENANT.user_id)
    )

    spaces = await ctx["vector_store"].list_spaces()
    assert [tag for space in spaces for tag in space.embed_models] == ["chosen:2b"]


async def test_the_fake_queue_records_enqueues() -> None:
    queue = FakeJobQueue()

    job_id = await queue.enqueue(JOB_NAME, "abc", tenant_id="t")

    assert job_id is not None
    assert queue.calls == [(JOB_NAME, ("abc",), {"tenant_id": "t"})]


async def test_a_repeated_job_id_queues_one_job() -> None:
    # A double-submitted upload must not parse and embed the same file twice.
    queue = FakeJobQueue()

    first = await queue.enqueue(JOB_NAME, job_id="attachment:1")
    second = await queue.enqueue(JOB_NAME, job_id="attachment:1")

    assert first == "attachment:1"
    assert second is None
    assert len(queue.calls) == 1


async def test_an_unreachable_queue_raises_the_seam_error(monkeypatch) -> None:
    # A route turns this into a 503: the work was never accepted, so reporting
    # it as queued would be a lie.
    async def _refuse(*_args, **_kwargs):
        raise RedisConnectionError("connection refused")

    monkeypatch.setattr(queue_module, "create_pool", _refuse)

    with pytest.raises(JobQueueUnavailableError):
        await ArqJobQueue("redis://localhost:6379/0").enqueue(JOB_NAME)


async def test_the_pool_is_built_once_and_reused(monkeypatch) -> None:
    class _Pool:
        def __init__(self) -> None:
            self.jobs: list[str] = []

        async def enqueue_job(self, name, *args, _job_id=None, **kwargs):
            self.jobs.append(name)
            return type("Job", (), {"job_id": _job_id or "generated"})()

    pool = _Pool()
    builds = 0

    async def _create(*_args, **_kwargs):
        nonlocal builds
        builds += 1
        return pool

    monkeypatch.setattr(queue_module, "create_pool", _create)
    queue = ArqJobQueue("redis://localhost:6379/0")

    assert builds == 0  # construction opens no connection
    assert await queue.enqueue(JOB_NAME) == "generated"
    assert await queue.enqueue(JOB_NAME, job_id="x") == "x"
    assert builds == 1
    assert pool.jobs == [JOB_NAME, JOB_NAME]
