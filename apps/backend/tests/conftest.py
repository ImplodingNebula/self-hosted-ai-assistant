"""Shared test fixtures. Real model calls are replaced by the FakeProvider.

The database is in-memory sqlite (aiosqlite) with the schema created straight
from ``SQLModel.metadata`` - fast, hermetic, no postgres needed. StaticPool
pins the single in-memory database to one connection so every session sees the
same data.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

import app.models  # noqa: F401  - register tables on SQLModel.metadata before create_all
import app.workers.queue as queue_module
from app.api.deps import get_model_provider
from app.core.concurrency import StreamLimiter
from app.core.config import get_settings
from app.db.seed import seed_defaults
from app.main import create_app
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.providers.rag.qdrant_store import QdrantVectorStore
from app.services.earned import clear_earned_cache
from app.services.model_facts import clear_facts_cache
from app.services.model_identity import clear_identity_cache
from app.services.model_policy import clear_policy_cache
from app.services.vector_spaces import clear_vector_space_cache
from app.storage.local import FakeStorageBackend
from app.workers.queue import FakeJobQueue


@pytest.fixture(autouse=True)
def _isolate_identity_cache():
    """Drop the installed-digest cache around every test.

    Keyed on the PROVIDER NAME alone with a multi-minute TTL, and every suite
    here runs as ``fake`` while inventing its own digests - so without this a
    warm-path or measured-think read in one test resolves a declaration
    against another test's weights. The chat path is immune (it hands in the
    digests from its own listing); the paths that are not are exactly the ones
    a cross-suite order change would break, silently and only sometimes.
    """
    clear_identity_cache()
    yield
    clear_identity_cache()


@pytest.fixture(autouse=True)
def _fresh_earned_cache() -> Iterator[None]:
    """Isolate the module-level earned-verdict caches between tests."""
    clear_earned_cache()
    yield
    clear_earned_cache()


@pytest.fixture(autouse=True)
def _fresh_policy_cache() -> Iterator[None]:
    """Isolate the module-level policy-declaration cache between tests.

    Same five-minute window as the earned lookup's, and the same hazard: one
    test's stored declaration would otherwise answer for every later test that
    names the same tag on the same provider.
    """
    clear_policy_cache()
    yield
    clear_policy_cache()


@pytest.fixture(autouse=True)
def _fresh_vector_space_cache() -> Iterator[None]:
    """Isolate the cached stranded-vectors report between tests.

    Its TTL is fifteen minutes, so without this one test's store would answer
    for every later one - and a suite is comfortably shorter than the window.
    """
    clear_vector_space_cache()
    yield
    clear_vector_space_cache()


@pytest.fixture(autouse=True)
def _fresh_facts_cache() -> Iterator[None]:
    """Isolate the cached runtime capability reports between tests.

    Keyed on (provider name, tag) with a five-minute TTL, so without this one
    test's scripted FakeProvider would answer for every later test that names
    the same tag - the same hazard the two cache fixtures above exist for,
    now on the chat path itself (window resolution and routing both probe).
    """
    clear_facts_cache()
    yield
    clear_facts_cache()


class RealJobQueueReached(BaseException):
    """A test tried to open a real Redis pool. See :func:`_no_real_job_queue`.

    Derived from BaseException on purpose: the callers that hand work to the
    queue swallow ``Exception`` so a background hand-off can never break a
    finished request, which would swallow this too and leave the test green
    while it talked to the developer's own Redis.
    """


@pytest.fixture(autouse=True)
def _no_real_job_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make "the harness forgot ``app.state.job_queue``" fail loudly.

    Every route that hands work to a worker depends on ``get_job_queue``, whose
    fallback builds a real :class:`ArqJobQueue`. A test app that does not set
    ``app.state.job_queue`` therefore reaches for Redis: on a machine without
    one that is ten seconds of connect retries per request (green, but slow and
    silent), and on a developer's box with Redis published it is worse - the
    test enqueues real jobs. Blocking the connect turns both into an immediate,
    named failure, so a new harness cannot quietly acquire a live queue.

    A test that genuinely exercises :class:`ArqJobQueue` patches ``create_pool``
    itself; that patch is applied after this one and wins.
    """

    async def _refuse(*_args: object, **_kwargs: object) -> None:
        raise RealJobQueueReached(
            "This test reached the real job queue. Set app.state.job_queue = FakeJobQueue() "
            "on the test app (or pass a fake queue to the service under test)."
        )

    monkeypatch.setattr(queue_module, "create_pool", _refuse)


@pytest.fixture
def fake_provider() -> FakeProvider:
    """A default fake model provider."""
    return FakeProvider()


@pytest.fixture
def fake_storage() -> FakeStorageBackend:
    """An in-memory object store."""
    return FakeStorageBackend()


@pytest.fixture
def fake_job_queue() -> FakeJobQueue:
    """A job queue that records enqueues instead of reaching Redis."""
    return FakeJobQueue()


def make_memory_store() -> QdrantVectorStore:
    """A vector store over qdrant-client's in-process local mode.

    Runs the real client code paths (filters, counts, delete-by-filter)
    without a server, so API tests exercise genuine store behavior while
    staying hermetic.
    """
    from qdrant_client import AsyncQdrantClient

    return QdrantVectorStore(client=AsyncQdrantClient(":memory:"))


@pytest.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    """An in-memory sqlite engine with the full schema created."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    # sqlite ignores FK constraints (incl. ON DELETE CASCADE) unless the
    # pragma is enabled per connection - postgres enforces them natively.
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fks(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_sessionmaker(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """A session factory over the test engine, with the default tenant seeded."""
    maker = async_sessionmaker(db_engine, expire_on_commit=False)
    await seed_defaults(maker)
    return maker


@pytest.fixture
async def client(
    fake_provider: FakeProvider,
    fake_storage: FakeStorageBackend,
    fake_job_queue: FakeJobQueue,
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncClient]:
    """An httpx client bound to the app in-process, with the model faked.

    ``app.state`` is populated by hand because ASGITransport does not run the
    lifespan - endpoints then use the sqlite sessionmaker instead of the lazy
    postgres fallback, the in-memory vector store instead of a network
    Qdrant, and the fake embedder instead of the resolved one.

    Storage and the job queue are faked for the same reason and one more: their
    lazy fallbacks build a real ``LocalFsStorage`` rooted at STORAGE_LOCAL_DIR
    and a real Redis pool, so a route test would write to the host and hang on
    a connect.
    """
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: fake_provider
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(get_settings().max_streams_per_tenant)
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.storage = fake_storage
    app.state.job_queue = fake_job_queue
    # The rate limiter counts through the same seam the lifespan hands it in
    # production, so route tests run against the SHIPPED configuration rather
    # than against a limiter that quietly failed open. The budgets are per
    # minute and the queue is per test, so no single test comes near them.
    app.state.rate_counter = fake_job_queue
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
    app.dependency_overrides.clear()
