"""Worker tests: the TTL sweep job and its cron cadence mapping."""

from qdrant_client import AsyncQdrantClient

from app.providers.rag.base import ChunkPayload, ChunkPoint
from app.providers.rag.qdrant_store import QdrantVectorStore
from app.workers.main import WorkerSettings, _sweep_minutes, sweep_expired_vectors

SETTING = "RAG_SWEEP_INTERVAL_MINUTES"


def test_sweep_minutes_even_divisors_pass_through() -> None:
    assert _sweep_minutes(10, setting=SETTING) == {0, 10, 20, 30, 40, 50}
    assert _sweep_minutes(30, setting=SETTING) == {0, 30}
    assert _sweep_minutes(1, setting=SETTING) == set(range(60))


def test_sweep_minutes_rounds_down_uneven_intervals() -> None:
    assert _sweep_minutes(45, setting=SETTING) == {0, 30}
    assert _sweep_minutes(7, setting=SETTING) == {0, 6, 12, 18, 24, 30, 36, 42, 48, 54}
    assert _sweep_minutes(0, setting=SETTING) == set(range(60))


def test_sweep_minutes_hourly_and_beyond() -> None:
    assert _sweep_minutes(60, setting=SETTING) == {0}
    assert _sweep_minutes(90, setting=SETTING) == {0}


def _point(ttl_at: float) -> ChunkPoint:
    return ChunkPoint(
        id="7c9a8a10-0000-5000-8000-000000000001",
        vector=[0.1] * 8,
        payload=ChunkPayload(
            tenant_id="00000000-0000-0000-0000-000000000001",
            session_id="00000000-0000-0000-0000-0000000000aa",
            doc_id="d",
            text="x",
            created_at=1.0,
            ttl_at=ttl_at,
            embed_model="fake-embed",
        ),
    )


async def test_sweep_job_removes_expired_points() -> None:
    store = QdrantVectorStore(client=AsyncQdrantClient(":memory:"))
    await store.ensure_ready("fake-embed", 8)
    await store.upsert(
        "session", "fake-embed", "00000000-0000-0000-0000-000000000001", [_point(ttl_at=100.0)]
    )
    swept = await sweep_expired_vectors({"vector_store": store})
    assert swept == 1
    assert await sweep_expired_vectors({"vector_store": store}) == 0
    await store.aclose()


async def test_sweep_job_tolerates_unreachable_store() -> None:
    store = QdrantVectorStore(url="http://127.0.0.1:1", timeout=1)
    assert await sweep_expired_vectors({"vector_store": store}) == 0
    await store.aclose()


def test_worker_settings_shape() -> None:
    # The compose healthcheck and cron cadence depend on these staying set.
    assert WorkerSettings.health_check_interval == 60
    # Three crons: the session-vector TTL sweep, the attachment garbage
    # collector, and the sweep that fails uploads no worker is processing.
    assert len(WorkerSettings.cron_jobs) == 3
