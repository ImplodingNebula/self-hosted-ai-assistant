"""Job-enqueue seam: the JobQueue abstraction over arq.

Handing work to a background worker is the one thing a request does that must
not depend on arq being importable, reachable, or even real. A route calls
:meth:`JobQueue.enqueue` and gets back a job id or a typed unavailability
error; nothing about Redis, pickling, or arq's job protocol crosses this line.

Same shape as the storage, RAG and search seams: an ABC with one abstract async
method, plain-Exception seam errors, and an in-module ``Fake`` beside the real
implementation so route tests need no Redis.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from redis.exceptions import RedisError


class JobQueueError(Exception):
    """Base class for job-queue seam failures."""


class JobQueueUnavailableError(JobQueueError):
    """Raised when the queue backend cannot be reached.

    A route turns this into a 503: the work was never accepted, so telling the
    client it was queued would be a lie.
    """


class JobQueue(ABC):
    """The job-enqueue seam. First impl: :class:`ArqJobQueue`."""

    @abstractmethod
    async def ping(self) -> None:
        """Check that the queue backend is reachable, and do nothing else.

        The readiness probe asks this. A deployment whose queue is down still
        accepts uploads it can never parse, so the API must report itself
        unready rather than look healthy while every background job is
        silently dropped.

        Raises:
            JobQueueUnavailableError: The queue backend is unreachable.
        """
        raise NotImplementedError

    @abstractmethod
    async def enqueue(
        self, job_name: str, /, *args: Any, job_id: str | None = None, **kwargs: Any
    ) -> str | None:
        """Queue ``job_name`` for a background worker to run.

        Args:
            job_name: The worker function's registered name.
            *args: Positional arguments passed to the worker function.
            job_id: A caller-chosen id that makes the enqueue idempotent. A
                second enqueue with the same id is dropped while the first is
                queued or running AND while its result is still cached (arq keeps
                a finished job's result for ``keep_result`` seconds, one hour by
                default), so a reused id will not re-run for that window even
                after the job finishes. Omit for at-least-once semantics.
            **kwargs: Keyword arguments passed to the worker function.

        Returns:
            The job id, or None when ``job_id`` names a job that is already
            queued, running, or whose recent result is still cached - a
            duplicate submission, not an error.

        Raises:
            JobQueueUnavailableError: The queue backend is unreachable.
        """
        raise NotImplementedError


class ArqJobQueue(JobQueue):
    """arq over Redis; the connection pool opens on first enqueue.

    Construction performs no I/O, matching every other singleton factory, so a
    Redis that is still booting cannot block or crash the API's startup. The
    first enqueue pays the connect, and a Redis outage surfaces there as
    :class:`JobQueueUnavailableError` rather than as an unhandled arq error.
    """

    def __init__(self, redis_url: str) -> None:
        self._settings = RedisSettings.from_dsn(redis_url)
        self._pool: ArqRedis | None = None
        # Concurrent first-enqueues would otherwise each build a pool and all
        # but one would leak (only the last assignment to _pool survives).
        self._lock = asyncio.Lock()

    async def _get_pool(self) -> ArqRedis:
        """Return the shared pool, connecting on first use."""
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is None:
                try:
                    self._pool = await create_pool(self._settings)
                except (RedisError, OSError) as exc:
                    raise JobQueueUnavailableError("The job queue is unavailable.") from exc
        return self._pool

    async def ping(self) -> None:
        """Round-trip the queue's Redis; see :meth:`JobQueue.ping`."""
        pool = await self._get_pool()
        try:
            await pool.ping()
        except (RedisError, OSError) as exc:
            raise JobQueueUnavailableError("The job queue is unavailable.") from exc

    async def incr_window(self, key: str, ttl_seconds: int) -> int:
        """Increment a shared expiring counter; satisfies ``core.ratelimit.RateCounter``.

        Not part of the JobQueue contract - enqueueing is - but the process
        already holds one Redis connection pool here, and the rate limiter
        needs a counter on exactly that Redis. Lending the pool is cheaper than
        opening a second one for a single INCR per request.

        The increment and the expiry go in ONE pipeline, so a limited request
        costs a single round trip. The expiry is re-armed on every hit rather
        than only on creation: asking whether the key is new would cost the
        round trip this saves, and the key already names its own window, so a
        later expiry changes nothing about what it counts.

        Args:
            key: The counter's name, already scoped to its window.
            ttl_seconds: How long the counter lives.

        Returns:
            The counter's value after this increment.

        Raises:
            JobQueueUnavailableError: The backend is unreachable.
        """
        pool = await self._get_pool()
        try:
            pipeline = pool.pipeline(transaction=False)
            pipeline.incr(key)
            pipeline.expire(key, ttl_seconds)
            count, _expired = await pipeline.execute()
        except (RedisError, OSError) as exc:
            raise JobQueueUnavailableError("The job queue is unavailable.") from exc
        return int(count)

    # ---- KeyValueStore -------------------------------------------------
    # Also not part of the JobQueue contract, and here for the same reason
    # incr_window is: the process holds exactly one Redis pool and the login
    # sessions need an expiring key-value store on it. See
    # app.services.auth.store.KeyValueStore for the protocol and for why this
    # one FAILS CLOSED where the rate limiter's counter fails open.

    async def kv_get(self, key: str) -> str | None:
        """Read a key; see :meth:`app.services.auth.store.KeyValueStore.kv_get`."""
        pool = await self._get_pool()
        try:
            value = await pool.get(key)
        except (RedisError, OSError) as exc:
            raise JobQueueUnavailableError("The job queue is unavailable.") from exc
        if value is None:
            return None
        return value.decode() if isinstance(value, bytes) else str(value)

    async def kv_set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Write a key with an expiry; see :class:`~app.services.auth.store.KeyValueStore`."""
        pool = await self._get_pool()
        try:
            await pool.set(key, value, ex=ttl_seconds)
        except (RedisError, OSError) as exc:
            raise JobQueueUnavailableError("The job queue is unavailable.") from exc

    async def kv_expire(self, key: str, ttl_seconds: int) -> None:
        """Re-arm a key's expiry; see :class:`~app.services.auth.store.KeyValueStore`."""
        pool = await self._get_pool()
        try:
            await pool.expire(key, ttl_seconds)
        except (RedisError, OSError) as exc:
            raise JobQueueUnavailableError("The job queue is unavailable.") from exc

    async def kv_delete(self, key: str) -> None:
        """Delete a key; see :class:`~app.services.auth.store.KeyValueStore`."""
        pool = await self._get_pool()
        try:
            await pool.delete(key)
        except (RedisError, OSError) as exc:
            raise JobQueueUnavailableError("The job queue is unavailable.") from exc

    async def kv_consume(self, key: str) -> str | None:
        """Atomically read and delete a key.

        GETDEL, one command, so two concurrent callbacks bearing the same
        ``state`` cannot both succeed - which is the entire guarantee that makes
        a login's state single-use. A GET followed by a DEL would be a race both
        of them win.
        """
        pool = await self._get_pool()
        try:
            value = await pool.getdel(key)
        except (RedisError, OSError) as exc:
            raise JobQueueUnavailableError("The job queue is unavailable.") from exc
        if value is None:
            return None
        return value.decode() if isinstance(value, bytes) else str(value)

    async def enqueue(
        self, job_name: str, /, *args: Any, job_id: str | None = None, **kwargs: Any
    ) -> str | None:
        """Queue a job on arq; see :meth:`JobQueue.enqueue`."""
        pool = await self._get_pool()
        try:
            job = await pool.enqueue_job(job_name, *args, _job_id=job_id, **kwargs)
        except (RedisError, OSError) as exc:
            raise JobQueueUnavailableError("The job queue is unavailable.") from exc
        # arq returns None when _job_id collides with a job that is still queued,
        # running, or recently finished (its result is cached): the work is
        # already handled, which is exactly what the caller wanted.
        return None if job is None else job.job_id

    async def aclose(self) -> None:
        """Release the pool (called by the app lifespan on shutdown)."""
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None


class FakeJobQueue(JobQueue):
    """Records enqueues instead of performing them; for tests.

    Honours ``job_id`` deduplication so a test can assert that a double
    submission queues one job, exactly as arq would.
    """

    def __init__(self, *, unavailable: bool = False) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.job_ids: set[str] = set()
        self.counters: dict[str, int] = {}
        self.values: dict[str, str] = {}
        self.unavailable = unavailable

    async def kv_get(self, key: str) -> str | None:
        """Read from the in-memory map; see :class:`~app.services.auth.store.KeyValueStore`."""
        if self.unavailable:
            raise JobQueueUnavailableError("The job queue is unavailable.")
        return self.values.get(key)

    async def kv_set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Write to the in-memory map, ignoring the TTL.

        Expiry is not simulated: an instance is as short-lived as the test
        holding it, and a test that needs an EXPIRED key deletes it, which is
        what expiry looks like from the reader's side anyway.
        """
        if self.unavailable:
            raise JobQueueUnavailableError("The job queue is unavailable.")
        self.values[key] = value

    async def kv_expire(self, key: str, ttl_seconds: int) -> None:
        """Accept and ignore a re-arm; see :meth:`kv_set`."""
        if self.unavailable:
            raise JobQueueUnavailableError("The job queue is unavailable.")

    async def kv_delete(self, key: str) -> None:
        """Remove a key; see :class:`~app.services.auth.store.KeyValueStore`."""
        if self.unavailable:
            raise JobQueueUnavailableError("The job queue is unavailable.")
        self.values.pop(key, None)

    async def kv_consume(self, key: str) -> str | None:
        """Read and remove in one step, as GETDEL does."""
        if self.unavailable:
            raise JobQueueUnavailableError("The job queue is unavailable.")
        return self.values.pop(key, None)

    async def ping(self) -> None:
        """Answer as the configured availability dictates; see :meth:`JobQueue.ping`."""
        if self.unavailable:
            raise JobQueueUnavailableError("The job queue is unavailable.")

    async def incr_window(self, key: str, ttl_seconds: int) -> int:
        """Count in memory; see :meth:`ArqJobQueue.incr_window`.

        The TTL is accepted and ignored: every key already carries its window,
        so a new window is a new key and an expired one is simply never read
        again. An instance is as short-lived as the test holding it.
        """
        if self.unavailable:
            raise JobQueueUnavailableError("The job queue is unavailable.")
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def enqueue(
        self, job_name: str, /, *args: Any, job_id: str | None = None, **kwargs: Any
    ) -> str | None:
        """Record the call; see :meth:`JobQueue.enqueue`."""
        if self.unavailable:
            raise JobQueueUnavailableError("The job queue is unavailable.")
        if job_id is not None and job_id in self.job_ids:
            return None
        if job_id is not None:
            self.job_ids.add(job_id)
        self.calls.append((job_name, args, kwargs))
        return job_id or f"fake-job-{len(self.calls)}"
