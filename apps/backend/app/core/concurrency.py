"""Per-principal concurrency limits for long-lived chat streams.

One process-wide :class:`StreamLimiter` lives on ``app.state``. A plain
counter dict guarded by an ``asyncio.Lock`` is enough here: streams are few
and long-lived, so contention on the lock is negligible, and (unlike a
semaphore per key) the dict never leaks entries for principals that stopped
streaming.
"""

from __future__ import annotations

import asyncio
from uuid import UUID


class StreamLimiter:
    """Caps the concurrent streams one user of one tenant may hold.

    The budget is per USER within a workspace, not per workspace. Keyed on the
    tenant as well because a user id is only unique inside its tenant, and a
    key that dropped the tenant would let two workspaces collide on it.

    Why per user: a single long agentic turn can hold a slot for minutes, and a
    workspace-wide budget of two means one colleague running two of them locks
    everyone else out of the assistant entirely. The cap's PURPOSE is to fail a
    request fast rather than queue it behind a GPU that serializes generations
    anyway - which is a statement about one person's own requests, not about
    how many people may use the deployment at once.

    Single-user deployments are unaffected: every request carries the same pair
    of ids today, so the counter behaves exactly as the per-tenant one did.

    Callers must pair every successful :meth:`acquire` with exactly one
    :meth:`release` (a ``try/finally`` around the stream). Over-releasing is
    tolerated - the count never goes below zero - so a double-release bug
    degrades to a slightly generous limit instead of a poisoned counter.
    """

    def __init__(self, max_per_user: int) -> None:
        self._max = max_per_user
        self._lock = asyncio.Lock()
        self._active: dict[tuple[UUID, UUID], int] = {}

    async def acquire(self, tenant_id: UUID, user_id: UUID) -> bool:
        """Try to claim a stream slot; return False when this user is at cap."""
        key = (tenant_id, user_id)
        async with self._lock:
            current = self._active.get(key, 0)
            if current >= self._max:
                return False
            self._active[key] = current + 1
            return True

    async def release(self, tenant_id: UUID, user_id: UUID) -> None:
        """Return a stream slot; entries at zero are dropped to avoid growth."""
        key = (tenant_id, user_id)
        async with self._lock:
            current = self._active.get(key, 0)
            if current <= 1:
                self._active.pop(key, None)
            else:
                self._active[key] = current - 1

    def any_active(self) -> bool:
        """Whether anyone is currently streaming (GPU-priority signal).

        Deliberately not per principal: the question it answers is "is the GPU
        busy generating", and the GPU is shared by the whole deployment.

        Lock-free on purpose: a single dict read is atomic under the event
        loop, and callers only use this as a soft hint to pause background
        embedding work - a stale answer costs one extra pause, nothing more.
        """
        return bool(self._active)
