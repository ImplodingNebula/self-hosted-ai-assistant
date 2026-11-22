"""Sessions and pending logins - the two short-lived records a login needs.

Both live in Redis, and neither gets a table. A session is a key with an expiry
by nature: it is created, it is read, it slides, and it disappears - which is a
row whose every operation would be a write and whose cleanup would be another
background sweep. A pending login is even shorter (ten minutes) and must be
consumed ATOMICALLY, which is one Redis verb and a transaction plus a lock in a
relational store.

Sessions are SERVER-SIDE on purpose, rather than a signed stateless token. A
stateless token cannot be revoked: signing out would clear a cookie the holder
could simply put back, and a demotion or a removal would not take effect until
the token expired. Here the record IS the authority, so ``POST /auth/logout``
ends the session for real and the role is re-read from the database on every
request rather than frozen into the cookie at sign-in.

**This store fails CLOSED, and that is the opposite of the rate limiter beside
it.** An unreachable counter lets a request through - refusing traffic because
bookkeeping is down would turn a Redis blink into a site outage. An unreachable
session store must NOT do the same: "we cannot check who you are, so come in"
is not a degraded mode, it is an open door. So a Redis outage with login on
means requests are refused (503) until Redis returns.

The backing store is reached through :class:`KeyValueStore`, a protocol the
process's existing Redis-backed job queue already satisfies - the same
borrowing the rate limiter does, and for the same reason: one connection pool
per process, not three.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

# Namespaces, so the auth keys are recognisable in a shared Redis and cannot
# collide with the rate limiter's counters or arq's own bookkeeping.
SESSION_KEY_PREFIX = "session:"
PENDING_KEY_PREFIX = "authpending:"

# How long a started login stays completable. Long enough to read a consent
# screen and pick an account, short enough that an abandoned attempt is not a
# replayable artifact lying around for hours.
PENDING_TTL_SECONDS = 600

# 32 bytes of entropy, URL-safe - the same size the OAuth state uses. This value
# IS the credential: anything that holds it is the signed-in person.
_SESSION_ID_BYTES = 32


@runtime_checkable
class KeyValueStore(Protocol):
    """An expiring key-value store with an ATOMIC read-and-delete.

    Deliberately not a seam of its own, for the same reason
    :class:`~app.core.ratelimit.RateCounter` is not: the process already holds
    one Redis pool (the job queue's), and these five verbs are what a session
    needs from it. Nothing here is Redis-shaped - "set with an expiry, read,
    re-arm, delete, and take-once" is a capability any store can offer.

    ``kv_consume`` is the one that must be atomic. A pending login read and then
    deleted in two steps is a race two concurrent callbacks both win, which is
    exactly the authorization-code-injection door single-use ``state`` exists to
    shut.
    """

    async def kv_get(self, key: str) -> str | None:
        """Return the value at ``key``, or None when it is absent or expired."""
        ...

    async def kv_set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store ``value`` at ``key`` with an expiry."""
        ...

    async def kv_expire(self, key: str, ttl_seconds: int) -> None:
        """Re-arm ``key``'s expiry without rewriting its value."""
        ...

    async def kv_delete(self, key: str) -> None:
        """Remove ``key`` if it is there."""
        ...

    async def kv_consume(self, key: str) -> str | None:
        """Atomically return ``key``'s value AND delete it (single use)."""
        ...


@dataclass(frozen=True, slots=True)
class PendingLogin:
    """What a started login must remember until its callback arrives.

    Attributes:
        nonce: Binds the eventual id_token to THIS attempt.
        code_verifier: The PKCE secret whose challenge went to the provider.
        next_path: Where to land afterwards - always a relative path.
    """

    nonce: str
    code_verifier: str
    next_path: str

    def dumps(self) -> str:
        """Serialize for storage."""
        return json.dumps(
            {"nonce": self.nonce, "code_verifier": self.code_verifier, "next": self.next_path}
        )

    @classmethod
    def loads(cls, raw: str) -> PendingLogin | None:
        """Parse a stored value, or None when it is not a usable record.

        None rather than an exception: a malformed value is indistinguishable
        from an absent one to the caller, which must refuse the callback either
        way, and a stored blob is not worth a distinct failure path.
        """
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        nonce, verifier = data.get("nonce"), data.get("code_verifier")
        if not isinstance(nonce, str) or not isinstance(verifier, str):
            return None
        next_path = data.get("next")
        return cls(
            nonce=nonce,
            code_verifier=verifier,
            next_path=next_path if isinstance(next_path, str) else "/",
        )


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """A signed-in session: two ids and two timestamps, nothing more.

    IDS ONLY, deliberately. The role is NOT in here: it is read from the users
    row on every authenticated request, which is what makes a demotion take
    effect on the next request rather than whenever the session happens to
    expire. Caching it here would quietly turn a server-side session back into a
    stateless token.
    """

    user_id: UUID
    tenant_id: UUID
    created_at: datetime
    last_seen: datetime

    def dumps(self) -> str:
        """Serialize for storage."""
        return json.dumps(
            {
                "user_id": str(self.user_id),
                "tenant_id": str(self.tenant_id),
                "created_at": self.created_at.isoformat(),
                "last_seen": self.last_seen.isoformat(),
            }
        )

    @classmethod
    def loads(cls, raw: str) -> SessionRecord | None:
        """Parse a stored session, or None when it is not a usable record."""
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        try:
            user_id = UUID(str(data["user_id"]))
            tenant_id = UUID(str(data["tenant_id"]))
            created_at = datetime.fromisoformat(str(data["created_at"]))
            last_seen = datetime.fromisoformat(str(data["last_seen"]))
        except (KeyError, ValueError, TypeError):
            return None
        return cls(user_id=user_id, tenant_id=tenant_id, created_at=created_at, last_seen=last_seen)


def new_session_id() -> str:
    """A fresh opaque session id (32 random bytes, URL-safe)."""
    return secrets.token_urlsafe(_SESSION_ID_BYTES)


def session_key(session_id: str) -> str:
    """The store key holding a session."""
    return f"{SESSION_KEY_PREFIX}{session_id}"


def pending_key(state: str) -> str:
    """The store key holding a pending login, named by its ``state``."""
    return f"{PENDING_KEY_PREFIX}{state}"


async def put_pending(store: KeyValueStore, state: str, pending: PendingLogin) -> None:
    """Stash a started login under its state value."""
    await store.kv_set(pending_key(state), pending.dumps(), PENDING_TTL_SECONDS)


async def take_pending(store: KeyValueStore, state: str) -> PendingLogin | None:
    """Consume a pending login - ONCE. A second callback with the same state gets None."""
    raw = await store.kv_consume(pending_key(state))
    return None if raw is None else PendingLogin.loads(raw)


async def create_session(
    store: KeyValueStore, *, user_id: UUID, tenant_id: UUID, ttl_seconds: int
) -> str:
    """Mint a session for a signed-in user and return its id."""
    now = datetime.now(UTC)
    session_id = new_session_id()
    record = SessionRecord(user_id=user_id, tenant_id=tenant_id, created_at=now, last_seen=now)
    await store.kv_set(session_key(session_id), record.dumps(), ttl_seconds)
    return session_id


async def read_session(
    store: KeyValueStore, session_id: str, *, ttl_seconds: int
) -> SessionRecord | None:
    """Read a session and slide its expiry, or None when there is none.

    The slide is a bare EXPIRE rather than a rewrite: it costs one round trip
    and touches no value, so an active person is never signed out mid-work while
    the record itself stays exactly as it was minted. ``last_seen`` in the stored
    value is therefore the sign-in time until something rewrites it - the
    expiry, not that field, is what governs the session's life.
    """
    raw = await store.kv_get(session_key(session_id))
    if raw is None:
        return None
    record = SessionRecord.loads(raw)
    if record is None:
        # A value that will not parse is not a session; drop it rather than
        # leaving an unreadable key to be re-read on every request.
        await store.kv_delete(session_key(session_id))
        return None
    await store.kv_expire(session_key(session_id), ttl_seconds)
    return record


async def delete_session(store: KeyValueStore, session_id: str) -> None:
    """End a session immediately - what sign-out means on a server-side session."""
    await store.kv_delete(session_key(session_id))
