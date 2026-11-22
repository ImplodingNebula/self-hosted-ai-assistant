"""The per-grant renewal lock - one tiny module both the flow and the tool path import.

A sealed OAuth grant is mutated from two directions: the tool path renews it on
demand mid-turn, and an admin can disconnect it at any moment. Both must
serialize on the SAME lock or a disconnect can land between a renewal's read and
its write, revoking a token that has already been rotated away and then having
the renewal RE-INSERT a bundle holding the new, still-live tokens - a grant the
admin believes is severed.

It lives here rather than in either caller because the import graph is one-way:
``services.tools.mcp_sources`` imports ``services.oauth.flow``, so the flow can
never import back. A module both can import is the only shape that works.

Scope, stated plainly: this is a PROCESS-LOCAL lock. It orders the two paths
inside one worker and nothing more; across workers the durable guarantee is the
update-only write (:meth:`~app.services.secret_store.SecretStore.store_existing_only`),
which turns a renewal into a deleted row into a no-op no matter which process
ran it. The lock is the cheap half; that method is the load-bearing one.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

# Keyed by the grant's SECRET NAME (which embeds the server's stable identity),
# so the dict stays as small as the number of OAuth-connected servers.
_grant_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def grant_lock(secret_name: str) -> asyncio.Lock:
    """The lock serializing every mutation of one sealed grant.

    Args:
        secret_name: The SecretStore name the grant's bundle lives under.

    Returns:
        The process-wide lock for that grant - the same object for every
        caller, which is the entire point.
    """
    return _grant_locks[secret_name]
