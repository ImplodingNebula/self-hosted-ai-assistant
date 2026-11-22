"""Two general adapters over the tool seam: a fixed source, and a TTL cache.

Neither knows what a tool does or where it lives - both take
:class:`~app.providers.tools.base.ToolSource` s and hand back one. That is the
point: :class:`StaticToolSource` is how the in-process tools (web search,
knowledge search) join the SAME registry as a remote MCP server, and
:class:`CachingToolSource` wraps any source whose listing is expensive - today
only the MCP ones. Nothing above the seam can tell a wrapped source from a bare
one, which is what keeps "MCP is a source of tools, not the architecture" true
in code rather than only in the docs.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable

from app.providers.tools.base import Tool, ToolContext, ToolSource


class StaticToolSource(ToolSource):
    """A source over a fixed, already-decided list of tools."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        """Bind the source to a fixed tool list.

        Args:
            tools: The tools this source always offers. Copied on construction,
                so a generator is consumed once here rather than emptying itself
                on the first listing.
        """
        self._tools = list(tools)

    async def list_tools(self, context: ToolContext) -> list[Tool]:
        """Return the fixed tools, ignoring ``context``.

        The context is deliberately unused: which local tools a turn may see is
        a POLICY decision (the user's web-search switch, the agency clamp, the
        tool-result budget) that ``chat_service`` makes before it builds this
        source. Re-deciding it here would put the same gate in two places, and
        the seam's job is composition, not policy.

        Args:
            context: The caller's identity and turn. Unused; the parameter is
                the seam contract.

        Returns:
            A copy of the tool list, so a caller that mutates what it got back
            cannot edit this source's offering for every later turn.
        """
        return list(self._tools)


class CachingToolSource(ToolSource):
    """A TTL cache over a wrapped source's LISTING (not its tool calls).

    It exists for one measured reason:
    :meth:`~app.providers.mcp.source.McpToolSource.list_tools` opens a full
    session per call (connect, initialize, list), and the tool loop lists once
    per TURN. Uncached, every agentic turn pays a network round trip to the
    sidecar before the model produces a token, and a HANGING sidecar costs the
    full list timeout on EVERY turn. Cached, that hang is amortized across an
    error window instead - but ONLY when ``error_ttl_seconds`` outlasts the
    wrapped source's listing timeout, a relationship this class cannot see well
    enough to check. Read ``error_ttl_seconds`` in :meth:`__init__` before
    trusting the previous sentence.

    SECURITY - the cache is keyed on NOTHING, which assumes the wrapped
    source's LISTING is identity-INDEPENDENT. That holds because an MCP server
    lists under its static, workspace-level header: every caller provokes
    byte-for-byte the same listing, so serving one caller's listing to another
    leaks nothing. Per-user credentials (slice 6b) do NOT break this: they are
    injected per tool CALL, which this cache never touches, and a LISTING keeps
    using the workspace header on purpose precisely so this stays true (see
    :meth:`~app.providers.mcp.source.McpToolSource.list_tools`).

    Per-TENANT server rows (slice 6b PR2) do not break it either, and the reason
    is structural rather than a key: the DB registry gives each
    ``(tenant, server)`` its OWN wrapped source and its OWN cache instance (see
    :class:`~app.services.tools.mcp_sources.McpSourceProvider`), so a single
    cache never spans two tenants - it can only ever hold the one tenant-server
    listing it was built for. That is why this class stays correct UNKEYED even
    though which servers exist is now per-tenant.

    It stops holding only if a server's LISTING is ever made a function of WHO
    asked WITHIN a tenant - then this cache would serve Alice's tool list to Bob,
    and the key MUST include the principal (``tenant_id``, ``user_id``) from
    :class:`~app.providers.tools.base.ToolContext`, or caching must be disabled
    for that server. This is the same class of bug the locked
    one-client-per-tool-call rule exists to prevent - a shared object silently
    carrying one principal's reach into another principal's request, with no
    downstream guard positioned to catch it.

    The CACHED :class:`~app.providers.tools.base.Tool` objects are safe to hand
    to any caller today for a separate reason worth keeping true: an
    :class:`~app.providers.mcp.source.McpTool` holds only its spec and its
    source, and identity arrives per call via ``ToolContext``. A tool that
    captured its caller at listing time would be unsafe to cache at all, not
    merely unsafe to cache unkeyed.
    """

    def __init__(
        self,
        inner: ToolSource,
        *,
        ttl_seconds: float,
        error_ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wrap ``inner`` in a TTL cache.

        Args:
            inner: The source whose listing is cached.
            ttl_seconds: How long a successful, non-empty listing is reused.
                ``0`` is a supported KILL SWITCH: an entry stored at ``t``
                expires at ``t``, so every listing misses and the cache is
                inert. It is the knob an operator debugging a live sidecar
                reaches for first, and it works only because :meth:`_cached`
                treats the expiry instant ITSELF as expired - load-bearing, not
                incidental. ``0`` disables caching of GOOD listings only; pair
                it with ``error_ttl_seconds=0`` to cache nothing at all.
            error_ttl_seconds: How long an EMPTY listing is reused. Two
                relationships matter, and neither is enforced here.

                It must EXCEED the wrapped source's LISTING timeout, or the
                class docstring's promise is simply false: a hanging server is
                dialled "once per error window" only if the window outlasts the
                hang. A 30s error TTL against a 30s list timeout blocks a turn
                at t=0 for 30s, expires at t=45, then blocks the turn at t=45
                for another 30s - a two-thirds duty cycle still hanging. The
                shipped pairing holds only because ``McpToolSource`` takes a
                SEPARATE ``list_timeout_seconds``: ``mcp_list_timeout_seconds``
                is 5.0 against ``mcp_list_cache_error_seconds`` of 30.0, so a
                hanging sidecar costs 5s of latency per 35s window.

                It should also be SHORTER than ``ttl_seconds``, so a sidecar
                that recovers is picked up long before a good listing would
                have expired - see :meth:`list_tools`.
            clock: The time source, monotonic seconds. Injected so tests drive
                expiry without sleeping. Monotonic on purpose: a wall clock
                stepping backwards (NTP, a DST-naive host) would freeze an
                entry as unexpirable, and stepping forwards would drop a warm
                cache mid-turn.
        """
        # Both TTL relationships above are CALLER-ENFORCED, deliberately. The
        # one that actually costs a user latency - error TTL vs the inner's
        # listing timeout - is UNCHECKABLE from here: the seam hands over a
        # ToolSource, which exposes no timeout to read. A guard could therefore
        # only check the cosmetic ordering against ttl_seconds, which would
        # read as "the TTLs are validated" while the load-bearing half went on
        # being silent - worse than no guard, because it invites trust. And the
        # blast radius does not earn one: either order still serves CORRECT
        # tools (the security invariant is the cache KEY, not the TTLs), so a
        # swap is a mistuned cache, not a broken one - not worth turning a
        # tuning typo into a refusal to boot.
        self._inner = inner
        self._ttl_seconds = ttl_seconds
        self._error_ttl_seconds = error_ttl_seconds
        self._clock = clock
        self._lock = asyncio.Lock()
        # (expires_at, tools); None until the first listing lands.
        self._entry: tuple[float, list[Tool]] | None = None

    def _cached(self) -> list[Tool] | None:
        """Return the unexpired cached listing, or None."""
        entry = self._entry
        # ">=", never ">": an entry is dead AT its expiry instant. That is what
        # makes the documented ttl_seconds=0 kill switch mean "never cache"
        # (stored at t, expires at t) rather than "cache FOREVER" - the frozen
        # tool list a ">" would silently ship.
        if entry is None or self._clock() >= entry[0]:
            return None
        return entry[1]

    async def list_tools(self, context: ToolContext) -> list[Tool]:
        """Return the wrapped source's tools, from cache when one is warm.

        An EMPTY listing is cached for the SHORTER ``error_ttl_seconds``
        because ``McpToolSource`` never raises - it logs and returns ``[]`` on
        any failure - so an empty listing is indistinguishable from a dead
        server. Both readings want the same treatment from opposite directions:
        a dead sidecar must not be dialled on every single turn, and it must
        come back quickly once it is healthy again. A genuinely empty server
        pays a slightly chattier poll, which is the cheap side of the trade.

        A raising ``inner`` propagates and is NOT cached. ``McpToolSource``
        cannot raise, but the ABC allows it and a future source might, and
        caching a raise would promote a transient bug into an outage lasting
        the whole TTL. The registry's caller already treats a raising source as
        a turn-level failure, so the honest thing is to let the next turn try
        again. The price is that N concurrent turns against a raising source
        SERIALIZE: each lock winner calls ``inner`` while holding the mutex and
        stores no entry for the next to find, so the single flight below
        degrades to a queue. Inert while the only wrapped source cannot raise,
        and the fix for a source that can is a cheap negative result inside the
        lock - NOT caching the failure, which is the trade this paragraph
        already refused.

        Args:
            context: The caller's identity and turn, passed through to the
                wrapped source on a miss. NOT part of the cache key - see the
                class docstring's security note before changing that.

        Returns:
            A copy of the cached or freshly listed tools.
        """
        cached = self._cached()
        if cached is not None:
            # Deliberately OUTSIDE the lock: a warm read is the common path
            # (once per turn, per source), and taking the mutex here would
            # serialize every concurrent turn behind one another for nothing.
            return list(cached)
        async with self._lock:
            # Re-check under the lock: this is the single flight. N turns
            # arriving on a cold cache all queue here, and the ones that lose
            # the race find the winner's listing already stored - so a cold
            # cache costs ONE session with the sidecar, not N.
            cached = self._cached()
            if cached is not None:
                return list(cached)
            tools = await self._inner.list_tools(context)
            ttl = self._ttl_seconds if tools else self._error_ttl_seconds
            self._entry = (self._clock() + ttl, list(tools))
            return list(tools)
