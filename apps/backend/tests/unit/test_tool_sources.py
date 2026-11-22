"""Unit tests for the static and caching tool-source adapters."""

from __future__ import annotations

import asyncio

import pytest

from app.providers.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSource,
    ToolSpec,
)
from app.services.tools.sources import CachingToolSource, StaticToolSource

CTX = ToolContext(tenant_id="t1", user_id="u1", conversation_id="c1", session_id="s1")


class _FakeTool(Tool):
    """A trivial named tool; only its spec matters to these tests."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self._name, description=f"fake {self._name}")

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        return ToolResult(content=self._name)


class _Clock:
    """A hand-driven monotonic clock, so expiry is tested without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _CountingSource(ToolSource):
    """Counts its listings and returns whatever `tools` currently holds."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = tools
        self.calls = 0

    async def list_tools(self, context: ToolContext) -> list[Tool]:
        self.calls += 1
        return list(self.tools)


class _GatedSource(_CountingSource):
    """Blocks inside the listing until the gate opens, to hold N calls in flight."""

    def __init__(self, tools: list[Tool]) -> None:
        super().__init__(tools)
        self.gate = asyncio.Event()

    async def list_tools(self, context: ToolContext) -> list[Tool]:
        self.calls += 1
        await self.gate.wait()
        return list(self.tools)


class _RaisingSource(ToolSource):
    """A source whose listing always fails."""

    def __init__(self) -> None:
        self.calls = 0

    async def list_tools(self, context: ToolContext) -> list[Tool]:
        self.calls += 1
        raise ToolError("listing exploded")


def _names(tools: list[Tool]) -> list[str]:
    return [tool.spec.name for tool in tools]


def _caching(inner: ToolSource, clock: _Clock) -> CachingToolSource:
    return CachingToolSource(inner, ttl_seconds=30.0, error_ttl_seconds=5.0, clock=clock)


# --- StaticToolSource ---


async def test_static_source_offers_its_tools() -> None:
    source = StaticToolSource([_FakeTool("a"), _FakeTool("b")])
    assert _names(await source.list_tools(CTX)) == ["a", "b"]


async def test_static_source_returns_a_copy() -> None:
    # A caller mutating what it got back must not edit the source's offering.
    source = StaticToolSource([_FakeTool("a")])
    listed = await source.list_tools(CTX)
    listed.clear()
    assert _names(await source.list_tools(CTX)) == ["a"]


async def test_static_source_copies_the_iterable_it_was_given() -> None:
    # A generator would otherwise be drained by the first listing.
    source = StaticToolSource(_FakeTool(name) for name in ("a", "b"))
    assert _names(await source.list_tools(CTX)) == ["a", "b"]
    assert _names(await source.list_tools(CTX)) == ["a", "b"]


# --- CachingToolSource ---


async def test_warm_cache_does_not_re_list_inner() -> None:
    clock = _Clock()
    inner = _CountingSource([_FakeTool("a")])
    source = _caching(inner, clock)

    assert _names(await source.list_tools(CTX)) == ["a"]
    clock.advance(29.0)
    assert _names(await source.list_tools(CTX)) == ["a"]
    assert inner.calls == 1


async def test_cache_expires_after_ttl_seconds() -> None:
    clock = _Clock()
    inner = _CountingSource([_FakeTool("a")])
    source = _caching(inner, clock)

    await source.list_tools(CTX)
    clock.advance(31.0)
    inner.tools = [_FakeTool("b")]

    assert _names(await source.list_tools(CTX)) == ["b"]
    assert inner.calls == 2


async def test_entry_is_expired_at_exactly_its_expiry_instant() -> None:
    # The boundary the 31.0-past-a-30.0-TTL test jumps clean over: clock() lands
    # EXACTLY on expires_at. Dead, not alive - the same ">=" the ttl=0 kill
    # switch below rests on, so this pins the rule at its only ambiguous point.
    clock = _Clock()
    inner = _CountingSource([_FakeTool("a")])
    source = _caching(inner, clock)

    await source.list_tools(CTX)  # stored at t=0.0, so expires_at == 30.0
    inner.tools = [_FakeTool("b")]

    clock.advance(30.0)  # exactly expires_at; both floats are exact here
    assert _names(await source.list_tools(CTX)) == ["b"]
    assert inner.calls == 2


async def test_ttl_seconds_zero_never_caches_a_good_listing() -> None:
    # The documented kill switch, and the ONLY thing pinning the ">=" in
    # _cached: an entry stored at t expires AT t, so every listing is a miss.
    # Under ">" this flips silently from "never cache" to "cache forever" - a
    # permanently frozen tool list, which is the opposite of what the operator
    # who reached for ttl=0 was trying to do.
    clock = _Clock()
    inner = _CountingSource([_FakeTool("a")])
    source = CachingToolSource(inner, ttl_seconds=0.0, error_ttl_seconds=0.0, clock=clock)

    for expected_calls in (1, 2, 3):
        assert _names(await source.list_tools(CTX)) == ["a"]
        assert inner.calls == expected_calls

    # The clock never moved: the kill switch must not depend on time passing.
    assert clock.now == 0.0


async def test_error_ttl_seconds_zero_never_caches_an_empty_listing() -> None:
    # ttl_seconds=0 kills caching of GOOD listings only; a dead sidecar's []
    # would still be held for error_ttl_seconds. Both zero == cache nothing.
    clock = _Clock()
    inner = _CountingSource([])
    source = CachingToolSource(inner, ttl_seconds=0.0, error_ttl_seconds=0.0, clock=clock)

    for expected_calls in (1, 2, 3):
        assert await source.list_tools(CTX) == []
        assert inner.calls == expected_calls


async def test_miss_path_listing_is_copied_per_caller() -> None:
    # The copy on the MISS path guards the stored entry: a caller mutating what
    # the cold call handed back must not edit what the cache goes on serving.
    clock = _Clock()
    inner = _CountingSource([_FakeTool("a")])
    source = _caching(inner, clock)

    listed = await source.list_tools(CTX)
    listed.clear()

    assert _names(await source.list_tools(CTX)) == ["a"]
    assert inner.calls == 1


async def test_warm_path_listing_is_copied_per_caller() -> None:
    # The WARM read is the object under test, and it needs its own test: the
    # miss-path copy above keeps that one green even if the warm path hands out
    # the internal list, so only this one pins `return list(cached)`.
    clock = _Clock()
    inner = _CountingSource([_FakeTool("a")])
    source = _caching(inner, clock)

    await source.list_tools(CTX)  # populate via the miss path
    warm = await source.list_tools(CTX)  # the warm read
    warm.clear()

    assert _names(await source.list_tools(CTX)) == ["a"]
    assert inner.calls == 1


async def test_empty_listing_uses_the_shorter_error_ttl_and_recovers() -> None:
    # McpToolSource returns [] on ANY failure, so an empty listing may be a dead
    # server: hold it briefly (do not dial a dead sidecar every turn) but let it
    # come back long before ttl_seconds would.
    clock = _Clock()
    inner = _CountingSource([])
    source = _caching(inner, clock)

    assert await source.list_tools(CTX) == []
    clock.advance(4.0)
    assert await source.list_tools(CTX) == []
    assert inner.calls == 1

    # Past the SHORT error TTL, and far short of the 30s success TTL.
    clock.advance(2.0)
    inner.tools = [_FakeTool("a")]
    assert _names(await source.list_tools(CTX)) == ["a"]
    assert inner.calls == 2

    # The recovered listing is a success, so it now holds for the full TTL.
    clock.advance(10.0)
    assert _names(await source.list_tools(CTX)) == ["a"]
    assert inner.calls == 2


async def test_concurrent_cold_listings_single_flight() -> None:
    clock = _Clock()
    inner = _GatedSource([_FakeTool("a")])
    source = _caching(inner, clock)

    tasks = [asyncio.create_task(source.list_tools(CTX)) for _ in range(5)]
    # Let every task reach the inner call or queue on the single-flight lock
    # before the gate opens; without the lock all five would be inside inner.
    for _ in range(3):
        await asyncio.sleep(0)
    inner.gate.set()
    results = await asyncio.gather(*tasks)

    assert inner.calls == 1
    assert all(_names(result) == ["a"] for result in results)


async def test_raising_inner_propagates_and_is_not_cached() -> None:
    # Caching a raise would turn a transient bug into a full-TTL outage.
    clock = _Clock()
    inner = _RaisingSource()
    source = _caching(inner, clock)

    for _ in range(2):
        with pytest.raises(ToolError):
            await source.list_tools(CTX)
    assert inner.calls == 2


async def test_recovers_after_a_raising_listing() -> None:
    clock = _Clock()
    inner = _CountingSource([_FakeTool("a")])
    source = _caching(inner, clock)
    await source.list_tools(CTX)
    clock.advance(31.0)

    # The listing fails once the entry has expired: the stale entry must not be
    # served, and the failure must not be cached either.
    failing = _RaisingSource()
    source._inner = failing  # noqa: SLF001
    with pytest.raises(ToolError):
        await source.list_tools(CTX)

    source._inner = inner  # noqa: SLF001
    assert _names(await source.list_tools(CTX)) == ["a"]
    assert inner.calls == 2
