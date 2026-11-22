"""Threshold-gated, in-process top-K tool selection.

Driven with a counting fake embedder that maps text to vectors from a lookup
table, so ranking is exact and assertable rather than approximate: the tests can
name the expected order instead of hoping a hash embedder cooperates. The call
counter is what proves the two properties that matter most - that the fast path
is INERT (no embed at or below the threshold) and that the description cache
actually caches (no re-embed on the second turn).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.providers.base import ProviderUnavailableError
from app.providers.rag.base import Embedder
from app.providers.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from app.services.tools.local import SEARCH_KNOWLEDGE_TOOL, WEB_SEARCH_TOOL
from app.services.tools.selection import (
    CORE_TOOL_NAMES,
    ToolVectorCache,
    cosine,
    select_tools,
)

pytestmark = pytest.mark.anyio

# A tiny hand-built vector space: three orthogonal axes so cosines are exact.
QUERY = "what is the weather"
VECTORS: dict[str, list[float]] = {
    QUERY: [1.0, 0.0, 0.0],
    "forecasts and temperature": [1.0, 0.0, 0.0],  # cosine 1.0
    "climate records": [0.6, 0.8, 0.0],  # cosine 0.6
    "stock prices": [0.0, 1.0, 0.0],  # cosine 0.0
    "issue tracking": [0.0, 0.0, 1.0],  # cosine 0.0
    "web_search": [0.0, 1.0, 0.0],  # core, deliberately ranks last
    "search_knowledge": [0.0, 1.0, 0.0],  # core, deliberately ranks last
}


class StubTool(Tool):
    """A tool that is nothing but a spec - selection never runs one."""

    def __init__(self, name: str, description: str = "", *, side_effecting: bool = False) -> None:
        self._spec = ToolSpec(name=name, description=description, side_effecting=side_effecting)

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        raise AssertionError("selection must never RUN a tool")


class CountingEmbedder(Embedder):
    """Deterministic lookup embedder that counts every call it receives."""

    def __init__(self, *, raises: bool = False, model: str = "fake-embed") -> None:
        self._raises = raises
        self._model = model
        self.embed_texts_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return 3

    @property
    def calls(self) -> int:
        """Every call of any kind - what the inert fast path must leave at zero."""
        return len(self.embed_texts_calls) + len(self.embed_query_calls)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(list(texts))
        if self._raises:
            raise ProviderUnavailableError("embedder down")
        # An unknown text becomes the zero vector, which scores 0.0 - the same
        # floor an absent vector gets, so unknown text can never outrank a match.
        return [list(VECTORS.get(text, [0.0, 0.0, 0.0])) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        if self._raises:
            raise ProviderUnavailableError("embedder down")
        return list(VECTORS.get(text, [0.0, 0.0, 0.0]))


class NonNumericEmbedder(Embedder):
    """An embedder whose vectors are structurally valid but hold no numbers.

    Not a strawman: the provider seam only checks that the embeddings body is a
    list of the right LENGTH (ollama_provider.py, `/api/embed`), never that the
    elements are numbers - so a proxy or a half-broken backend answering
    `{"embeddings": [null, null]}` clears every guard on the way in and hands a
    list of None straight to the ranking.

    This is the case that separates "the embed call failed" from "the embed call
    SUCCEEDED and returned rubbish": nothing raises until the arithmetic inside
    cosine(), which is why the scoring - not just the awaits - has to sit inside
    the fail-soft guard.
    """

    def __init__(self, model: str = "fake-embed") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return 3

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[None, None, None] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [None, None, None]


def core_tools() -> list[StubTool]:
    """The two always-offered tools, described so they rank LAST for QUERY."""
    return [
        StubTool(WEB_SEARCH_TOOL, "web_search"),
        StubTool(SEARCH_KNOWLEDGE_TOOL, "search_knowledge"),
    ]


def names(tools: list[Tool]) -> list[str]:
    return [tool.spec.name for tool in tools]


async def select(tools: list[Tool], **overrides: object) -> list[Tool]:
    """Call select_tools with the tests' defaults: above threshold, ranking on."""
    kwargs: dict = {
        "query": QUERY,
        "embedder": CountingEmbedder(),
        "threshold": 2,
        "top_k": 2,
        "cache": ToolVectorCache(),
    }
    kwargs.update(overrides)
    return await select_tools(tools, **kwargs)


# --- the inert fast path -----------------------------------------------------


async def test_at_or_below_threshold_returns_every_tool_and_never_embeds() -> None:
    embedder = CountingEmbedder()
    tools = [*core_tools(), StubTool("weather", "forecasts and temperature")]

    selected = await select_tools(
        tools,
        query=QUERY,
        embedder=embedder,
        threshold=3,  # exactly len(tools): the boundary is inclusive
        top_k=1,
        cache=ToolVectorCache(),
    )

    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL, "weather"]
    # The whole point of the threshold: a turn that pays no embed today must not
    # start paying one (the dev embedder is CPU-pinned; a cold load is 6-8s).
    assert embedder.calls == 0


async def test_below_threshold_leaves_the_cache_untouched() -> None:
    embedder = CountingEmbedder()
    cache = ToolVectorCache()
    tools = [StubTool("weather", "forecasts and temperature")]

    await select_tools(tools, query=QUERY, embedder=embedder, threshold=8, top_k=1, cache=cache)

    assert embedder.calls == 0
    # A later ABOVE-threshold turn must still find the cache cold: the fast path
    # did no IO at all, it did not merely skip the ranking.
    assert cache._vectors == {}


# --- ranking -----------------------------------------------------------------


async def test_core_tools_are_offered_even_when_they_rank_last() -> None:
    tools = [
        *core_tools(),  # both score 0.0 against QUERY, by construction
        StubTool("weather", "forecasts and temperature"),
        StubTool("climate", "climate records"),
    ]

    selected = await select(tools, top_k=1)

    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL, "weather"]
    assert CORE_TOOL_NAMES.issubset(set(names(selected)))


async def test_closest_non_core_tools_win_in_score_order() -> None:
    tools = [
        StubTool("stocks", "stock prices"),  # 0.0
        StubTool("climate", "climate records"),  # 0.6
        StubTool("weather", "forecasts and temperature"),  # 1.0
        StubTool("issues", "issue tracking"),  # 0.0
    ]

    selected = await select(tools, top_k=3)

    assert names(selected) == ["weather", "climate", "stocks"]


async def test_top_k_bounds_the_non_core_count_only() -> None:
    tools = [
        *core_tools(),
        StubTool("weather", "forecasts and temperature"),
        StubTool("climate", "climate records"),
        StubTool("stocks", "stock prices"),
    ]

    selected = await select(tools, top_k=1)

    # top_k is the NON-CORE budget: 2 core + 1 ranked, not 1 in total.
    assert len(selected) == 3
    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL, "weather"]


async def test_top_k_of_zero_offers_the_core_tools_only() -> None:
    embedder = CountingEmbedder()
    tools = [*core_tools(), StubTool("weather", "forecasts and temperature")]

    selected = await select(tools, top_k=0, embedder=embedder)

    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL]
    # Nothing to rank means nothing to embed.
    assert embedder.calls == 0


async def test_equal_scores_keep_registry_order() -> None:
    # Three tools that all score 0.0 against QUERY: nothing but the tie-break
    # decides their order, so a flapping sort would show up here.
    tools = [
        StubTool("stocks", "stock prices"),
        StubTool("issues", "issue tracking"),
        StubTool("unknown", "no vector for this text"),
    ]

    first = await select(tools, top_k=3)
    second = await select(tools, top_k=3)

    assert names(first) == ["stocks", "issues", "unknown"]
    assert names(second) == names(first)


async def test_a_tool_with_no_description_ranks_by_its_name() -> None:
    tools = [
        StubTool("stocks", "stock prices"),
        StubTool("weather", ""),  # falls back to the name, not the empty string
        StubTool("issues", "issue tracking"),
    ]
    embedder = CountingEmbedder()

    await select(tools, top_k=3, embedder=embedder)

    assert embedder.embed_texts_calls == [["stock prices", "weather", "issue tracking"]]


# --- fail-soft ---------------------------------------------------------------


async def test_no_embedder_falls_back_to_registry_order() -> None:
    tools = [
        *core_tools(),
        StubTool("stocks", "stock prices"),
        StubTool("weather", "forecasts and temperature"),
    ]

    selected = await select(tools, embedder=None, top_k=1)

    # Registry order, NOT rank order: "stocks" leads even though "weather" is
    # the better match, because without an embedder there is nothing to rank on.
    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL, "stocks"]


async def test_a_raising_embedder_falls_back_and_logs_instead_of_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tools = [
        *core_tools(),
        StubTool("stocks", "stock prices"),
        StubTool("weather", "forecasts and temperature"),
    ]

    with caplog.at_level(logging.WARNING, logger="app.services.tools.selection"):
        selected = await select(tools, embedder=CountingEmbedder(raises=True), top_k=1)

    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL, "stocks"]
    assert "Tool selection failed" in caplog.text
    assert "ProviderUnavailableError" in caplog.text


class SlowEmbedder(Embedder):
    """An embedder that never answers in time - the wedged CPU-pinned instance."""

    def __init__(self, delay: float = 30.0) -> None:
        self._delay = delay

    @property
    def model(self) -> str:
        return "slow-embed"

    @property
    def dim(self) -> int:
        return 3

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(self._delay)
        raise AssertionError("the deadline must fire first")

    async def embed_query(self, text: str) -> list[float]:
        await asyncio.sleep(self._delay)
        raise AssertionError("the deadline must fire first")


async def test_a_slow_embedder_degrades_the_turn_instead_of_stalling_it(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Selection runs BEFORE the first token, on the same CPU-pinned embedder the
    # chat model contends with, and TOOL_TIMEOUT_SECONDS governs tool CALLS -
    # nothing else bounds this path. Without the deadline the turn waits on the
    # embedder for as long as it feels like taking.
    monkeypatch.setattr("app.services.tools.selection._EMBED_TIMEOUT_SECONDS", 0.05)
    tools = [
        *core_tools(),
        StubTool("stocks", "stock prices"),
        StubTool("weather", "forecasts and temperature"),
    ]

    with caplog.at_level(logging.WARNING, logger="app.services.tools.selection"):
        selected = await asyncio.wait_for(
            select(tools, embedder=SlowEmbedder(), top_k=1), timeout=5.0
        )

    # Registry order - the same degradation a failing embedder gets.
    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL, "stocks"]
    assert "Tool selection failed" in caplog.text
    assert "TimeoutError" in caplog.text


async def test_only_a_bounded_number_of_tools_is_ever_embedded(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The candidate set comes from REMOTE listings, so its size is not this
    # process's to trust: a server advertising thousands of tools would otherwise
    # put one unbounded embedding batch on the turn the listing cache expires.
    monkeypatch.setattr("app.services.tools.selection._MAX_RANKED_TOOLS", 2)
    embedder = CountingEmbedder()
    tools = [
        *core_tools(),
        StubTool("stocks", "stock prices"),
        StubTool("issues", "issue tracking"),
        # The best match for QUERY, deliberately past the cap: it is never
        # embedded, so it cannot be ranked.
        StubTool("weather", "forecasts and temperature"),
    ]

    with caplog.at_level(logging.WARNING, logger="app.services.tools.selection"):
        selected = await select(tools, embedder=embedder, top_k=1)

    assert embedder.embed_texts_calls == [["stock prices", "issue tracking"]]
    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL, "stocks"]
    assert "ranking only the first 2" in caplog.text


async def test_a_non_numeric_vector_falls_back_and_logs_instead_of_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The embed calls SUCCEED here - it is the scoring that blows up, one frame
    # later and outside every await. The fail-soft guard promises "a malformed
    # vector" degrades; this is the test that makes the promise true rather than
    # aspirational, and it fails the moment the ranking moves back out of the try.
    tools = [
        *core_tools(),
        StubTool("stocks", "stock prices"),
        StubTool("weather", "forecasts and temperature"),
    ]

    with caplog.at_level(logging.WARNING, logger="app.services.tools.selection"):
        selected = await select(tools, embedder=NonNumericEmbedder(), top_k=1)

    # Registry order, so "stocks" leads: there is no ranking to lead with.
    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL, "stocks"]
    assert "Tool selection failed" in caplog.text
    assert "TypeError" in caplog.text


async def test_an_empty_query_falls_back_without_embedding() -> None:
    embedder = CountingEmbedder()
    tools = [
        *core_tools(),
        StubTool("stocks", "stock prices"),
        StubTool("weather", "forecasts and temperature"),
    ]

    selected = await select(tools, query="   ", embedder=embedder, top_k=1)

    assert names(selected) == [WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL, "stocks"]
    assert embedder.calls == 0


# --- the description-vector cache --------------------------------------------


async def test_the_same_tools_re_embed_nothing_on_a_second_turn() -> None:
    embedder = CountingEmbedder()
    cache = ToolVectorCache()
    tools = [
        StubTool("stocks", "stock prices"),
        StubTool("weather", "forecasts and temperature"),
        StubTool("climate", "climate records"),
    ]

    first = await select(tools, embedder=embedder, cache=cache, top_k=3)
    # ONE batched call for three tools, not one call per tool.
    assert len(embedder.embed_texts_calls) == 1
    assert embedder.embed_texts_calls[0] == [
        "stock prices",
        "forecasts and temperature",
        "climate records",
    ]

    second = await select(tools, embedder=embedder, cache=cache, top_k=3)

    assert names(second) == names(first)
    # The document side is cached; only the query side is paid again per turn.
    assert len(embedder.embed_texts_calls) == 1
    assert len(embedder.embed_query_calls) == 2


async def test_a_changed_description_under_the_same_name_re_embeds() -> None:
    embedder = CountingEmbedder()
    cache = ToolVectorCache()

    await select([StubTool("a", "stock prices")], embedder=embedder, cache=cache, threshold=0)
    # Same NAME, new description - the remote listing is mutable and untrusted,
    # so the hash in the key is what stops a stale vector misranking it forever.
    await select(
        [StubTool("a", "forecasts and temperature")], embedder=embedder, cache=cache, threshold=0
    )

    assert embedder.embed_texts_calls == [["stock prices"], ["forecasts and temperature"]]


async def test_a_different_embedder_model_re_embeds() -> None:
    cache = ToolVectorCache()
    tools = [StubTool("a", "stock prices")]

    first = CountingEmbedder(model="embed-v1")
    await select(tools, embedder=first, cache=cache, threshold=0)
    # A different model is a different vector SPACE, so the cached vector is not
    # stale - it is meaningless against the new query vector.
    second = CountingEmbedder(model="embed-v2")
    await select(tools, embedder=second, cache=cache, threshold=0)

    assert second.embed_texts_calls == [["stock prices"]]


async def test_the_cache_is_bounded_against_description_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.tools.selection._MAX_CACHE_ENTRIES", 4)
    embedder = CountingEmbedder()
    cache = ToolVectorCache()

    # A server rotating its description on every listing mints a new key each
    # turn; the cache must not grow for the life of the process on remote input.
    for turn in range(20):
        await select(
            [StubTool("a", f"description revision {turn}")],
            embedder=embedder,
            cache=cache,
            threshold=0,
        )

    assert len(cache._vectors) <= 4


# --- cosine ------------------------------------------------------------------


def test_cosine_scores_identical_orthogonal_and_zero_vectors() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([3.0, 0.0], [9.0, 0.0]) == pytest.approx(1.0)  # magnitude-free
    assert cosine([-1.0, 0.0], [1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_guards_the_zero_norm_case_instead_of_dividing_by_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0
    assert cosine([], [1.0]) == 0.0
