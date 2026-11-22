"""Unit tests for NoopReranker and the Infinity sidecar client."""

import httpx

from app.providers.rag.base import ScoredChunk
from app.providers.rag.reranker import InfinityReranker, NoopReranker

CHUNKS = [
    ScoredChunk(id="p1", score=0.9, text="alpha", doc_id="d1"),
    ScoredChunk(id="p2", score=0.8, text="beta", doc_id="d2"),
    ScoredChunk(id="p3", score=0.7, text="gamma", doc_id="d3"),
]


def build_reranker(handler) -> InfinityReranker:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return InfinityReranker("http://mock:7997", "bge-reranker-v2-m3", client=client)


async def test_noop_keeps_dense_order_and_caps() -> None:
    result = await NoopReranker().rerank("q", CHUNKS, top_k=2)
    assert [c.id for c in result] == ["p1", "p2"]


async def test_infinity_reorders_and_replaces_scores() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.4},
                    {"index": 1, "relevance_score": 0.1},
                ]
            },
        )

    result = await build_reranker(handler).rerank("q", CHUNKS, top_k=2)
    assert [c.id for c in result] == ["p3", "p1"]
    assert result[0].score == 0.95


async def test_infinity_http_error_falls_back_to_dense() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    result = await build_reranker(handler).rerank("q", CHUNKS, top_k=2)
    assert [c.id for c in result] == ["p1", "p2"]


async def test_infinity_connect_error_falls_back_to_dense() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    result = await build_reranker(handler).rerank("q", CHUNKS, top_k=3)
    assert [c.id for c in result] == ["p1", "p2", "p3"]


async def test_infinity_malformed_body_falls_back_to_dense() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 99, "relevance_score": 1.0}]})

    result = await build_reranker(handler).rerank("q", CHUNKS, top_k=2)
    assert [c.id for c in result] == ["p1", "p2"]


async def test_infinity_duplicate_index_falls_back_to_dense() -> None:
    # A repeated index would duplicate the same passage (and its citation)
    # downstream, so the whole response is distrusted: dense order wins.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.8},
                ]
            },
        )

    result = await build_reranker(handler).rerank("q", CHUNKS, top_k=2)
    # The dense top_k slice comes back untouched - no duplicated passage.
    assert result == CHUNKS[:2]


async def test_infinity_empty_input_short_circuits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected for empty input")

    assert await build_reranker(handler).rerank("q", [], top_k=5) == []
