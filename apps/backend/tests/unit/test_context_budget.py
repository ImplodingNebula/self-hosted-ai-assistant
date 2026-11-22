"""Unit tests for context-window budget math."""

from app.providers.rag.base import ScoredChunk
from app.services.context_budget import (
    ANSWER_RESERVE_TOKENS,
    THINKING_ANSWER_RESERVE_TOKENS,
    approx_tokens,
    rag_budget_tokens,
    trim_chunks_to_budget,
)


def chunk(text: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(id="p1", score=score, text=text, doc_id="d1")


def test_approx_tokens_rounds_up() -> None:
    assert approx_tokens("") == 0
    assert approx_tokens("abcd") == 1
    assert approx_tokens("abcde") == 2


def test_budget_without_num_ctx_is_the_configured_ceiling() -> None:
    assert (
        rag_budget_tokens(
            1200, num_ctx=None, system_tokens=500, history_tokens=4000, message_tokens=100
        )
        == 1200
    )


def test_budget_shrinks_to_window_headroom() -> None:
    budget = rag_budget_tokens(
        1200,
        num_ctx=4096,
        system_tokens=300,
        history_tokens=2000,
        message_tokens=100,
    )
    assert budget == 4096 - ANSWER_RESERVE_TOKENS - 300 - 2000 - 100


def test_budget_never_negative() -> None:
    assert (
        rag_budget_tokens(
            1200, num_ctx=2048, system_tokens=500, history_tokens=4000, message_tokens=100
        )
        == 0
    )


def test_generous_window_keeps_configured_budget() -> None:
    assert (
        rag_budget_tokens(
            1200, num_ctx=32768, system_tokens=300, history_tokens=1000, message_tokens=50
        )
        == 1200
    )


def test_thinking_reserve_and_overhead_shrink_headroom() -> None:
    kwargs = {"num_ctx": 4096, "system_tokens": 300, "history_tokens": 500, "message_tokens": 100}
    normal = rag_budget_tokens(
        5000, reserve_tokens=ANSWER_RESERVE_TOKENS, overhead_tokens=280, **kwargs
    )
    thinking = rag_budget_tokens(
        5000, reserve_tokens=THINKING_ANSWER_RESERVE_TOKENS, overhead_tokens=280, **kwargs
    )
    assert normal == 4096 - ANSWER_RESERVE_TOKENS - 280 - 300 - 500 - 100
    assert thinking == 4096 - THINKING_ANSWER_RESERVE_TOKENS - 280 - 300 - 500 - 100
    assert normal - thinking == THINKING_ANSWER_RESERVE_TOKENS - ANSWER_RESERVE_TOKENS
    # Nonzero overhead reduces the headroom by exactly its own size.
    no_overhead = rag_budget_tokens(
        5000, reserve_tokens=ANSWER_RESERVE_TOKENS, overhead_tokens=0, **kwargs
    )
    assert no_overhead - normal == 280


def test_thinking_reserve_zeroes_a_window_the_normal_reserve_fits() -> None:
    # Mirrors the chat-path scenario: a 2048 window leaves grounding room
    # under the normal reserve, and none at all under the thinking reserve.
    kwargs = {
        "num_ctx": 2048,
        "system_tokens": 326,
        "history_tokens": 0,
        "message_tokens": 10,
        "overhead_tokens": 280,
    }
    assert rag_budget_tokens(1200, reserve_tokens=ANSWER_RESERVE_TOKENS, **kwargs) > 0
    assert rag_budget_tokens(1200, reserve_tokens=THINKING_ANSWER_RESERVE_TOKENS, **kwargs) == 0


def test_trim_keeps_whole_chunks_in_rank_order() -> None:
    chunks = [chunk("a" * 400), chunk("b" * 400), chunk("c" * 400)]
    kept = trim_chunks_to_budget(chunks, budget_tokens=200)
    assert [c.text[0] for c in kept] == ["a", "b"]


def test_trim_zero_budget_keeps_nothing() -> None:
    assert trim_chunks_to_budget([chunk("hello")], budget_tokens=0) == []


def test_trim_stops_at_first_overflow() -> None:
    # A small chunk after an overflowing one is still dropped: the ranking
    # order is the relevance order, so skipping ahead would promote weaker
    # context above stronger context.
    chunks = [chunk("a" * 400), chunk("b" * 4000), chunk("c" * 4)]
    kept = trim_chunks_to_budget(chunks, budget_tokens=200)
    assert len(kept) == 1
