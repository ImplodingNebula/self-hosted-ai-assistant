"""Context-window budgeting: cheap token estimates and the per-turn split.

One approximation rules the whole platform: 1 token ~ 4 characters. It is
deterministic, free, and close enough for truncation decisions (the exact
tokenizer varies per model anyway). History capping, chunk sizing, and the
retrieval budget below all share it so their numbers compose.

The per-turn context envelope (docs/ARCHITECTURE.md section 6): the system
prompt is reserved first, the answer gets a fixed reserve, retrieved context
gets at most RAG_TOKEN_BUDGET, and history keeps whatever was already capped
into it - trimmed further only when the model's window (num_ctx) cannot hold
the total.
"""

from __future__ import annotations

from app.providers.rag.base import ScoredChunk

# Approximate chars-per-token shared by every budget in the platform.
APPROX_CHARS_PER_TOKEN = 4

# Tokens held back for the model's own answer when fitting against num_ctx.
# Thinking turns reserve more: the reasoning stream shares the window with
# the answer, and overrunning it makes the runtime evict the prompt HEAD -
# the system prompt first, then the oldest history - degrading exactly the
# turns that packed the window fullest.
ANSWER_RESERVE_TOKENS = 1024
THINKING_ANSWER_RESERVE_TOKENS = 2048


def approx_tokens(text: str) -> int:
    """Estimate the token count of ``text`` (1 token ~ 4 chars, rounded up)."""
    return (len(text) + APPROX_CHARS_PER_TOKEN - 1) // APPROX_CHARS_PER_TOKEN


def rag_budget_tokens(
    configured_budget: int,
    *,
    num_ctx: int | None,
    system_tokens: int,
    history_tokens: int,
    message_tokens: int,
    reserve_tokens: int = ANSWER_RESERVE_TOKENS,
    overhead_tokens: int = 0,
) -> int:
    """Return how many tokens retrieved context may use this turn.

    The configured budget is the ceiling; when the model's context window is
    known, the budget also shrinks to what actually fits after the system
    prompt, history, the user message, the grounding scaffolding, and the
    answer reserve. Zero means grounding must be skipped outright - shipping
    truncated context would emit citations for text the model never saw. An
    unknown window (a runtime that would not report one) trusts the configured
    ceiling alone.

    Args:
        configured_budget: RAG_TOKEN_BUDGET from settings.
        num_ctx: The model's context window in tokens, if known.
        system_tokens: Estimated tokens of the system prompt.
        history_tokens: Estimated tokens of the replayed history.
        message_tokens: Estimated tokens of the user message.
        reserve_tokens: Tokens held back for the model's own output.
        overhead_tokens: Tokens the grounding scaffolding itself costs.

    Returns:
        The token budget for retrieved chunks (>= 0).
    """
    budget = max(0, configured_budget)
    if num_ctx is not None:
        headroom = (
            num_ctx
            - reserve_tokens
            - overhead_tokens
            - system_tokens
            - history_tokens
            - message_tokens
        )
        budget = min(budget, max(0, headroom))
    return budget


def history_budget_tokens(
    configured_budget: int,
    *,
    num_ctx: int,
    system_tokens: int,
    message_tokens: int,
    reserve_tokens: int = ANSWER_RESERVE_TOKENS,
) -> int:
    """Return how many tokens replayed history may use this turn.

    The sibling of :func:`rag_budget_tokens`, one layer earlier in the
    envelope, and it exists because the configured budget alone is not a
    guarantee of anything. ``HISTORY_TOKEN_BUDGET`` defaults to 4000 tokens
    against a window that a workspace ceiling may set as low as 4096: the
    system prompt and the answer reserve then do not fit beside it, and the
    runtime evicts the prompt HEAD to make room - taking the persona and the
    injection-defense layer with it, silently, on a turn that still answers.

    Capping history against the WINDOW rather than against a fixed number is
    what makes "the assembled prompt fits" a property of the code instead of a
    property of the operator's configuration. Retrieval is budgeted after this
    and reads the trimmed history, so the two compose rather than double-count.

    Args:
        configured_budget: HISTORY_TOKEN_BUDGET from settings - a ceiling, and
            the only limit when it is already the smaller of the two.
        num_ctx: The window this turn will request. Always known (see
            :func:`app.services.model_facts.resolve_num_ctx`).
        system_tokens: Estimated tokens of the system prompt.
        message_tokens: Estimated tokens of the user message.
        reserve_tokens: Tokens held back for the model's own output; a
            thinking turn reserves more.

    Returns:
        The token budget for replayed history (>= 0). Zero means the window
        holds the system prompt, the question and the answer and nothing else,
        which is a short window doing the only honest thing available to it.
    """
    headroom = num_ctx - system_tokens - message_tokens - reserve_tokens
    return max(0, min(max(0, configured_budget), headroom))


def trim_chunks_to_budget(chunks: list[ScoredChunk], budget_tokens: int) -> list[ScoredChunk]:
    """Keep the best-scored whole chunks that fit within ``budget_tokens``.

    Chunks arrive ranked (best first) and are kept greedily in that order; a
    chunk that would overflow the budget is dropped along with everything
    after it, so the context never ends on a truncated passage.
    """
    kept: list[ScoredChunk] = []
    used = 0
    for chunk in chunks:
        cost = approx_tokens(chunk.text)
        if used + cost > budget_tokens:
            break
        kept.append(chunk)
        used += cost
    return kept
