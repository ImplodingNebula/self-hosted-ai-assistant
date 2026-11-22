"""Thinking policy: decide whether a chat turn should use reasoning."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from app.services.model_resolution import ThinkMode

# Verbs and phrases that usually signal a task worth reasoning about. Kept
# conservative on purpose: false positives cost latency on every turn.
_REASONING_HINT = re.compile(
    r"\b(prove|derive|debug|analy[sz]e|compare|design|plan|optimi[sz]e|calculate"
    r"|solve|refactor|architecture|algorithm|complexity|step[ -]by[ -]step"
    r"|trade[ -]?offs?)\b",
    re.IGNORECASE,
)

# Auto-mode thresholds. Long messages or a hint verb with some substance suggest
# a task that benefits from reasoning; anything else stays on the fast path.
_LONG_MESSAGE_CHARS = 600
_HINT_MIN_CHARS = 120
_MIN_QUESTION_MARKS = 3


def resolve_thinking(
    mode: Literal["auto", "on", "off"],
    message: str,
    history: Sequence = (),
) -> bool:
    """Decide whether this turn should run on the thinking model.

    This is the v1 heuristic, replaced by the orchestrator in a later phase.
    ``on`` and ``off`` are explicit user choices and always win. ``auto`` uses
    deterministic, conservative signals - the bias is toward the fast
    non-thinking path.

    Args:
        mode: The client-requested thinking mode.
        message: The new user message for this turn.
        history: Prior turns; unused by the v1 heuristic, accepted so the
            signature survives the orchestrator swap.

    Returns:
        True when the turn should use reasoning.
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    stripped = message.strip()
    if len(stripped) >= _LONG_MESSAGE_CHARS:
        return True
    if "```" in message:
        return True
    if message.count("?") >= _MIN_QUESTION_MARKS:
        return True
    return bool(_REASONING_HINT.search(message)) and len(stripped) >= _HINT_MIN_CHARS


def normalize_think(think: bool, model_mode: ThinkMode) -> bool | None:
    """Reconcile the turn's think decision with what the target model can do.

    An always-thinking model (e.g. the Qwen3 2507 thinking checkpoint) reasons
    no matter what; sending ``think=false`` makes the server skip parsing the
    reasoning, which then leaks raw ``<think>`` text into the visible answer.
    Forcing ``True`` keeps the reasoning on the thinking channel where the UI
    collapses it. A never-thinking model gets ``None`` (flag omitted) so the
    request cannot be rejected for carrying a flag the model does not support.
    Hybrid and unknown models pass the decision through explicitly - the
    provider's retry-without-think handles any model that still objects.

    Args:
        think: The resolved thinking decision for this turn.
        model_mode: The target model's :data:`~app.services.model_resolution.ThinkMode`.

    Returns:
        The think value to place on :class:`GenerationOptions` (None = omit).
    """
    if model_mode == "always":
        return True
    if model_mode == "never":
        return None
    return think
