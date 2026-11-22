"""Agency policy: decide how much a chat turn lets the model drive.

How autonomous a turn is - whether the model is handed tools and left to decide
when to search the web or the stored records, or handed nothing and run through
the deterministic pre-tool pipeline - is a routing decision resolved here, the
same shape as :func:`app.services.thinking.resolve_thinking` and
:func:`app.services.web_search.resolve_web_search`.

The one rule that matters: **every input can only clamp DOWN**. The per-model
ceiling (a stored operator declaration), what the model has actually earned
from the tool-check harness, the ``AGENCY_MODE`` deployment setting, the
request, and any number of runtime floors are each an independent claim, and the
turn runs in the least autonomous of them. Nothing here can raise a turn's
autonomy above what every one of its inputs allowed - a property test pins
exactly that, because an escalation bug here would be a privilege-escalation bug.

:func:`resolve_agency` is the pure clamp; :func:`resolve_tool_offer` builds on it
to also break the thinking-vs-tools tie and gate the tool set by the user's
switches, returning the whole per-turn decision the chat service acts on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.services.model_resolution import AgencyMode

# Least to most autonomous. The resolver works entirely in terms of this rank,
# so "clamp down" is literally "take the minimum rank".
_AGENCY_RANK: dict[AgencyMode, int] = {
    "deterministic": 0,
    "assisted": 1,
    "agentic": 2,
}
_MODE_BY_RANK: dict[int, AgencyMode] = {rank: mode for mode, rank in _AGENCY_RANK.items()}


def agency_rank(mode: AgencyMode) -> int:
    """Return the ordinal of ``mode`` on the least-to-most-autonomous scale."""
    return _AGENCY_RANK[mode]


def parse_agency_mode(value: str) -> AgencyMode | None:
    """Turn the ``AGENCY_MODE`` setting into a clamp, or None for "no clamp".

    ``AGENCY_MODE`` accepts the three real modes plus ``"auto"``, which means
    "defer to the model's ceiling" - i.e. do not clamp on this axis. A real mode
    becomes a clamp at that mode; ``"auto"`` (and, defensively, any unexpected
    string that the config Literal would already have rejected) becomes None.

    Args:
        value: The configured ``AGENCY_MODE`` string.

    Returns:
        The clamp to apply for the deployment, or None to leave this axis open.
    """
    if value in _AGENCY_RANK:
        return value  # type: ignore[return-value]  # membership narrows to AgencyMode
    return None


def resolve_agency(
    *,
    ceiling: AgencyMode,
    earned: AgencyMode,
    env_mode: AgencyMode | None,
    request: AgencyMode | None = None,
    runtime_floors: Sequence[AgencyMode] = (),
) -> AgencyMode:
    """Resolve the mode a turn runs in as the minimum over every constraint.

    Each argument is an independent claim about the most autonomy this turn may
    have; the result is the least autonomous of them, so no single input can
    raise the outcome above what another allowed.

    Args:
        ceiling: The per-model resolved ceiling (a stored declaration, else
            what the runtime reports; ``deterministic`` for an
            unreadable one). A probe never raises it.
        earned: The latest measured verdict from the tool-check harness. A
            never-probed, never-declared model is ``deterministic``, which
            structurally denies the loop to a model that cannot emit a tool call.
        env_mode: The deployment clamp from ``AGENCY_MODE`` (see
            :func:`parse_agency_mode`); None means "auto" - defer to the ceiling.
        request: An optional per-request clamp (a developer/testing affordance,
            not a composer control); None means the request does not constrain it.
        runtime_floors: Zero or more floors decided while assembling the turn -
            no tool survived the user's switches, no room in the context window
            for a tool result, a reasoning turn on a model that cannot reason and
            call tools at once. Any of them can force ``deterministic``.

    Returns:
        The agency mode to run the turn in.
    """
    # ceiling and earned are non-Optional BY CONTRACT: an unknown or never-probed
    # model must be resolved to a concrete "deterministic" by the caller (see
    # effective_agency_ceiling, and the earned-verdict lookup that reads the probe
    # log), NEVER passed here as None. A None on either of these two axes would be
    # filtered out as "no clamp" below and let the remaining axes drive - which on
    # the ceiling/earned axes is an escalation, not a relaxation. env_mode and
    # request are the genuinely optional axes (None = auto = no clamp).
    candidates: list[AgencyMode | None] = [ceiling, earned, env_mode, request, *runtime_floors]
    ranks = [_AGENCY_RANK[mode] for mode in candidates if mode is not None]
    if not ranks:
        return "deterministic"
    return _MODE_BY_RANK[min(ranks)]


@dataclass(frozen=True)
class ToolOffer:
    """The resolved decision for one turn: how it drives, and what it is offered.

    Attributes:
        mode: The agency mode the turn actually runs in, after clamping AND the
            thinking-vs-tools arbitration. It is ``"deterministic"`` whenever no
            tool survives - a model that cannot drive, a deployment clamp, a
            reasoning turn that beat the tools, or an empty tool set - so the
            caller reads a single field to know whether the pre-tool pipeline
            applies verbatim.
        think: Whether the turn should reason, AFTER arbitration. In an agentic
            turn on a model that cannot reason and call tools at once, an ``auto``
            thinking decision is turned OFF so the tools survive; an explicit
            ``on`` keeps reasoning and drops the tools instead. The caller routes
            the model on this value (never the raw heuristic) and normalizes it
            against the model's think mode.
        offer_web: Offer the ``web_search`` tool this turn.
        offer_knowledge: Offer the ``search_knowledge`` tool this turn.
        agentic: The turn drives itself - suppress the deterministic web
            pre-fetch and the ambient knowledge base (named and inferred files
            are still retrieved), and stand a ``tools_not_used`` notice when the
            model answers without calling a tool. False for ``assisted`` (which
            keeps the pre-fetch AND offers tools) and ``deterministic``.
    """

    mode: AgencyMode
    think: bool
    offer_web: bool
    offer_knowledge: bool
    agentic: bool


def _deterministic(think: bool) -> ToolOffer:
    """The no-tools outcome: the pre-tool pipeline, carrying the think decision."""
    return ToolOffer(
        mode="deterministic", think=think, offer_web=False, offer_knowledge=False, agentic=False
    )


def resolve_tool_offer(
    *,
    ceiling: AgencyMode,
    earned: AgencyMode,
    env_mode: AgencyMode | None,
    request_mode: AgencyMode | None,
    tools_native: bool,
    thinking_with_tools: bool,
    requested_thinking: Literal["auto", "on", "off"],
    heuristic_think: bool,
    web_offered: bool,
    knowledge_offered: bool,
) -> ToolOffer:
    """Decide a turn's agency mode, its think flag, and which tools it is offered.

    Composes :func:`resolve_agency` (the clamp-down over every autonomy claim)
    with two further gates that make the result honest:

    - **Thinking vs tools.** Only when the model cannot reason and call tools in
      the same turn (``thinking_with_tools`` false, e.g. qwen3 on Ollama,
      ollama/ollama#10976) is there a conflict to break. In ``agentic`` an
      ``auto`` thinking decision yields to the tools (reasoning off, tools kept)
      while an explicit ``on`` wins (reasoning kept, tools dropped, the turn runs
      deterministic). In ``assisted`` the tie breaks the other way - a thinking
      turn keeps its reasoning and is offered no tools - so assisted stays
      byte-for-byte the pre-tool behaviour on any turn that would have reasoned.
      When ``thinking_with_tools`` is true there is nothing to arbitrate and the
      heuristic think decision passes through untouched.

    - **The tool set follows the user's switches.** ``web_offered`` and
      ``knowledge_offered`` already fold in the request switches, the workspace
      settings, and whether each collaborator is wired. If neither survives, the
      turn has no tools to offer and runs deterministic - which is what keeps the
      locked "web search off = fully local" literally true, since a turn with the
      web switch off never reaches the model with a live-fetch tool.

    The one escalation-proof rule from :func:`resolve_agency` still holds end to
    end: every branch here can only turn a more-autonomous outcome into a
    less-autonomous one.

    Args:
        ceiling: The per-model resolved ceiling.
        earned: The measured verdict, folded per the active regime; a
            declaration alone reaches assisted and no further.
        env_mode: The ``AGENCY_MODE`` deployment clamp, or None for "auto".
        request_mode: An optional per-request clamp (testing affordance).
        tools_native: Whether the model's chat template can take a tool list at
            all; false forces ``deterministic`` (a mechanical floor).
        thinking_with_tools: Whether the model can reason and call tools at once.
        requested_thinking: The turn's effective thinking switch (auto|on|off;
            the request when explicit, else the stored default) - the
            explicit/auto distinction is what the arbitration turns on.
        heuristic_think: The think decision from :func:`resolve_thinking`.
        web_offered: Whether ``web_search`` may be offered (switch + wiring).
        knowledge_offered: Whether ``search_knowledge`` may be offered (wiring).

    Returns:
        The resolved :class:`ToolOffer` for the turn.
    """
    # A model whose template cannot take a tool list is a hard deterministic
    # floor, not merely an empty offer.
    runtime_floors: tuple[AgencyMode, ...] = () if tools_native else ("deterministic",)
    mode = resolve_agency(
        ceiling=ceiling,
        earned=earned,
        env_mode=env_mode,
        request=request_mode,
        runtime_floors=runtime_floors,
    )
    if mode == "deterministic":
        return _deterministic(heuristic_think)

    # Thinking vs tools. Only a model that cannot do both at once forces a choice.
    if thinking_with_tools:
        think = heuristic_think
    elif mode == "agentic":
        # auto -> tools win (reason off, keep tools); explicit on -> thinking wins.
        if requested_thinking == "on":
            return _deterministic(think=True)
        think = False
    else:  # assisted: a thinking turn keeps reasoning and is offered no tools.
        if heuristic_think:
            return _deterministic(think=True)
        think = False

    offer_web = web_offered
    offer_knowledge = knowledge_offered
    if not (offer_web or offer_knowledge):
        # Nothing survives the switches: no tool to offer, so the turn is
        # deterministic (the tools_offered == 0 floor).
        return _deterministic(think)
    return ToolOffer(
        mode=mode,
        think=think,
        offer_web=offer_web,
        offer_knowledge=offer_knowledge,
        agentic=mode == "agentic",
    )
