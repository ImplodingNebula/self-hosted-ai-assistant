"""Unit tests for the agency resolver, the AGENCY_MODE parser, and the offer."""

from __future__ import annotations

import itertools

import pytest

from app.services.agency import (
    agency_rank,
    parse_agency_mode,
    resolve_agency,
    resolve_tool_offer,
)
from app.services.model_resolution import AgencyMode

_MODES: tuple[AgencyMode, ...] = ("deterministic", "assisted", "agentic")


def _offer(**overrides):
    """resolve_tool_offer with agentic-capable defaults, overridable per test."""
    kwargs = {
        "ceiling": "agentic",
        "earned": "agentic",
        "env_mode": "agentic",
        "request_mode": None,
        "tools_native": True,
        "thinking_with_tools": False,
        "requested_thinking": "auto",
        "heuristic_think": False,
        "web_offered": True,
        "knowledge_offered": True,
    }
    kwargs.update(overrides)
    return resolve_tool_offer(**kwargs)


# --- parse_agency_mode ---


def test_auto_is_no_clamp() -> None:
    assert parse_agency_mode("auto") is None


def test_real_modes_parse_to_themselves() -> None:
    assert parse_agency_mode("deterministic") == "deterministic"
    assert parse_agency_mode("assisted") == "assisted"
    assert parse_agency_mode("agentic") == "agentic"


def test_unexpected_value_is_no_clamp() -> None:
    # The config Literal rejects these before they reach here; treated as "auto".
    assert parse_agency_mode("banana") is None
    assert parse_agency_mode("") is None


# --- resolve_agency: the clamp-down rule ---


def test_all_agentic_stays_agentic() -> None:
    assert resolve_agency(ceiling="agentic", earned="agentic", env_mode="agentic") == "agentic"


def test_env_clamps_below_ceiling() -> None:
    assert resolve_agency(ceiling="agentic", earned="agentic", env_mode="assisted") == "assisted"


def test_earned_clamps_below_ceiling_and_env() -> None:
    # A model that has not earned the loop cannot be pushed into it by the env.
    assert (
        resolve_agency(ceiling="agentic", earned="deterministic", env_mode="agentic")
        == "deterministic"
    )


def test_auto_env_defers_to_ceiling() -> None:
    # env_mode None ("auto") does not clamp: the ceiling (and earned) decide.
    assert resolve_agency(ceiling="assisted", earned="agentic", env_mode=None) == "assisted"


def test_request_clamps_further() -> None:
    assert (
        resolve_agency(
            ceiling="agentic", earned="agentic", env_mode="agentic", request="deterministic"
        )
        == "deterministic"
    )


def test_request_none_does_not_clamp() -> None:
    assert (
        resolve_agency(ceiling="agentic", earned="agentic", env_mode="agentic", request=None)
        == "agentic"
    )


def test_a_runtime_floor_forces_deterministic() -> None:
    # e.g. no tool survived the user's switches, or no room for a tool result.
    assert (
        resolve_agency(
            ceiling="agentic",
            earned="agentic",
            env_mode="agentic",
            runtime_floors=["deterministic"],
        )
        == "deterministic"
    )


def test_no_constraints_is_deterministic() -> None:
    # Degenerate case: nothing said anything -> the safe floor.
    assert resolve_agency(ceiling=None, earned=None, env_mode=None) == "deterministic"  # type: ignore[arg-type]


# --- the property that a bug here would be a privilege-escalation bug ---


def test_result_never_exceeds_the_minimum_of_every_input() -> None:
    """Exhaustively: the outcome equals the least-autonomous non-None input."""
    floor_options: list[list[AgencyMode]] = [[], ["deterministic"], ["assisted"], ["agentic"]]
    for ceiling, earned in itertools.product(_MODES, _MODES):
        for env_mode in (*_MODES, None):
            for request in (*_MODES, None):
                for floors in floor_options:
                    result = resolve_agency(
                        ceiling=ceiling,
                        earned=earned,
                        env_mode=env_mode,
                        request=request,
                        runtime_floors=floors,
                    )
                    present = [
                        m for m in (ceiling, earned, env_mode, request, *floors) if m is not None
                    ]
                    expected_rank = min(agency_rank(m) for m in present)
                    assert agency_rank(result) == expected_rank
                    # And, restated as the invariant that matters: never above
                    # any single input.
                    for m in present:
                        assert agency_rank(result) <= agency_rank(m)


# --- resolve_tool_offer: the whole per-turn decision ---


def test_agentic_auto_offers_both_tools_and_suppresses_thinking() -> None:
    # The shipped default path: agentic, auto thinking, a model that cannot do
    # both -> tools win, reasoning off, ambient suppressed, notice armed.
    offer = _offer()
    assert offer.mode == "agentic"
    assert offer.agentic is True
    assert offer.think is False
    assert offer.offer_web and offer.offer_knowledge


def test_agentic_explicit_thinking_beats_tools_and_runs_deterministic() -> None:
    # thinking "on" is a deliberate ask: keep the reasoning, drop the tools.
    offer = _offer(requested_thinking="on", heuristic_think=True)
    assert offer.mode == "deterministic"
    assert offer.think is True
    assert not offer.offer_web and not offer.offer_knowledge
    assert offer.agentic is False


def test_thinking_with_tools_model_keeps_reasoning_and_tools() -> None:
    # A model that CAN interleave: no arbitration, thinking passes through.
    offer = _offer(thinking_with_tools=True, requested_thinking="on", heuristic_think=True)
    assert offer.mode == "agentic"
    assert offer.think is True
    assert offer.offer_web and offer.offer_knowledge


def test_assisted_thinking_turn_keeps_reasoning_and_offers_no_tools() -> None:
    # assisted stays byte-for-byte the pre-tool behaviour on a turn that reasons.
    offer = _offer(env_mode="assisted", heuristic_think=True)
    assert offer.mode == "deterministic"
    assert offer.think is True
    assert not offer.offer_web and not offer.offer_knowledge


def test_assisted_non_thinking_turn_offers_tools_but_is_not_agentic() -> None:
    # assisted keeps the ambient pre-fetch (agentic False) AND offers tools.
    offer = _offer(env_mode="assisted", heuristic_think=False)
    assert offer.mode == "assisted"
    assert offer.agentic is False
    assert offer.offer_web and offer.offer_knowledge


def test_web_switch_off_drops_only_the_web_tool() -> None:
    offer = _offer(web_offered=False)
    assert offer.mode == "agentic"
    assert offer.offer_web is False
    assert offer.offer_knowledge is True


def test_no_tool_survives_the_switches_falls_to_deterministic() -> None:
    # The tools_offered == 0 floor: nothing to offer -> the pre-tool pipeline.
    offer = _offer(web_offered=False, knowledge_offered=False)
    assert offer.mode == "deterministic"
    assert offer.agentic is False


def test_a_non_native_template_can_never_be_offered_tools() -> None:
    # Mechanical floor: a template that cannot take a tool list is deterministic
    # regardless of the agentic ceiling.
    offer = _offer(tools_native=False)
    assert offer.mode == "deterministic"
    assert not offer.offer_web and not offer.offer_knowledge


def test_env_clamp_to_deterministic_offers_nothing() -> None:
    offer = _offer(env_mode="deterministic")
    assert offer.mode == "deterministic"
    assert offer.agentic is False


def test_request_clamp_below_env_wins() -> None:
    # A per-request assisted clamp on an agentic deployment: assisted behaviour.
    offer = _offer(request_mode="assisted", heuristic_think=False)
    assert offer.mode == "assisted"
    assert offer.agentic is False
    assert offer.offer_web and offer.offer_knowledge


@pytest.mark.parametrize("requested", ["auto", "off"])
def test_agentic_non_explicit_thinking_always_yields_to_tools(requested) -> None:
    # Whether auto happened to pick thinking or the user left it off, an agentic
    # turn that is not an explicit "on" keeps its tools and reasons not.
    offer = _offer(requested_thinking=requested, heuristic_think=True)
    assert offer.think is False
    assert offer.offer_web and offer.offer_knowledge
