"""Unit tests for the v1 thinking heuristic and the think-flag normalization."""

from app.services.thinking import normalize_think, resolve_thinking


def test_on_always_thinks() -> None:
    assert resolve_thinking("on", "hi") is True


def test_off_never_thinks_even_on_reasoning_text() -> None:
    message = "Prove the algorithm complexity step by step. " * 30
    assert resolve_thinking("off", message) is False


def test_auto_trivial_message_stays_fast() -> None:
    assert resolve_thinking("auto", "hello there") is False


def test_auto_long_message_thinks() -> None:
    assert resolve_thinking("auto", "a" * 600) is True


def test_auto_just_below_length_threshold_stays_fast() -> None:
    assert resolve_thinking("auto", "a" * 599) is False


def test_auto_code_fence_thinks() -> None:
    assert resolve_thinking("auto", "why does this fail?\n```py\nprint(x)\n```") is True


def test_auto_three_questions_thinks() -> None:
    assert resolve_thinking("auto", "what? why? how?") is True


def test_auto_two_questions_stays_fast() -> None:
    assert resolve_thinking("auto", "what? why not") is False


def test_auto_reasoning_verb_with_substance_thinks() -> None:
    message = (
        "Compare the trade-offs between using a message queue and direct HTTP "
        "calls for ingesting records into the vector store in this platform."
    )
    assert len(message.strip()) >= 120
    assert resolve_thinking("auto", message) is True


def test_auto_reasoning_verb_too_short_stays_fast() -> None:
    assert resolve_thinking("auto", "debug this") is False


def test_auto_spelling_variants_match() -> None:
    message = "Analyse and optimise the retrieval pipeline latency " + "x" * 80
    assert resolve_thinking("auto", message) is True


# --- think-flag normalization against the target model's capabilities ---


def test_always_model_forces_think_true() -> None:
    # Sending think=false to an always-thinking model makes the server skip
    # parsing the reasoning, leaking raw think text into the answer.
    assert normalize_think(False, "always") is True
    assert normalize_think(True, "always") is True


def test_never_model_omits_the_flag() -> None:
    assert normalize_think(False, "never") is None
    assert normalize_think(True, "never") is None


def test_hybrid_model_passes_decision_through() -> None:
    assert normalize_think(False, "hybrid") is False
    assert normalize_think(True, "hybrid") is True


def test_unknown_model_passes_decision_through() -> None:
    assert normalize_think(False, "unknown") is False
    assert normalize_think(True, "unknown") is True
