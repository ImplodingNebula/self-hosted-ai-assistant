"""The code-shipped quirk seed: exact beats glob, alias-aware, gates nothing."""

from __future__ import annotations

import re

import pytest

from app.services.embedder_check import PrefixChoice, resolve_query_prefix
from app.services.known_models import KNOWN_MODELS, quirks_for


def test_an_exact_tag_answers_for_itself() -> None:
    entry = quirks_for("qwen3:4b-instruct")
    assert entry is not None
    assert (entry.think_mode, entry.thinking_model) == ("never", "qwen3:4b")


def test_a_family_glob_answers_for_a_member_nothing_names() -> None:
    assert quirks_for("qwen3:8b").think_mode == "hybrid"
    assert quirks_for("qwen3:30b-a3b").think_mode == "hybrid"


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        # The 2507 split, which is why exact-beats-glob is a RULE rather than a
        # side effect of declaration order: both of these sit inside the
        # `qwen3:*` hybrid family, and neither of them is hybrid.
        ("qwen3:4b", "always"),
        ("qwen3:4b-instruct", "never"),
    ],
)
def test_an_exact_pattern_beats_a_family_glob(tag: str, expected: str) -> None:
    assert quirks_for(tag).think_mode == expected
    # ...and the family really does claim them, which is what makes the
    # override meaningful rather than vacuous.
    assert quirks_for("qwen3:9b").think_mode == "hybrid"


def test_an_exact_pattern_wins_even_when_the_glob_is_declared_first() -> None:
    # `qwen3-embedding:*` is declared ABOVE `qwen3-embedding:0.6b`, so a plain
    # in-order scan would hand the 0.6b tag the family row and lose its floors.
    patterns = [entry.pattern for entry in KNOWN_MODELS]
    assert patterns.index("qwen3-embedding:*") < patterns.index("qwen3-embedding:0.6b")
    entry = quirks_for("qwen3-embedding:0.6b")
    assert (entry.min_score, entry.rel_margin) == (0.35, 0.10)


def test_exact_matching_is_alias_aware() -> None:
    # Both sides are written-down configuration, so the same model spelled two
    # ways is one model - the same rule every other tag comparison between two
    # configured values follows (app.providers.base.tag_aliases).
    assert quirks_for("library/qwen3:4b-instruct").think_mode == "never"


def test_a_tag_nothing_describes_answers_nothing() -> None:
    assert quirks_for("mistral-small:24b") is None
    assert quirks_for("") is None


@pytest.mark.parametrize("tag", ["gemma3:12b", "deepseek-r1:7b"])
def test_a_family_nobody_measured_here_is_deliberately_absent(tag: str) -> None:
    # These two were considered and left out. An unmeasured think_mode is not
    # the harmless kind of stale - it outranks the runtime's own template
    # report, so a wrong guess silences a fact the serving stack was willing to
    # state. Absence sends the tag to the template inference instead, which is
    # the weaker answer and the honest one.
    assert quirks_for(tag) is None


def test_hf_packaging_carries_a_note_and_no_policy() -> None:
    entry = quirks_for("hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M")
    assert entry is not None
    assert "tool_calls" in entry.note
    # It GATES NOTHING: the seed warns about the packaging and declares no
    # ceiling, no think mode, and nothing else a resolver reads as permission.
    assert entry.think_mode is None
    assert entry.thinking_model is None


def test_every_entry_carries_dated_provenance() -> None:
    # The whole contract of the file: a reader who disagrees with a line can go
    # find the measurement it names, and see how old it is.
    for entry in KNOWN_MODELS:
        assert entry.provenance.strip(), entry.pattern
        assert re.search(r"\b(19|20)\d{2}\b", entry.provenance), entry.pattern


def test_one_family_embeds_its_queries_on_one_scale() -> None:
    # A vector space is only comparable to itself, and the relevance floors
    # beside these entries are calibrated against the PREFIXED similarity
    # scale. The exact entry shadows the family glob outright rather than
    # merging with it, so the specific tag has to repeat the wrapper - and a
    # copy that drifted by one character would silently re-scale every query
    # made through it while the floors stayed where they are.
    family = quirks_for("qwen3-embedding:4b")
    specific = quirks_for("qwen3-embedding:0.6b")
    assert family is not None and specific is not None
    assert family.query_prefix
    assert specific.query_prefix == family.query_prefix


def test_the_seed_is_the_only_source_of_a_query_wrapper() -> None:
    # It used to be one of two, and the second - a committed per-deployment
    # file - could name a different string for the same weights. Live retrieval
    # read one and the embedder harness read the other, so a check could
    # validate floors against a scale nothing queried on. The harness resolves
    # through here now, and a tag the seed does not describe embeds RAW rather
    # than borrowing a neighbour's wrapper.
    assert resolve_query_prefix("qwen3-embedding:0.6b") == PrefixChoice(
        prefix=quirks_for("qwen3-embedding:0.6b").query_prefix, source="known_models"
    )
    assert resolve_query_prefix("mistral-small:24b") == PrefixChoice(prefix="", source="none")
