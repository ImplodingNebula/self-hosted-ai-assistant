"""Unit tests for the usage-gated citation helpers (prompts.py).

Grounding OFFERS passages; the answer decides. These pin down the marker
parsing that decides which offered sources are emitted and persisted.
"""

from app.providers.base import Citation
from app.services.prompts import (
    referenced_indexes,
    renumber_used_citations,
    used_citations,
)


def _citation(index: int) -> Citation:
    return Citation(id=f"c{index}", index=index, source=f"Source {index}", snippet="snippet")


def test_referenced_indexes_parses_single_and_grouped_markers() -> None:
    assert referenced_indexes("See [1] and [2, 3]; also [2] again.") == {1, 2, 3}


def test_referenced_indexes_ignores_wide_bracketed_numbers() -> None:
    # Years and array-ish literals are not source markers: the pattern caps at
    # two digits, the plausible passage range.
    assert referenced_indexes("In [2026] the array[100] grew by [999].") == set()


def test_referenced_indexes_ignores_code_spans() -> None:
    # Bracketed subscripts inside fenced blocks or inline code are not source
    # references; markers in the surrounding prose still count.
    answer = "Use `arr[1]` inline.\n```python\nprint(sys.argv[1], items[2])\n```\nSee [3]."
    assert referenced_indexes(answer) == {3}


def test_referenced_indexes_ignores_prose_subscripts() -> None:
    # A word character before the bracket marks a subscript, not a citation.
    assert referenced_indexes("the value argv[1] is read first") == set()


def test_referenced_indexes_counts_adjacent_chains() -> None:
    assert referenced_indexes("Backed by [1][2].") == {1, 2}


def test_referenced_indexes_unterminated_fence_strips_to_end() -> None:
    # A partial answer saved on disconnect can end mid-code-block; everything
    # after the opening fence is code, so its brackets do not count.
    assert referenced_indexes("Start [1].\n```python\nx = argv[2]") == {1}


def test_referenced_indexes_empty_answer() -> None:
    assert referenced_indexes("") == set()
    assert referenced_indexes("No markers here.") == set()


def test_used_citations_keeps_only_referenced_in_offer_order() -> None:
    offered = [_citation(1), _citation(2), _citation(3)]
    kept = used_citations(offered, "Backed by [3], see also [1].")
    assert [c.index for c in kept] == [1, 3]


def test_used_citations_without_markers_cites_nothing() -> None:
    assert used_citations([_citation(1)], "Hello.") == []


def test_used_citations_marker_with_no_matching_passage_selects_nothing() -> None:
    # A model echoing "[7]" from history cannot fabricate a citation: only
    # offered indexes count.
    assert used_citations([_citation(1)], "As shown in [7].") == []


def test_used_citations_with_no_offer_is_empty() -> None:
    assert used_citations([], "Even with a [1] marker.") == []


def test_renumber_maps_first_reference_to_one() -> None:
    # A reply citing only the fourth offered passage must display [1], not [4].
    offered = [_citation(i) for i in range(1, 5)]
    kept, rewritten = renumber_used_citations(offered, "Air quality is unhealthy [4].")
    assert [(c.index, c.id) for c in kept] == [(1, "c4")]
    assert rewritten == "Air quality is unhealthy [1]."


def test_renumber_orders_by_first_reference_not_offer_order() -> None:
    offered = [_citation(i) for i in range(1, 5)]
    kept, rewritten = renumber_used_citations(offered, "First [3], then [1], then [3] again.")
    assert [(c.index, c.id) for c in kept] == [(1, "c3"), (2, "c1")]
    assert rewritten == "First [1], then [2], then [1] again."


def test_renumber_rewrites_grouped_markers() -> None:
    offered = [_citation(i) for i in range(1, 5)]
    kept, rewritten = renumber_used_citations(offered, "Backed by [4, 2].")
    assert [(c.index, c.id) for c in kept] == [(1, "c4"), (2, "c2")]
    assert rewritten == "Backed by [1, 2]."


def test_renumber_leaves_unoffered_echo_markers_alone() -> None:
    # "[7]" echoed from history maps to nothing; it stays as-is rather than
    # being renumbered into a marker that would falsely match a real source.
    offered = [_citation(1), _citation(2)]
    kept, rewritten = renumber_used_citations(offered, "See [2] and [7].")
    assert [(c.index, c.id) for c in kept] == [(1, "c2")]
    assert rewritten == "See [1] and [7]."


def test_renumber_mixed_group_keeps_unoffered_number() -> None:
    offered = [_citation(i) for i in range(1, 5)]
    _, rewritten = renumber_used_citations(offered, "Backed by [4, 7].")
    assert rewritten == "Backed by [1, 7]."


def test_renumber_ignores_code_spans() -> None:
    offered = [_citation(i) for i in range(1, 5)]
    answer = "Use `arr[3]` and read [3].\n```python\nprint(sys.argv[3])\n```"
    kept, rewritten = renumber_used_citations(offered, answer)
    assert [(c.index, c.id) for c in kept] == [(1, "c3")]
    assert rewritten == "Use `arr[3]` and read [1].\n```python\nprint(sys.argv[3])\n```"


def test_renumber_identity_returns_answer_unchanged() -> None:
    offered = [_citation(1), _citation(2), _citation(3)]
    answer = "See [1] then [2]."
    kept, rewritten = renumber_used_citations(offered, answer)
    assert [(c.index, c.id) for c in kept] == [(1, "c1"), (2, "c2")]
    assert rewritten is answer


def test_renumber_without_markers_or_offer_is_empty() -> None:
    assert renumber_used_citations([_citation(1)], "Hello.") == ([], "Hello.")
    assert renumber_used_citations([], "Even with a [1] marker.") == ([], "Even with a [1] marker.")
