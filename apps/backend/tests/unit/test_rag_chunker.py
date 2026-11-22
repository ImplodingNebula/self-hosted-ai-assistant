"""Unit tests for the deterministic token-aware chunker."""

from app.providers.rag.chunker import TextChunker


def make_chunker(chunk_tokens: int = 25, overlap_tokens: int = 5) -> TextChunker:
    return TextChunker(chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)


def test_empty_text_yields_no_chunks() -> None:
    assert make_chunker().chunk("", doc_id="d1") == []
    assert make_chunker().chunk("   \n\n  ", doc_id="d1") == []


def test_short_text_is_one_chunk_with_metadata() -> None:
    chunks = make_chunker().chunk("A short note.", doc_id="d1", title="Note")
    assert len(chunks) == 1
    assert chunks[0].text == "A short note."
    assert chunks[0].index == 0
    assert chunks[0].doc_id == "d1"
    assert chunks[0].title == "Note"


def test_long_text_splits_with_sequential_indexes() -> None:
    sentences = " ".join(f"Sentence number {i} adds some words." for i in range(20))
    chunks = make_chunker().chunk(sentences, doc_id="d1")
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.text


def test_chunking_is_deterministic() -> None:
    text = " ".join(f"Sentence {i} with content." for i in range(30))
    first = make_chunker().chunk(text, doc_id="d1")
    second = make_chunker().chunk(text, doc_id="d1")
    assert [c.text for c in first] == [c.text for c in second]


def test_consecutive_chunks_share_overlap() -> None:
    text = " ".join(f"Fact {i} is stated here." for i in range(30))
    chunks = make_chunker(chunk_tokens=25, overlap_tokens=10).chunk(text, doc_id="d1")
    assert len(chunks) >= 2
    # The second chunk starts with the tail piece of the first.
    first_tail = chunks[0].text.rsplit(".", 2)[-2].strip()
    assert first_tail
    assert first_tail in chunks[1].text


def test_zero_overlap_produces_disjoint_chunks() -> None:
    text = " ".join(f"Item {i} noted." for i in range(40))
    chunks = make_chunker(chunk_tokens=20, overlap_tokens=0).chunk(text, doc_id="d1")
    assert len(chunks) >= 2
    rebuilt = " ".join(c.text for c in chunks)
    assert rebuilt.count("Item 3 noted.") == 1


def test_oversized_unbroken_word_is_hard_cut() -> None:
    blob = "x" * 500
    chunks = make_chunker(chunk_tokens=25).chunk(blob, doc_id="d1")
    assert len(chunks) >= 2
    assert all(len(c.text) <= 25 * 4 + 1 for c in chunks)


def test_paragraphs_within_budget_stay_intact() -> None:
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = make_chunker(chunk_tokens=100).chunk(text, doc_id="d1")
    assert len(chunks) == 1
    assert chunks[0].text == "First paragraph here. Second paragraph here."


def test_internal_whitespace_is_normalized() -> None:
    chunks = make_chunker().chunk("spaced    out\ttext\nhere", doc_id="d1")
    assert chunks[0].text == "spaced out text here"
