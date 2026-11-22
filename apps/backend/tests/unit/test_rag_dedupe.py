"""Tests for source-level chunk de-duplication (one grounded passage/citation per source).

A long web page or document splits into many chunks that all carry the same
``url``/``doc_id``; retrieval can return several of them, which previously became
several identical-looking citations. ``dedupe_chunks_by_source`` collapses those
to the best chunk per source so a source is grounded and cited exactly once.
"""

from app.providers.rag.base import ScoredChunk
from app.services.rag_service import dedupe_chunks_by_source


def _chunk(chunk_id: str, score: float, doc_id: str, *, url: str | None = None) -> ScoredChunk:
    return ScoredChunk(id=chunk_id, score=score, text="passage", doc_id=doc_id, url=url)


def test_dedupe_keeps_best_chunk_per_doc_id() -> None:
    # Two chunks of one document (no URL) collapse to the first (best-ranked) one.
    chunks = [
        _chunk("a1", 0.9, "doc-1"),
        _chunk("a2", 0.8, "doc-1"),
        _chunk("b1", 0.7, "doc-2"),
    ]
    assert [c.id for c in dedupe_chunks_by_source(chunks)] == ["a1", "b1"]


def test_dedupe_collapses_same_url_across_doc_ids() -> None:
    # A page returned twice (distinct chunk/doc ids, same URL) is one source.
    chunks = [
        _chunk("a1", 0.9, "doc-1", url="https://example.org/a"),
        _chunk("a2", 0.8, "doc-2", url="https://example.org/a"),
    ]
    assert [c.id for c in dedupe_chunks_by_source(chunks)] == ["a1"]


def test_dedupe_url_canonicalization_collapses_cosmetic_variants() -> None:
    # Trailing slash, host case, and a tracking param all resolve to one source.
    chunks = [
        _chunk("a1", 0.9, "doc-1", url="https://Example.org/a/"),
        _chunk("a2", 0.8, "doc-2", url="https://example.org/a?utm_source=news"),
    ]
    assert [c.id for c in dedupe_chunks_by_source(chunks)] == ["a1"]


def test_dedupe_keeps_distinct_sources_in_order() -> None:
    chunks = [
        _chunk("a1", 0.9, "doc-1", url="https://example.org/a"),
        _chunk("b1", 0.8, "doc-2", url="https://example.org/b"),
        _chunk("c1", 0.7, "doc-3"),
    ]
    assert [c.id for c in dedupe_chunks_by_source(chunks)] == ["a1", "b1", "c1"]


def test_dedupe_does_not_merge_distinct_query_pages() -> None:
    # A meaningful (non-tracking) query param distinguishes pages: keep both.
    chunks = [
        _chunk("a1", 0.9, "doc-1", url="https://example.org/item?id=1"),
        _chunk("a2", 0.8, "doc-2", url="https://example.org/item?id=2"),
    ]
    assert [c.id for c in dedupe_chunks_by_source(chunks)] == ["a1", "a2"]


def test_dedupe_malformed_url_falls_back_to_raw() -> None:
    # A URL urlsplit cannot parse (unclosed IPv6 bracket) must not raise and abort
    # the turn: it falls back to the raw string as the key, so two identical
    # malformed URLs still collapse and a different one is kept.
    chunks = [
        _chunk("a1", 0.9, "doc-1", url="http://[::1"),
        _chunk("a2", 0.8, "doc-2", url="http://[::1"),
        _chunk("b1", 0.7, "doc-3", url="http://[::2"),
    ]
    assert [c.id for c in dedupe_chunks_by_source(chunks)] == ["a1", "b1"]


def test_dedupe_empty_input() -> None:
    assert dedupe_chunks_by_source([]) == []
