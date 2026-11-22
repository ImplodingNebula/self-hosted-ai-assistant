"""Deterministic token-aware text chunking.

Splitting is structural - paragraphs first, sentences inside oversized
paragraphs, hard character cuts only for pathological single "words" - and
budgets use the same cheap chars-per-token approximation as the rest of the
platform (see app.services.context_budget). No tokenizer dependency by design:
exact token counts vary per model anyway, and truncation-grade accuracy is all
chunking needs.
"""

from __future__ import annotations

import re

from app.providers.rag.base import Chunk, Chunker
from app.services.context_budget import APPROX_CHARS_PER_TOKEN

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
# Sentence-ish boundaries: terminal punctuation followed by whitespace. Good
# enough for packing; mid-abbreviation splits merely move a boundary.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class TextChunker(Chunker):
    """Packs paragraphs/sentences into chunks of ~``chunk_tokens`` tokens.

    Consecutive chunks share a small tail overlap so a fact straddling a
    boundary is still retrievable from at least one chunk. Overlap is clamped
    below half the chunk budget - a larger overlap could re-emit more than it
    advances and stall progress.
    """

    def __init__(self, chunk_tokens: int = 350, overlap_tokens: int = 50) -> None:
        self._chunk_chars = max(1, chunk_tokens) * APPROX_CHARS_PER_TOKEN
        overlap_chars = max(0, overlap_tokens) * APPROX_CHARS_PER_TOKEN
        self._overlap_chars = min(overlap_chars, self._chunk_chars // 2)

    def chunk(self, text: str, *, doc_id: str, title: str | None = None) -> list[Chunk]:
        """Split ``text`` into ordered, overlapping chunks.

        Args:
            text: The document text; whitespace-only input yields no chunks.
            doc_id: Caller-scoped document id stamped on every chunk.
            title: Optional human-readable title stamped on every chunk.

        Returns:
            Chunks in document order with sequential ``index`` values.
        """
        pieces = self._split(text)
        if not pieces:
            return []
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for piece in pieces:
            # +1 for the joining space; harmless slack on the first piece.
            if current and current_len + len(piece) + 1 > self._chunk_chars:
                chunks.append(" ".join(current))
                current, current_len = self._overlap_seed(current)
            current.append(piece)
            current_len += len(piece) + 1
        if current:
            chunks.append(" ".join(current))
        return [
            Chunk(text=chunk_text, index=i, doc_id=doc_id, title=title)
            for i, chunk_text in enumerate(chunks)
        ]

    def _split(self, text: str) -> list[str]:
        """Break text into packable pieces, each within the chunk budget."""
        pieces: list[str] = []
        for paragraph in _PARAGRAPH_SPLIT.split(text):
            paragraph = " ".join(paragraph.split())
            if not paragraph:
                continue
            if len(paragraph) <= self._chunk_chars:
                pieces.append(paragraph)
                continue
            for sentence in _SENTENCE_SPLIT.split(paragraph):
                if not sentence:
                    continue
                if len(sentence) <= self._chunk_chars:
                    pieces.append(sentence)
                else:
                    # A single over-budget "sentence" (no punctuation, e.g. a
                    # log dump): hard-cut so every piece is packable.
                    pieces.extend(
                        sentence[i : i + self._chunk_chars]
                        for i in range(0, len(sentence), self._chunk_chars)
                    )
        return pieces

    def _overlap_seed(self, finished: list[str]) -> tuple[list[str], int]:
        """Return the tail pieces of a finished chunk that seed the next one."""
        if self._overlap_chars <= 0:
            return [], 0
        seed: list[str] = []
        seed_len = 0
        for piece in reversed(finished):
            if seed_len + len(piece) + 1 > self._overlap_chars:
                break
            seed.insert(0, piece)
            seed_len += len(piece) + 1
        return seed, seed_len
