"""Tests for the plain-text parser: decoding, sniffing, and what counts as text."""

from __future__ import annotations

import codecs

import pytest

from app.parsing.text import PlainTextParser, decode_text, looks_like_text

PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


async def test_parses_source_code_as_one_document() -> None:
    parser = PlainTextParser()

    docs = await parser.parse(b"def main():\n    pass\n", filename="main.py")

    assert len(docs) == 1
    assert docs[0].ordinal == 0
    assert docs[0].title is None
    assert "def main()" in docs[0].text


async def test_a_file_with_no_text_yields_no_documents() -> None:
    parser = PlainTextParser()

    # Whitespace only: embedding it would add a chunk that grounds nothing.
    assert await parser.parse(b"   \n\t\n", filename="empty.txt") == []
    assert await parser.parse(b"", filename="empty.txt") == []


def test_plain_utf8_round_trips() -> None:
    assert decode_text("hello wörld".encode()) == "hello wörld"


def test_empty_bytes_decode_to_empty_string() -> None:
    assert decode_text(b"") == ""


# A wrong decode is worse than a failure: mojibake gets embedded and cited as if
# it were real content. Each of these must come back exactly.
@pytest.mark.parametrize(
    ("encoding", "bom"),
    [
        ("utf-8-sig", codecs.BOM_UTF8),
        ("utf-16-le", codecs.BOM_UTF16_LE),
        ("utf-16-be", codecs.BOM_UTF16_BE),
        ("utf-32-le", codecs.BOM_UTF32_LE),
        ("utf-32-be", codecs.BOM_UTF32_BE),
    ],
)
def test_byte_order_marks_are_honored(encoding: str, bom: bytes) -> None:
    text = "héllo wörld"
    raw = text.encode(encoding) if encoding == "utf-8-sig" else bom + text.encode(encoding)

    assert decode_text(raw) == text


def test_utf32_le_is_not_mistaken_for_utf16_le() -> None:
    """UTF-32-LE begins with the UTF-16-LE mark, so mark order is load-bearing."""
    raw = codecs.BOM_UTF32_LE + "abc".encode("utf-32-le")

    assert decode_text(raw) == "abc"


def test_legacy_single_byte_encoding_is_identified() -> None:
    # cp1252, no mark, invalid as UTF-8: the detector must recognise it rather
    # than the lossy fallback turning the curly quotes into replacement chars.
    raw = "the “quoted” café".encode("cp1252")

    decoded = decode_text(raw)

    assert "caf" in decoded
    assert "�" not in decoded


def test_undecodable_bytes_degrade_rather_than_raise() -> None:
    # An invalid UTF-8 sequence with no byte-order mark. A file the app cannot
    # identify must still yield readable content: an error the user cannot act
    # on is worse than a few replacement characters.
    decoded = decode_text(b"\xc3\x28 valid ascii tail")

    assert isinstance(decoded, str)
    assert "valid ascii tail" in decoded


def test_a_leading_utf16_mark_wins_over_the_ascii_that_follows() -> None:
    """Bytes opening with a UTF-16-LE mark ARE UTF-16, however text-like the tail looks.

    Guards the mark check against being 'fixed' into a heuristic that prefers
    whichever decode yields more ASCII.
    """
    raw = codecs.BOM_UTF16_LE + "hi".encode("utf-16-le")

    assert decode_text(raw) == "hi"


def test_looks_like_text_accepts_text_and_empty() -> None:
    assert looks_like_text(b"") is True
    assert looks_like_text(b"#!/usr/bin/env python\n") is True


def test_looks_like_text_rejects_a_binary_header() -> None:
    assert looks_like_text(PNG_HEADER) is False


def test_looks_like_text_accepts_utf16_despite_its_nul_bytes() -> None:
    raw = codecs.BOM_UTF16_LE + "hello".encode("utf-16-le")

    # UTF-16 pads ASCII with NULs, so the mark must be checked before the NUL
    # scan or every UTF-16 file would be refused as binary.
    assert b"\x00" in raw
    assert looks_like_text(raw) is True


def test_a_nul_beyond_the_sniff_window_is_not_inspected() -> None:
    # Bounded detection cost: only the prefix decides.
    raw = b"a" * 70000 + b"\x00"

    assert looks_like_text(raw) is True


def test_handles_the_whole_text_family_and_declared_types() -> None:
    parser = PlainTextParser()

    assert parser.handles("text/x-rust", "") is True
    assert parser.handles("application/json", "") is True
    assert parser.handles(None, ".py") is True
    assert parser.handles("application/pdf", ".pdf") is False
    assert parser.handles(None, "") is False
