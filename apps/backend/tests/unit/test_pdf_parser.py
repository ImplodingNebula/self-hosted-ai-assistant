"""PDF parsing: per-page text, blank-page handling, timeout, and dispatch."""

from __future__ import annotations

import pytest

from app.parsing.base import ParseFailedError
from app.parsing.pdf import PdfParser
from app.parsing.registry import default_registry
from app.parsing.text import PlainTextParser


def _make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF, one page per entry (empty string = blank page)."""
    n = len(pages)
    font = 3
    page_nums = [4 + 2 * i for i in range(n)]
    content_nums = [5 + 2 * i for i in range(n)]
    kids = " ".join(f"{p} 0 R" for p in page_nums)
    objs: list[tuple[int, bytes]] = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    for i, text in enumerate(pages):
        objs.append(
            (
                page_nums[i],
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content_nums[i]} 0 R >>"
                ).encode(),
            )
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objs.append(
            (
                content_nums[i],
                b"<< /Length "
                + str(len(stream)).encode()
                + b" >>\nstream\n"
                + stream
                + b"\nendstream",
            )
        )
    objs.sort()
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num, body in objs:
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    max_num = max(offsets)
    out += f"xref\n0 {max_num + 1}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, max_num + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {max_num + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    return bytes(out)


async def test_each_page_becomes_one_ordered_document() -> None:
    data = _make_pdf(["Alpha section.", "Beta section.", "Gamma section."])

    docs = await PdfParser().parse(data, filename="report.pdf")

    assert [d.ordinal for d in docs] == [0, 1, 2]
    assert [d.title for d in docs] == ["page 1", "page 2", "page 3"]
    assert [d.text.strip() for d in docs] == ["Alpha section.", "Beta section.", "Gamma section."]


async def test_a_blank_page_is_skipped_but_page_numbers_are_kept() -> None:
    # A page with no text layer (blank, or image-only) is left out rather than
    # embedded as whitespace, but the pages that remain keep their real numbers.
    data = _make_pdf(["First page text.", "", "Third page text."])

    docs = await PdfParser().parse(data, filename="report.pdf")

    assert [d.ordinal for d in docs] == [0, 2]
    assert [d.title for d in docs] == ["page 1", "page 3"]


async def test_a_pdf_with_no_text_returns_no_documents() -> None:
    docs = await PdfParser().parse(_make_pdf(["", ""]), filename="scan.pdf")
    assert docs == []


async def test_empty_bytes_return_no_documents() -> None:
    assert await PdfParser().parse(b"", filename="empty.pdf") == []


async def test_a_corrupt_pdf_fails_rather_than_crashing() -> None:
    with pytest.raises(ParseFailedError):
        await PdfParser().parse(b"%PDF-1.4\nthis is not a real pdf\n%%EOF", filename="broken.pdf")


async def test_a_parse_that_runs_past_the_timeout_is_killed() -> None:
    # A timeout smaller than the child process can even start guarantees the
    # timeout branch, proving a hung parse is terminated and reported, not hung.
    data = _make_pdf(["Some content."])
    with pytest.raises(ParseFailedError):
        await PdfParser(timeout_seconds=0.001).parse(data, filename="slow.pdf")


async def test_a_pdf_with_too_many_pages_is_refused() -> None:
    # A page-count bomb is refused rather than walked page by page.
    data = _make_pdf(["one", "two", "three"])
    with pytest.raises(ParseFailedError):
        await PdfParser(max_pages=2).parse(data, filename="bomb.pdf")


def test_sniff_recognises_the_pdf_signature_only_at_the_start() -> None:
    parser = PdfParser()
    assert parser.sniff(b"%PDF-1.7\n...") is True
    assert parser.sniff(b"\xef\xbb\xbf%PDF-1.4 after a BOM") is True  # short prefix allowed
    assert parser.sniff(b"just some text") is False
    assert parser.sniff(b"") is False
    # A text file that merely mentions the signature further in is NOT a PDF.
    assert parser.sniff(b"This note explains the %PDF- magic bytes in detail.") is False


def test_registry_dispatches_pdfs_to_the_pdf_parser() -> None:
    registry = default_registry()
    by_type = registry.resolve(filename="x", content_type="application/pdf")
    by_ext = registry.resolve(filename="report.PDF", content_type=None)
    assert isinstance(by_type, PdfParser)
    assert isinstance(by_ext, PdfParser)


def test_a_mislabelled_pdf_is_claimed_by_its_signature_not_the_text_parser() -> None:
    # No .pdf extension, wrong content type: only the magic-byte sniff saves it
    # from decoding to garbage in the text parser.
    data = _make_pdf(["Real PDF content."])
    parser = default_registry().resolve(
        filename="mystery.bin", content_type="application/octet-stream", data=data
    )
    assert isinstance(parser, PdfParser)
    assert not isinstance(parser, PlainTextParser)
