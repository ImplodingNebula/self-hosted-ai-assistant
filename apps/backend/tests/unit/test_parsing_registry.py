"""Tests for the parser registry: dispatch, precedence, and honest refusal."""

from __future__ import annotations

from typing import ClassVar

import pytest

from app.parsing.base import DocumentParser, ParsedDocument, UnsupportedFileError
from app.parsing.registry import (
    ParserRegistry,
    default_registry,
    extension_of,
    normalize_media_type,
)
from app.parsing.text import PlainTextParser

PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


class _SvgParser(DocumentParser):
    """A stand-in for a specific format whose bytes happen to be text."""

    media_types: ClassVar[frozenset[str]] = frozenset({"image/svg+xml"})
    extensions: ClassVar[frozenset[str]] = frozenset({".svg"})

    async def parse(self, data, *, filename, media_type=None) -> list[ParsedDocument]:
        return [ParsedDocument(ordinal=0, text="svg")]


def test_normalize_media_type_strips_parameters_and_case() -> None:
    assert normalize_media_type("Text/Plain; charset=utf-8") == "text/plain"
    assert normalize_media_type("  APPLICATION/JSON  ") == "application/json"
    assert normalize_media_type("") is None
    assert normalize_media_type(None) is None


def test_extension_of_handles_dotfiles_paths_and_case() -> None:
    assert extension_of("main.PY") == ".py"
    assert extension_of(".env") == ".env"
    assert extension_of("archive.tar.gz") == ".gz"
    assert extension_of("noext") == ""
    # A filename is a label, never a path: only the final component is read.
    assert extension_of("../../etc/passwd") == ""
    assert extension_of(r"C:\evil\notes.txt") == ".txt"


def test_resolves_by_media_type() -> None:
    registry = default_registry()

    parser = registry.resolve(filename="whatever", content_type="text/x-go")

    assert isinstance(parser, PlainTextParser)


def test_resolves_by_extension_when_no_media_type_is_declared() -> None:
    registry = default_registry()

    # Browsers frequently send no Content-Type for a source file.
    assert isinstance(registry.resolve(filename="lib.rs"), PlainTextParser)


def test_an_unlisted_extension_still_parses_when_the_bytes_are_text() -> None:
    registry = default_registry()

    # ".nut" is in no list anywhere. This fallback is what makes "source code in
    # any language" true instead of a list that must be endlessly extended.
    parser = registry.resolve(filename="script.nut", data=b"function main() {}")

    assert isinstance(parser, PlainTextParser)


def test_a_declared_handler_wins_over_the_text_sniff() -> None:
    registry_with_svg = ParserRegistry([_SvgParser(), PlainTextParser()])

    # SVG is markup, so its bytes sniff as text. Because _SvgParser declares the
    # extension, dispatch reaches it before the sniff fallback ever runs.
    parser = registry_with_svg.resolve(filename="logo.svg", data=b"<svg></svg>")

    assert isinstance(parser, _SvgParser)


def test_the_sniff_fallback_runs_only_after_every_declared_handler() -> None:
    registry_with_svg = ParserRegistry([_SvgParser(), PlainTextParser()])

    # No extension and no declared type, so nothing claims it: only the sniff
    # can decide, and it routes the text bytes to the text parser.
    parser = registry_with_svg.resolve(filename="mystery", data=b"plain content")

    assert isinstance(parser, PlainTextParser)


def test_registration_order_is_precedence() -> None:
    registry = ParserRegistry([PlainTextParser()])
    registry.register(_SvgParser())

    # PlainTextParser declares image/svg+xml too, and was registered first.
    assert isinstance(registry.resolve(filename="logo.svg"), PlainTextParser)


def test_a_binary_file_is_refused_honestly() -> None:
    registry = default_registry()

    # A binary format nothing here reads: unknown type and extension, no signature
    # any parser claims, and a NUL byte so it is not mistaken for text either.
    audio = b"\x00\x01\x02\x03BINARYAUDIO\xff\x00\xfb"
    with pytest.raises(UnsupportedFileError) as exc:
        registry.resolve(filename="track.mp3", content_type="audio/mpeg", data=audio)

    assert "track.mp3" in str(exc.value)


async def test_a_lying_content_type_loses_to_the_signature() -> None:
    """What the bytes ARE beats what the upload SAYS they are.

    text/plain is the type a browser sends when it recognises nothing, so it turns up
    on genuinely binary files constantly - it is a guess, and a signature is a fact.
    Honouring the guess would send a real format to the reader that reads anything,
    badly: the bytes decode to mojibake, the row goes READY, and nonsense is embedded
    and retrieved under the user's own filename.

    So the text parser's claim is HELD while the signatures are tried, and this file is
    claimed by the one that recognises it.
    """
    from app.parsing.image import ImageParser

    registry = default_registry()

    parser = registry.resolve(filename="evil.txt", content_type="text/plain", data=PNG_HEADER)

    assert isinstance(parser, ImageParser)


async def test_an_unrecognised_mislabel_degrades_rather_than_raising() -> None:
    registry = default_registry()
    # No signature matches these bytes, so the held text claim stands. Dispatch does not
    # police that, and does not need to: the parser degrades to a lossy string instead
    # of raising, and no filename or type ever decides where bytes are written (that is
    # the storage seam, which mints its own keys).
    junk = bytes(range(256)) * 4

    parser = registry.resolve(filename="evil.txt", content_type="text/plain", data=junk)
    assert isinstance(parser, PlainTextParser)

    docs = await parser.parse(junk, filename="evil.txt", media_type="text/plain")

    assert all(isinstance(doc.text, str) for doc in docs)


def test_bytes_omitted_means_unknown_not_empty() -> None:
    registry = default_registry()

    # An empty file is text, but "no bytes supplied" is not a claim that the
    # file is empty - it disables the sniff so an unmatched file is refused
    # rather than silently handed to the text parser.
    assert isinstance(registry.resolve(filename="anything", data=b""), PlainTextParser)
    with pytest.raises(UnsupportedFileError):
        registry.resolve(filename="mystery.bin")


def test_an_empty_registry_refuses_everything() -> None:
    with pytest.raises(UnsupportedFileError):
        ParserRegistry().resolve(filename="notes.txt", data=b"plain")
