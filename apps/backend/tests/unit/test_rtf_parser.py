"""RTF: the format that looks like text, and what goes wrong when it is treated as text."""

import pytest

from app.parsing.registry import default_registry
from app.parsing.rtf import RtfParser
from app.parsing.text import PlainTextParser

# Fixtures are BUILT, never committed - and here they are just strings, because an RTF
# is ASCII from end to end. That is the whole problem the parser exists to solve.

HEADER = r"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0\fnil\fcharset0 Calibri;}}"


def make_rtf(body: str) -> bytes:
    """Wrap ``body`` in a minimal but realistic RTF document."""
    return (HEADER + body + "}").encode("latin-1")


async def test_an_rtf_document_reads_as_its_prose() -> None:
    data = make_rtf(r"\pard\f0\fs22 The rollout begins on Tuesday.\par Rollback takes ten minutes.")

    documents = await RtfParser().parse(data, filename="plan.rtf")

    assert len(documents) == 1
    assert "The rollout begins on Tuesday." in documents[0].text
    assert "Rollback takes ten minutes." in documents[0].text
    # One unit, so the filename already names it - a title would just repeat.
    assert documents[0].title is None


async def test_the_markup_is_not_read_back_as_prose() -> None:
    """The bug this parser closes, stated as a test.

    An RTF is plain ASCII, so it passes every test for "this looks like text" and, with
    no parser of its own, reaches the plain-text fallback - which reads it, successfully,
    as its own markup. The file then looks ready, retrieves, and gets CITED, quoting a
    font table at somebody who asked what their document said. A silent wrong answer,
    which is worse than a refusal.
    """
    data = make_rtf(r"\pard\f0 The rollout begins on Tuesday.\par")

    text = (await RtfParser().parse(data, filename="plan.rtf"))[0].text

    assert "The rollout begins on Tuesday." in text
    for control_word in ("rtf1", "fonttbl", "Calibri", "ansicpg1252", "pard"):
        assert control_word not in text

    # And for contrast: what the fallback would have produced from the same bytes.
    fallback = (await PlainTextParser().parse(data, filename="plan.rtf"))[0].text
    assert "fonttbl" in fallback


async def test_a_pictures_hex_payload_never_reaches_the_index() -> None:
    # An embedded image is megabytes of hex digits inside a \pict group. To a retriever
    # that is not gibberish, it is WORDS - a wall of high-scoring nonsense that crowds
    # the real passages out of every result. Destination groups are skipped whole.
    picture = r"{\pict\pngblip\picw100\pich100 " + "89504e470d0a1a0a" * 500 + "}"
    data = make_rtf(r"\pard Chart follows." + picture + r"\par The trend is up.")

    text = (await RtfParser().parse(data, filename="report.rtf"))[0].text

    assert "Chart follows." in text
    assert "The trend is up." in text
    assert "89504e47" not in text
    assert "pngblip" not in text


async def test_accented_text_survives_with_its_fallback_character_discarded() -> None:
    r"""The detail a hand-rolled reader reliably gets wrong.

    Each \uN escape is followed by fallback characters for readers too old to understand
    it, and \ucN says how many to throw away. Keep them, and every accented word in a
    European document arrives with a stray "?" welded to it - which is not a cosmetic
    complaint: it is a different token, so the word no longer matches the query that
    should have found it. \'hh, meanwhile, is a raw byte in the document's own code page.
    """
    data = make_rtf(r"\uc1\pard Caf\'e9 in Z\u252?rich costs \'80 5.")

    text = (await RtfParser().parse(data, filename="menu.rtf"))[0].text

    # \'e9 and \'80 are cp1252 bytes; \u252 is a unicode escape whose "?" fallback the
    # \uc1 count says to drop.
    assert "Café" in text
    assert "Zürich" in text
    assert "€" in text
    assert "?" not in text


async def test_deeply_nested_groups_do_not_take_down_the_reader() -> None:
    # A brace is one byte, so a small file can nest a hundred thousand groups deep. A
    # recursive-descent reader meets its recursion limit and dies; this one does not.
    data = make_rtf(r"\pard " + "{" * 100_000 + "buried" + "}" * 100_000 + r"\par Visible.")

    documents = await RtfParser().parse(data, filename="nested.rtf")

    assert "Visible." in documents[0].text


async def test_a_truncated_document_yields_what_was_read() -> None:
    # RTF is a stream of groups, so a file cut short simply stops. Whatever was read
    # before the cut is real text, and returning it beats refusing the file for a brace.
    data = (HEADER + r"\pard The first half is intact.\par The second half is cu").encode()

    documents = await RtfParser().parse(data, filename="cut.rtf")

    assert "The first half is intact." in documents[0].text


async def test_the_registry_claims_an_rtf_by_extension_and_by_its_signature() -> None:
    """The registry ORDER is what makes this parser reachable at all.

    Every other binary format is claimed by a signature that no text file has. An RTF's
    signature is text. So the fallback would happily take it, and the only thing standing
    between an uploaded document and a citation full of control words is that RtfParser
    is registered ahead of PlainTextParser and sniffs the file first.
    """
    registry = default_registry()
    data = make_rtf(r"\pard Prose.")

    assert isinstance(registry.resolve(filename="a.rtf"), RtfParser)
    assert isinstance(registry.resolve(filename="a.rtf", content_type="text/rtf"), RtfParser)
    # No extension, and declared as plain text - it still must not reach the fallback.
    assert isinstance(
        registry.resolve(filename="attachment", content_type="text/plain", data=data), RtfParser
    )


async def test_a_note_that_merely_mentions_rtf_is_still_a_text_file() -> None:
    # The signature is anchored at the very start, so a text file whose FIRST LINE talks
    # about "{\rtf" is not claimed. Claiming it would be this parser's own bug, in
    # reverse: a readable note fed to a reader that would strip it to nothing.
    note = b"Notes on the format: an RTF begins with {\\rtf1 and ends with a brace."

    parser = default_registry().resolve(filename="notes.txt", data=note)

    assert isinstance(parser, PlainTextParser)


async def test_an_empty_file_yields_nothing_rather_than_failing() -> None:
    assert await RtfParser().parse(b"", filename="empty.rtf") == []


async def test_a_document_with_no_prose_yields_nothing() -> None:
    # A header, a font table, and not one word. There is nothing to ground an answer on,
    # and an empty document is more honest than a document of whitespace.
    assert await RtfParser().parse(make_rtf(r"\pard"), filename="blank.rtf") == []


@pytest.mark.parametrize("extension", [".odt", ".ods", ".odp", ".epub", ".rtf"])
async def test_every_new_format_is_reachable_from_the_registry(extension: str) -> None:
    # A parser nobody can route to is a parser that does not exist. This is the one test
    # that fails if a future edit drops a registration.
    assert not isinstance(default_registry().resolve(filename=f"a{extension}"), PlainTextParser)
