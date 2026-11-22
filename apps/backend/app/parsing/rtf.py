r"""Rich Text Format: a document that looks like text and is not.

RTF is the odd one out. Every other binary format this app reads is obviously
binary - a PDF, a zip, an image - and a mislabelled one is caught by its signature
before it can be mistaken for prose. An RTF is different: it is plain ASCII, all the
way down, so it slips through every test for "does this look like text" and arrives
at the plain-text parser looking perfectly respectable.

And the plain-text parser reads it. That is the bug this closes. Without a parser of
its own, an uploaded ``.rtf`` was ingested as its own markup - font tables, colour
tables, control words and all - and the file then looked ready, retrieved, and got
CITED, quoting ``\fonttbl{\f0\fnil\fcharset0 Calibri;}`` at a person who had asked
what their document said. A silent wrong answer, which is worse than a refusal.

Reading it properly means honouring a few things that are easy to get wrong, and
this is why the parse is a dependency rather than a hand-rolled loop:

- **Destination groups must be skipped whole.** ``\fonttbl``, ``\colortbl``,
  ``\stylesheet``, ``\info``, ``\pict`` and any ``\*\...`` group hold metadata, not
  text - and a picture's group holds its pixels as hex. Emitting those would poison
  the index with megabytes of gibberish that reads, to a retriever, like words.
- **``\uc`` says how many characters to throw away.** Each ``\uN`` unicode escape is
  followed by fallback characters for readers too old to understand it, and the file
  states how many. Keep them and every accented word in a European document arrives
  with a stray ``?`` welded to it.
- **``\'hh`` is a raw byte in the document's own code page**, which the document
  declares and which is not always the one that would be guessed.

The reader runs in a killable child process (see :mod:`app.parsing.isolation`).
There is no native code here at all - and it makes no difference: a pure-Python loop
over hostile input cannot be interrupted in a thread any more than a C one can, and
the point of the child is that it can simply be killed.
"""

from __future__ import annotations

from typing import ClassVar

from app.parsing.base import DocumentParser, ParsedDocument
from app.parsing.isolation import run_isolated

# Generous for a large document, bounded so a pathological one cannot run forever.
# The same value the other document formats use.
_DEFAULT_PARSE_TIMEOUT_SECONDS = 60.0

# What this format is called in a message the person who uploaded it reads.
_WHAT = "RTF document"

# Every RTF begins with this. The version digit varies, so the marker stops short of
# it. Anchored at the very start: a note that happens to MENTION "{\rtf" in its text
# is a text file, and claiming it here would be exactly the mistake this parser
# exists to prevent, in reverse.
_SIGNATURE = b"{\\rt"

# The most text one document may yield. RTF does not amplify - an escape is several
# bytes and becomes one character - so this cannot be reached by a file within the
# upload cap. It is a backstop, not a bound anyone should hit.
_MAX_TEXT_CHARS = 32 * 1024 * 1024


def read_rtf(data: bytes) -> str:
    r"""Extract an RTF document's text; runs in the child process.

    Top-level so the spawn bootstrap can import it, with the library imported here
    rather than at module scope because the child re-imports this module.

    The bytes are decoded as latin-1, which cannot fail: RTF's own syntax is ASCII,
    and everything outside it is escaped (``\'hh``, ``\uN``) precisely so the file
    survives a byte-oriented reader. Decoding permissively hands the escapes to the
    parser intact, which is the only reader that knows what code page they are in.
    """
    from striprtf.striprtf import rtf_to_text

    # errors="ignore" on a decode that cannot fail is belt and braces: a truncated
    # multi-byte tail must not take the whole document down.
    text = rtf_to_text(data.decode("latin-1", errors="ignore"), errors="ignore")
    return text[:_MAX_TEXT_CHARS]


class RtfParser(DocumentParser):
    """Reads a Rich Text Format document as one unit of text."""

    media_types: ClassVar[frozenset[str]] = frozenset(
        {
            "application/rtf",
            "text/rtf",
            "application/x-rtf",
            "text/richtext",
        }
    )
    extensions: ClassVar[frozenset[str]] = frozenset({".rtf"})

    def __init__(self, *, timeout_seconds: float = _DEFAULT_PARSE_TIMEOUT_SECONDS) -> None:
        """Create a parser that gives one document ``timeout_seconds`` to be read."""
        self._timeout = timeout_seconds

    def sniff(self, data: bytes) -> bool:
        """Claim a file whose bytes say RTF, whatever it was called.

        This matters more here than for any other format. An RTF is ASCII, so a
        mislabelled one (sent as ``text/plain``, or with no extension at all) passes
        the "looks like text" test and would otherwise fall through to the plain-text
        parser - which reads it, successfully, as its own markup. The signature is
        the only thing standing between that file and a citation full of control
        words.
        """
        return data.startswith(_SIGNATURE)

    async def parse(
        self, data: bytes, *, filename: str, media_type: str | None = None
    ) -> list[ParsedDocument]:
        """Extract the document's text; an empty document yields nothing.

        One unit, like a Word document: RTF has page breaks, but they are a
        formatting instruction rather than a structure, so there is no natural seam
        to cite. Length-splitting downstream does the rest.

        A truncated document is not an error. RTF is a stream of groups, and a file
        cut short simply stops - whatever was read before the cut is real text, and
        returning it beats refusing the file for the sake of a brace.

        Raises:
            ParseFailedError: The document could not be read, or took too long.
        """
        if not data:
            return []
        text = await run_isolated(read_rtf, data, timeout_seconds=self._timeout, what=_WHAT)
        if not text.strip():
            return []
        return [ParsedDocument(ordinal=0, text=text)]
