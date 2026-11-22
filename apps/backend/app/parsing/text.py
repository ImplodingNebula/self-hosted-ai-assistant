"""Parser for plain text: prose, source code in any language, and data formats.

This is the parser that makes a general-purpose assistant feel general. An
assistant that rejects a `.py` file, a `.csv`, or a config file reads as broken,
and none of those need a real parser - they need to be decoded correctly and
handed on as text.

Decoding is the whole problem. A source file may arrive as UTF-8, as UTF-8 with
a byte-order mark, as UTF-16 from a Windows editor, or in a legacy single-byte
encoding, and decoding it wrongly turns the text into plausible-looking garbage
that then gets embedded and cited. The strategy is: honor a byte-order mark when
one is present, else try strict UTF-8, else let charset-normalizer identify the
encoding, and only then fall back to a lossy decode.
"""

from __future__ import annotations

import codecs
from typing import ClassVar

from charset_normalizer import from_bytes

from app.parsing.base import DocumentParser, ParsedDocument

# Byte-order marks, longest first: UTF-32-LE starts with the UTF-16-LE mark, so
# testing the shorter one first would misidentify every UTF-32-LE file.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)

# How much of a file is inspected to decide whether it is text at all, and to
# identify its encoding. Detection cost is bounded; the decode still covers the
# whole file with whatever encoding this prefix implies.
_SNIFF_BYTES = 65536

# Structured-data and markup types that are plain text underneath. Extracting
# their semantics (a table per sheet, a document tree) is a later parser's job;
# reading them as text already grounds an answer usefully.
_DATA_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/x-ndjson",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
        "application/javascript",
        "application/x-sh",
        "application/sql",
        "application/x-python",
        "application/x-python-code",
        "image/svg+xml",  # markup, not a raster image
    }
)

# Extensions worth claiming outright. This list is a convenience, NOT the
# mechanism: a file whose extension is missing here still reaches this parser
# when its bytes look like text (see `looks_like_text`), which is what makes
# "source code in any language" true rather than a list to keep extending.
_PROSE_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".rst", ".adoc", ".tex", ".log"})
_DATA_EXTENSIONS = frozenset(
    {
        ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml", ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".conf", ".env", ".properties", ".svg",
    }
)  # fmt: skip
_CODE_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts",
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cs", ".go", ".rs", ".rb",
        ".php", ".swift", ".m", ".mm", ".scala", ".clj", ".ex", ".exs", ".erl",
        ".hs", ".lua", ".pl", ".r", ".jl", ".dart", ".zig", ".nim", ".v",
        ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
        ".sql", ".graphql", ".proto", ".tf", ".hcl",
        ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
        ".dockerfile", ".gradle", ".cmake", ".mk",
    }
)  # fmt: skip
_EXTENSIONS = _PROSE_EXTENSIONS | _DATA_EXTENSIONS | _CODE_EXTENSIONS


def looks_like_text(data: bytes) -> bool:
    """Return whether ``data`` is plausibly text rather than a binary blob.

    A NUL byte in the inspected prefix is the tell: no text encoding this parser
    handles produces one, while almost every binary format does within its
    header. UTF-16 and UTF-32 are the exception - their ASCII characters are
    padded with NULs - so a byte-order mark is honored first.

    An empty file is text: it is a valid, if useless, upload.
    """
    if not data:
        return True
    if any(data.startswith(bom) for bom, _ in _BOMS):
        return True
    return b"\x00" not in data[:_SNIFF_BYTES]


def decode_text(data: bytes) -> str:
    """Decode ``data`` to text, identifying its encoding as reliably as possible.

    Decoding wrongly is worse than failing: mojibake embeds and cites as if it
    were real content. The order below tries the outcomes most likely to be
    exactly right before the one that merely cannot fail.

    Args:
        data: The file's raw bytes.

    Returns:
        The decoded text, with a lossy replacement as the last resort so a file
        the app cannot identify still yields readable content rather than an
        error the user cannot act on.
    """
    if not data:
        return ""
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            # utf-8-sig consumes its own mark; the UTF-16/32 codecs do not when
            # the byte order is named explicitly, so the mark is sliced off.
            body = data if encoding == "utf-8-sig" else data[len(bom) :]
            return body.decode(encoding, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    best = from_bytes(data[:_SNIFF_BYTES]).best()
    if best is not None:
        try:
            return data.decode(best.encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            pass
    return data.decode("utf-8", errors="replace")


class PlainTextParser(DocumentParser):
    """Reads any file whose bytes are text: prose, source code, data formats."""

    media_types: ClassVar[frozenset[str]] = _DATA_MEDIA_TYPES
    extensions: ClassVar[frozenset[str]] = _EXTENSIONS

    def handles(self, media_type: str | None, extension: str) -> bool:
        """Claim the whole ``text/*`` family, plus the declared types and extensions."""
        if media_type is not None and media_type.startswith("text/"):
            return True
        return super().handles(media_type, extension)

    async def parse(
        self, data: bytes, *, filename: str, media_type: str | None = None
    ) -> list[ParsedDocument]:
        """Decode the file into a single document, or none when it holds no text.

        Pure-Python decoding bounded by the upload size cap, so it does not need
        a worker thread the way the native PDF and image parsers will.
        """
        text = decode_text(data)
        if not text.strip():
            return []
        return [ParsedDocument(ordinal=0, text=text)]
