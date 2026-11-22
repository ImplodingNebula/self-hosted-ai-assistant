"""Turn HTML into the text a person would actually read.

Two very different callers need the same thing. The web fetcher pulls a page off
the internet to ground an answer, and an EPUB chapter is XHTML in a zip - both
arrive as markup and have to become readable prose before anything can chunk,
embed or cite them. One extractor serves both, because a second copy of this logic
would immediately drift from it, and the subtleties here were each paid for once
already: a stray close tag that underflows the skip depth, a page that omits its
optional ``</head>``, a ``<main>`` region that turns out to be empty.

What it does is drop the parts of a document nobody reads - scripts, styles, page
chrome - and, when a page marks a semantic ``<main>`` region, keep only that. It
is a lightweight readability pass on the standard library's own parser, with no
third-party dependency and no browser.

**The skip set is a parameter, because the same tag means different things in the
two worlds.** On a web page, ``<header>`` is a site banner and ``<aside>`` is a
promo rail; in a book, ``<header>`` is the chapter's own title block and
``<aside epub:type="footnote">`` is the author's footnote. Dropping those on the
web keeps menus out of an answer; dropping them in a book throws away the text.

Parsing is deliberately LENIENT. Both sources are untrusted and real-world markup
is broken constantly, so a parse error salvages whatever was collected rather than
failing the document. Note the parser resolves character references but NOT entity
declarations, which is why an XHTML chapter can be read here safely while the same
bytes would need a hardened parser to go through an XML reader.
"""

from __future__ import annotations

import contextlib
from html.parser import HTMLParser

# Tags whose text is code/markup noise or page chrome, never page content.
# "title" is skipped by NAME (not via a <head> region) because html.parser does
# no implied-end-tag construction: a page that omits the optional </head> would
# otherwise leave a <head> skip region open forever and drop the entire body.
# nav/header/footer/aside/form are skipped as chrome: on real content pages they
# are navigation menus, section taxonomies, cookie/subscription banners and the
# like, which otherwise dominate the extract and bury (or, under the char cap,
# truncate away) the actual content. What content pages lose this way is what a
# reader ignores; what SPA landing pages "lose" was never real text anyway.
WEB_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "title",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
    }
)

# A book is not a web page, and two of those tags mean the opposite thing in one:
# <header> is usually the chapter's own title block ("Chapter Three / The Return"),
# and <aside epub:type="footnote"> is where EPUB 3 puts footnotes - which is often
# where the argument actually is. Skipping them would quietly delete the heading and
# the notes from every chapter. <nav> stays skipped, and that is a feature: it means
# a table-of-contents document sitting in the reading order extracts to nothing and
# is dropped, instead of being indexed as prose and cited as if it said something.
EPUB_SKIP_TAGS = WEB_SKIP_TAGS - {"header", "aside"}

# The semantic main-content region. When a page marks one, extraction keeps ONLY
# its text (the readability heuristic): the primary content, not the surrounding
# page. Only <main> qualifies - NOT <article>, which HTML5 also sanctions for
# teaser/related/comment cards, so preferring an <article> region would let a
# promo block eclipse a primary body that sits in plain <div>s. <main> is unique
# per page and unambiguously the principal content.
_MAIN_TAGS = frozenset({"main"})

# The headings a chapter's title is taken from, in the order a document uses them.
_HEADING_TAGS = ("h1", "h2", "h3")

# Tags after which a line break preserves readability (keeps items apart).
_BLOCK_TAGS = frozenset(
    {
        "p",
        "br",
        "div",
        "li",
        "ul",
        "ol",
        "section",
        "article",
        "main",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
    }
)


class _TextExtractor(HTMLParser):
    """Collect visible text: drop chrome/code, prefer the semantic main region.

    Text is accumulated twice - the whole (de-chromed) body, and separately the
    text inside any <main> region. :meth:`get_text` returns the main-only text
    when a region actually held content, else the full body, so a page with an
    empty or absent <main> degrades to the whole document.

    The document's ``<title>`` and its first heading are captured as they go by,
    so a caller that wants to LABEL the document (an EPUB chapter needs a title)
    gets one from the same single pass. The title is captured even though it is
    skipped from the text stream, which is the point: it names the chapter without
    being repeated inside it.
    """

    def __init__(self, *, skip_tags: frozenset[str] = WEB_SKIP_TAGS) -> None:
        """Create an extractor that drops ``skip_tags`` and their contents."""
        super().__init__(convert_charrefs=True)
        self._skip_tags = skip_tags
        self._all: list[str] = []
        self._main: list[str] = []
        self._skip_depth = 0
        self._main_depth = 0
        self._main_has_text = False
        self._title: list[str] = []
        self._heading: list[str] = []
        self._in_title = False
        self._in_heading = False
        self._heading_done = False

    def _emit(self, text: str) -> None:
        """Append text to the body stream, and to the main stream when inside one."""
        self._all.append(text)
        if self._main_depth > 0:
            self._main.append(text)

    def handle_starttag(self, tag: str, attrs: object) -> None:
        """Enter a skip region for chrome tags; open main regions; break blocks."""
        if tag == "title":
            # Captured whether or not it is skipped from the body, so the label
            # survives even though the words do not repeat in the text.
            self._in_title = True
        if tag in _HEADING_TAGS and not self._heading_done and self._skip_depth == 0:
            self._in_heading = True
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if tag in _MAIN_TAGS:
            self._main_depth += 1
        if tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        """Leave a skip/main region; break lines on closing block tags."""
        if tag == "title":
            self._in_title = False
        if tag in _HEADING_TAGS and self._in_heading:
            self._in_heading = False
            self._heading_done = True
        if tag in self._skip_tags:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if tag in _MAIN_TAGS and self._main_depth:
            self._main_depth -= 1
        if tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        """Keep text unless inside a skipped (chrome) region."""
        if self._in_title:
            # Before the skip check, deliberately: "title" is normally IN the skip
            # set, so a title's text never reaches the line below.
            self._title.append(data)
        if self._skip_depth:
            return
        if self._in_heading:
            self._heading.append(data)
        self._emit(data)
        if self._main_depth > 0 and data.strip():
            self._main_has_text = True

    def get_text(self) -> str:
        """Return collapsed text: the main region if it had content, else the body."""
        parts = self._main if self._main_has_text else self._all
        lines = ("".join(parts)).splitlines()
        cleaned = [" ".join(line.split()) for line in lines]
        return "\n".join(line for line in cleaned if line)

    def get_title(self) -> str | None:
        """Return the document's ``<title>``, collapsed, or None when it has none."""
        title = " ".join("".join(self._title).split())
        return title or None

    def get_heading(self) -> str | None:
        """Return the text of the document's FIRST heading, or None when it has none."""
        heading = " ".join("".join(self._heading).split())
        return heading or None


def _parse(html: str, skip_tags: frozenset[str]) -> _TextExtractor:
    """Feed ``html`` to an extractor, salvaging whatever a broken document yields."""
    parser = _TextExtractor(skip_tags=skip_tags)
    # Lenient parse: any malformed-markup error salvages the text collected so far.
    with contextlib.suppress(Exception):
        parser.feed(html)
    return parser


def extract_text(html: str, *, max_chars: int, skip_tags: frozenset[str] = WEB_SKIP_TAGS) -> str:
    """Extract readable text from an HTML document, capped at ``max_chars``.

    Chrome (nav/header/footer/aside/form) is dropped and, when the page marks a
    <main> region, only that region's text is kept - so grounding sees the
    primary content, not the surrounding menus. Malformed markup never raises -
    the stdlib parser is lenient and any parse error yields whatever was collected.
    """
    return _parse(html, skip_tags).get_text()[:max_chars]


def extract_document(
    html: str, *, max_chars: int, skip_tags: frozenset[str] = WEB_SKIP_TAGS
) -> tuple[str, str | None]:
    """Extract ``(text, label)`` from an HTML document in ONE pass.

    The label is what a citation should say about this document: its first heading
    if it has one, else its ``<title>``. A chapter's own heading beats its title
    tag, because a book's every chapter often carries the same title tag (the
    book's) while the heading is the thing that differs - and it is the difference
    that makes a citation useful.

    Returns:
        The readable text (capped), and the label, which is None when the document
        offers neither a heading nor a title.
    """
    parser = _parse(html, skip_tags)
    return parser.get_text()[:max_chars], parser.get_heading() or parser.get_title()
