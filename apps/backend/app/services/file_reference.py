"""Work out which of a conversation's files a message is actually asking about.

``attachment_ids`` says which files a turn CARRIES, and the composer only ever
sends what is sitting in its tray. The tray empties the moment a message goes,
so every turn afterwards carries nothing - and a perfectly ordinary follow-up
("what does the screenshot say?") reaches the backend with no file on it at all.

Ordinary retrieval was supposed to cover exactly that: the file's text is still
in the conversation's session scope, so a question about it should simply find
it. It does not, and the reason is worth stating plainly, because it is not
something a better floor value would fix. A question ABOUT a file shares almost
no words with what is INSIDE it - "what does the screenshot say" against a page
of terminal commands - so the passage scores below the relevance floor and is
dropped. Lowering the floor until it survives would let every weak match through
on every other turn.

So the reference is recognised in the words instead. A message that names a file,
or points at one unmistakably, re-attaches it to the turn, and it is then read
exactly as though it had been in the tray: retrieval by record id, which does not
apply the floor at all.

Deterministic, and biased toward matching - but only because the cost of matching
wrongly is kept low on purpose. A pointer is a READING of the words, and readings
are sometimes wrong: "the file system in linux" and "the image tag in html" both
look like someone pointing at an upload. So an inferred file is offered to the
answer and nothing more. It is never announced, and if it turns out to hold
nothing the turn says nothing about it, because a guess that misses must not
accuse ("this answer does not use screenshot.png" on a reply about linux). Only a
file the user genuinely named - by id, on the turn - is a promise worth admitting
to breaking. See ``chat_service`` for where that line is drawn.

What is left, then, is a bounded slice of context a model can ignore - against a
false negative, which is the original bug: a file the user is plainly asking about,
answered as though it were not there.

Same "a heuristic now, a capable model later" shape the thinking and web-search
decisions already use - Phase 5 gives a tool-calling model a real way to ask for a
file by name, and a 4B model has none.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.models import Attachment
from app.services.attachment_ingest import display_name

# How many files one message can pull back in. A bound, not a judgement: the
# retrieval budget is shared, and a conversation with thirty uploads must not be
# able to spend all of it on a single vague "the files".
MAX_REFERENCED_FILES = 3

# The shortest stem worth matching on its own. Below this a filename is initials
# or a number ("1.png", "a.txt") and would fire on almost any sentence.
_MIN_STEM_CHARS = 4

# The words people use for a file when they are not naming it. Doubles as a
# blocklist for stem matching: a file called `image.png` must not be dragged in
# by "can you generate an image", so a stem that is merely one of these generic
# nouns is never treated as a NAME. The pointer rule below still catches an
# actual reference to it.
_FILE_NOUNS = (
    "attachment",
    "attachments",
    "csv",
    # No "doc"/"docs": in a thread about code, "check the docs" means the
    # documentation far more often than it means an uploaded file, and a pointer
    # is only worth having while it is more right than wrong.
    "document",
    "documents",
    "file",
    "files",
    "image",
    "images",
    "photo",
    "photos",
    "picture",
    "pictures",
    "scan",
    "scans",
    "screenshot",
    "screenshots",
    "pdf",
    "pdfs",
    "sheet",
    "spreadsheet",
    "slides",
)

# "the file", "that screenshot", "my uploaded document", "the last two images".
# A determiner is required: it is what separates pointing at a specific file
# ("summarise the document") from talking about files in general ("how do I read
# a document in python"), which must not drag anything in.
_POINTER = re.compile(
    r"\b(?:the|that|this|these|those|my|your)\s+"
    r"(?:(?:first|second|last|previous|earlier|other|attached|uploaded|shared|sent|above)\s+){0,2}"
    r"(?:" + "|".join(_FILE_NOUNS) + r")\b",
    re.IGNORECASE,
)


def _stem(filename: str) -> str:
    """The filename without its extension, lowercased."""
    return filename.rsplit(".", 1)[0].strip().lower()


def _standalone(needle: str, haystack: str) -> bool:
    """Whether ``needle`` occurs in ``haystack`` as a whole name, not inside a longer one.

    A plain substring test is not enough in either direction: `report.pdf` would
    match inside "summarise q3-report.pdf" (a DIFFERENT file), and `a.py` would
    match inside "data.python". The fences allow the punctuation a sentence puts
    around a filename - a trailing full stop, a comma, quotes - while refusing the
    characters a longer filename would be built from.
    """
    return (
        re.search(rf"(?<![\w.\-]){re.escape(needle)}(?![\w\-])", haystack, re.IGNORECASE)
        is not None
    )


def _names_the_file(message: str, filename: str) -> bool:
    """Whether ``message`` refers to ``filename`` by name.

    The full name (with extension) always counts. The bare stem counts too - a
    person writes "what is in test ss" as often as "test ss.png" - but only when
    it is long enough to be distinctive, and is not simply a generic word for a
    file. Neither counts when it merely sits inside a longer name.
    """
    if _standalone(filename, message):
        return True
    stem = _stem(filename)
    if len(stem) < _MIN_STEM_CHARS or stem in _FILE_NOUNS:
        return False
    return _standalone(stem, message)


def resolve_referenced_files(
    message: str,
    files: Sequence[Attachment],
    *,
    limit: int = MAX_REFERENCED_FILES,
) -> list[Attachment]:
    """Return the files in ``files`` that ``message`` is asking about.

    Naming a file wins outright: if the message says "test ss.png", that is the
    file it means, and no amount of vague pointing elsewhere in the sentence
    changes it. Only when nothing is named does an unmistakable pointer ("the
    screenshot", "that pdf") fall back to the most recent uploads, because "the
    file" nearly always means the last one.

    Args:
        message: The user's message, as typed.
        files: The conversation's files, oldest first, already filtered to the
            ones that could actually be read (ready, and not already on the turn).
        limit: The most files one message may pull back in.

    Returns:
        The referenced files, newest first for a pointer and in the order they
        were uploaded for an explicit name. Empty when the message points at
        nothing - the overwhelmingly common case.
    """
    if not files:
        return []
    named = [file for file in files if _names_the_file(message, file.filename)]
    if named:
        return named[:limit]
    if _POINTER.search(message):
        return list(reversed(files))[:limit]
    return []


def implicit_file_question(files: Sequence[Attachment]) -> str:
    """The question a turn is asking when it carries files and no words at all.

    Attaching a file and pressing send, with nothing typed, is a real request -
    it is how every mainstream assistant is used, and it is the natural reply to
    "attach the logs and I will take a look". The model still has to be asked
    something, so the files become the ask. This never reaches the transcript:
    the stored message stays empty, because putting words in someone's mouth is
    worse than an empty bubble.
    """
    names = ", ".join(display_name(file) for file in files)
    if not names:
        return ""
    if len(files) == 1:
        return f"I have attached {names}. Read it and tell me what it contains."
    return f"I have attached these files: {names}. Read them and tell me what they contain."
