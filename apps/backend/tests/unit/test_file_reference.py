"""Which of a conversation's files a message is asking about.

The rule has to earn two things at once, and they pull against each other. It
must catch the follow-up the user actually types - "what does the screenshot
say?", minutes after uploading it, with an empty tray and no ids on the turn -
because that is the case ordinary retrieval provably cannot serve (a question
about a file shares no words with the text inside it, so the relevance floor
throws the passage away). And it must not drag a file into a turn that was never
about it, because that spends the context budget on noise.

The bias is deliberate: a false positive costs a slice of context a model can
ignore, a false negative is the bug. These tests pin where that line sits.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.models import Attachment
from app.services.file_reference import (
    MAX_REFERENCED_FILES,
    implicit_file_question,
    resolve_referenced_files,
)

TENANT = uuid4()


def _file(filename: str) -> Attachment:
    return Attachment(
        id=uuid4(),
        tenant_id=TENANT,
        user_id=TENANT,
        conversation_id=uuid4(),
        filename=filename,
        content_type="text/plain",
        size_bytes=10,
        storage_key=f"{TENANT}/{uuid4()}/{uuid4()}",
    )


def _names(files: list[Attachment]) -> list[str]:
    return [file.filename for file in files]


# ------------------------------------------------------------- naming a file


@pytest.mark.parametrize(
    "message",
    [
        "what does test ss.png say?",
        "What does TEST SS.PNG say?",  # case is not a reference
        "open test ss and read it",  # the stem alone is how people write it
    ],
)
def test_a_message_that_names_a_file_reaches_it(message: str) -> None:
    files = [_file("test ss.png"), _file("notes.md")]

    assert _names(resolve_referenced_files(message, files)) == ["test ss.png"]


def test_a_stem_inside_a_longer_word_is_not_a_reference() -> None:
    # "report" must not fire on "reportedly". The stem match is word-bounded, or
    # a file called `a.txt` would be referenced by half the English language.
    files = [_file("report.pdf")]

    assert resolve_referenced_files("reportedly the deadline moved", files) == []


def test_a_generically_named_file_is_not_dragged_in_by_the_word_itself() -> None:
    # `image.png` has the stem "image", which is also just the word for a file of
    # its kind. Matching on it would mean any sentence about images pulls this one
    # in. Only the full name counts, and the pointer rule still covers a genuine
    # reference to it.
    files = [_file("image.png")]

    assert resolve_referenced_files("can you generate an image for me", files) == []
    assert _names(resolve_referenced_files("what is in image.png", files)) == ["image.png"]


def test_a_short_stem_is_ignored() -> None:
    # "1.png" -> stem "1", which appears in almost any message carrying a number.
    files = [_file("1.png")]

    assert resolve_referenced_files("there were 1 or 2 problems", files) == []


# ----------------------------------------------------------- pointing at one


@pytest.mark.parametrize(
    "message",
    [
        "summarise the file I attached",
        "what does the screenshot say?",
        "read that pdf again",
        "go through my uploaded document",
        "what was in the last image?",
    ],
)
def test_a_message_that_points_at_a_file_reaches_the_most_recent(message: str) -> None:
    files = [_file("old.txt"), _file("recent.png")]

    # "the file" nearly always means the last one.
    assert _names(resolve_referenced_files(message, files))[0] == "recent.png"


@pytest.mark.parametrize(
    "message",
    [
        "how do I read a file in python?",  # no determiner: talking ABOUT files
        "what is a screenshot?",
        "hello",
        "write me a document template",
    ],
)
def test_talking_about_files_in_general_reaches_nothing(message: str) -> None:
    files = [_file("old.txt"), _file("recent.png")]

    assert resolve_referenced_files(message, files) == []


def test_a_named_file_wins_over_a_vague_pointer_elsewhere() -> None:
    # "the file" would have taken the newest; the message says which one it means.
    files = [_file("turbines.txt"), _file("newest.png")]

    referenced = resolve_referenced_files("compare the file to turbines.txt", files)

    assert _names(referenced) == ["turbines.txt"]


def test_a_vague_pointer_cannot_pull_in_the_whole_conversation() -> None:
    # The retrieval budget is shared. A thread with a dozen uploads must not be
    # able to spend all of it because someone wrote "the files".
    files = [_file(f"doc{n}.txt") for n in range(10)]

    referenced = resolve_referenced_files("summarise the files", files)

    assert len(referenced) == MAX_REFERENCED_FILES


def test_nothing_to_reference_is_not_an_error() -> None:
    assert resolve_referenced_files("what does the screenshot say?", []) == []


# ------------------------------------------------- a turn with no words at all


def test_files_alone_become_the_question() -> None:
    question = implicit_file_question([_file("logs.txt")])

    assert "logs.txt" in question
    # It has to READ like an instruction, or a model handed an empty turn and a
    # wall of context will simply say hello back.
    assert question.endswith(".")
    assert "Read it" in question


def test_several_files_alone_ask_about_all_of_them() -> None:
    question = implicit_file_question([_file("a.txt"), _file("b.csv")])

    assert "a.txt" in question
    assert "b.csv" in question
    assert "Read them" in question


def test_no_files_asks_nothing() -> None:
    # The schema rejects a turn with neither words nor files, so this is only ever
    # reached with something to say. It must not invent a question regardless.
    assert implicit_file_question([]) == ""


# --------------------------------------------- readings that would be wrong


@pytest.mark.parametrize(
    "message",
    [
        # A file NAME must be the whole name, not a piece of a longer one.
        "summarise q3-report.pdf",
        "how does data.python handle this",
    ],
)
def test_a_name_inside_a_longer_name_is_a_different_file(message: str) -> None:
    files = [_file("report.pdf"), _file("a.py")]

    assert resolve_referenced_files(message, files) == []


def test_the_docs_means_documentation() -> None:
    # In a thread about code, "check the docs" almost never means an upload. A
    # pointer is only worth having while it is more right than wrong.
    files = [_file("notes.txt")]

    assert resolve_referenced_files("check the docs for the asyncio api", files) == []


def test_a_file_that_is_named_with_punctuation_around_it_still_matches() -> None:
    # The fences must allow what a sentence puts around a filename, or the rule
    # would only work when the name happens to be the last word.
    files = [_file("report.pdf")]

    assert _names(resolve_referenced_files("what does report.pdf say?", files)) == ["report.pdf"]
    assert _names(resolve_referenced_files('open "report.pdf", please', files)) == ["report.pdf"]


def test_the_question_asked_on_a_users_behalf_would_match_itself() -> None:
    """Pins the trap, rather than pretending it is not there.

    A turn of files and no words is asked on the user's behalf ("I have attached
    these files: a.png, b.png. Read them..."). That sentence points at files in
    exactly the way this rule looks for - so it MATCHES, and if it were ever fed
    back in it would drag every other file in the conversation onto a turn that
    already names the ones it means.

    The rule cannot defend against that; only the caller can, by reading the words
    the user really typed. So this asserts the danger is real, and
    test_chat_attachments pins the caller getting it right. Weakening the pointer
    until this string stops matching would be fixing the wrong end.
    """
    older = [_file("unrelated.pdf")]
    self_referential = implicit_file_question([_file("a.png"), _file("b.png")])

    assert resolve_referenced_files(self_referential, older) == older
