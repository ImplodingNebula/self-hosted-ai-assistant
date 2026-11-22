"""The killable parse subprocess: it must really kill, and really notice a death.

Native document code (PDFium, lxml, Tesseract) cannot be interrupted. A thread
running it cannot be unwound and an asyncio timeout will not touch it, so the
only way to bound a runaway parse is to run it in a process and terminate that.
Everything here is about the two ways that contract can quietly fail:

- the timeout fires but the child lives on, burning a core for the life of the
  worker - which is exactly the disaster the subprocess exists to prevent, and
  which a test asserting only "an error was raised" would not notice; and
- the child DIES (a segfault in the native library, or the kernel's
  out-of-memory killer) and says nothing at all, so the parent waits out the
  entire deadline and then blames the file for being slow. It was not slow. The
  reader died.

The targets are top-level because the spawn bootstrap re-imports this module in
the child.
"""

from __future__ import annotations

import os
import signal
import time

import pytest

from app.parsing.base import ParseFailedError, ParserOutOfMemoryError
from app.parsing.isolation import _await_result, run_isolated

# Long enough that a test which is supposed to finish early has room to prove it,
# and short enough that a genuine hang does not stall the suite for minutes.
_GENEROUS_TIMEOUT = 30.0


def _answer(value: str) -> str:
    return f"read: {value}"


def _explode() -> str:
    raise ValueError("this file is nonsense")


def _hang_forever(pid_path: str) -> str:
    """Record the child's pid, then never return. Stands in for a runaway parse."""
    with open(pid_path, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
        handle.flush()
        os.fsync(handle.fileno())
    while True:
        time.sleep(0.05)
    return ""  # unreachable


def _die_hard() -> str:
    """Get killed the way the kernel's out-of-memory killer kills: uncatchably."""
    os.kill(os.getpid(), signal.SIGKILL)
    return ""  # unreachable


def _crash_natively() -> str:
    """Exit without a word, the way a segfaulting native library does."""
    os._exit(3)
    return ""  # unreachable


def _answer_unpicklably() -> object:
    """Return something that cannot cross a process boundary."""
    return lambda: "a local function pickles nowhere"


def _answer_at_length() -> str:
    """Return more text than a pipe holds at once - a long document's worth."""
    return "x" * (4 * 1024 * 1024)


async def test_a_parse_that_works_returns_its_result() -> None:
    result = await run_isolated(_answer, "a page", timeout_seconds=_GENEROUS_TIMEOUT, what="PDF")

    assert result == "read: a page"


async def test_a_parse_that_raises_reports_why() -> None:
    # The traceback cannot cross a process boundary, so the child hands the reason
    # back over the queue. It has to survive the trip.
    with pytest.raises(ParseFailedError, match="nonsense"):
        await run_isolated(_explode, timeout_seconds=_GENEROUS_TIMEOUT, what="PDF")


@pytest.mark.skipif(os.name != "posix", reason="pid liveness check is posix-only")
async def test_a_runaway_parse_is_actually_killed(tmp_path) -> None:
    """The point of the whole module, and the part a weaker test would miss.

    Asserting that the timeout raises proves nothing about the child: a parent
    that gave up while the child ran on would raise exactly the same error, and
    the runaway native loop this exists to contain would keep a core pinned for
    the life of the worker. So the child's pid is checked directly - it must be
    gone.
    """
    pid_file = tmp_path / "child.pid"

    # Comfortably past the spawn cost: a fresh interpreter takes a second or so to
    # boot and re-import, and a deadline shorter than that would fire before the
    # child ever reached the code under test.
    with pytest.raises(ParseFailedError, match="took too long"):
        await run_isolated(_hang_forever, str(pid_file), timeout_seconds=6.0, what="PDF")

    pid = int(pid_file.read_text(encoding="utf-8"))
    # By the time the error reaches here the parent has terminated the child AND
    # joined it, so it is not merely dying - it is reaped and gone. Signal 0 asks
    # whether a pid exists without delivering anything, and it must not find this
    # one.
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL is posix-only")
async def test_a_reader_killed_by_the_kernel_is_noticed_at_once() -> None:
    """A dead child must not be mistaken for a slow one.

    Waiting for the answer alone cannot tell them apart: a child killed by the
    out-of-memory killer sends NOTHING, so the parent would block for the whole
    deadline - up to two minutes at the OCR timeout, three times over - holding an
    ingest slot doing nothing, and then report that the file "took too long to
    read". The elapsed-time assertion is the real one here: the error message could
    be right by luck, but finishing in under a second against a thirty second
    deadline can only mean the death itself was seen.
    """
    started = time.monotonic()
    with pytest.raises(ParserOutOfMemoryError) as raised:
        await run_isolated(_die_hard, timeout_seconds=_GENEROUS_TIMEOUT, what="image")
    elapsed = time.monotonic() - started

    assert elapsed < _GENEROUS_TIMEOUT / 3, "the wait ran on after the child was already dead"
    assert "took too long" not in str(raised.value)
    # Rendering a large page at OCR resolution is the most memory-hungry thing
    # this app does, so naming the cause is worth the words.
    assert "ran out of memory" in str(raised.value)


def test_a_killed_reader_is_not_a_verdict_on_the_file() -> None:
    """The type carries the retry decision, so it is worth pinning on its own.

    Every OTHER failure in this module says the file is unreadable and is recorded
    as final. This one says the machine was out of memory, which the same file may
    well survive on the next attempt - so it must stay OUTSIDE the parsing-error
    family that the ingest service treats as terminal.
    """
    assert not issubclass(ParserOutOfMemoryError, ParseFailedError)
    assert not isinstance(ParserOutOfMemoryError("x"), ParseFailedError)


async def test_a_reader_that_dies_silently_is_reported_as_a_crash() -> None:
    started = time.monotonic()
    with pytest.raises(ParseFailedError) as raised:
        await run_isolated(_crash_natively, timeout_seconds=_GENEROUS_TIMEOUT, what="Word document")
    elapsed = time.monotonic() - started

    assert elapsed < _GENEROUS_TIMEOUT / 3
    assert "Word document" in str(raised.value)
    assert "took too long" not in str(raised.value)


async def test_an_answer_that_cannot_cross_the_boundary_is_reported_not_awaited() -> None:
    # The child cannot send what it read, so it sends the reason instead - which is
    # the whole difference between a file that fails with an explanation and one
    # that fails with a shrug. The parent must not sit out the deadline waiting for
    # an answer that is never coming, either.
    started = time.monotonic()
    with pytest.raises(ParseFailedError) as raised:
        await run_isolated(_answer_unpicklably, timeout_seconds=_GENEROUS_TIMEOUT, what="PDF")
    elapsed = time.monotonic() - started

    assert elapsed < _GENEROUS_TIMEOUT / 3
    assert "took too long" not in str(raised.value)
    assert "pickl" in str(raised.value).lower(), "the reason did not survive the trip"


class _TruncatedPipe:
    """A pipe holding the front of a message whose sender died mid-write."""

    def poll(self, timeout: float | None = None) -> bool:
        """Say there is something to read - there is, just not all of it."""
        return True

    def recv(self) -> tuple[str, object]:
        """Fail exactly as the standard library does on a half-written message."""
        raise OSError("got end of file during message")


class _KilledProcess:
    """A child the kernel's out-of-memory killer has already taken."""

    # Only meaningful where SIGKILL exists; the one test using it skips elsewhere.
    exitcode = -getattr(signal, "SIGKILL", 9)

    def join(self, timeout: float | None = None) -> None:
        """Already reaped; nothing to wait for."""


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL is posix-only")
def test_a_reader_killed_mid_answer_is_still_recognised_as_a_memory_kill() -> None:
    """The half-written answer, which is where a big file's reader actually dies.

    A pipe carries a long document's text in pieces - it holds only tens of KB at
    a time - so a reader killed while handing back a big answer leaves the front of
    a message behind and nothing more. The standard library reports THAT as an
    OSError, not the EOFError it raises when nothing arrived at all, and the
    difference is not cosmetic: miss it, and a memory kill is reported to the
    person who uploaded the file as an unavailable service, on the one path where
    the reader's death is most likely of all.

    Posix-only, like its sibling above, and for the same reason: what it pins is
    the classification of a SIGKILL exit code, and the classifier can only make
    that call where SIGKILL exists. Windows has no such signal and no kernel
    out-of-memory killer to raise it, so there the death is correctly reported as
    an ordinary crash - the deployment target is a Linux container either way.
    """
    with pytest.raises(ParserOutOfMemoryError):
        _await_result(_TruncatedPipe(), _KilledProcess(), 30.0, "PDF")


async def test_a_long_answer_comes_back_whole() -> None:
    # A pipe holds some tens of KB before it blocks, so anything bigger crosses in
    # chunks while the parent drains the other end. A real document's text is far
    # past that, and a parent that waited for the child to exit before reading
    # would deadlock on every one of them.
    result = await run_isolated(_answer_at_length, timeout_seconds=_GENEROUS_TIMEOUT, what="PDF")

    assert len(result) == 4 * 1024 * 1024
