"""Run an untrusted parse where it can actually be killed.

Every real document format is read by native code - PDFium for PDFs, lxml for
the office formats - and native code cannot be interrupted. An asyncio timeout
will not touch it, and a thread running it cannot be unwound: once a C call
enters a pathological loop, the thread is gone for the life of the process. A
hostile or merely malformed file could therefore wedge an ingest worker for good.

A process, on the other hand, can be terminated. So the parse runs in a child
that is killed if it overruns, and the worker survives to fail the file honestly.
The ingest job's own ``job_timeout`` is the outer bound, and it cannot fire while
a native call holds the thread - which is precisely why the inner process exists.

The answer comes back over a one-way pipe rather than a queue, and that choice is
load-bearing. A queue hands the item to a feeder thread, so a child that is killed
mid-hand-off leaves a truncated message behind with the write end still open in
the parent - and a read of it blocks forever, in exactly the case (a child killed
by the kernel) this module exists to survive. A pipe whose only write end lives in
the child turns that same death into an EOF, which is a thing that can be caught.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import signal
import time
from collections.abc import Callable
from typing import Any

from app.parsing.base import ParseFailedError, ParserOutOfMemoryError, ParsingError

logger = logging.getLogger(__name__)

# Spawn, not fork: the ingest worker is an async process, and forking one copies
# whatever locks its other threads happen to hold, which can deadlock the child.
# Spawn starts a clean interpreter - and is the only option on Windows. The cost
# is that the target must be a top-level function whose module is import-light,
# because the child re-imports it.
_MP_CONTEXT = multiprocessing.get_context("spawn")

# How long to wait for a terminated child to actually die before giving up on it.
_JOIN_TIMEOUT_SECONDS = 5.0

# How often the wait re-checks the pipe. Short enough that an answer (or a death)
# is picked up at once, long enough that the check costs nothing next to the parse
# it is watching.
_POLL_SECONDS = 0.1


def _child(target: Callable[..., Any], args: tuple[Any, ...], conn: Any) -> None:
    """Run ``target`` in the child process and hand its outcome back.

    Failures are SENT rather than raised: an exception raised here dies with the
    child, because a traceback does not cross a process boundary. That includes
    the failure to send itself - a result that cannot be pickled raises inside
    the send, and the reason for it is worth more to the parent than silence.

    A reader that KNOWS what is wrong with the file says so, and its sentence is
    passed through whole. Anything else is a library's exception, and the parent
    wraps it, because "PackageNotFoundError" is not a sentence anybody wants to
    read about the document they just uploaded.
    """
    try:
        conn.send(("ok", target(*args)))
    except ParsingError as exc:
        with contextlib.suppress(Exception):
            conn.send(("fail", str(exc)))
    except Exception as exc:  # noqa: BLE001 - handed to the parent as a seam error
        # If even this cannot be sent the pipe itself is gone, and the parent sees
        # the EOF that closing it produces - which it already knows how to report.
        with contextlib.suppress(Exception):
            conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _crash_error(exitcode: int | None, what: str) -> Exception:
    """Turn a dead child's exit code into the error that best explains it.

    The distinction that matters is transient versus terminal, because the caller
    acts on it: a killed reader is a statement about the MACHINE and is retried,
    while every other death is a statement about the FILE and is final.
    """
    sigkill = getattr(signal, "SIGKILL", None)
    if sigkill is not None and exitcode == -sigkill:
        # Nothing inside the child can catch this. In a container with a memory
        # limit it is overwhelmingly the kernel's out-of-memory killer, which is a
        # real risk here: rendering a large page at OCR resolution is the most
        # memory-hungry thing this app does. The same file may well read perfectly
        # once whatever else was competing for memory has finished, so this one is
        # raised as the transient failure it is.
        return ParserOutOfMemoryError(f"The {what} could not be read (it ran out of memory).")
    if exitcode is None:
        reason = "it stopped responding"
    elif exitcode < 0:
        reason = f"it crashed (signal {-exitcode})"
    elif exitcode == 0:
        # It exited cleanly having said nothing at all - not even the reason it
        # could not answer, which the child sends for itself. Not a crash, but not
        # an answer either, and there is nothing left to say about it.
        reason = "it produced nothing"
    else:
        reason = f"it exited unexpectedly (code {exitcode})"
    return ParseFailedError(f"The {what} could not be read ({reason}).")


def _await_result(conn: Any, proc: Any, timeout: float, what: str) -> tuple[str, Any]:
    """Wait for the child's answer, or for the child to die trying.

    Waiting for the answer alone is not enough, and the gap is not theoretical. A
    child killed by the kernel, or felled by a segfault in the native library it
    exists to contain, sends NOTHING - so a plain blocking read would wait out the
    entire deadline (up to two minutes at the OCR timeout, across three retries),
    hold an ingest slot the whole time doing nothing, and then report that the file
    "took too long to read". Which is false: it took no time at all. The reader died.

    The pipe closes when the child dies, so its death arrives at the very poll that
    was waiting for its answer - immediately, and whether it had begun to reply or
    not. Both halves of "not" are caught here, because the standard library tells
    them apart: nothing of the message arrived is an ``EOFError``, while a message
    that STARTED and then stopped is an ``OSError`` ("got end of file during
    message"). The second is not the exotic one. A long document's text is written
    in pieces - the pipe holds only tens of KB at a time - so a reader killed while
    handing back a big answer ends exactly there, and treating that as an ordinary
    error would report a memory kill as an unavailable service.

    The one thing not bounded is the read itself: once the first byte arrives, the
    rest is read without a deadline. A child that froze mid-write (SIGSTOP, a cgroup
    freezer) would hold the thread. That is accepted - a child that is writing is a
    child that is alive and answering, and the alternative is reimplementing the
    framing this pipe gives for free.
    """
    deadline = time.monotonic() + timeout
    while not conn.poll(min(_POLL_SECONDS, max(0.0, deadline - time.monotonic()))):
        if time.monotonic() >= deadline:
            logger.warning("%s parse exceeded %.1fs; terminating the reader", what, timeout)
            raise ParseFailedError(f"The {what} took too long to read.")
    try:
        return conn.recv()
    except (EOFError, OSError):
        # The only write end went away without a complete message on it: the child
        # died before it could answer, or while it was still answering. Reap it
        # first, so the exit code that explains the death is there to be read.
        proc.join(_JOIN_TIMEOUT_SECONDS)
        raise _crash_error(proc.exitcode, what) from None


def _run(target: Callable[..., Any], args: tuple[Any, ...], timeout: float, what: str) -> Any:
    """Run ``target(*args)`` in a killable child; return its result.

    Raises:
        ParseFailedError: The file could not be read, it killed the reader, or
            reading it ran past ``timeout``. The child is terminated either way.
        ParserOutOfMemoryError: The reader was killed from outside - see
            :func:`_crash_error`. Transient, so the caller may retry it.
    """
    receiver, sender = _MP_CONTEXT.Pipe(duplex=False)
    try:
        proc = _MP_CONTEXT.Process(target=_child, args=(target, args, sender), daemon=True)
        proc.start()
    except BaseException:
        receiver.close()
        raise
    finally:
        # Closed whatever happened. The child duplicated its own write end inside
        # start(), so after that this copy is dead weight - and worse than dead:
        # left open it keeps the pipe alive, and a dead child's silence would never
        # arrive here as an end of file.
        sender.close()
    try:
        status, payload = _await_result(receiver, proc, timeout, what)
    finally:
        proc.terminate()
        proc.join(_JOIN_TIMEOUT_SECONDS)
        receiver.close()
    if status == "fail":
        # The reader diagnosed the file itself. Its sentence is already the one to
        # show, so it crosses the boundary unwrapped.
        raise ParseFailedError(payload)
    if status == "err":
        raise ParseFailedError(f"The {what} could not be read ({payload}).")
    return payload


async def run_isolated(
    target: Callable[..., Any],
    *args: Any,
    timeout_seconds: float,
    what: str,
) -> Any:
    """Extract from a file in a separate, killable process.

    The deadline is an argument rather than an ``asyncio.timeout`` around the
    call, and it has to be: cancelling the coroutine that waits would abandon the
    child, not kill it, and the runaway native parse this exists to contain would
    go on burning a core for the life of the worker. Only the process holding it
    can be terminated, so only the code holding that process can enforce a limit.

    Args:
        target: A TOP-LEVEL function (the spawn bootstrap re-imports its module,
            so it must be picklable and its module cheap to import).
        *args: Passed to ``target``; must be picklable, and so must its result.
        timeout_seconds: How long the child gets before it is killed.
        what: What is being read ("PDF", "Word document"), for the error message
            the person who uploaded it eventually reads.

    Raises:
        ParseFailedError: The file is unreadable, or it ran past the deadline.
        ParserOutOfMemoryError: The reader was killed for want of memory. Unlike
            every other failure here this one is about the machine, not the file,
            so it is transient and worth retrying.
    """
    return await asyncio.to_thread(_run, target, args, timeout_seconds, what)
