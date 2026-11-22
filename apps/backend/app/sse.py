"""Server-Sent Events (SSE) formatting and stream helpers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress

from pydantic import BaseModel

# SSE comment frame: clients ignore lines starting with ":" per the SSE spec,
# but proxies and browsers see traffic and keep the idle socket open.
KEEP_ALIVE_FRAME = ": keep-alive\n\n"


def sse_event(event: str, data: BaseModel | dict) -> str:
    """Format one SSE frame: an event name plus a JSON ``data`` line."""
    payload = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


async def with_heartbeat(frames: AsyncIterator[str], interval: float) -> AsyncIterator[str]:
    """Re-yield ``frames`` unchanged, inserting keep-alive comments in quiet gaps.

    A producer task forwards the source into a queue and the consumer polls the
    queue with a timeout, emitting :data:`KEEP_ALIVE_FRAME` whenever ``interval``
    seconds pass without a frame. This shape - rather than a
    ``wait_for``-cancel around ``anext(frames)`` - matters: cancelling ``anext``
    on timeout can drop a frame that was already on its way out.

    ``interval <= 0`` disables heartbeats (plain passthrough). Producer
    exceptions propagate to the consumer; closing this generator cancels the
    producer and closes ``frames`` so upstream cleanup always runs.

    Args:
        frames: The source SSE frames.
        interval: Seconds of silence tolerated before each keep-alive comment.

    Yields:
        Source frames in order, interleaved with keep-alive comment frames.
    """
    if interval <= 0:
        try:
            async for frame in frames:
                yield frame
        finally:
            # Same guarantee as the heartbeat path: an early close must reach
            # the source's cleanup immediately, not at garbage collection.
            aclose = getattr(frames, "aclose", None)
            if aclose is not None:
                await aclose()
        return

    # None is the end-of-stream sentinel; real frames are always strings. The
    # queue is unbounded so the sentinel put in the producer's finally can
    # never block (a bounded queue could deadlock a cancelled producer).
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _produce() -> None:
        try:
            async for frame in frames:
                await queue.put(frame)
        finally:
            # Reached on success, failure, and cancellation alike: the
            # consumer must always wake up and stop polling.
            await queue.put(None)

    producer = asyncio.create_task(_produce())
    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                yield KEEP_ALIVE_FRAME
                continue
            if frame is None:
                # Re-raises the producer's exception when it ended with one.
                await producer
                return
            yield frame
    finally:
        producer.cancel()
        # The producer's own failure was already surfaced on the normal path;
        # here it is secondary to whatever is closing the stream.
        with suppress(BaseException):
            await producer
        # Deterministically close the source: a cancelled producer may have
        # left it suspended, and its finally blocks (partial-reply
        # persistence, limiter release) must not wait for garbage collection.
        aclose = getattr(frames, "aclose", None)
        if aclose is not None:
            await aclose()
