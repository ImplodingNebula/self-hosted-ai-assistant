"""Size-bounded reads for every OAuth document fetched from a remote host.

Each document this package fetches - protected-resource metadata, an
authorization server's metadata, a token response, an identity lookup - is
small, and every one of them comes from a host named by REMOTE input: the
metadata a registered server serves decides where the next request goes. A
buffered read of an unbounded body is therefore an out-of-memory kill any
registered server can trigger, in the single process that also serves every
live chat stream.

An oversized body is REFUSED, never truncated. Truncating a JSON document does
not yield a smaller document, it yields a parse error wearing a disguise, and
the caller would report "malformed metadata" for what is really an attack.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

# Generous by two orders of magnitude: a real authorization-server metadata
# document is one to three KB, and a token response a few hundred bytes.
MAX_DOCUMENT_BYTES = 64 * 1024


class OversizedResponseError(Exception):
    """A remote document exceeded the byte cap and was refused unread."""

    def __init__(self, what: str, limit: int) -> None:
        """Record what was being read and the cap it broke, never the body."""
        super().__init__(f"the {what} exceeded {limit} bytes and was refused")
        self.what = what
        self.limit = limit


async def read_capped(
    response: httpx.Response, *, what: str, limit: int = MAX_DOCUMENT_BYTES
) -> bytes:
    """Read a streamed response body, refusing it past ``limit`` bytes.

    Args:
        response: An OPEN streaming response (from ``client.stream(...)``); a
            buffered response has already paid the cost this guards against.
        what: What is being read, for the error message. Never a URL - it may
            embed a credential.
        limit: The byte ceiling.

    Returns:
        The complete body.

    Raises:
        OversizedResponseError: The body ran past ``limit``. Raised as soon as
            the cap is crossed, so the excess is never accumulated.
    """
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > limit:
            raise OversizedResponseError(what, limit)
    return bytes(body)


async def read_json_capped(
    response: httpx.Response, *, what: str, limit: int = MAX_DOCUMENT_BYTES
) -> Any:
    """Read a streamed response body under a byte cap and parse it as JSON.

    Raises:
        OversizedResponseError: The body ran past ``limit``.
        ValueError: The body was not valid JSON - the same failure
            ``httpx.Response.json`` raises, so call sites are unchanged.
    """
    return json.loads(await read_capped(response, what=what, limit=limit))
