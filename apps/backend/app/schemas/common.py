"""Field types shared by the HTTP schemas.

Small enough to live in one module, and it must: the DTOs import each other (a
conversation carries its attachments), so a shared type parked in either one
would be a cycle waiting to happen.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BeforeValidator


def _assume_utc(value: object) -> object:
    """Treat naive datetimes as UTC.

    All rows are written with UTC timestamps, but sqlite (the test fixture
    backend) drops the offset on readback while postgres preserves it; pinning
    the zone here keeps the wire format ISO-8601 UTC on both.
    """
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


UtcDatetime = Annotated[datetime, BeforeValidator(_assume_utc)]
"""A datetime that always serializes with an explicit UTC offset.

Every timestamp the API returns must use this. Clients compare them ACROSS DTOs -
the composer decides which uploaded files are still an unsent draft by asking
whether an attachment is newer than the last message in the conversation - and a
response that stamped one of the two without an offset would have that comparison
silently read it as local time, skewing it by the reader's own UTC offset.
"""
