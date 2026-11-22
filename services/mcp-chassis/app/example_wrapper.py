"""The example MCP wrapper: fronts a mock external API-backed datastore.

A real wrapper translates each tool call into an HTTP request against an
upstream API (the datastore's own JS/HTTP surface - a ticket system, a CRM,
an orders service). This example stands that upstream in with an in-process
dict, so the container runs anywhere with no credentials and no network.
Deployment-specific wrappers live on other branches; this one exists to
exercise the chassis pattern end to end.
"""

from __future__ import annotations

import logging
import os
import threading

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# The mock upstream. What a deployment reaches over HTTP lives here as a
# module-level dict; ids are stable so tests and demos can name them.
RECORDS: dict[str, dict[str, str]] = {
    "rec-001": {
        "title": "Printer offline after firmware update",
        "body": (
            "The office printer dropped off the network after the 2.1 firmware "
            "update. Rolling back to 2.0 restores it."
        ),
        "status": "open",
    },
    "rec-002": {
        "title": "Password reset link expires too fast",
        "body": (
            "Reset emails carry links that expire in five minutes; users on slow "
            "mail servers never make it in time."
        ),
        "status": "open",
    },
    "rec-003": {
        "title": "CSV export drops accented names",
        "body": (
            "Customer names with accents export as question marks. The export "
            "encodes latin-1 instead of utf-8."
        ),
        "status": "closed",
    },
}

# FastMCP runs sync tools on worker threads, so id allocation plus insert is
# a check-then-act; the lock keeps two concurrent creates from colliding.
_records_lock = threading.Lock()


class Record(BaseModel):
    """One record as structured output.

    The field names ARE the platform's citable-records contract: a read tool
    whose ``structuredContent`` is a ``records`` list of these gets ingested
    into the conversation's session store and cited by ``[n]`` marker, instead
    of being handed to the model as opaque text. ``text`` carries the full body
    (not a display snippet) precisely because the full text is what grounds and
    cites well - surfacing structured records rather than a formatted string is
    the whole reason this returns a model.
    """

    id: str
    title: str
    text: str


class RecordList(BaseModel):
    """The structured return the read tools share.

    ``records`` is the citable payload. ``message`` carries the human-readable
    note for the no-match and unknown-id cases, where there is nothing to cite
    but the model still needs to be told what happened and how to recover; the
    platform ingests ``records`` and ignores ``message``, so the note never
    pollutes the cited set.
    """

    records: list[Record]
    message: str | None = None


def _next_record_id() -> str:
    """The next free ``rec-NNN`` id; scans past collisions rather than assuming density."""
    n = len(RECORDS) + 1
    while (candidate := f"rec-{n:03d}") in RECORDS:
        n += 1
    return candidate


def build_example_server(allowed_hosts: list[str]) -> FastMCP:
    """Builds the ``example-datastore`` FastMCP server.

    Args:
        allowed_hosts: Host header values the transport accepts. Explicit
            because the SDK's DNS-rebinding protection ships with an EMPTY
            allow list - without this, every request (localhost included) is
            rejected 421 before the protocol layer sees it.

    Returns:
        A stateless Streamable-HTTP FastMCP server exposing the example tools.
    """
    if os.environ.get("EXAMPLE_API_BASE_URL"):
        # The env var is reserved for a real upstream client; main ships only
        # the mock, so say so out loud instead of silently ignoring config.
        logger.warning(
            "EXAMPLE_API_BASE_URL is set, but a real upstream client is not "
            "implemented on main; the built-in mock store is used instead."
        )

    server = FastMCP(
        "example-datastore",
        stateless_http=True,
        transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts),
    )

    @server.tool(
        description=(
            "Search the example datastore for records whose title or body "
            "contains the query text. Returns the matching records with their "
            "id, title, and full text."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def search_records(query: str) -> RecordList:
        """Substring search over record titles and bodies."""
        needle = query.lower()
        matches = [
            Record(id=record_id, title=record["title"], text=record["body"])
            for record_id, record in RECORDS.items()
            if needle in record["title"].lower() or needle in record["body"].lower()
        ]
        if not matches:
            return RecordList(records=[], message=f"No records match {query!r}.")
        return RecordList(records=matches)

    @server.tool(
        description="Fetch one record's full text (title, status, body) by its record id.",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def get_record(record_id: str) -> RecordList:
        """One record's full text; unknown ids get a message, not an exception."""
        record = RECORDS.get(record_id)
        if record is None:
            return RecordList(
                records=[],
                message=f"No record with id {record_id!r}. Use search_records to find valid ids.",
            )
        text = f"status: {record['status']}\n\n{record['body']}"
        return RecordList(records=[Record(id=record_id, title=record["title"], text=text)])

    # Deliberately UNANNOTATED: creating a record is side-effecting, and the
    # platform fails closed on a missing readOnlyHint - marking it read-only
    # would let it run without the confirmation flow.
    @server.tool(
        description="Create a new record in the example datastore and return its new record id.",
    )
    def create_record(title: str, body: str) -> str:
        """Inserts a record into the mock store."""
        with _records_lock:
            record_id = _next_record_id()
            RECORDS[record_id] = {"title": title, "body": body, "status": "open"}
        return f"Created record {record_id}."

    return server
