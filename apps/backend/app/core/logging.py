"""Structured JSON logging with request-id correlation.

One JSON object per line on stderr, so ``docker compose logs`` and any log
shipper parse output without extra configuration. The request id assigned by
``app.core.middleware.RequestIdMiddleware`` lives in a contextvar here, where
both the formatter and the problem+json error handlers can read it without
importing middleware internals.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

from app.core.config import Settings

# Set once per request by RequestIdMiddleware and deliberately never reset:
# the catch-all Exception handler runs in Starlette's outermost error
# middleware - above the request-id middleware - and must still see the id
# while the exception unwinds. Servers run each request in its own task
# context, so values cannot leak across concurrent requests.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Return the current request id, or None outside a request."""
    return request_id_var.get()


# Attributes present on every LogRecord; anything else arrived via ``extra=``
# and belongs in the JSON payload (e.g. the access line's method/status).
_RESERVED_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "taskName",
}


class JsonLineFormatter(logging.Formatter):
    """Format each record as one compact JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the record, carrying request_id and exception text when present."""
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # default=str: an unserializable ``extra`` value must never kill a log line.
        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_logging(settings: Settings) -> None:
    """Install the JSON handler on the root logger (idempotent).

    Only the root logger is touched: uvicorn's own loggers keep their handlers
    and ``propagate=False`` wiring, so server startup/access output stays
    intact while every application log flows through the JSON handler. Calling
    this again (tests build many apps) refreshes the level without stacking
    duplicate handlers.

    Args:
        settings: Application settings; ``log_level`` sets the root level
            (unknown names fall back to INFO rather than crashing startup).
    """
    root = logging.getLogger()
    level = logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)
    root.setLevel(level)
    if not any(isinstance(handler.formatter, JsonLineFormatter) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLineFormatter())
        root.addHandler(handler)
