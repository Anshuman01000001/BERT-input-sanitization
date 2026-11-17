"""Structured logging.

Every moderation decision is written as a single JSON object so that the log
stream can be shipped to a store and replayed for moderator review and player
appeals without any parsing.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from datetime import UTC, datetime

# Attributes present on every ``LogRecord``; anything else was passed in via
# ``extra=`` and belongs in the structured payload.
_STANDARD_FIELDS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line.

    ``ensure_ascii`` is off so non-ASCII chat text stays readable, which
    leaves U+2028, U+2029 and U+0085 unescaped. Python's ``str.splitlines()``
    breaks on all three, so a JSON-lines consumer would see one record split
    into two broken fragments. They are escaped explicitly below.
    """

    _LINE_BREAKERS = {
        " ": "\\u2028",
        " ": "\\u2029",
        "": "\\u0085",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        rendered = json.dumps(payload, default=str, ensure_ascii=False)
        for character, escaped in self._LINE_BREAKERS.items():
            rendered = rendered.replace(character, escaped)
        return rendered


class RecurringErrorReporter:
    """Report a recurring failure without flooding the log.

    A Redis outage is expected and handled: the service keeps classifying and
    says so in its responses. The operator needs to know it happened, not to
    read the same multi-line traceback on every request, which would bury the
    moderation audit trail that the log exists for.

    The first occurrence is reported in full; after that the same condition
    is reported at most once per interval.
    """

    def __init__(self, interval_s: float = 60.0) -> None:
        self._interval = interval_s
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def should_report(self, key: str) -> bool:
        """True if this condition has not been reported recently."""
        now = time.monotonic()
        with self._lock:
            if now - self._last.get(key, 0.0) < self._interval:
                return False
            self._last[key] = now
            return True


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit JSON lines to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True


# Shared across the service so that one outage is reported once, not once
# per module that happens to touch Redis.
RECURRING_ERRORS = RecurringErrorReporter()
