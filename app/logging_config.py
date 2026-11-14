"""Structured logging.

Every moderation decision is written as a single JSON object so that the log
stream can be shipped to a store and replayed for moderator review and player
appeals without any parsing.
"""

from __future__ import annotations

import json
import logging
import sys
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
