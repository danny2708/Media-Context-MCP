"""Logging that cannot corrupt the MCP STDIO stream.

STDOUT belongs exclusively to the JSON-RPC framing. Every handler here writes to
STDERR (plus an optional file). :func:`configure_logging` also re-points the root
logger, so a chatty third-party library that calls ``print``-free ``logging`` can
never leak into the protocol.

Named ``logging_setup`` rather than ``logging`` on purpose: a module named
``logging`` inside the package shadows the stdlib for anything doing a relative
import, and the failure mode is baffling.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("media_mcp_request_id", default=None)

_SECRET_PLACEHOLDER = "<redacted>"
_CONFIGURED = False


def new_request_id() -> str:
    """Short correlation id attached to every log line of one tool call."""
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or "-"
        return True


class _StructuredFormatter(logging.Formatter):
    """One JSON object per line -- greppable, and safe for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Install STDERR (and optional file) logging. Idempotent."""
    global _CONFIGURED

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(_StructuredFormatter())
    stderr_handler.addFilter(_RequestIdFilter())
    root.addHandler(stderr_handler)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(_StructuredFormatter())
            file_handler.addFilter(_RequestIdFilter())
            root.addHandler(file_handler)
        except OSError as exc:
            # A broken log file must never stop the server; say so on STDERR and move on.
            root.warning("Could not open log file %s: %s", log_file, exc)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log with structured fields.

    Values are passed through :func:`scrub` so a stray settings object or provider
    payload cannot put an API key into the log.
    """
    logger.log(level, message, extra={"fields": {k: scrub(v) for k, v in fields.items()}})


_SECRET_HINTS = ("api_key", "apikey", "authorization", "token", "secret", "password", "bearer")


def scrub(value: Any) -> Any:
    """Best-effort redaction of secret-shaped data before it reaches a log sink."""
    if isinstance(value, dict):
        return {
            key: (
                _SECRET_PLACEHOLDER
                if any(hint in str(key).lower() for hint in _SECRET_HINTS)
                else scrub(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [scrub(item) for item in value]
    return value
