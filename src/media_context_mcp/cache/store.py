"""On-disk cache store: JSON metadata + content, one file per entry.

Design points, all deliberate:

* **Atomic writes.** Entries are written to a temp file in the same directory and
  moved into place with ``os.replace``, which is atomic on POSIX and NTFS. A
  concurrent reader sees either the old entry or the new one, never a torn file.
* **Corruption is a miss.** An unreadable or schema-mismatched entry is deleted
  and treated as absent; a broken cache must degrade to slow, never to wrong.
* **Only successes are stored.** The pipeline never calls ``put`` for failures --
  auth errors, timeouts, empty or invalid provider responses -- so a transient
  failure can never be replayed from disk as if it were an answer.
* **No secrets.** The stored payload is the serialised ProcessorResult plus
  bookkeeping. API keys are not part of the key material or the payload.
* **No database.** A directory of self-contained JSON files needs no migrations,
  survives partial deletion, and is trivially inspectable. At the scale of a
  per-workspace media cache this beats SQLite on operational simplicity.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger

_LOGGER = get_logger(__name__)

ENTRY_SCHEMA_VERSION = 2


class CacheStore:
    """A directory of ``<key>.json`` entries with TTL and size-based cleanup."""

    def __init__(
        self,
        directory: Path,
        *,
        enabled: bool = True,
        max_bytes: int = 1024 * 1024 * 1024,
        ttl_days: int = 30,
    ) -> None:
        self._directory = directory
        self._enabled = enabled
        self._max_bytes = max_bytes
        self._ttl_seconds = ttl_days * 86400
        self._ready = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def directory(self) -> Path:
        return self._directory

    def _ensure_directory(self) -> bool:
        if self._ready:
            return True
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._ready = True
            return True
        except OSError as exc:
            _LOGGER.warning("Cache directory unavailable (%s); caching disabled", exc)
            self._enabled = False
            return False

    def _path_for(self, key: str) -> Path:
        # Keys are hex SHA-256, so they are always filesystem-safe; the assert
        # guards against a future caller passing something else.
        assert all(ch in "0123456789abcdef" for ch in key), "cache key must be hex"
        return self._directory / f"{key}.json"

    # ------------------------------------------------------------------ get --

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the stored payload, or ``None`` on miss/expiry/corruption."""
        if not self._enabled or not self._ensure_directory():
            return None
        path = self._path_for(key)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            _LOGGER.warning("Cache read failed for %s: %s", key[:12], exc)
            return None

        try:
            entry = json.loads(raw)
            if entry.get("schema") != ENTRY_SCHEMA_VERSION:
                raise ValueError(f"schema {entry.get('schema')} != {ENTRY_SCHEMA_VERSION}")
            stored_at = float(entry["stored_at"])
            payload = entry["payload"]
        except (ValueError, KeyError, TypeError) as exc:
            _LOGGER.warning("Corrupt cache entry %s (%s); deleting", key[:12], exc)
            with contextlib.suppress(OSError):
                path.unlink()
            return None

        if self._ttl_seconds > 0 and time.time() - stored_at > self._ttl_seconds:
            with contextlib.suppress(OSError):
                path.unlink()
            return None

        # Touch the mtime so size-based cleanup evicts least-recently-used first.
        with contextlib.suppress(OSError):
            os.utime(path, None)
        return payload

    # ------------------------------------------------------------------ put --

    def put(self, key: str, payload: dict[str, Any]) -> bool:
        """Store a payload atomically. Failures disable nothing and raise nothing."""
        if not self._enabled or not self._ensure_directory():
            return False
        entry = {
            "schema": ENTRY_SCHEMA_VERSION,
            "stored_at": time.time(),
            "payload": payload,
        }
        path = self._path_for(key)
        try:
            # Temp file in the same directory so os.replace stays on one filesystem.
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{key[:12]}.", suffix=".tmp", dir=self._directory
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(entry, handle, ensure_ascii=False)
                os.replace(temp_name, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temp_name)
                raise
        except OSError as exc:
            _LOGGER.warning("Cache write failed for %s: %s", key[:12], exc)
            return False
        return True

    # -------------------------------------------------------------- cleanup --

    def cleanup(self) -> dict[str, int]:
        """Drop expired entries, then trim to the size budget, oldest-touched first.

        Returns counters for diagnostics. Safe to call at any time; errors on
        individual files are skipped.
        """
        stats = {"expired": 0, "evicted": 0, "kept": 0, "bytes": 0}
        if not self._ensure_directory():
            return stats

        now = time.time()
        entries: list[tuple[float, int, Path]] = []
        for path in self._directory.glob("*.json"):
            try:
                stat_result = path.stat()
            except OSError:
                continue
            age = now - stat_result.st_mtime
            if self._ttl_seconds > 0 and age > self._ttl_seconds:
                with contextlib.suppress(OSError):
                    path.unlink()
                    stats["expired"] += 1
                continue
            entries.append((stat_result.st_mtime, stat_result.st_size, path))

        entries.sort()  # oldest mtime first
        total = sum(size for _, size, _ in entries)
        while total > self._max_bytes and entries:
            _, size, path = entries.pop(0)
            with contextlib.suppress(OSError):
                path.unlink()
                stats["evicted"] += 1
                total -= size
        stats["kept"] = len(entries)
        stats["bytes"] = total
        return stats

    def clear(self) -> int:
        """Delete every entry. Returns the number removed."""
        if not self._ensure_directory():
            return 0
        removed = 0
        for path in self._directory.glob("*.json"):
            with contextlib.suppress(OSError):
                path.unlink()
                removed += 1
        return removed
