"""Workspace sandbox.

Every file the server reads passes through :func:`resolve_media_path`. The order of
operations matters and is fixed:

1. reject syntactically hostile input (null bytes, URL schemes, empty strings);
2. expand ``~``  -- but never environment variables, which would let a caller
   smuggle a path in through the server's own environment;
3. make the path absolute, resolving it against each allowed root when relative;
4. canonicalise with ``realpath`` so that symlinks, ``..`` segments, junctions and
   8.3 short names are all collapsed *before* any check;
5. only then verify ancestry against the resolved roots;
6. verify the target is a regular file.

Ancestry is checked with :func:`os.path.normcase` on both sides rather than a raw
string prefix: ``/data/workspace-evil`` must not pass because ``/data/workspace``
is allowed, and on Windows ``C:\\Work`` and ``c:\\work`` are the same directory.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from ..errors import (
    ErrorCode,
    FileNotFoundError_,
    InvalidArgumentError,
    NotARegularFileError,
    PathNotAllowedError,
)

# Anything with a scheme is rejected outright in the MVP: remote fetching is
# disabled (SSRF surface), and 'file://' would only be a second spelling of a
# local path that bypasses the reader's intuition about what is being asked.
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
_DATA_URI_RE = re.compile(r"^data:", re.IGNORECASE)

_WSL_STYLE_RE = re.compile(r"^/mnt/[a-zA-Z]/")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_UNC_RE = re.compile(r"^\\\\[^\\]")


def _normcase(path: Path) -> str:
    return os.path.normcase(str(path))


def is_within(candidate: Path, root: Path) -> bool:
    """True when ``candidate`` is ``root`` itself or lives underneath it.

    Both sides must already be resolved. Uses path-component ancestry, so a shared
    textual prefix is not enough.
    """
    candidate_norm = _normcase(candidate)
    root_norm = _normcase(root)
    if candidate_norm == root_norm:
        return True
    try:
        return os.path.commonpath([candidate_norm, root_norm]) == root_norm
    except ValueError:
        # Different drives on Windows, or mixing absolute and relative.
        return False


def describe_platform_mismatch(raw_path: str) -> str | None:
    """Explain a path that belongs to the *other* side of a WSL boundary.

    We refuse to translate automatically. ``/mnt/d/x`` and ``D:\\x`` are the same
    bytes on disk but different strings to the sandbox, and a silent rewrite would
    mean the allowed-root check validates a path the caller never asked for.
    """
    if os.name == "nt" and _WSL_STYLE_RE.match(raw_path):
        drive = raw_path[5]
        remainder = raw_path[7:]
        return (
            f"'{raw_path}' is a WSL-style path but this server runs on Windows. "
            f"Use '{drive.upper()}:\\{remainder.replace('/', chr(92))}' instead."
        )
    if os.name != "nt" and _WINDOWS_DRIVE_RE.match(raw_path):
        drive = raw_path[0].lower()
        remainder = raw_path[3:].replace("\\", "/")
        return (
            f"'{raw_path}' is a Windows path but this server runs on a POSIX host. "
            f"Use '/mnt/{drive}/{remainder}' instead."
        )
    return None


def validate_path_syntax(raw_path: str) -> str:
    """Reject inputs that must never reach the filesystem layer."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise InvalidArgumentError(
            "path must be a non-empty string.",
            hint="Pass an absolute path, or a path relative to an allowed root.",
        )
    if "\x00" in raw_path:
        raise InvalidArgumentError(
            "path contains a null byte.",
            hint="Remove embedded null bytes from the path.",
        )
    if _DATA_URI_RE.match(raw_path):
        raise InvalidArgumentError(
            "data: URIs are not supported.",
            hint="Write the media to a file inside an allowed root and pass its path.",
        )
    if _URL_SCHEME_RE.match(raw_path):
        scheme = raw_path.split("://", 1)[0]
        raise InvalidArgumentError(
            f"Remote and scheme-qualified paths are disabled ('{scheme}://').",
            hint=(
                "Download the file into an allowed root first, then pass the local path. "
                "Remote fetching is intentionally disabled in this release."
            ),
        )
    if _UNC_RE.match(raw_path):
        raise PathNotAllowedError(
            "UNC network paths are not accepted.",
            hint="Copy the file to a local directory listed in MEDIA_MCP_ALLOWED_ROOTS.",
        )
    return raw_path.strip()


def _candidate_absolute_paths(raw_path: str, roots: list[Path]) -> list[Path]:
    """Absolute candidates for a possibly-relative input."""
    expanded = Path(os.path.expanduser(raw_path))
    if expanded.is_absolute():
        return [expanded]
    # A relative path is interpreted against each allowed root, in order. The CWD is
    # deliberately not consulted: an MCP server's CWD is whatever the client happened
    # to spawn it in, which is not a meaningful sandbox anchor.
    return [root / expanded for root in roots]


def resolve_media_path(raw_path: str, roots: list[Path]) -> Path:
    """Resolve ``raw_path`` to a canonical file inside one of ``roots``.

    Raises a :class:`~media_context_mcp.errors.MediaContextError` subclass on any
    failure; never returns a path that has not been fully validated.
    """
    cleaned = validate_path_syntax(raw_path)

    if not roots:
        raise PathNotAllowedError(
            "No allowed roots are configured, so no file may be read.",
            code=ErrorCode.CONFIGURATION_ERROR,
            hint=(
                "Set MEDIA_MCP_ALLOWED_ROOTS in the MCP server's environment to the "
                "workspace directory, then restart the client."
            ),
        )

    mismatch = describe_platform_mismatch(cleaned)
    if mismatch:
        raise PathNotAllowedError(
            mismatch,
            hint="Paths are never translated across the WSL boundary automatically, "
            "because a silent rewrite would validate a path you did not ask for.",
            details={"given_path": cleaned},
        )

    candidates = _candidate_absolute_paths(cleaned, roots)

    resolved: Path | None = None
    for candidate in candidates:
        try:
            # strict=False so a missing file still canonicalises; existence is checked
            # below, after the sandbox test, so we never leak whether an out-of-root
            # file exists.
            canonical = Path(os.path.realpath(candidate))
        except (OSError, RuntimeError) as exc:
            raise InvalidArgumentError(
                f"path could not be resolved: {exc}",
                hint="Check for an invalid or excessively long path.",
            ) from exc
        if any(is_within(canonical, root) for root in roots):
            resolved = canonical
            break
        if resolved is None:
            resolved = canonical  # remember the first for the error message

    if resolved is None or not any(is_within(resolved, root) for root in roots):
        raise PathNotAllowedError(
            "The requested path is outside every allowed root.",
            hint=(
                "Allowed roots are: "
                + ", ".join(str(root) for root in roots)
                + ". Move the file inside one of them, or add its directory to "
                "MEDIA_MCP_ALLOWED_ROOTS and restart the server."
            ),
            details={"given_path": cleaned, "resolved_path": str(resolved) if resolved else None},
        )

    if not resolved.exists():
        raise FileNotFoundError_(
            f"No such file: {resolved}",
            hint="Check the spelling, or list the directory to confirm the file name.",
            details={"resolved_path": str(resolved)},
        )

    try:
        info = resolved.stat()
    except OSError as exc:
        raise FileNotFoundError_(
            f"Could not stat file: {exc}",
            hint="The file may have been removed or its permissions changed.",
        ) from exc

    if stat.S_ISDIR(info.st_mode):
        raise NotARegularFileError(
            f"Path is a directory, not a file: {resolved}",
            hint="Pass the path of a single media or document file.",
        )
    if not stat.S_ISREG(info.st_mode):
        raise NotARegularFileError(
            f"Path is not a regular file (fifo, socket, or device): {resolved}",
            hint="Only regular files can be analysed.",
        )

    return resolved
