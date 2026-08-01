"""Path sandbox tests: every escape route the spec names, plus a few it doesn't."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from media_context_mcp.errors import (
    FileNotFoundError_,
    InvalidArgumentError,
    NotARegularFileError,
    PathNotAllowedError,
)
from media_context_mcp.security.paths import is_within, resolve_media_path


@pytest.fixture()
def sandbox(tmp_path: Path) -> tuple[Path, Path]:
    """An allowed root containing one file, next to an out-of-root file."""
    root = tmp_path / "workspace"
    root.mkdir()
    inside = root / "notes.txt"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("outside", encoding="utf-8")
    return root, outside


def test_valid_absolute_path_inside_root(sandbox):
    root, _ = sandbox
    resolved = resolve_media_path(str(root / "notes.txt"), [root])
    assert resolved.name == "notes.txt"


def test_relative_path_resolves_against_root(sandbox):
    root, _ = sandbox
    resolved = resolve_media_path("notes.txt", [root])
    assert resolved == Path(os.path.realpath(root / "notes.txt"))


def test_traversal_attempt_rejected(sandbox):
    root, _ = sandbox
    with pytest.raises(PathNotAllowedError):
        resolve_media_path("../secret.txt", [root])


def test_absolute_path_outside_root_rejected(sandbox):
    root, outside = sandbox
    with pytest.raises(PathNotAllowedError):
        resolve_media_path(str(outside), [root])


def test_prefix_sibling_directory_rejected(tmp_path: Path):
    """/x/workspace-evil must not pass because /x/workspace is allowed."""
    root = tmp_path / "workspace"
    root.mkdir()
    evil = tmp_path / "workspace-evil"
    evil.mkdir()
    victim = evil / "file.txt"
    victim.write_text("x", encoding="utf-8")
    with pytest.raises(PathNotAllowedError):
        resolve_media_path(str(victim), [root])


@pytest.mark.skipif(
    sys.platform == "win32" and not os.environ.get("CI"),
    reason="symlink creation on Windows needs developer mode or admin",
)
def test_symlink_escape_rejected(sandbox):
    root, outside = sandbox
    link = root / "innocent.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("cannot create symlinks in this environment")
    with pytest.raises(PathNotAllowedError):
        resolve_media_path(str(link), [root])


def test_nonexistent_file(sandbox):
    root, _ = sandbox
    with pytest.raises(FileNotFoundError_):
        resolve_media_path("missing.txt", [root])


def test_existence_not_leaked_outside_root(sandbox, tmp_path: Path):
    """Outside the root, 'exists' and 'does not exist' must be indistinguishable."""
    root, outside = sandbox
    with pytest.raises(PathNotAllowedError):
        resolve_media_path(str(outside), [root])
    with pytest.raises(PathNotAllowedError):
        resolve_media_path(str(tmp_path / "never-created.txt"), [root])


def test_directory_rejected(sandbox):
    root, _ = sandbox
    with pytest.raises(NotARegularFileError):
        resolve_media_path(str(root), [root])


def test_url_rejected(sandbox):
    root, _ = sandbox
    for url in ("https://example.com/a.png", "http://x/y.pdf", "file:///etc/passwd",
                "ftp://host/file", "data:image/png;base64,AAAA"):
        with pytest.raises(InvalidArgumentError):
            resolve_media_path(url, [root])


def test_unc_path_rejected(sandbox):
    root, _ = sandbox
    with pytest.raises(PathNotAllowedError):
        resolve_media_path(r"\\server\share\file.txt", [root])


def test_null_byte_rejected(sandbox):
    root, _ = sandbox
    with pytest.raises(InvalidArgumentError):
        resolve_media_path("notes\x00.txt", [root])


def test_empty_path_rejected(sandbox):
    root, _ = sandbox
    with pytest.raises(InvalidArgumentError):
        resolve_media_path("   ", [root])


def test_no_roots_configured_is_config_error(sandbox):
    root, _ = sandbox
    from media_context_mcp.errors import ErrorCode

    with pytest.raises(PathNotAllowedError) as excinfo:
        resolve_media_path(str(root / "notes.txt"), [])
    assert excinfo.value.code is ErrorCode.CONFIGURATION_ERROR


@pytest.mark.skipif(sys.platform != "win32", reason="WSL-path advice is Windows-side")
def test_wsl_style_path_gets_translation_advice(sandbox):
    root, _ = sandbox
    with pytest.raises(PathNotAllowedError) as excinfo:
        resolve_media_path("/mnt/d/Work/project/screenshot.png", [root])
    assert "D:\\" in excinfo.value.message


def test_is_within_case_insensitive_on_windows(tmp_path: Path):
    if sys.platform != "win32":
        pytest.skip("case-folding check is Windows behaviour")
    lower = Path(str(tmp_path).lower())
    upper = Path(str(tmp_path).upper())
    assert is_within(lower / "a", upper)
