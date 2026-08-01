"""Configuration parsing and resource-limit primitives."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from media_context_mcp.config import Settings
from media_context_mcp.errors import (
    FileTooLargeError,
    InvalidPageSelectionError,
)
from media_context_mcp.security.limits import enforce_file_size, parse_page_selection

# ------------------------------------------------------------------- config --


def test_roots_pathsep_parsing(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    settings = Settings(allowed_roots=os.pathsep.join([str(a), str(b)]))
    assert settings.resolved_roots() == [a.resolve(), b.resolve()]


def test_roots_json_parsing(tmp_path: Path):
    a = tmp_path / "a"
    a.mkdir()
    settings = Settings(allowed_roots=f'["{str(a).replace(chr(92), chr(92) * 2)}"]')
    assert settings.resolved_roots() == [a.resolve()]


def test_empty_roots_is_fatal_problem():
    settings = Settings(allowed_roots=[])
    fatal = settings.fatal_problems()
    assert any("ALLOWED_ROOTS" in problem.field for problem in fatal)


def test_huggingface_preset_fills_base_url():
    settings = Settings(
        allowed_roots=[],
        vision_provider="huggingface",
        vision_model="org/some-vlm",
        vision_api_key="hf_x",
    )
    assert settings.effective_vision_base_url == "https://router.huggingface.co/v1"
    assert settings.vision_configured


def test_explicit_base_url_beats_preset():
    settings = Settings(
        allowed_roots=[],
        vision_provider="huggingface",
        vision_base_url="https://my-proxy.local/v1",
        vision_model="m",
        vision_api_key="k",
    )
    assert settings.effective_vision_base_url == "https://my-proxy.local/v1"


def test_hf_policy_suffix_is_explicit_and_not_doubled():
    settings = Settings(
        allowed_roots=[],
        vision_provider="huggingface",
        vision_model="org/vlm",
        vision_api_key="k",
        hf_provider_policy="fastest",
    )
    assert settings.effective_vision_model == "org/vlm:fastest"
    already = settings.model_copy(update={"vision_model": "org/vlm:together"})
    assert already.effective_vision_model == "org/vlm:together"


def test_cloud_gate_defaults_closed():
    settings = Settings(
        allowed_roots=[],
        vision_base_url="https://x/v1",
        vision_model="m",
        vision_api_key="k",
    )
    assert settings.vision_configured
    assert not settings.allow_cloud_vision
    assert not settings.cloud_vision_usable


def test_redacted_dump_never_contains_key():
    settings = Settings(allowed_roots=[], vision_api_key="sk-SUPERSECRET")
    dump = str(settings.redacted_dump())
    assert "SUPERSECRET" not in dump
    assert "<set>" in dump


# -------------------------------------------------------------------- pages --


@pytest.mark.parametrize(
    ("expression", "total", "expected"),
    [
        ("1", 5, [1]),
        ("1-3", 5, [1, 2, 3]),
        ("1,3,5", 5, [1, 3, 5]),
        ("1-2,4-5", 5, [1, 2, 4, 5]),
        (" 2 , 4 ", 5, [2, 4]),
        ("3,1,3", 5, [1, 3]),  # duplicates collapse, order normalises
        (None, 3, [1, 2, 3]),
    ],
)
def test_page_selection_valid(expression, total, expected):
    assert parse_page_selection(expression, total, max_pages=30) == expected


@pytest.mark.parametrize(
    "expression",
    ["0", "-1", "5-2", "abc", "1-", "-3", "1,,x", "99", "1-99"],
)
def test_page_selection_invalid(expression):
    with pytest.raises(InvalidPageSelectionError):
        parse_page_selection(expression, total_pages=5, max_pages=30)


def test_page_selection_caps_at_max_pages():
    assert parse_page_selection(None, total_pages=100, max_pages=4) == [1, 2, 3, 4]


# --------------------------------------------------------------------- size --


def test_file_size_limit(tmp_path: Path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 2048)
    assert enforce_file_size(big, max_bytes=4096) == 2048
    with pytest.raises(FileTooLargeError):
        enforce_file_size(big, max_bytes=1024)
