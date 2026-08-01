"""MCP protocol tests: a real server subprocess over STDIO.

These verify the transport contract itself: the tool is discoverable, a valid
call returns the documented shape, an invalid call returns a structured error,
and nothing but JSON-RPC ever appears on STDOUT.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _frame(payload: dict) -> bytes:
    # mcp's stdio transport is newline-delimited JSON.
    return (json.dumps(payload) + "\n").encode("utf-8")


def run_stdio_session(messages: list[dict], env_extra: dict[str, str], timeout: int = 60):
    """Spawn the real server, send messages, return (parsed stdout lines, stderr)."""
    env = os.environ.copy()
    env.update(env_extra)
    process = subprocess.Popen(
        [sys.executable, "-m", "media_context_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    stdin_data = b"".join(_frame(message) for message in messages)
    try:
        stdout, stderr = process.communicate(stdin_data, timeout=timeout)
    finally:
        if process.poll() is None:
            process.kill()

    parsed = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        # Every STDOUT line MUST be valid JSON-RPC; anything else is corruption.
        parsed.append(json.loads(line))
    return parsed, stderr.decode("utf-8", errors="replace")


INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
LIST_TOOLS = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


@pytest.fixture(scope="module")
def server_env(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("mcp-root")
    (root / "hello.txt").write_text("hello from mcp test\nsecond line\n", encoding="utf-8")
    return {
        "MEDIA_MCP_ALLOWED_ROOTS": str(root),
        "MEDIA_MCP_CACHE_DIR": str(tmp_path_factory.mktemp("mcp-cache")),
        "MEDIA_MCP_LOG_LEVEL": "INFO",
        # ensure no ambient vision config bleeds into the test
        "MEDIA_MCP_VISION_BASE_URL": "",
        "MEDIA_MCP_VISION_API_KEY": "",
        "MEDIA_MCP_VISION_MODEL": "",
    }


def test_initialize_and_tool_discovery(server_env):
    replies, stderr = run_stdio_session([INITIALIZE, INITIALIZED, LIST_TOOLS], server_env)
    by_id = {reply.get("id"): reply for reply in replies if "id" in reply}

    assert 1 in by_id, f"no initialize reply; stderr:\n{stderr[-2000:]}"
    assert by_id[1]["result"]["serverInfo"]["name"] == "media-context-mcp"

    tools = by_id[2]["result"]["tools"]
    names = [tool["name"] for tool in tools]
    assert names == ["analyze_media"]
    schema = tools[0].get("inputSchema") or tools[0].get("input_schema")
    assert "path" in schema["properties"]
    assert "question" in schema["properties"]


def test_valid_call_succeeds_and_stdout_is_pure(server_env):
    call = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "analyze_media",
            "arguments": {"path": "hello.txt", "question": None},
        },
    }
    replies, stderr = run_stdio_session([INITIALIZE, INITIALIZED, call], server_env)
    by_id = {reply.get("id"): reply for reply in replies if "id" in reply}
    result = by_id[3]["result"]
    assert not result.get("isError", False)

    structured = result.get("structuredContent") or result.get("structured_content")
    assert structured["success"] is True
    assert structured["processing"]["processor"] == "text"
    assert "hello from mcp test" in structured["markdown"]
    assert structured["cache_key"]
    # logs went to stderr, not stdout (stdout purity is asserted by the JSON
    # parse inside run_stdio_session)
    assert "media-context-mcp starting" in stderr


def test_invalid_call_returns_structured_error(server_env):
    call = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "analyze_media",
            "arguments": {"path": "../../outside.txt"},
        },
    }
    replies, _ = run_stdio_session([INITIALIZE, INITIALIZED, call], server_env)
    by_id = {reply.get("id"): reply for reply in replies if "id" in reply}
    result = by_id[4]["result"]
    structured = result.get("structuredContent") or result.get("structured_content")
    assert structured["success"] is False
    assert structured["error"]["code"] == "PATH_NOT_ALLOWED"
    assert "hint" in structured["error"]


def test_vision_mode_without_provider_is_honest_over_mcp(server_env):
    call = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "analyze_media",
            "arguments": {"path": "hello.txt", "mode": "vision"},
        },
    }
    replies, _ = run_stdio_session([INITIALIZE, INITIALIZED, call], server_env)
    by_id = {reply.get("id"): reply for reply in replies if "id" in reply}
    structured = (by_id[5]["result"].get("structuredContent")
                  or by_id[5]["result"].get("structured_content"))
    assert structured["success"] is False
    # honest failure: mode=vision on a txt is MODE_NOT_APPLICABLE (vision doesn't
    # apply to text categories) -- importantly, NOT a fabricated success
    assert structured["error"]["code"] in {"MODE_NOT_APPLICABLE", "VISION_NOT_CONFIGURED"}
