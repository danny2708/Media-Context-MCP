"""Optional real-provider smoke test.

Never runs in the default suite: it requires the ``requires_vision`` marker to be
selected explicitly AND real credentials in the environment.

    pytest -m requires_vision

Environment needed:
    MEDIA_MCP_VISION_BASE_URL (or MEDIA_MCP_VISION_PROVIDER=huggingface)
    MEDIA_MCP_VISION_API_KEY
    MEDIA_MCP_VISION_MODEL
    MEDIA_MCP_ALLOW_CLOUD_VISION=true
"""

from __future__ import annotations

import os

import pytest

from conftest import make_pipeline, make_settings
from media_context_mcp.models import AnalyzeMediaRequest
from media_context_mcp.providers.openai_compatible import build_vision_provider
from media_context_mcp.providers.tesseract import build_ocr_backend

pytestmark = pytest.mark.requires_vision

_HAVE_CREDENTIALS = bool(
    os.environ.get("MEDIA_MCP_VISION_API_KEY")
    and os.environ.get("MEDIA_MCP_VISION_MODEL")
    and (
        os.environ.get("MEDIA_MCP_VISION_BASE_URL")
        or os.environ.get("MEDIA_MCP_VISION_PROVIDER") == "huggingface"
    )
    and os.environ.get("MEDIA_MCP_ALLOW_CLOUD_VISION", "").lower() == "true"
)


@pytest.mark.skipif(not _HAVE_CREDENTIALS, reason="live vision credentials not configured")
async def test_live_terminal_screenshot(fixtures, fixtures_root, tmp_path):
    from media_context_mcp.config import Settings

    env_settings = Settings()  # reads the real environment
    settings = make_settings(
        fixtures_root,
        tmp_path / "cache",
        vision_provider=env_settings.vision_provider,
        vision_base_url=env_settings.vision_base_url,
        vision_api_key=env_settings.vision_api_key,
        vision_model=env_settings.vision_model,
        allow_cloud_vision=True,
    )
    ocr = build_ocr_backend("tesseract", None)
    pipeline = make_pipeline(
        settings,
        ocr=ocr if ocr.availability()[0] else None,
        vision=build_vision_provider(settings),
    )
    try:
        result = await pipeline.analyze(
            AnalyzeMediaRequest(
                path=str(fixtures["terminal_png"]),
                question="Read the visible error message and identify the likely cause.",
                max_chars=30_000,
            )
        )
    finally:
        await pipeline.aclose()

    assert result.success
    # the fixture's error code must have been read from the image
    assert "TS2345" in result.markdown, (
        "live model failed to read the error code; output:\n" + result.markdown[:2000]
    )
    assert result.processing.model
