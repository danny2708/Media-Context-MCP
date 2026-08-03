"""Integration tests verifying runtime execution path wiring for UI Spatial Perception."""

import pytest
from PIL import Image

from media_context_mcp.models import AnalyzeMediaRequest, VisionProfile
from media_context_mcp.processors.image import ImageProcessor
from media_context_mcp.processors.imaging import PreprocessConfig, prepare_for_vision_multipass
from media_context_mcp.prompts import select_profile
from media_context_mcp.providers.base import VisionImage
from media_context_mcp.providers.openai_compatible import OpenAICompatibleVisionProvider


def test_explicit_vision_profile_overrides_question_heuristic():
    """Verify explicit_profile parameter forces selecting the targeted profile over question heuristic."""
    # Generic question without UI keywords
    generic_question = "What is in this image?"

    # Without explicit profile -> falls back to 'general'
    profile_default = select_profile(generic_question)
    assert profile_default.key == "general"

    # With explicit profile -> forces 'ui_alignment'
    profile_explicit = select_profile(generic_question, explicit_profile="ui_alignment")
    assert profile_explicit.key == "ui_alignment"

    # With explicit profile -> forces 'ui_grounding'
    profile_grounding = select_profile(generic_question, explicit_profile="ui_grounding")
    assert profile_grounding.key == "ui_grounding"


def test_multipass_long_screenshot_populates_spatial_metadata():
    """Verify prepare_for_vision_multipass attaches role, sequence_index, and coordinates."""
    # Create a 1000x5000px vertical screenshot
    img = Image.new("RGB", (1000, 5000), color="white")
    config = PreprocessConfig(max_pixels=25_000_000, max_dimension=2048)

    payloads, notes = prepare_for_vision_multipass(img, config, label="test_long")
    assert len(payloads) >= 3  # 1 overview pass + 2 detail tiles
    assert "Multi-pass long image processing" in notes[0]

    overview = payloads[0]
    assert overview.role == "overview"
    assert overview.sequence_index == 1
    assert overview.source_x == 0
    assert overview.source_y == 0
    assert overview.source_width == 1000
    assert overview.source_height == 5000

    tile1 = payloads[1]
    assert tile1.role == "detail"
    assert tile1.sequence_index == 2
    assert tile1.source_x == 0
    assert tile1.source_y == 0
    assert tile1.source_width == 1000
    assert tile1.source_height == 2048


def test_openai_compatible_provider_builds_interleaved_tile_manifests():
    """Verify build_payload interleaves textual manifest descriptions before image URLs."""
    provider = OpenAICompatibleVisionProvider(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o",
    )

    overview_img = VisionImage(
        data=b"fake-bytes-1",
        mime_type="image/png",
        width=400,
        height=2000,
        role="overview",
        sequence_index=1,
        source_x=0,
        source_y=0,
        source_width=1000,
        source_height=5000,
        original_width=1000,
        original_height=5000,
    )

    tile_img = VisionImage(
        data=b"fake-bytes-2",
        mime_type="image/png",
        width=1000,
        height=2048,
        role="detail",
        sequence_index=2,
        source_x=0,
        source_y=0,
        source_width=1000,
        source_height=2048,
        original_width=1000,
        original_height=5000,
    )

    payload = provider.build_payload(
        images=[overview_img, tile_img],
        prompt="Analyze UI layout",
        system=None,
        max_output_tokens=1024,
    )

    user_content = payload["messages"][0]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "Analyze UI layout"

    # Interleaved manifest for Image 1 (Overview)
    assert user_content[1]["type"] == "text"
    assert "Image 1 of 2: Downscaled global overview pass of the full 1000x5000 screenshot" in user_content[1]["text"]
    assert user_content[2]["type"] == "image_url"

    # Interleaved manifest for Image 2 (Detail Tile)
    assert user_content[3]["type"] == "text"
    assert "Image 2 of 2: Native-resolution detail tile covering x=0-1000, y=0-2048" in user_content[3]["text"]
    assert user_content[4]["type"] == "image_url"
