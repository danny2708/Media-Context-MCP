"""Image preprocessing: orientation, limits, format policy, tiling."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from media_context_mcp.errors import FileTooLargeError, ImageDecodeFailedError
from media_context_mcp.processors.imaging import (
    MAX_TILES,
    PreprocessConfig,
    open_image,
    prepare_for_ocr,
    prepare_for_vision,
)
from media_context_mcp.security.limits import enforce_image_pixels

CONFIG = PreprocessConfig(max_pixels=40_000_000, max_dimension=1024, max_bytes=10_000_000)


def save(tmp_path: Path, image: Image.Image, name: str = "img.png", **kwargs) -> Path:
    path = tmp_path / name
    image.save(path, **kwargs)
    return path


def test_open_and_pixel_limit(tmp_path: Path):
    path = save(tmp_path, Image.new("RGB", (100, 50), (1, 2, 3)))
    image = open_image(path, max_pixels=40_000_000)
    assert image.size == (100, 50)
    with pytest.raises((FileTooLargeError, ImageDecodeFailedError)):
        open_image(path, max_pixels=100)  # 100x50 = 5000 > 100


def test_pixel_enforcement_math():
    with pytest.raises(FileTooLargeError):
        enforce_image_pixels(10_000, 10_000, 40_000_000)
    enforce_image_pixels(1000, 1000, 40_000_000)  # no raise


def test_not_an_image_rejected(tmp_path: Path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"this is not a png at all")
    with pytest.raises(ImageDecodeFailedError):
        open_image(path, max_pixels=40_000_000)


def test_exif_orientation_applied(tmp_path: Path):
    """A 100x40 JPEG with orientation=6 (90° CW) must open as 40x100."""
    image = Image.new("RGB", (100, 40), (200, 100, 50))
    exif = image.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW
    path = tmp_path / "rotated.jpg"
    image.save(path, format="JPEG", exif=exif)
    opened = open_image(path, max_pixels=40_000_000)
    assert opened.size == (40, 100)


def test_alpha_converted_safely(tmp_path: Path):
    path = save(tmp_path, Image.new("RGBA", (10, 10), (255, 0, 0, 128)))
    opened = open_image(path, max_pixels=40_000_000)
    assert opened.mode == "RGB"


def test_small_image_not_downscaled_and_stays_png():
    image = Image.new("RGB", (600, 400), (255, 255, 255))
    payloads, notes = prepare_for_vision(image, CONFIG)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload.mime_type == "image/png"
    assert (payload.width, payload.height) == (600, 400)
    assert not payload.was_downscaled
    assert notes == []


def test_oversized_pageish_image_downscaled_with_note():
    image = Image.new("RGB", (2000, 1500), (255, 255, 255))
    payloads, notes = prepare_for_vision(image, CONFIG)
    assert len(payloads) == 1
    assert max(payloads[0].width, payloads[0].height) == 1024
    assert payloads[0].original_width == 2000
    assert payloads[0].was_downscaled
    assert any("downscaled" in note for note in notes)


def test_long_screenshot_tiled_in_order():
    image = Image.new("RGB", (800, 3000), (255, 255, 255))
    payloads, notes = prepare_for_vision(image, CONFIG)
    assert 1 < len(payloads) <= MAX_TILES
    assert [p.tile_index for p in payloads] == list(range(1, len(payloads) + 1))
    # tiles keep native width -- no downscaling
    assert all(p.width == 800 for p in payloads)
    assert any("tiles" in note for note in notes)
    # source regions recorded in labels
    assert all("y=" in (p.label or "") for p in payloads)


def test_absurdly_long_image_falls_back_to_downscale():
    image = Image.new("RGB", (600, 30000), (255, 255, 255))
    payloads, notes = prepare_for_vision(image, CONFIG)
    assert len(payloads) == 1  # too many tiles would be needed; downscaled instead
    assert payloads[0].was_downscaled


def test_forced_jpeg_format():
    config = PreprocessConfig(
        max_pixels=40_000_000, max_dimension=1024, max_bytes=10_000_000, image_format="jpeg"
    )
    payloads, _ = prepare_for_vision(Image.new("RGB", (100, 100)), config)
    assert payloads[0].mime_type == "image/jpeg"


def test_ocr_payload_native_resolution_png():
    image = Image.new("RGB", (3000, 500), (0, 0, 0))
    payload = prepare_for_ocr(image, CONFIG)
    assert payload.mime_type == "image/png"
    assert (payload.width, payload.height) == (3000, 500)  # never downscaled for OCR


def test_byte_budget_forces_recompression():
    import random

    random.seed(7)
    noisy = Image.new("RGB", (900, 900))
    noisy.putdata(
        [(random.randrange(256), random.randrange(256), random.randrange(256))
         for _ in range(900 * 900)]
    )
    config = PreprocessConfig(
        max_pixels=40_000_000, max_dimension=1024, max_bytes=120_000, image_format="auto"
    )
    payloads, _ = prepare_for_vision(noisy, config)
    assert len(payloads[0].data) <= 120_000
