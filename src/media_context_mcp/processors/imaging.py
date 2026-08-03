"""Image preprocessing shared by the OCR, vision and PDF processors.

Everything here happens in memory -- no temporary files are written, so there is
nothing to protect on disk and nothing to clean up.

Rules that drive the choices:

* **Decode defensively.** Pixel limits are checked from the header before the
  image is rasterised, so a decompression bomb is rejected without ever being
  expanded in memory.
* **Prefer PNG for text.** Almost every image this server sees contains text --
  terminal output, code, UI labels, scanned pages. JPEG artefacts around glyph
  edges cost OCR accuracy and confuse vision models, so re-encoding is lossless
  unless the payload would exceed the configured byte budget
  (``MEDIA_MCP_VISION_IMAGE_FORMAT`` forces one or the other).
* **Downscale only when forced**, and record both the original and processed
  dimensions so the response can say what the model actually saw.
* **Tile instead of destroying small text.** A long screenshot downscaled to fit
  a provider limit becomes unreadable; cutting it into a few overlapping tiles
  keeps glyphs at native size. Tiling is bounded (``MAX_TILES``) and each tile
  carries its index and source region.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from ..errors import ImageDecodeFailedError
from ..providers.base import VisionImage
from ..security.limits import apply_pillow_guard, enforce_image_pixels

# Hard cap on tiles cut from one image: each tile is one more provider request.
MAX_TILES = 4

# Overlap between adjacent tiles so a line of text cut by a tile boundary is
# fully readable in at least one tile.
TILE_OVERLAP_PX = 96

# Aspect ratio beyond which an image is considered "long" (a full-page scroll
# screenshot) and tiling beats downscaling.
_TILING_ASPECT_THRESHOLD = 2.5


@dataclass(frozen=True)
class PreprocessConfig:
    """The knobs; mirrors the MEDIA_MCP_VISION_* image settings."""

    max_pixels: int
    max_dimension: int = 4096
    max_bytes: int = 10_485_760
    image_format: str = "auto"  # auto | png | jpeg


def open_image(path: Path, max_pixels: int) -> Image.Image:
    """Open and normalise an image file, enforcing the pixel budget first."""
    apply_pillow_guard(max_pixels)
    try:
        with Image.open(path) as probe:
            width, height = probe.size
            enforce_image_pixels(width, height, max_pixels)
            probe.load()
            # EXIF orientation is applied here so downstream code never has to think
            # about a phone photo that is secretly rotated.
            image = ImageOps.exif_transpose(probe)
            # Alpha and palette modes convert to RGB; L (grayscale) is kept, it is
            # already ideal for OCR and smaller to send.
            return image.convert("RGB") if image.mode not in {"RGB", "L"} else image.copy()
    except UnidentifiedImageError as exc:
        raise ImageDecodeFailedError(
            f"Not a decodable image: {path.name}",
            hint="The file may be corrupt or may not be the format its extension claims.",
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageDecodeFailedError(
            f"Image rejected by the decompression-bomb guard: {exc}",
            hint="Raise MEDIA_MCP_MAX_IMAGE_PIXELS only if you trust the source.",
        ) from exc
    except OSError as exc:
        raise ImageDecodeFailedError(
            f"Could not read image {path.name}: {exc}",
            hint="Check that the file is complete and readable.",
        ) from exc


def _encode(
    image: Image.Image,
    config: PreprocessConfig,
    *,
    label: str,
    source_page: int | None,
    tile_index: int | None,
    original_size: tuple[int, int],
) -> VisionImage:
    """Encode with the configured format policy.

    ``auto``: PNG first (lossless -- text stays crisp), high-quality JPEG only when
    the PNG would blow the byte budget. The MIME type is set from the encoding that
    actually happened, never from a file extension.
    """
    fmt = config.image_format
    data: bytes
    mime: str

    if fmt in {"auto", "png"}:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        data, mime = buffer.getvalue(), "image/png"
        if fmt == "auto" and len(data) > config.max_bytes:
            fmt = "jpeg"
    if fmt == "jpeg":
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
        data, mime = buffer.getvalue(), "image/jpeg"

    if len(data) > config.max_bytes:
        # Still too big: downscale by area until under budget. This is a last
        # resort; the tiler should normally have prevented it.
        scale = (config.max_bytes / len(data)) ** 0.5 * 0.95
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        shrunk = image.resize(new_size, Image.Resampling.LANCZOS)
        return _encode(
            shrunk,
            config,
            label=label,
            source_page=source_page,
            tile_index=tile_index,
            original_size=original_size,
        )

    return VisionImage(
        data=data,
        mime_type=mime,
        width=image.width,
        height=image.height,
        label=label,
        source_page=source_page,
        tile_index=tile_index,
        original_width=original_size[0],
        original_height=original_size[1],
    )


def prepare_for_ocr(
    image: Image.Image,
    config: PreprocessConfig,
    *,
    label: str = "",
    source_page: int | None = None,
) -> VisionImage:
    """Encode losslessly at native resolution.

    OCR accuracy tracks glyph height in pixels, so downscaling before OCR directly
    costs recognition quality. The byte budget does not apply -- OCR is local.
    """
    ocr_config = PreprocessConfig(
        max_pixels=config.max_pixels,
        max_dimension=config.max_dimension,
        max_bytes=2**31,  # effectively unlimited; the pixel guard already ran
        image_format="png",
    )
    return _encode(
        image,
        ocr_config,
        label=label,
        source_page=source_page,
        tile_index=None,
        original_size=image.size,
    )


def prepare_for_vision(
    image: Image.Image,
    config: PreprocessConfig,
    *,
    label: str = "",
    source_page: int | None = None,
) -> tuple[list[VisionImage], list[str]]:
    """Produce the payload(s) for a vision request. Returns ``(images, notes)``.

    Strategy, in order of preference:

    1. The image already fits ``max_dimension`` -> send it untouched.
    2. It is oversized but roughly page-shaped -> downscale, note the loss.
    3. It is oversized and *long* (tall scroll capture / wide panorama) -> cut into
       up to ``MAX_TILES`` overlapping tiles along the long axis at native
       resolution, so small text survives. Order and source regions are recorded
       on each tile.
    """
    notes: list[str] = []
    original_size = image.size
    longest = max(image.size)

    if longest <= config.max_dimension:
        return (
            [
                _encode(
                    image,
                    config,
                    label=label,
                    source_page=source_page,
                    tile_index=None,
                    original_size=original_size,
                )
            ],
            notes,
        )

    aspect = max(image.size) / max(1, min(image.size))
    if aspect >= _TILING_ASPECT_THRESHOLD:
        tiles = _tile_long_image(image, config, label=label, source_page=source_page)
        if tiles:
            notes.append(
                f"The image ({original_size[0]}x{original_size[1]}) exceeds the "
                f"{config.max_dimension}px limit and is strongly elongated; it was cut "
                f"into {len(tiles)} overlapping tiles at native resolution instead of "
                "being downscaled, to keep small text legible. Tiles are ordered "
                "top-to-bottom/left-to-right."
            )
            return tiles, notes

    scale = config.max_dimension / longest
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    notes.append(
        f"The image was downscaled from {original_size[0]}x{original_size[1]} to "
        f"{new_size[0]}x{new_size[1]} to fit the {config.max_dimension}px provider "
        "limit. Very small text may have become unreadable; if exact small text "
        "matters, use mode='ocr' (OCR runs at native resolution)."
    )
    return (
        [
            _encode(
                resized,
                config,
                label=label,
                source_page=source_page,
                tile_index=None,
                original_size=original_size,
            )
        ],
        notes,
    )


def prepare_for_vision_multipass(
    image: Image.Image,
    config: PreprocessConfig,
    *,
    label: str = "",
    source_page: int | None = None,
) -> tuple[list[VisionImage], list[str]]:
    """Produce multi-pass payloads for a long image: Overview Pass + Detailed Tile Passes."""
    notes: list[str] = []
    original_size = image.size
    aspect = max(image.size) / max(1, min(image.size))

    if aspect >= _TILING_ASPECT_THRESHOLD and max(image.size) > config.max_dimension:
        # Pass 1: Global low-res overview
        scale = config.max_dimension / max(image.size)
        overview_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        overview_img = image.resize(overview_size, Image.Resampling.LANCZOS)
        overview_encoded = _encode(
            overview_img,
            config,
            label=f"{label} [overview pass]" if label else "overview pass",
            source_page=source_page,
            tile_index=0,  # 0 indicates global overview
            original_size=original_size,
        )

        # Pass 2: Overlapping native detailed tiles
        tiles = _tile_long_image(image, config, label=label, source_page=source_page)
        if tiles:
            notes.append(
                f"Multi-pass long image processing: overview pass ({overview_size[0]}x{overview_size[1]}) "
                f"plus {len(tiles)} overlapping native tiles for detailed spatial perception."
            )
            return [overview_encoded] + tiles, notes

    return prepare_for_vision(image, config, label=label, source_page=source_page)


def _tile_long_image(
    image: Image.Image,
    config: PreprocessConfig,
    *,
    label: str,
    source_page: int | None,
) -> list[VisionImage]:
    """Cut a long image into <= MAX_TILES overlapping strips along its long axis.

    Returns ``[]`` when even MAX_TILES tiles cannot keep each strip within the
    dimension limit -- the caller then falls back to downscaling.
    """
    vertical = image.height >= image.width
    long_edge = image.height if vertical else image.width
    short_edge = image.width if vertical else image.height

    if short_edge > config.max_dimension:
        return []  # both axes oversized; strips would still be too big

    step = config.max_dimension - TILE_OVERLAP_PX
    import math

    tile_count = math.ceil((long_edge - TILE_OVERLAP_PX) / step)
    if tile_count > MAX_TILES:
        return []
    tile_count = max(1, tile_count)

    tiles: list[VisionImage] = []
    for index in range(tile_count):
        start = index * step
        end = min(start + config.max_dimension, long_edge)
        if vertical:
            box = (0, start, image.width, end)
            region = f"y={start}-{end}"
        else:
            box = (start, 0, end, image.height)
            region = f"x={start}-{end}"
        crop = image.crop(box)
        tile_label = f"{label} [tile {index + 1}/{tile_count}, {region}]" if label else (
            f"tile {index + 1}/{tile_count}, {region}"
        )
        tiles.append(
            _encode(
                crop,
                config,
                label=tile_label,
                source_page=source_page,
                tile_index=index + 1,
                original_size=image.size,
            )
        )
    return tiles
