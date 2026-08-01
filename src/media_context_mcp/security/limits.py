"""Resource limits.

These guard the process against inputs that are cheap to send and expensive to
process: decompression bombs, 500-megapixel PNGs, thousand-page PDFs, and
conversions that never terminate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

from PIL import Image

from ..errors import (
    FileTooLargeError,
    ImageDecodeFailedError,
    InvalidPageSelectionError,
    ProcessingTimeoutError,
)

T = TypeVar("T")

# Pillow's own bomb guard raises DecompressionBombError past this; we set it from
# configuration so the limit is one number the operator controls.
_DEFAULT_PIXEL_LIMIT = 40_000_000


def enforce_file_size(path: Path, max_bytes: int) -> int:
    """Reject oversized files before anything reads or decodes them."""
    size = path.stat().st_size
    if size > max_bytes:
        raise FileTooLargeError(
            f"File is {size / 1_048_576:.1f} MB which exceeds the "
            f"{max_bytes / 1_048_576:.0f} MB limit.",
            hint=(
                "Raise MEDIA_MCP_MAX_FILE_MB if this file is genuinely needed, or split "
                "the document and analyse a page range with the 'pages' parameter."
            ),
            details={"size_bytes": size, "max_bytes": max_bytes},
        )
    return size


def apply_pillow_guard(max_pixels: int | None) -> None:
    """Point Pillow's decompression-bomb guard at our configured limit."""
    Image.MAX_IMAGE_PIXELS = max_pixels or _DEFAULT_PIXEL_LIMIT


def enforce_image_pixels(width: int, height: int, max_pixels: int) -> None:
    if width <= 0 or height <= 0:
        raise ImageDecodeFailedError(
            f"Image reports a non-positive size ({width}x{height}).",
            hint="The file is probably corrupt; try re-exporting it.",
        )
    pixels = width * height
    if pixels > max_pixels:
        raise FileTooLargeError(
            f"Image is {width}x{height} ({pixels:,} pixels), above the "
            f"{max_pixels:,} pixel limit.",
            hint=(
                "Downscale the image before analysis, or raise "
                "MEDIA_MCP_MAX_IMAGE_PIXELS if you trust the source."
            ),
            details={"width": width, "height": height, "max_pixels": max_pixels},
        )


def parse_page_selection(expression: str | None, total_pages: int, max_pages: int) -> list[int]:
    """Parse a 1-based page expression into a sorted list of 1-based page numbers.

    Accepts ``"3"``, ``"1-5"``, ``"1,3,7"`` and ``"1-3,8-10"``. Whitespace is
    ignored. Returns every page (capped at ``max_pages``) when ``expression`` is
    ``None``.
    """
    if total_pages <= 0:
        raise InvalidPageSelectionError(
            "The document reports zero pages.",
            hint="The file may be empty or corrupt.",
        )

    if expression is None or not expression.strip():
        selected = list(range(1, total_pages + 1))
        return selected[:max_pages]

    pages: set[int] = set()
    for chunk in expression.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, _, end_text = token.partition("-")
            start_text, end_text = start_text.strip(), end_text.strip()
            if not start_text.isdigit() or not end_text.isdigit():
                raise InvalidPageSelectionError(
                    f"Malformed page range '{token}'.",
                    hint="Use forms like '1', '1-5', '1,3,7' or '1-3,8-10' with 1-based numbers.",
                )
            start, end = int(start_text), int(end_text)
            if start > end:
                raise InvalidPageSelectionError(
                    f"Page range '{token}' runs backwards.",
                    hint="Write ranges low-to-high, e.g. '2-6'.",
                )
            pages.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise InvalidPageSelectionError(
                    f"'{token}' is not a page number.",
                    hint="Use forms like '1', '1-5', '1,3,7' or '1-3,8-10' with 1-based numbers.",
                )
            pages.add(int(token))

    if not pages:
        raise InvalidPageSelectionError(
            f"Page selection '{expression}' selected nothing.",
            hint="Provide at least one page, e.g. pages='1'.",
        )

    if min(pages) < 1:
        raise InvalidPageSelectionError(
            "Page numbers are 1-based; 0 and negatives are invalid.",
            hint="The first page is page 1.",
        )

    out_of_range = sorted(page for page in pages if page > total_pages)
    if out_of_range:
        raise InvalidPageSelectionError(
            f"Pages {out_of_range} do not exist; the document has {total_pages} page(s).",
            hint=f"Select pages between 1 and {total_pages}.",
            details={"total_pages": total_pages, "out_of_range": out_of_range},
        )

    return sorted(pages)[:max_pages]


async def run_with_timeout(awaitable: Awaitable[T], seconds: float, what: str) -> T:
    """Await with a hard deadline, converting expiry into a stable error code.

    ``asyncio.timeout`` cancels the inner task, so a well-behaved processor stops
    working rather than continuing in the background after we have given up.
    """
    try:
        async with asyncio.timeout(seconds):
            return await awaitable
    except TimeoutError as exc:
        raise ProcessingTimeoutError(
            f"{what} exceeded the {seconds:.0f}s time limit.",
            hint=(
                "Retry with a narrower 'pages' selection or detail='compact', or raise "
                "MEDIA_MCP_PROCESS_TIMEOUT_SECONDS."
            ),
            details={"timeout_seconds": seconds},
        ) from exc
