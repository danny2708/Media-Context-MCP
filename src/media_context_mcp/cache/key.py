"""Cache key derivation.

The key answers one question: *would this request, run again, produce the same
result?* Anything that can change the output is part of the key; anything that
cannot is left out so it does not fragment the cache.

Included: file content hash (never the path -- the same bytes at two paths are the
same document), mode, detail, normalised page selection, normalised question,
processor name+version, prompt version, OCR backend/version/languages, whether OCR
may be sent to the cloud, requested provider/model/route, image-preprocessing
parameters, PDF density threshold and render DPI, and the vision output-token cap.

Deliberately excluded: ``max_chars`` -- truncation is applied when reading the
cache, so callers with different output budgets share one entry; the API key --
secrets must never be hashed into something that lands on disk; and the resolved
file path.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

from ..config import Settings
from ..models import AnalyzeMediaRequest

# Bump when the key layout itself changes; makes every old entry an automatic miss.
KEY_SCHEMA_VERSION = "2"


def normalise_question(question: str | None) -> str:
    """Collapse trivial variation so equivalent questions share a cache entry."""
    if not question:
        return ""
    text = unicodedata.normalize("NFC", question)
    return " ".join(text.lower().split())


def normalise_pages(pages: str | None) -> str:
    """Canonical form of a page expression: '1,3-5 ' and '1, 3-5' must match."""
    if not pages:
        return ""
    return ",".join(part.strip() for part in pages.split(",") if part.strip())


def build_cache_key(
    *,
    sha256: str,
    request: AnalyzeMediaRequest,
    processor: str,
    processor_version: str,
    prompt_version: str,
    settings: Settings,
    ocr_backend: str,
    ocr_version: str,
) -> str:
    """Deterministic key over everything that shapes the output."""
    material: dict[str, Any] = {
        "schema": KEY_SCHEMA_VERSION,
        "sha256": sha256,
        "mode": request.mode,
        "vision_profile": request.vision_profile,
        "detail": request.detail,
        "pages": normalise_pages(request.pages),
        "question": hashlib.sha256(
            normalise_question(request.question).encode("utf-8")
        ).hexdigest(),
        "processor": processor,
        "processor_version": processor_version,
        "prompt_version": prompt_version,
        # OCR configuration
        "ocr_backend": ocr_backend,
        "ocr_version": ocr_version,
        "ocr_languages": settings.ocr_languages,
        "send_ocr_to_cloud": settings.send_ocr_to_cloud,
        # Vision configuration -- requested identity only; never the key.
        "vision_provider": settings.vision_provider,
        "vision_base_url": settings.effective_vision_base_url,
        "vision_model": settings.effective_vision_model,
        "vision_route": settings.vision_route,
        "vision_max_output_tokens": settings.vision_max_output_tokens,
        "allow_cloud_vision": settings.allow_cloud_vision,
        # Routing policy knobs that change which backend sees the data.
        "auto_image_strategy": settings.auto_image_strategy,
        "pdf_min_chars_per_page": settings.pdf_min_chars_per_page,
        # Image processing parameters.
        "pdf_render_dpi": settings.pdf_render_dpi,
        "vision_max_dimension": settings.vision_max_dimension,
        "vision_max_image_bytes": settings.vision_max_image_bytes,
        "vision_image_format": settings.vision_image_format,
        "max_image_pixels": settings.max_image_pixels,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
