"""Routing heuristics.

Kept as pure functions over plain data so that every rule is unit-testable without
touching the filesystem, a model, or a network. The chosen processor and the reason
for choosing it are both reported back to the caller -- an agent that disagrees with
the routing can override it with ``mode``.

Documented limitations
----------------------
* We do **not** inspect image content to decide between OCR and vision. Deciding
  "is this image only machine-readable text?" reliably requires decoding and
  analysing the image, which costs about as much as just running the processor.
  So plain images go to vision when a vision provider is configured (it degrades
  gracefully on text-only images; OCR does not degrade gracefully on a UI
  screenshot) and to OCR otherwise. §9 of the specification explicitly allows this.
* The PDF text/scan split uses a characters-per-page threshold. It misclassifies
  legitimately sparse pages (a title page, a full-page figure with a caption) as
  scanned. The cost is a wasted OCR/vision pass, not a wrong answer, and every
  such decision appears in ``processing.fallbacks_used``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import ModeNotApplicableError, OcrNotConfiguredError, VisionNotConfiguredError
from ..models import AnalysisMode, MediaCategory, MediaInfo

# Processor identifiers, also reported in ProcessingInfo.processor.
TEXT = "text"
MARKITDOWN = "markitdown"
PDF = "pdf"
OCR = "ocr"
VISION = "vision"

# PDF sub-strategies, resolved per page inside the PDF processor.
STRATEGY_AUTO = "auto"
STRATEGY_TEXT = "text"
STRATEGY_OCR = "ocr"
STRATEGY_VISION = "vision"


@dataclass(frozen=True)
class Capabilities:
    """What this deployment can actually do, resolved at request time."""

    vision_available: bool
    ocr_available: bool
    ocr_unavailable_reason: str | None = None
    image_default_mode: str = "vision"


@dataclass(frozen=True)
class RoutingDecision:
    processor: str
    reason: str
    pdf_strategy: str = STRATEGY_AUTO
    warnings: list[str] = field(default_factory=list)


_MARKITDOWN_CATEGORIES = {
    MediaCategory.OFFICE,
    MediaCategory.HTML,
    MediaCategory.DATA,
    MediaCategory.EMAIL,
    MediaCategory.EBOOK,
    MediaCategory.NOTEBOOK,
}


def _no_visual_backend_error(category: MediaCategory) -> Exception:
    return VisionNotConfiguredError(
        "This input needs a visual backend, but neither a vision provider nor a "
        "working OCR backend is available.",
        hint=(
            "Configure a vision provider (MEDIA_MCP_VISION_BASE_URL, "
            "MEDIA_MCP_VISION_MODEL, MEDIA_MCP_VISION_API_KEY) or install the "
            "tesseract binary and set MEDIA_MCP_OCR_BACKEND=tesseract. Run "
            "'media-context-mcp doctor' to see which one is missing."
        ),
        details={"category": category.value},
    )


def _image_route(caps: Capabilities) -> RoutingDecision:
    """Auto routing for a standalone image."""
    prefer_vision = caps.image_default_mode != "ocr"

    if prefer_vision and caps.vision_available:
        return RoutingDecision(
            processor=VISION,
            reason=(
                "Image routed to vision: OCR alone cannot describe layout, component "
                "state, colour, icons or spatial relationships."
            ),
        )
    if not prefer_vision and caps.ocr_available:
        return RoutingDecision(
            processor=OCR,
            reason="Image routed to OCR because MEDIA_MCP_IMAGE_DEFAULT_MODE=ocr.",
        )
    if caps.ocr_available:
        return RoutingDecision(
            processor=OCR,
            reason="Image routed to OCR because no vision provider is configured.",
            warnings=[
                "No vision provider is configured, so this image was processed with OCR "
                "only. OCR reads characters; it does not report layout, component state, "
                "colours, icons or spatial relationships. Treat the result as text "
                "extraction, not as a description of the image."
            ],
        )
    if caps.vision_available:
        return RoutingDecision(
            processor=VISION,
            reason="Image routed to vision because no OCR backend is available.",
        )
    raise _no_visual_backend_error(MediaCategory.IMAGE)


def decide_route(
    info: MediaInfo,
    mode: AnalysisMode,
    caps: Capabilities,
) -> RoutingDecision:
    """Choose a processor for ``info`` under ``mode``.

    Raises ``MODE_NOT_APPLICABLE`` when the caller forces a mode the input cannot
    satisfy, rather than silently doing something else.
    """
    category = info.category

    if mode == "document":
        if category is MediaCategory.IMAGE:
            raise ModeNotApplicableError(
                "mode='document' cannot process a standalone image: there is no "
                "embedded text layer to extract.",
                hint="Use mode='ocr' for text-heavy images or mode='vision' for "
                "screenshots, charts and diagrams.",
                details={"category": category.value},
            )
        if category is MediaCategory.PDF:
            return RoutingDecision(
                processor=PDF,
                reason="mode='document' forces the embedded PDF text layer with no fallback.",
                pdf_strategy=STRATEGY_TEXT,
            )
        return _document_route(info)

    if mode == "ocr":
        if category is MediaCategory.IMAGE:
            if not caps.ocr_available:
                raise OcrNotConfiguredError(
                    "mode='ocr' was requested but no OCR backend is available: "
                    f"{caps.ocr_unavailable_reason or 'backend disabled'}.",
                    hint=(
                        "Install tesseract and ensure it is on PATH (or set "
                        "MEDIA_MCP_TESSERACT_CMD), then set "
                        "MEDIA_MCP_OCR_BACKEND=tesseract. Run 'media-context-mcp doctor' "
                        "to verify."
                    ),
                )
            return RoutingDecision(processor=OCR, reason="mode='ocr' was requested explicitly.")
        if category is MediaCategory.PDF:
            if not caps.ocr_available:
                raise OcrNotConfiguredError(
                    "mode='ocr' was requested but no OCR backend is available: "
                    f"{caps.ocr_unavailable_reason or 'backend disabled'}.",
                    hint="Install tesseract, or use mode='document' to read the PDF's "
                    "embedded text layer instead.",
                )
            return RoutingDecision(
                processor=PDF,
                reason="mode='ocr' forces rasterise-then-OCR for every selected page.",
                pdf_strategy=STRATEGY_OCR,
            )
        raise ModeNotApplicableError(
            f"mode='ocr' does not apply to {category.value} files, which already "
            "contain machine-readable text.",
            hint="Use mode='auto' or mode='document' for this file type.",
            details={"category": category.value},
        )

    if mode == "vision":
        if not caps.vision_available:
            raise VisionNotConfiguredError(
                "mode='vision' was requested but no vision provider is configured.",
                hint=(
                    "Set MEDIA_MCP_VISION_BASE_URL, MEDIA_MCP_VISION_MODEL and "
                    "MEDIA_MCP_VISION_API_KEY in the MCP server's environment, then "
                    "restart the client. Nothing was analysed visually."
                ),
            )
        if category is MediaCategory.IMAGE:
            return RoutingDecision(
                processor=VISION, reason="mode='vision' was requested explicitly."
            )
        if category is MediaCategory.PDF:
            return RoutingDecision(
                processor=PDF,
                reason="mode='vision' forces page rendering and visual analysis.",
                pdf_strategy=STRATEGY_VISION,
            )
        raise ModeNotApplicableError(
            f"mode='vision' does not apply to {category.value} files; they are not "
            "rendered to pixels by this server.",
            hint="Export the file to PDF or an image if you need visual analysis of it.",
            details={"category": category.value},
        )

    # --- auto ---------------------------------------------------------------
    if category is MediaCategory.IMAGE:
        return _image_route(caps)
    if category is MediaCategory.PDF:
        return RoutingDecision(
            processor=PDF,
            reason=(
                "PDF routed to text extraction with a per-page quality check; pages "
                "below the text-density threshold fall back to OCR or vision."
            ),
            pdf_strategy=STRATEGY_AUTO,
        )
    return _document_route(info)


def _document_route(info: MediaInfo) -> RoutingDecision:
    if info.category is MediaCategory.TEXT:
        return RoutingDecision(
            processor=TEXT,
            reason="Plain-text file read directly; no conversion needed.",
        )
    if info.category in _MARKITDOWN_CATEGORIES:
        return RoutingDecision(
            processor=MARKITDOWN,
            reason=f"{info.extension or info.mime_type} converted to Markdown by MarkItDown.",
        )
    raise ModeNotApplicableError(
        f"No document processor handles {info.category.value} files.",
        hint="Convert the file to PDF, an Office format, or plain text.",
        details={"category": info.category.value},
    )


def page_needs_fallback(
    char_count: int,
    threshold: int,
    printable_ratio: float,
) -> bool:
    """Decide whether one PDF page's extracted text is good enough to trust.

    Two independent failure shapes are caught:

    * too little text -- a scanned page yields a handful of stray characters;
    * text that is mostly unprintable -- a broken or non-embedded font encoding
      produces plenty of characters, all of them garbage.
    """
    if char_count < threshold:
        return True
    return printable_ratio < 0.60


def printable_ratio(text: str) -> float:
    """Share of characters that are printable or ordinary whitespace."""
    if not text:
        return 0.0
    good = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return good / len(text)
