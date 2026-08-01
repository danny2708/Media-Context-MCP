"""Routing heuristics.

Kept as pure functions over plain data so that every rule is unit-testable without
touching the filesystem, a model, or a network. The chosen processor and the reason
for choosing it are both reported back to the caller -- an agent that disagrees with
the routing can override it with ``mode``.

OCR and vision are complementary, not alternatives:

* OCR extracts exact visible characters, locally, and is the preferred source for
  exact text (error messages, code, identifiers, paths, versions).
* A vision model understands layout, relationships, visual state, charts and
  diagrams -- and may hallucinate exact strings.

The image plan therefore has three shapes, selected by
``MEDIA_MCP_AUTO_IMAGE_STRATEGY`` plus a small deterministic intent classifier
over the caller's question:

* ``ocr_only`` -- run OCR, never touch the cloud.
* ``vision`` -- call the vision provider; OCR runs first as supporting context
  when available (and is sent along only if MEDIA_MCP_SEND_OCR_TO_CLOUD allows).
* ``hybrid`` -- run OCR; if the request is clearly text-extraction-oriented and
  OCR quality turns out sufficient, answer from OCR alone; otherwise escalate to
  vision with the OCR candidate attached. The quality half of that decision is
  runtime data, so it lives in the image processor -- routing only sets the plan.

Documented limitations
----------------------
* Intent classification is keyword-based. A question that describes a visual task
  in unusual words falls back to ``unknown`` intent, which in hybrid mode means
  vision is used when permitted -- the safe default, since vision degrades
  gracefully on text-only images while OCR cannot answer a visual question at all.
* The PDF text/scan split uses a characters-per-page threshold. It misclassifies
  legitimately sparse pages (a title page, a full-page figure) as scanned. The
  cost is a wasted OCR/vision pass, not a wrong answer, and every such decision
  appears in ``processing.fallbacks_used``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..errors import (
    CloudVisionDisabledError,
    ModeNotApplicableError,
    OcrNotConfiguredError,
    VisionNotConfiguredError,
)
from ..models import AnalysisMode, MediaCategory, MediaInfo

# Processor identifiers, also reported in ProcessingInfo.processor.
TEXT = "text"
MARKITDOWN = "markitdown"
PDF = "pdf"
IMAGE = "image"

# PDF sub-strategies, resolved per page inside the PDF processor.
STRATEGY_AUTO = "auto"
STRATEGY_TEXT = "text"
STRATEGY_OCR = "ocr"
STRATEGY_VISION = "vision"

# Image plans, executed by the image processor.
PLAN_OCR_ONLY = "ocr_only"
PLAN_VISION = "vision"
PLAN_HYBRID = "hybrid"

# Question intents.
INTENT_TEXT_EXTRACTION = "text_extraction"
INTENT_VISUAL_SEMANTICS = "visual_semantics"
INTENT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Capabilities:
    """What this deployment can actually do, resolved at request time."""

    vision_configured: bool
    cloud_vision_allowed: bool
    ocr_available: bool
    ocr_unavailable_reason: str | None = None
    auto_image_strategy: str = "hybrid"

    @property
    def vision_usable(self) -> bool:
        return self.vision_configured and self.cloud_vision_allowed


@dataclass(frozen=True)
class RoutingDecision:
    processor: str
    reason: str
    pdf_strategy: str = STRATEGY_AUTO
    image_plan: str = PLAN_HYBRID
    intent: str = INTENT_UNKNOWN
    warnings: list[str] = field(default_factory=list)


_MARKITDOWN_CATEGORIES = {
    MediaCategory.OFFICE,
    MediaCategory.HTML,
    MediaCategory.DATA,
    MediaCategory.EMAIL,
    MediaCategory.EBOOK,
    MediaCategory.NOTEBOOK,
}

# --- question intent ---------------------------------------------------------

# "Clearly text-extraction-oriented": read/extract/transcribe exact content.
_TEXT_EXTRACTION_RE = re.compile(
    r"\b(read|extract|transcribe|copy|quote|what (does|do|is) the .{0,30}"
    r"(say|text|message|output)|exact (text|error|message|command|output|code|value)|"
    r"error message|the command shown|word for word|verbatim)\b",
    re.IGNORECASE,
)

# Requires spatial/visual/semantic reasoning that characters alone cannot answer.
_VISUAL_SEMANTICS_RE = re.compile(
    r"\b(why|layout|align(ed|ment)?|spacing|overlap|position|spatial|arrang|"
    r"which .{0,30}(component|element|button|item|node) is|selected|highlighted|"
    r"disabled|enabled|active|focused|checked|"
    r"explain|interpret|describe the (chart|graph|diagram|ui|screen|flow)|"
    r"mermaid|flow ?chart|diagram|relationship|connect(ed|ion)|edge|arrow|"
    r"compare|difference|trend|look(s)? (like|right|wrong|off)|design|style|"
    r"color|colour|icon|state of)\b",
    re.IGNORECASE,
)


def classify_intent(question: str | None) -> str:
    """Deterministic keyword intent classifier. Transparent by design; not ML."""
    if not question or not question.strip():
        return INTENT_UNKNOWN
    # Visual wins when both match: "read the error AND tell me why the modal is
    # misaligned" needs vision; OCR output alone cannot cover it.
    if _VISUAL_SEMANTICS_RE.search(question):
        return INTENT_VISUAL_SEMANTICS
    if _TEXT_EXTRACTION_RE.search(question):
        return INTENT_TEXT_EXTRACTION
    return INTENT_UNKNOWN


# --- vision availability errors ---------------------------------------------


def vision_unusable_error(caps: Capabilities) -> Exception:
    """The precise reason vision cannot run right now, as a typed error."""
    if not caps.vision_configured:
        return VisionNotConfiguredError(
            "No vision provider is configured.",
            hint=(
                "Set MEDIA_MCP_VISION_MODEL and MEDIA_MCP_VISION_API_KEY (plus "
                "MEDIA_MCP_VISION_BASE_URL, or MEDIA_MCP_VISION_PROVIDER=huggingface "
                "for the router preset), then restart the client. Nothing was "
                "analysed visually."
            ),
        )
    return CloudVisionDisabledError(
        "Cloud vision is disabled by configuration (MEDIA_MCP_ALLOW_CLOUD_VISION=false).",
        hint=(
            "Visual semantic analysis was skipped because sending images to a remote "
            "provider requires explicit opt-in -- screenshots may contain secrets or "
            "proprietary code. Set MEDIA_MCP_ALLOW_CLOUD_VISION=true to permit it, or "
            "use mode='ocr' for local-only text extraction."
        ),
    )


def _ocr_unavailable_error(caps: Capabilities, context_hint: str) -> OcrNotConfiguredError:
    return OcrNotConfiguredError(
        "No OCR backend is available: "
        f"{caps.ocr_unavailable_reason or 'backend disabled'}.",
        hint=(
            "Install tesseract and ensure it is on PATH (or set "
            "MEDIA_MCP_TESSERACT_CMD), then set MEDIA_MCP_OCR_BACKEND=tesseract. "
            f"{context_hint} Run 'media-context-mcp doctor' to verify."
        ),
    )


# --- image routing -----------------------------------------------------------


def _auto_image_route(question: str | None, caps: Capabilities) -> RoutingDecision:
    intent = classify_intent(question)
    strategy = caps.auto_image_strategy

    if not caps.ocr_available and not caps.vision_usable:
        # Neither backend: fail with the more actionable of the two errors.
        raise vision_unusable_error(caps)

    if strategy == "ocr_first" or not caps.vision_usable:
        if caps.ocr_available:
            plan = PLAN_HYBRID if caps.vision_usable else PLAN_OCR_ONLY
            reason = (
                "Image routed OCR-first"
                + ("" if caps.vision_usable else " (vision unavailable)")
                + "; vision escalation "
                + ("permitted for visual questions." if caps.vision_usable else "not possible.")
            )
            warnings = []
            if not caps.vision_usable and intent == INTENT_VISUAL_SEMANTICS:
                warnings = [
                    "The question needs visual-semantic understanding (layout, state, "
                    "relationships), but only local OCR is available. The result below "
                    "is a character transcription and cannot answer the visual part. "
                    + (
                        "Set MEDIA_MCP_ALLOW_CLOUD_VISION=true to enable the configured "
                        "vision provider."
                        if caps.vision_configured
                        else "Configure a vision provider to get visual analysis."
                    )
                ]
            return RoutingDecision(
                processor=IMAGE,
                reason=reason,
                image_plan=plan,
                intent=intent,
                warnings=warnings,
            )
        return RoutingDecision(
            processor=IMAGE,
            reason="Image routed to vision: no OCR backend is available.",
            image_plan=PLAN_VISION,
            intent=intent,
        )

    if strategy == "vision_first":
        return RoutingDecision(
            processor=IMAGE,
            reason="Image routed to vision (MEDIA_MCP_AUTO_IMAGE_STRATEGY=vision_first); "
            "OCR runs first as supporting context when available.",
            image_plan=PLAN_VISION,
            intent=intent,
        )

    # hybrid (default): OCR always runs; a clearly text-extraction question with
    # good OCR is answered locally, everything else escalates to vision with the
    # OCR candidate attached.
    return RoutingDecision(
        processor=IMAGE,
        reason=(
            "Image routed hybrid: OCR runs locally first; the request "
            f"was classified as {intent.replace('_', '-')}, so "
            + (
                "OCR alone will be used if its quality is sufficient."
                if intent == INTENT_TEXT_EXTRACTION
                else "vision will interpret the image with the OCR text as an "
                "untrusted candidate transcription."
            )
        ),
        image_plan=PLAN_HYBRID,
        intent=intent,
    )


# --- top-level dispatch ------------------------------------------------------


def decide_route(
    info: MediaInfo,
    mode: AnalysisMode,
    question: str | None,
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
                raise _ocr_unavailable_error(caps, "mode='ocr' was requested explicitly.")
            return RoutingDecision(
                processor=IMAGE,
                reason="mode='ocr': local OCR only; no cloud call will be made.",
                image_plan=PLAN_OCR_ONLY,
                intent=classify_intent(question),
            )
        if category is MediaCategory.PDF:
            if not caps.ocr_available:
                raise _ocr_unavailable_error(
                    caps, "Or use mode='document' to read the PDF's embedded text layer."
                )
            return RoutingDecision(
                processor=PDF,
                reason="mode='ocr' forces rasterise-then-OCR for every selected page; "
                "no cloud call will be made.",
                pdf_strategy=STRATEGY_OCR,
            )
        raise ModeNotApplicableError(
            f"mode='ocr' does not apply to {category.value} files, which already "
            "contain machine-readable text.",
            hint="Use mode='auto' or mode='document' for this file type.",
            details={"category": category.value},
        )

    if mode == "vision":
        if not caps.vision_usable:
            raise vision_unusable_error(caps)
        if category is MediaCategory.IMAGE:
            return RoutingDecision(
                processor=IMAGE,
                reason="mode='vision' was requested explicitly; OCR runs first as "
                "supporting context when available.",
                image_plan=PLAN_VISION,
                intent=classify_intent(question),
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
        return _auto_image_route(question, caps)
    if category is MediaCategory.PDF:
        return RoutingDecision(
            processor=PDF,
            reason=(
                "PDF routed to text extraction with a per-page quality check; pages "
                "below the text-density threshold fall back to OCR, then vision when "
                "semantic understanding is required and permitted."
            ),
            pdf_strategy=STRATEGY_AUTO,
            intent=classify_intent(question),
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


# --- runtime quality checks (used by processors) -----------------------------


def page_needs_fallback(char_count: int, threshold: int, printable: float) -> bool:
    """Decide whether one PDF page's extracted text is good enough to trust.

    Two independent failure shapes are caught: too little text (a scanned page
    yields a handful of stray characters) and text that is mostly unprintable
    (a broken font encoding produces plenty of characters, all garbage).
    """
    if char_count < threshold:
        return True
    return printable < 0.60


def printable_ratio(text: str) -> float:
    """Share of characters that are printable or ordinary whitespace."""
    if not text:
        return 0.0
    good = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return good / len(text)


def ocr_quality_sufficient(char_count: int, mean_confidence: float | None) -> bool:
    """Is OCR output alone trustworthy enough to answer a text-extraction request?

    Requires both substance (enough characters that the image plausibly was text)
    and either decent engine confidence or no confidence data at all (some engines
    report none; absence of evidence is not treated as failure).
    """
    if char_count < 40:
        return False
    if mean_confidence is not None and mean_confidence < 0.65:
        return False
    return True
