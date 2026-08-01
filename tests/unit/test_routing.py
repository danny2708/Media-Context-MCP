"""Routing heuristics: intent classification and the mode/category matrix."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from media_context_mcp.errors import (
    CloudVisionDisabledError,
    ModeNotApplicableError,
    OcrNotConfiguredError,
    VisionNotConfiguredError,
)
from media_context_mcp.models import MediaCategory, MediaInfo
from media_context_mcp.routing.heuristics import (
    INTENT_TEXT_EXTRACTION,
    INTENT_UNKNOWN,
    INTENT_VISUAL_SEMANTICS,
    PLAN_HYBRID,
    PLAN_OCR_ONLY,
    PLAN_VISION,
    Capabilities,
    classify_intent,
    decide_route,
    ocr_quality_sufficient,
    page_needs_fallback,
)


def info_for(category: MediaCategory, extension: str = ".png") -> MediaInfo:
    return MediaInfo(
        path=Path("x" + extension),
        name="x" + extension,
        extension=extension,
        mime_type="application/octet-stream",
        category=category,
        size_bytes=10,
        modified_at=datetime.now(tz=UTC),
        sha256="0" * 64,
    )


FULL = Capabilities(vision_configured=True, cloud_vision_allowed=True, ocr_available=True)
OCR_ONLY = Capabilities(vision_configured=False, cloud_vision_allowed=False, ocr_available=True)
VISION_ONLY = Capabilities(vision_configured=True, cloud_vision_allowed=True, ocr_available=False)
GATED = Capabilities(vision_configured=True, cloud_vision_allowed=False, ocr_available=True)
NOTHING = Capabilities(vision_configured=False, cloud_vision_allowed=False, ocr_available=False)


# ------------------------------------------------------------------- intent --


@pytest.mark.parametrize(
    "question",
    [
        "Read the exact error message.",
        "Extract the command shown in the terminal.",
        "Transcribe the code.",
        "extract this table",
    ],
)
def test_text_extraction_intent(question):
    assert classify_intent(question) == INTENT_TEXT_EXTRACTION


@pytest.mark.parametrize(
    "question",
    [
        "Why is this modal layout incorrect?",
        "Which UI component is selected?",
        "Explain this chart.",
        "Convert this flowchart into Mermaid.",
        "Compare these elements spatially.",
        "Identify the relationship between the diagram nodes.",
    ],
)
def test_visual_semantics_intent(question):
    assert classify_intent(question) == INTENT_VISUAL_SEMANTICS


def test_unknown_intent():
    assert classify_intent(None) == INTENT_UNKNOWN
    assert classify_intent("hmm") == INTENT_UNKNOWN


def test_mixed_question_prefers_visual():
    q = "Read the error text and explain why the layout broke."
    assert classify_intent(q) == INTENT_VISUAL_SEMANTICS


# ----------------------------------------------------------------- document --


def test_txt_routes_to_text():
    decision = decide_route(info_for(MediaCategory.TEXT, ".txt"), "auto", None, FULL)
    assert decision.processor == "text"


def test_docx_routes_to_markitdown():
    decision = decide_route(info_for(MediaCategory.OFFICE, ".docx"), "auto", None, FULL)
    assert decision.processor == "markitdown"


def test_pdf_routes_to_pdf_auto():
    decision = decide_route(info_for(MediaCategory.PDF, ".pdf"), "auto", None, FULL)
    assert decision.processor == "pdf"
    assert decision.pdf_strategy == "auto"


# -------------------------------------------------------------------- image --


def test_auto_image_hybrid_default():
    decision = decide_route(
        info_for(MediaCategory.IMAGE), "auto", "Which button is disabled?", FULL
    )
    assert decision.processor == "image"
    assert decision.image_plan == PLAN_HYBRID
    assert decision.intent == INTENT_VISUAL_SEMANTICS


def test_auto_image_without_vision_degrades_to_ocr_with_warning():
    decision = decide_route(
        info_for(MediaCategory.IMAGE), "auto", "Why does the layout look wrong?", OCR_ONLY
    )
    assert decision.image_plan == PLAN_OCR_ONLY
    assert decision.warnings  # explains the visual part cannot be answered


def test_auto_image_cloud_gated_behaves_like_no_vision():
    decision = decide_route(info_for(MediaCategory.IMAGE), "auto", None, GATED)
    assert decision.image_plan == PLAN_OCR_ONLY


def test_auto_image_vision_first_strategy():
    caps = Capabilities(
        vision_configured=True,
        cloud_vision_allowed=True,
        ocr_available=True,
        auto_image_strategy="vision_first",
    )
    decision = decide_route(info_for(MediaCategory.IMAGE), "auto", None, caps)
    assert decision.image_plan == PLAN_VISION


def test_auto_image_nothing_available_raises():
    with pytest.raises(VisionNotConfiguredError):
        decide_route(info_for(MediaCategory.IMAGE), "auto", None, NOTHING)


# -------------------------------------------------------------- forced modes --


def test_mode_ocr_never_plans_vision():
    decision = decide_route(info_for(MediaCategory.IMAGE), "ocr", "explain layout", FULL)
    assert decision.image_plan == PLAN_OCR_ONLY


def test_mode_ocr_without_backend_raises():
    with pytest.raises(OcrNotConfiguredError):
        decide_route(info_for(MediaCategory.IMAGE), "ocr", None, VISION_ONLY)


def test_mode_vision_without_config_raises():
    with pytest.raises(VisionNotConfiguredError):
        decide_route(info_for(MediaCategory.IMAGE), "vision", None, OCR_ONLY)


def test_mode_vision_cloud_disabled_raises_distinct_code():
    with pytest.raises(CloudVisionDisabledError):
        decide_route(info_for(MediaCategory.IMAGE), "vision", None, GATED)


def test_mode_document_on_image_raises():
    with pytest.raises(ModeNotApplicableError):
        decide_route(info_for(MediaCategory.IMAGE), "document", None, FULL)


def test_mode_ocr_on_docx_raises():
    with pytest.raises(ModeNotApplicableError):
        decide_route(info_for(MediaCategory.OFFICE, ".docx"), "ocr", None, FULL)


def test_forced_mode_overrides_auto():
    decision = decide_route(info_for(MediaCategory.PDF, ".pdf"), "vision", None, FULL)
    assert decision.pdf_strategy == "vision"


# ------------------------------------------------------------------ quality --


def test_page_needs_fallback_thresholds():
    assert page_needs_fallback(10, 120, 1.0)
    assert not page_needs_fallback(500, 120, 0.99)
    assert page_needs_fallback(500, 120, 0.30)  # plenty of chars, all garbage


def test_ocr_quality_sufficient():
    assert ocr_quality_sufficient(200, 0.9)
    assert ocr_quality_sufficient(200, None)  # engines that report nothing
    assert not ocr_quality_sufficient(10, 0.9)
    assert not ocr_quality_sufficient(200, 0.4)
