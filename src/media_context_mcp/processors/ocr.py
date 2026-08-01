"""OCR processor for standalone images.

This processor answers exactly one question: *what characters are in this image?*
It does not describe layout, state or colour, and the result says so, because an
agent that mistakes an OCR dump for a description of a UI will draw confident wrong
conclusions from it.
"""

from __future__ import annotations

from ..errors import OcrNotConfiguredError
from ..models import (
    AnalyzeMediaRequest,
    EvidenceItem,
    EvidenceType,
    MediaCategory,
    MediaInfo,
    ProcessorResult,
)
from ..providers.base import ImageInput, OcrResult
from .base import ProcessingContext
from .imaging import open_image, prepare_for_ocr

OCR_SCOPE_NOTE = (
    "This is optical character recognition output: the characters found in the image. "
    "It carries no information about layout, component state, colour, icons or spatial "
    "relationships. Do not treat it as a description of what the image looks like."
)


class OcrProcessor:
    """Extracts text from an image with the configured OCR backend."""

    name = "ocr"
    version = "1.0.0"

    def supports(self, info: MediaInfo) -> bool:
        return info.category is MediaCategory.IMAGE

    async def process(
        self,
        request: AnalyzeMediaRequest,
        context: ProcessingContext,
    ) -> ProcessorResult:
        backend = context.ocr
        if backend is None:
            raise OcrNotConfiguredError(
                "No OCR backend is available.",
                hint="Set MEDIA_MCP_OCR_BACKEND=tesseract and install the tesseract binary.",
            )

        if request.pages:
            context.warn("The 'pages' parameter does not apply to images and was ignored.")

        image = open_image(context.info.path, context.settings.max_image_pixels)
        try:
            payload = prepare_for_ocr(image, label=context.info.name)
        finally:
            image.close()

        result = await backend.recognise(payload, context.settings.ocr_languages)
        return build_ocr_result(
            processor=self.name,
            version=self.version,
            result=result,
            payload=payload,
            request=request,
            note=OCR_SCOPE_NOTE,
        )


def build_ocr_result(
    *,
    processor: str,
    version: str,
    result: OcrResult,
    payload: ImageInput,
    request: AnalyzeMediaRequest,
    note: str,
) -> ProcessorResult:
    """Wrap raw OCR output in the standard result shape, unmodified."""
    text = result.text.rstrip()
    warnings = list(result.warnings)
    warnings.append(note)

    if not text.strip():
        warnings.append(
            "OCR found no readable text in this image. If the image is a UI, chart or "
            "diagram rather than a page of text, call analyze_media again with "
            "mode='vision' (a vision provider must be configured)."
        )

    # A fence keeps indentation and blank lines intact; without it a Markdown renderer
    # would collapse the very whitespace that makes a code screenshot readable.
    fence = "```"
    while fence in text:
        fence += "`"
    content = f"{fence}text\n{text}\n{fence}" if text.strip() else "_(no text recognised)_"

    confidence_note = (
        f", mean engine confidence {result.mean_confidence:.0%}"
        if result.mean_confidence is not None
        else ", confidence not reported by the engine"
    )
    summary = (
        f"OCR ({result.engine} {result.engine_version}, languages {result.languages}) "
        f"recovered {len(text):,} characters from a {payload.width}x{payload.height} "
        f"image{confidence_note}."
    )

    evidence = [
        EvidenceItem(
            type=EvidenceType.OCR,
            location=payload.label or "image",
            content=text[:1500] if text.strip() else "(no text recognised)",
            confidence=result.mean_confidence,
        )
    ]

    if request.question:
        warnings.append(
            "A question was supplied, but OCR cannot answer questions -- it only "
            "transcribes characters. The transcription above is the raw evidence; "
            "interpret it yourself, or use mode='vision' for an answered reading."
        )

    return ProcessorResult(
        processor=processor,
        processor_version=version,
        model=f"{result.engine}-{result.engine_version}",
        summary=summary,
        content_markdown=content,
        evidence=evidence,
        warnings=warnings,
        extra={
            "languages": result.languages,
            "mean_confidence": result.mean_confidence,
            "char_count": len(text),
        },
    )
