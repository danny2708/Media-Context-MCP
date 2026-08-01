"""OCR execution helpers shared by the image and PDF processors.

These functions answer exactly one question: *what characters are in this image?*
They never describe layout, state or colour, and the results say so, because an
agent that mistakes an OCR dump for a description of a UI will draw confident
wrong conclusions from it.
"""

from __future__ import annotations

from ..models import EvidenceItem, EvidenceType
from ..providers.base import OcrBackend, OcrResult, VisionImage

OCR_SCOPE_NOTE = (
    "This is optical character recognition output: the characters found in the image. "
    "It carries no information about layout, component state, colour, icons or spatial "
    "relationships. Do not treat it as a description of what the image looks like."
)


async def run_ocr(
    backend: OcrBackend,
    image: VisionImage,
    languages: str,
) -> OcrResult:
    """Run OCR on one image. Thin wrapper kept for symmetry and test seams."""
    return await backend.recognise(image, languages)


def fenced(text: str, language: str = "text") -> str:
    """Wrap text in a code fence that survives text containing backticks.

    A fence keeps indentation and blank lines intact; without it a Markdown
    renderer would collapse the very whitespace that makes a code screenshot
    readable.
    """
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}"


def ocr_evidence(result: OcrResult, image: VisionImage) -> EvidenceItem:
    text = result.text.rstrip()
    return EvidenceItem(
        type=EvidenceType.OCR,
        location=image.label or (f"page {image.source_page}" if image.source_page else "image"),
        content=text[:1500] if text.strip() else "(no text recognised)",
        confidence=result.mean_confidence,
    )


def ocr_summary(result: OcrResult, image: VisionImage) -> str:
    text = result.text.rstrip()
    confidence_note = (
        f", mean engine confidence {result.mean_confidence:.0%}"
        if result.mean_confidence is not None
        else ", confidence not reported by the engine"
    )
    return (
        f"OCR ({result.engine} {result.engine_version}, languages {result.languages}) "
        f"recovered {len(text):,} characters from a {image.width}x{image.height} "
        f"image{confidence_note}."
    )
