"""PDF processor built on PyMuPDF.

MarkItDown's PDF converter returns one flat string with no page boundaries, which
makes the ``pages`` parameter and per-page evidence impossible through it. PyMuPDF
gives page-addressed text extraction *and* page rendering for the scanned-page
fallback, so PDFs are handled here and MarkItDown keeps the Office formats.

Per-page strategy (``pdf_strategy`` from routing):

* ``text``   -- embedded text layer only, no fallback (mode='document').
* ``ocr``    -- rasterise and OCR every selected page, no cloud (mode='ocr').
* ``vision`` -- rasterise and visually analyse every selected page (mode='vision').
* ``auto``   -- text layer first; a page failing the density check falls back to
  OCR when available, then to vision when the question needs semantic reading and
  cloud vision is permitted. Every fallback is reported.

Pages are always selected *before* any rendering, and rendering happens one page
at a time with bounded concurrency -- a 500-page scan can never balloon into 500
parallel rasters or one giant multi-image provider request. Vision calls are one
request per page, merged afterwards with page-level evidence retained.
"""

from __future__ import annotations

import asyncio
import io

import fitz  # PyMuPDF
from PIL import Image

from ..errors import DocumentConversionFailedError
from ..models import (
    AnalyzeMediaRequest,
    EvidenceItem,
    EvidenceType,
    MediaCategory,
    MediaInfo,
    ProcessorResult,
)
from ..prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_vision_prompt,
    parse_vision_reply,
    select_profile,
)
from ..providers.base import OcrResult, VisionImage
from ..routing.heuristics import (
    INTENT_VISUAL_SEMANTICS,
    STRATEGY_OCR,
    STRATEGY_TEXT,
    STRATEGY_VISION,
    page_needs_fallback,
    printable_ratio,
)
from ..security.limits import parse_page_selection
from .base import ProcessingContext
from .imaging import PreprocessConfig, prepare_for_ocr, prepare_for_vision
from .image import render_vision_markdown
from .ocr import fenced, ocr_evidence

# At most this many pages are rasterised/processed concurrently.
_PAGE_CONCURRENCY = 2


class PdfProcessor:
    """Page-addressed PDF extraction with honest scanned-page fallback."""

    name = "pdf"
    version = f"2.0.0+pymupdf-{fitz.pymupdf_version}"

    def supports(self, info: MediaInfo) -> bool:
        return info.category is MediaCategory.PDF

    async def process(
        self,
        request: AnalyzeMediaRequest,
        context: ProcessingContext,
    ) -> ProcessorResult:
        # PyMuPDF is synchronous; opening and text extraction run on a worker
        # thread, then the async part (OCR/vision fallbacks) resumes on the loop.
        settings = context.settings
        strategy = context.decision.pdf_strategy
        path = str(context.info.path)

        def _open() -> tuple[int, list[tuple[int, str]]]:
            try:
                document = fitz.open(path)
            except Exception as exc:  # noqa: BLE001 - fitz raises plain RuntimeError
                raise DocumentConversionFailedError(
                    f"PyMuPDF could not open the PDF: {exc}",
                    hint="The file may be corrupt, encrypted, or not actually a PDF.",
                ) from exc
            try:
                if document.needs_pass:
                    raise DocumentConversionFailedError(
                        "The PDF is password-protected.",
                        hint="Decrypt the file first; password input is not supported.",
                    )
                total = document.page_count
                selected = parse_page_selection(request.pages, total, settings.max_pages)
                texts = [
                    (number, document.load_page(number - 1).get_text("text"))
                    for number in selected
                ]
                return total, texts
            finally:
                document.close()

        total_pages, page_texts = await asyncio.to_thread(_open)
        selected = [number for number, _ in page_texts]

        warnings: list[str] = []
        if request.pages is None and total_pages > len(selected):
            warnings.append(
                f"The document has {total_pages} pages; only the first {len(selected)} "
                f"were processed (MEDIA_MCP_MAX_PAGES={settings.max_pages}). Use the "
                "'pages' parameter to reach later pages."
            )

        # Classify each selected page.
        weak_pages: list[int] = []
        good_pages: dict[int, str] = {}
        for number, text in page_texts:
            stripped = text.strip()
            if strategy == STRATEGY_TEXT or not page_needs_fallback(
                len(stripped),
                settings.pdf_min_chars_per_page,
                printable_ratio(stripped),
            ):
                good_pages[number] = text
            else:
                weak_pages.append(number)

        if strategy in {STRATEGY_OCR, STRATEGY_VISION}:
            # Forced rasterisation of every selected page.
            weak_pages = selected
            good_pages = {}

        fallbacks_used: list[str] = []
        page_sections: dict[int, str] = {}
        evidence: list[EvidenceItem] = []
        model_used: str | None = None
        extra: dict[str, object] = {}

        for number in sorted(good_pages):
            text = good_pages[number].rstrip()
            page_sections[number] = f"### Page {number}\n\n{text}" if text else (
                f"### Page {number}\n\n_(blank page)_"
            )
            evidence.append(
                EvidenceItem(
                    type=EvidenceType.PAGE,
                    location=f"page {number}",
                    content=(text[:400] + " ...") if len(text) > 400 else text or "(blank)",
                )
            )

        if weak_pages:
            (
                fallback_sections,
                fallback_evidence,
                fallback_warnings,
                fallback_notes,
                model_used,
                extra,
            ) = await self._process_weak_pages(
                weak_pages, strategy, request, context
            )
            page_sections.update(fallback_sections)
            evidence.extend(fallback_evidence)
            warnings.extend(fallback_warnings)
            fallbacks_used.extend(fallback_notes)

        ordered = [page_sections[number] for number in sorted(page_sections)]
        content = "\n\n".join(ordered)

        text_page_count = len(good_pages)
        summary_bits = [
            f"PDF `{context.info.name}`: {total_pages} page(s), "
            f"{len(selected)} selected"
        ]
        if text_page_count:
            summary_bits.append(f"{text_page_count} read from the embedded text layer")
        if weak_pages:
            summary_bits.append(
                f"{len(weak_pages)} below the text-density threshold "
                f"({settings.pdf_min_chars_per_page} chars/page) handled by fallback"
            )
        summary = "; ".join(summary_bits) + "."

        extra.update(
            {
                "total_pages": total_pages,
                "selected_pages": selected,
                "text_layer_pages": sorted(good_pages),
                "fallback_pages": weak_pages,
                "strategy": strategy,
            }
        )

        return ProcessorResult(
            processor=self.name,
            processor_version=self.version,
            model=model_used,
            summary=summary,
            content_markdown=content,
            evidence=evidence,
            warnings=warnings,
            fallbacks_used=fallbacks_used,
            extra=extra,
        )

    # ------------------------------------------------------------ fallbacks --

    async def _process_weak_pages(
        self,
        pages: list[int],
        strategy: str,
        request: AnalyzeMediaRequest,
        context: ProcessingContext,
    ) -> tuple[dict[int, str], list[EvidenceItem], list[str], list[str], str | None, dict]:
        settings = context.settings
        preprocess = PreprocessConfig(
            max_pixels=settings.max_image_pixels,
            max_dimension=settings.vision_max_dimension,
            max_bytes=settings.vision_max_image_bytes,
            image_format=settings.vision_image_format,
        )

        use_vision = strategy == STRATEGY_VISION or (
            strategy != STRATEGY_OCR
            and context.vision is not None
            and (
                context.ocr is None
                or context.decision.intent == INTENT_VISUAL_SEMANTICS
            )
        )
        use_ocr = not use_vision and context.ocr is not None

        sections: dict[int, str] = {}
        evidence: list[EvidenceItem] = []
        warnings: list[str] = []
        notes: list[str] = []
        model_used: str | None = None
        extra: dict[str, object] = {}

        if not use_vision and not use_ocr:
            for number in pages:
                sections[number] = (
                    f"### Page {number}\n\n"
                    "_(this page has no usable embedded text and no OCR or vision "
                    "backend is available to read it)_"
                )
            warnings.append(
                f"Page(s) {pages} appear scanned or image-only, and neither OCR nor a "
                "permitted vision provider is available. Their content is NOT included. "
                "Install tesseract for local OCR, or configure a vision provider and "
                "set MEDIA_MCP_ALLOW_CLOUD_VISION=true."
            )
            notes.append("scanned pages skipped: no OCR or vision backend")
            return sections, evidence, warnings, notes, model_used, extra

        dpi = settings.pdf_render_dpi
        path = str(context.info.path)

        def _render(number: int) -> Image.Image:
            document = fitz.open(path)
            try:
                page = document.load_page(number - 1)
                pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
                return Image.open(io.BytesIO(pixmap.tobytes("png")))
            finally:
                document.close()

        semaphore = asyncio.Semaphore(_PAGE_CONCURRENCY)

        async def _handle(number: int) -> tuple[int, str, list[EvidenceItem], list[str]]:
            async with semaphore:
                pil = await asyncio.to_thread(_render, number)
                try:
                    if use_vision:
                        return await self._vision_page(
                            number, pil, preprocess, request, context
                        )
                    return await self._ocr_page(number, pil, preprocess, context)
                finally:
                    pil.close()

        results = await asyncio.gather(*(_handle(number) for number in pages))

        ocr_sent_any = (
            use_vision and context.ocr is not None and settings.send_ocr_to_cloud
        )
        for number, section, page_evidence, page_warnings in sorted(results):
            sections[number] = section
            evidence.extend(page_evidence)
            warnings.extend(w for w in page_warnings if w not in warnings)

        if use_vision:
            notes.append(
                f"pages {pages} rendered at {dpi} DPI and analysed by the vision "
                "provider (one request per page)"
            )
            model_used = context.vision.requested_model if context.vision else None
            extra = {
                "prompt_version": PROMPT_VERSION,
                "ocr_sent_to_cloud": ocr_sent_any,
                "provider": context.vision.provider_name if context.vision else None,
            }
        else:
            notes.append(f"pages {pages} rendered at {dpi} DPI and read by local OCR")
            if strategy != STRATEGY_OCR and context.vision is None:
                warnings.append(
                    "Scanned pages were read with local OCR only. OCR extracts "
                    "characters; layout, tables and figures on these pages were not "
                    "interpreted. Configure a vision provider and set "
                    "MEDIA_MCP_ALLOW_CLOUD_VISION=true for semantic reading."
                )
        return sections, evidence, warnings, notes, model_used, extra

    async def _ocr_page(
        self,
        number: int,
        pil: Image.Image,
        preprocess: PreprocessConfig,
        context: ProcessingContext,
    ) -> tuple[int, str, list[EvidenceItem], list[str]]:
        assert context.ocr is not None
        payload = prepare_for_ocr(
            pil, preprocess, label=f"page {number}", source_page=number
        )
        result: OcrResult = await context.ocr.recognise(
            payload, context.settings.ocr_languages
        )
        text = result.text.rstrip()
        section = (
            f"### Page {number} (OCR)\n\n" + (fenced(text) if text else "_(no text recognised)_")
        )
        return number, section, [ocr_evidence(result, payload)], list(result.warnings)

    async def _vision_page(
        self,
        number: int,
        pil: Image.Image,
        preprocess: PreprocessConfig,
        request: AnalyzeMediaRequest,
        context: ProcessingContext,
    ) -> tuple[int, str, list[EvidenceItem], list[str]]:
        assert context.vision is not None
        settings = context.settings
        warnings: list[str] = []

        payloads, notes = prepare_for_vision(
            pil, preprocess, label=f"page {number}", source_page=number
        )
        warnings.extend(notes)

        # OCR candidate for the page, when a backend exists and policy permits.
        ocr_candidate: str | None = None
        if context.ocr is not None and settings.send_ocr_to_cloud:
            try:
                ocr_payload = prepare_for_ocr(
                    pil, preprocess, label=f"page {number}", source_page=number
                )
                candidate = await context.ocr.recognise(
                    ocr_payload, settings.ocr_languages
                )
                if candidate.text.strip():
                    ocr_candidate = candidate.text
            except Exception:  # noqa: BLE001 - candidate is best-effort
                pass

        profile = select_profile(request.question, from_pdf_page=True)
        prompt = build_vision_prompt(
            profile,
            question=request.question,
            detail=request.detail,
            label=f"page {number} of {context.info.name}",
            ocr_candidate=ocr_candidate,
        )
        provider_result = await context.vision.analyze(
            images=payloads,
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_output_tokens=settings.vision_max_output_tokens,
            request_id=context.request_id,
        )
        warnings.extend(provider_result.warnings)

        analysis = parse_vision_reply(provider_result.content)
        body = render_vision_markdown(analysis, None)
        section = f"### Page {number} (visual analysis)\n\n{body}"

        evidence: list[EvidenceItem] = []
        for item in analysis.exact_text:
            evidence.append(
                EvidenceItem(
                    type=EvidenceType.VISUAL,
                    location=f"page {number}",
                    content=item[:1200],
                )
            )
        if not evidence:
            evidence.append(
                EvidenceItem(
                    type=EvidenceType.VISUAL,
                    location=f"page {number}",
                    content=(analysis.summary or provider_result.content[:400]),
                )
            )
        return number, section, evidence, warnings
