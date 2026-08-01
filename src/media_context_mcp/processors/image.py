"""Image processor: executes the routing layer's image plan.

Three plans exist (see ``routing.heuristics``):

* ``ocr_only``   -- local OCR, guaranteed no network.
* ``vision``     -- vision provider; OCR runs first as supporting context when a
                    backend is available.
* ``hybrid``     -- OCR first; a clearly text-extraction-oriented request with
                    sufficient OCR quality is answered locally, everything else
                    escalates to vision with the OCR text attached as an
                    *untrusted candidate transcription*.

The cloud gate is enforced before any provider call: this processor only receives
a vision provider object at all when the pipeline has already verified
``MEDIA_MCP_ALLOW_CLOUD_VISION=true``. OCR text rides along to the provider only
when ``MEDIA_MCP_SEND_OCR_TO_CLOUD`` permits.
"""

from __future__ import annotations

import time

from ..errors import OcrFailedError, OcrNotConfiguredError
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
    VisionAnalysis,
    build_vision_prompt,
    parse_vision_reply,
    select_profile,
)
from ..providers.base import OcrResult, VisionImage, VisionProviderResult
from ..routing.heuristics import (
    INTENT_TEXT_EXTRACTION,
    PLAN_OCR_ONLY,
    PLAN_VISION,
    ocr_quality_sufficient,
    vision_unusable_error,
)
from .base import ProcessingContext
from .imaging import PreprocessConfig, open_image, prepare_for_ocr, prepare_for_vision
from .ocr import OCR_SCOPE_NOTE, fenced, ocr_evidence, ocr_summary


class ImageProcessor:
    """Handles every standalone raster image."""

    name = "image"
    version = "2.0.0"

    def supports(self, info: MediaInfo) -> bool:
        return info.category is MediaCategory.IMAGE

    async def process(
        self,
        request: AnalyzeMediaRequest,
        context: ProcessingContext,
    ) -> ProcessorResult:
        if request.pages:
            context.warn("The 'pages' parameter does not apply to images and was ignored.")

        settings = context.settings
        plan = context.decision.image_plan
        intent = context.decision.intent

        preprocess = PreprocessConfig(
            max_pixels=settings.max_image_pixels,
            max_dimension=settings.vision_max_dimension,
            max_bytes=settings.vision_max_image_bytes,
            image_format=settings.vision_image_format,
        )

        image = open_image(context.info.path, settings.max_image_pixels)
        try:
            # --- OCR pass (all plans except pure-vision-without-backend) -----
            ocr_result: OcrResult | None = None
            ocr_payload: VisionImage | None = None
            if plan == PLAN_OCR_ONLY or context.ocr is not None:
                ocr_payload = prepare_for_ocr(image, preprocess, label=context.info.name)
                if plan == PLAN_OCR_ONLY:
                    if context.ocr is None:
                        raise OcrNotConfiguredError(
                            "No OCR backend is available.",
                            hint="Set MEDIA_MCP_OCR_BACKEND=tesseract and install the "
                            "tesseract binary.",
                        )
                    ocr_result = await context.ocr.recognise(
                        ocr_payload, settings.ocr_languages
                    )
                elif context.ocr is not None:
                    # Supporting context for vision/hybrid: an OCR failure here is
                    # a downgrade, not a request failure.
                    try:
                        ocr_result = await context.ocr.recognise(
                            ocr_payload, settings.ocr_languages
                        )
                    except (OcrFailedError, OcrNotConfiguredError) as exc:
                        context.warn(
                            f"OCR supporting pass failed ({exc.message}); continuing "
                            "with vision only."
                        )

            # --- resolve the plan into an outcome ----------------------------
            if plan == PLAN_OCR_ONLY:
                assert ocr_result is not None and ocr_payload is not None
                return self._ocr_only_result(ocr_result, ocr_payload, request)

            if plan != PLAN_VISION and ocr_result is not None:
                # hybrid: text-extraction intent + good OCR answers locally.
                text_length = len(ocr_result.text.strip())
                if intent == INTENT_TEXT_EXTRACTION and ocr_quality_sufficient(
                    text_length, ocr_result.mean_confidence
                ):
                    assert ocr_payload is not None
                    result = self._ocr_only_result(ocr_result, ocr_payload, request)
                    result.fallbacks_used.append(
                        "hybrid plan resolved with OCR alone: the question asks for "
                        "exact text and OCR quality was sufficient "
                        f"({text_length} chars"
                        + (
                            f", {ocr_result.mean_confidence:.0%} confidence)"
                            if ocr_result.mean_confidence is not None
                            else ")"
                        )
                    )
                    return result

            # --- vision pass --------------------------------------------------
            if context.vision is None:
                if ocr_result is not None:
                    # Vision needed but unusable; the router already warned. Give
                    # the OCR reading rather than nothing, clearly labelled.
                    assert ocr_payload is not None
                    result = self._ocr_only_result(ocr_result, ocr_payload, request)
                    result.warnings.append(
                        "Visual semantic analysis was skipped: "
                        + str(vision_unusable_error_message(context))
                    )
                    result.fallbacks_used.append(
                        "vision unavailable; degraded to OCR-only output"
                    )
                    return result
                raise vision_unusable_error_typed(context)

            vision_payloads, notes = prepare_for_vision(
                image, preprocess, label=context.info.name
            )
            for note in notes:
                context.warn(note)

            ocr_candidate: str | None = None
            ocr_sent = False
            if ocr_result is not None and ocr_result.text.strip():
                if settings.send_ocr_to_cloud:
                    ocr_candidate = ocr_result.text
                    ocr_sent = True
                else:
                    context.warn(
                        "OCR text was extracted locally but NOT sent to the vision "
                        "provider (MEDIA_MCP_SEND_OCR_TO_CLOUD=false); it is included "
                        "in the evidence below."
                    )

            profile = select_profile(request.question)
            prompt = build_vision_prompt(
                profile,
                question=request.question,
                detail=request.detail,
                label=context.info.name,
                ocr_candidate=ocr_candidate,
            )

            started = time.perf_counter()
            provider_result = await context.vision.analyze(
                images=vision_payloads,
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_output_tokens=settings.vision_max_output_tokens,
                request_id=context.request_id,
            )
            vision_ms = int((time.perf_counter() - started) * 1000)

            return self._vision_result(
                provider_result,
                vision_payloads,
                ocr_result,
                ocr_sent=ocr_sent,
                request=request,
                profile_key=profile.key,
                vision_ms=vision_ms,
            )
        finally:
            image.close()

    # ------------------------------------------------------------------ OCR --

    def _ocr_only_result(
        self,
        result: OcrResult,
        payload: VisionImage,
        request: AnalyzeMediaRequest,
    ) -> ProcessorResult:
        """Raw OCR output in the standard shape, unmodified and honestly scoped."""
        text = result.text.rstrip()
        warnings = list(result.warnings)
        warnings.append(OCR_SCOPE_NOTE)

        if not text.strip():
            warnings.append(
                "OCR found no readable text in this image. If the image is a UI, chart "
                "or diagram rather than a page of text, call analyze_media again with "
                "mode='vision' (a vision provider and cloud opt-in are required)."
            )

        content = fenced(text) if text.strip() else "_(no text recognised)_"

        if request.question:
            warnings.append(
                "A question was supplied, but OCR cannot answer questions -- it only "
                "transcribes characters. The transcription above is the raw evidence; "
                "interpret it yourself, or use mode='vision' for an answered reading."
            )

        return ProcessorResult(
            processor=self.name,
            processor_version=self.version,
            model=f"{result.engine}-{result.engine_version}",
            summary=ocr_summary(result, payload),
            content_markdown=content,
            evidence=[ocr_evidence(result, payload)],
            warnings=warnings,
            extra={
                "plan": "ocr_only",
                "languages": result.languages,
                "mean_confidence": result.mean_confidence,
                "char_count": len(text),
                "ocr_used": True,
                "ocr_sent_to_cloud": False,
            },
        )

    # --------------------------------------------------------------- vision --

    def _vision_result(
        self,
        provider_result: VisionProviderResult,
        payloads: list[VisionImage],
        ocr_result: OcrResult | None,
        *,
        ocr_sent: bool,
        request: AnalyzeMediaRequest,
        profile_key: str,
        vision_ms: int,
    ) -> ProcessorResult:
        analysis: VisionAnalysis = parse_vision_reply(provider_result.content)
        warnings = list(provider_result.warnings)

        if not analysis.structured:
            warnings.append(
                "The vision model did not follow the requested section layout; its "
                "reply is included as-is. Observation, interpretation and inference "
                "may be mixed together in it."
            )

        evidence: list[EvidenceItem] = []
        if ocr_result is not None and ocr_result.text.strip():
            # OCR evidence comes first: it is the exact-text ground truth. A vision
            # claim about an exact string that conflicts with OCR must not win
            # silently -- both are present, differently typed, for the caller.
            evidence.append(ocr_evidence(ocr_result, payloads[0]))
            warnings.extend(ocr_result.warnings)
        for item in analysis.exact_text:
            evidence.append(
                EvidenceItem(
                    type=EvidenceType.VISUAL,
                    location=payloads[0].label or "image",
                    content=item[:1500],
                )
            )
        for item in analysis.inferences:
            evidence.append(
                EvidenceItem(type=EvidenceType.INFERENCE, location=None, content=item[:800])
            )

        content = render_vision_markdown(analysis, ocr_result if not ocr_sent else None)

        served_by = provider_result.actual_model or provider_result.requested_model
        summary = analysis.summary or (
            f"Visual analysis of a {payloads[0].width}x{payloads[0].height} image "
            f"by {served_by}."
        )

        extra: dict[str, object] = {
            "plan": "vision",
            "profile": profile_key,
            "prompt_version": PROMPT_VERSION,
            "provider": provider_result.provider,
            "requested_model": provider_result.requested_model,
            "actual_model": provider_result.actual_model,
            "provider_route": provider_result.provider_route,
            "finish_reason": provider_result.finish_reason,
            "input_tokens": provider_result.input_tokens,
            "output_tokens": provider_result.output_tokens,
            "vision_duration_ms": vision_ms,
            "tiles": len(payloads),
            "ocr_used": ocr_result is not None,
            "ocr_sent_to_cloud": ocr_sent,
            "structured_reply": analysis.structured,
        }

        return ProcessorResult(
            processor=self.name,
            processor_version=self.version,
            model=served_by,
            summary=summary,
            answer=analysis.answer,
            content_markdown=content,
            evidence=evidence,
            warnings=warnings,
            extra=extra,
        )


def render_vision_markdown(
    analysis: VisionAnalysis, unsent_ocr: OcrResult | None
) -> str:
    """Render the parsed analysis back into clearly-labelled Markdown."""
    parts: list[str] = []

    if analysis.answer:
        parts.append(f"### Answer\n\n{analysis.answer}")
    if analysis.direct_observations:
        parts.append(
            "### Direct observations\n\n"
            + "\n".join(f"- {item}" for item in analysis.direct_observations)
        )
    if analysis.exact_text:
        joined = "\n\n".join(analysis.exact_text)
        parts.append(f"### Exact text\n\n{joined}")
    if analysis.visual_interpretation:
        parts.append(
            "### Visual interpretation\n\n"
            + "\n".join(f"- {item}" for item in analysis.visual_interpretation)
        )
    if analysis.inferences:
        parts.append(
            "### Inference (beyond what is literally visible)\n\n"
            + "\n".join(f"- {item}" for item in analysis.inferences)
        )
    if analysis.uncertainties:
        parts.append(
            "### Uncertain\n\n" + "\n".join(f"- {item}" for item in analysis.uncertainties)
        )
    if analysis.raw_text and not parts:
        parts.append(analysis.raw_text)

    if unsent_ocr is not None and unsent_ocr.text.strip():
        parts.append(
            "### Local OCR transcription (not shown to the vision model)\n\n"
            + fenced(unsent_ocr.text.rstrip())
        )

    return "\n\n".join(parts) if parts else "_(the vision model returned no usable content)_"


# Helper shims so the code above reads cleanly ---------------------------------


def vision_unusable_error_typed(context: ProcessingContext):
    from ..routing.heuristics import Capabilities

    caps = Capabilities(
        vision_configured=context.settings.vision_configured,
        cloud_vision_allowed=context.settings.allow_cloud_vision,
        ocr_available=context.ocr is not None,
    )
    return vision_unusable_error(caps)


def vision_unusable_error_message(context: ProcessingContext) -> str:
    error = vision_unusable_error_typed(context)
    return getattr(error, "message", str(error))
