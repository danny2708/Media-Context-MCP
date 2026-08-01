"""The analysis pipeline: validate -> detect -> route -> cache -> process -> render.

This module is deliberately independent of MCP. ``server.py`` adapts it to the
protocol and ``cli.py`` runs it headless -- which is what separates a media
processing failure from an MCP transport failure when debugging.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .cache import CacheStore, build_cache_key
from .config import Settings
from .errors import ConfigurationError, InvalidArgumentError
from .logging_setup import get_logger, log_event, new_request_id, set_request_id
from .media_info import detect_media, ensure_supported
from .models import (
    AnalyzeMediaRequest,
    AnalyzeMediaResult,
    ProcessorResult,
    RequestInfo,
)
from .processors.base import MediaProcessor, ProcessingContext
from .processors.image import ImageProcessor
from .processors.markitdown_adapter import MarkItDownProcessor
from .processors.pdf import PdfProcessor
from .processors.text import TextProcessor
from .prompts import PROMPT_VERSION
from .providers.base import OcrBackend, VisionProvider
from .providers.openai_compatible import build_vision_provider
from .providers.tesseract import build_ocr_backend
from .render import assemble_result
from .routing.heuristics import Capabilities, decide_route
from .security.limits import enforce_file_size, run_with_timeout
from .security.paths import resolve_media_path

_LOGGER = get_logger(__name__)


@dataclass
class Pipeline:
    """Holds the wired-up components. Build one per process via :func:`build_pipeline`."""

    settings: Settings
    cache: CacheStore
    processors: dict[str, MediaProcessor]
    ocr: OcrBackend | None
    vision: VisionProvider | None
    _cleanup_done: bool = False

    async def aclose(self) -> None:
        if self.vision is not None:
            await self.vision.aclose()

    # ------------------------------------------------------------------ run --

    async def analyze(self, request: AnalyzeMediaRequest) -> AnalyzeMediaResult:
        """Run one analysis. Raises MediaContextError subclasses on failure."""
        request_id = new_request_id()
        set_request_id(request_id)
        started = time.perf_counter()

        fatal = self.settings.fatal_problems()
        if fatal:
            problem = fatal[0]
            raise ConfigurationError(
                f"{problem.field}: {problem.message}",
                hint=problem.hint,
            )

        if request.max_chars < 1:
            raise InvalidArgumentError(
                "max_chars must be positive.",
                hint=f"Use a value between 500 and {self.settings.max_output_chars}.",
            )

        # 1. sandbox + existence + regular-file checks
        roots = self.settings.resolved_roots()
        path = resolve_media_path(request.path, roots)

        # 2. size gate before any read of the content
        enforce_file_size(path, self.settings.max_file_bytes)

        # 3. detection (includes the content hash)
        info = detect_media(path)
        ensure_supported(info)

        # 4. routing
        ocr_available, ocr_reason = (
            self.ocr.availability() if self.ocr is not None else (False, "backend disabled")
        )
        caps = Capabilities(
            vision_configured=self.settings.vision_configured,
            cloud_vision_allowed=self.settings.allow_cloud_vision,
            ocr_available=ocr_available,
            ocr_unavailable_reason=ocr_reason,
            auto_image_strategy=self.settings.auto_image_strategy,
        )
        decision = decide_route(info, request.mode, request.question, caps)
        processor = self.processors[decision.processor]

        # 5. cache lookup
        cache_key = build_cache_key(
            sha256=info.sha256,
            request=request,
            processor=processor.name,
            processor_version=processor.version,
            prompt_version=PROMPT_VERSION,
            settings=self.settings,
            ocr_backend=self.ocr.name if self.ocr else "none",
            ocr_version=self.ocr.version if self.ocr else "0",
        )

        cached_payload = None
        if not request.force_refresh:
            cached_payload = self.cache.get(cache_key)

        if cached_payload is not None:
            processor_result = ProcessorResult.model_validate(cached_payload)
            duration_ms = int((time.perf_counter() - started) * 1000)
            log_event(
                _LOGGER,
                logging.INFO,
                "analyze_media cache hit",
                processor=processor.name,
                cache="hit",
                duration_ms=duration_ms,
                media_type=info.category.value,
            )
            return assemble_result(
                processor_result=processor_result,
                source=info.to_source_info(),
                request_info=self._request_info(request),
                cached=True,
                duration_ms=duration_ms,
                cache_key=cache_key,
                extra_warnings=list(decision.warnings),
            )

        # 6. process, under the global timeout. The cloud gate is enforced here:
        # the processor only ever receives a vision provider when the operator
        # both configured one and opted in to cloud vision.
        vision = self.vision if self.settings.cloud_vision_usable else None
        context = ProcessingContext(
            settings=self.settings,
            info=info,
            decision=decision,
            logger=_LOGGER,
            request_id=request_id,
            vision=vision,
            ocr=self.ocr if ocr_available else None,
        )
        processor_result = await run_with_timeout(
            processor.process(request, context),
            self.settings.process_timeout_seconds,
            f"Processing {info.name} with the {processor.name} processor",
        )
        # Routing warnings + processing warnings both reach the caller.
        processor_result.warnings = [*context.warnings, *processor_result.warnings]

        # 7. store -- successes only; every failure path above raised.
        self.cache.put(cache_key, processor_result.model_dump(mode="json"))
        self._occasional_cleanup()

        duration_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            _LOGGER,
            logging.INFO,
            "analyze_media processed",
            processor=processor.name,
            processor_version=processor.version,
            cache="miss",
            duration_ms=duration_ms,
            media_type=info.category.value,
            routing_reason=decision.reason,
            fallbacks=processor_result.fallbacks_used,
        )

        return assemble_result(
            processor_result=processor_result,
            source=info.to_source_info(),
            request_info=self._request_info(request),
            cached=False,
            duration_ms=duration_ms,
            cache_key=cache_key,
            extra_warnings=list(decision.warnings),
        )

    def _request_info(self, request: AnalyzeMediaRequest) -> RequestInfo:
        return RequestInfo(
            question=request.question,
            mode=request.mode,
            pages=request.pages,
            detail=request.detail,
            max_chars=request.max_chars,
        )

    def _occasional_cleanup(self) -> None:
        """One cleanup pass per process lifetime, after the first write."""
        if not self._cleanup_done:
            self._cleanup_done = True
            stats = self.cache.cleanup()
            if stats["expired"] or stats["evicted"]:
                log_event(
                    _LOGGER,
                    logging.INFO,
                    "cache cleanup",
                    **stats,
                )


def build_pipeline(settings: Settings) -> Pipeline:
    """Wire the default components. Tests build their own with fakes injected."""
    ocr = build_ocr_backend(settings.ocr_backend, settings.tesseract_cmd)
    vision = build_vision_provider(settings)
    cache = CacheStore(
        settings.resolved_cache_dir(),
        enabled=settings.cache_enabled,
        max_bytes=settings.cache_max_bytes,
        ttl_days=settings.cache_ttl_days,
    )
    processors: dict[str, MediaProcessor] = {
        "text": TextProcessor(),
        "markitdown": MarkItDownProcessor(),
        "pdf": PdfProcessor(),
        "image": ImageProcessor(),
    }
    return Pipeline(
        settings=settings,
        cache=cache,
        processors=processors,
        ocr=ocr,
        vision=vision,
    )
