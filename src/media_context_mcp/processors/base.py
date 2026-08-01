"""Processor interface and the context handed to every processor.

Processors receive their backends through :class:`ProcessingContext` rather than
importing them. That is the dependency-injection seam the tests use: a fake vision
provider and a fake OCR backend make the whole pipeline deterministic and offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import Logger
from typing import Protocol, runtime_checkable

from ..config import Settings
from ..models import AnalyzeMediaRequest, MediaInfo, ProcessorResult
from ..providers.base import OcrBackend, VisionProvider
from ..routing.heuristics import RoutingDecision


@dataclass
class ProcessingContext:
    """Everything a processor may use, and nothing more."""

    settings: Settings
    info: MediaInfo
    decision: RoutingDecision
    logger: Logger
    request_id: str
    vision: VisionProvider | None = None
    ocr: OcrBackend | None = None
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        """Record a caller-visible warning. Duplicates are collapsed."""
        if message not in self.warnings:
            self.warnings.append(message)


@runtime_checkable
class MediaProcessor(Protocol):
    """Contract every processor implements."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str:
        """Bump this whenever output for identical input would change.

        The version is part of the cache key, so bumping it invalidates exactly the
        entries this processor produced and nothing else.
        """
        ...

    def supports(self, info: MediaInfo) -> bool: ...

    async def process(
        self,
        request: AnalyzeMediaRequest,
        context: ProcessingContext,
    ) -> ProcessorResult: ...
