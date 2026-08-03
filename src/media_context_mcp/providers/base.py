"""Backend interfaces.

Both interfaces are deliberately narrow. A processor knows it is asking "read the
characters in this image" or "look at these images and answer this"; it knows
nothing about HTTP, vendors, or engine flags. That is what makes the whole
pipeline testable without a network, and what lets a local VLM (Ollama, vLLM) be
added later as just another :class:`VisionProvider`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VisionImage:
    """One preprocessed raster handed to a backend (vision or OCR).

    ``data`` holds the encoded bytes; ``mime_type`` describes what the *bytes*
    are (set from the actual encoding performed, never inferred from a file
    extension). ``source_page`` is the 1-based page for rasters that came out of
    a document; ``tile_index`` orders tiles cut from one oversized image.
    """

    data: bytes
    mime_type: str
    width: int
    height: int
    label: str = ""
    source_page: int | None = None
    tile_index: int | None = None
    role: str = "detail"  # overview | detail
    sequence_index: int = 0
    source_x: int | None = None
    source_y: int | None = None
    source_width: int | None = None
    source_height: int | None = None
    original_width: int | None = None
    original_height: int | None = None

    @property
    def was_downscaled(self) -> bool:
        return (
            self.original_width is not None
            and (self.original_width, self.original_height) != (self.width, self.height)
        )


@dataclass
class VisionProviderResult:
    """A vision backend's reply, before any interpretation by this server.

    ``requested_model`` is what configuration asked for; ``actual_model`` and
    ``provider_route`` are what the API reported serving the request, when it
    reports anything -- many OpenAI-compatible implementations do not, and absent
    metadata stays ``None`` rather than being copied from the request.
    """

    content: str
    provider: str
    requested_model: str
    actual_model: str | None = None
    provider_route: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class OcrResult:
    """Characters recovered from an image, with the engine's own confidence."""

    text: str
    engine: str
    engine_version: str
    languages: str
    mean_confidence: float | None = None
    """0..1, or ``None`` when the engine reported nothing. Never fabricated."""
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class VisionProvider(Protocol):
    """A multimodal model endpoint, local or cloud.

    Implementations must: respect cancellation (no retry after the task is
    cancelled), bound their retries, and never log credentials or image bytes.
    """

    @property
    def provider_name(self) -> str: ...

    @property
    def requested_model(self) -> str: ...

    async def analyze(
        self,
        *,
        images: list[VisionImage],
        prompt: str,
        system: str | None = None,
        max_output_tokens: int,
        request_id: str,
    ) -> VisionProviderResult: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class OcrBackend(Protocol):
    """A local optical character recognition engine."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def availability(self) -> tuple[bool, str | None]:
        """``(available, reason_if_not)``. Must not raise."""
        ...

    async def recognise(self, image: VisionImage, languages: str) -> OcrResult: ...
