"""Public data contract for ``analyze_media``.

These models are the tool's schema. Renaming a field is a breaking change for any
agent that reads the structured content, so treat them as an API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AnalysisMode = Literal["auto", "document", "ocr", "vision"]
DetailLevel = Literal["compact", "normal", "full"]


class MediaCategory(str, Enum):
    """Coarse family of the input, decided before any processor runs."""

    IMAGE = "image"
    PDF = "pdf"
    OFFICE = "office"
    TEXT = "text"
    HTML = "html"
    DATA = "data"
    EMAIL = "email"
    EBOOK = "ebook"
    NOTEBOOK = "notebook"
    AUDIO = "audio"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"


class EvidenceType(str, Enum):
    """How a piece of evidence was obtained.

    This distinction is the point of the whole tool: an agent must be able to tell
    text that was read out of a file from text a model believed it saw in a picture.
    """

    TEXT = "text"
    PAGE = "page"
    SLIDE = "slide"
    SHEET = "sheet"
    REGION = "region"
    VISUAL = "visual"
    OCR = "ocr"
    INFERENCE = "inference"


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EvidenceType
    location: str | None = Field(
        default=None, description="Human-facing anchor, e.g. 'page 3', 'Sheet1', 'slide 2'."
    )
    content: str
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Backend-reported confidence. Absent when the backend reports none; "
        "never invented.",
    )


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    media_type: MediaCategory
    mime_type: str
    size_bytes: int
    modified_at: datetime | None = None
    sha256: str


class VisionProfile(str, Enum):
    """Explicit vision processing profile contract."""

    AUTO = "auto"
    GENERAL = "general"
    UI_STRUCTURE = "ui_structure"
    UI_ALIGNMENT = "ui_alignment"
    UI_GROUNDING = "ui_grounding"
    TERMINAL = "terminal"
    CHART = "chart"
    DIAGRAM = "diagram"
    SCANNED_DOCUMENT = "scanned_document"


class BoundingBox(BaseModel):
    """Normalized [x_min, y_min, x_max, y_max] coordinate in a 0-1000 space."""

    model_config = ConfigDict(extra="forbid")

    x_min: int = Field(ge=0, le=1000)
    y_min: int = Field(ge=0, le=1000)
    x_max: int = Field(ge=0, le=1000)
    y_max: int = Field(ge=0, le=1000)


class UiComponent(BaseModel):
    """Structured representation of a recognized UI element."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    type: str  # button, text_input, card, container, navbar, etc.
    bbox: BoundingBox | None = None
    children_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class UiLayout(BaseModel):
    """Structured inference of container CSS layout behavior."""

    model_config = ConfigDict(extra="forbid")

    display: str | None = None  # flex, grid, block
    direction: str | None = None  # row, column
    columns: int | None = None
    gap_estimate_px: int | None = None


class UiIssue(BaseModel):
    """Detected UI spatial misalignment, clipping, or overflow."""

    model_config = ConfigDict(extra="forbid")

    type: str  # vertical_misalignment, horizontal_misalignment, clipping, overflow
    components: list[str] = Field(default_factory=list)
    evidence: str
    severity: str = "medium"  # low, medium, high
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class UiSpatialAnalysis(BaseModel):
    """3-tier structured spatial perception payload."""

    model_config = ConfigDict(extra="forbid")

    observed_components: list[UiComponent] = Field(default_factory=list)
    inferred_layout: UiLayout | None = None
    issues: list[UiIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class RequestInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = None
    mode: AnalysisMode
    vision_profile: VisionProfile = VisionProfile.AUTO
    pages: str | None = None
    detail: DetailLevel
    max_chars: int


class ProcessingInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processor: str
    processor_version: str
    model: str | None = None
    cached: bool = False
    duration_ms: int = 0
    fallbacks_used: list[str] = Field(default_factory=list)


class TruncationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    truncated: bool
    original_chars: int | None = None
    returned_chars: int | None = None
    recovery_hint: str | None = None


class AnalyzeMediaResult(BaseModel):
    """Successful (or partially successful) analysis."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    source: SourceInfo
    request: RequestInfo
    processing: ProcessingInfo
    summary: str
    markdown: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    truncation: TruncationInfo | None = None
    cache_key: str


class AnalyzeMediaError(BaseModel):
    """Structured failure. ``success`` is always False -- never report a fake result."""

    model_config = ConfigDict(extra="forbid")

    success: Literal[False] = False
    error: dict[str, Any]
    markdown: str
    request: RequestInfo | None = None
    source: SourceInfo | None = None


class MediaInfo(BaseModel):
    """Detected facts about the input file, produced before routing."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    path: Path
    name: str
    extension: str
    mime_type: str
    category: MediaCategory
    size_bytes: int
    modified_at: datetime | None
    sha256: str
    sniffed_type: str | None = Field(
        default=None,
        description="Container type recognised from the file's magic bytes, when it "
        "disagrees with or refines the extension.",
    )

    def to_source_info(self) -> SourceInfo:
        return SourceInfo(
            path=str(self.path),
            name=self.name,
            media_type=self.category,
            mime_type=self.mime_type,
            size_bytes=self.size_bytes,
            modified_at=self.modified_at,
            sha256=self.sha256,
        )


class AnalyzeMediaRequest(BaseModel):
    """Normalised, validated tool arguments."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    path: str
    question: str | None = None
    mode: AnalysisMode = "auto"
    vision_profile: VisionProfile = VisionProfile.AUTO
    pages: str | None = None
    detail: DetailLevel = "normal"
    max_chars: int
    force_refresh: bool = False


class ProcessorResult(BaseModel):
    """What a processor hands back to the pipeline.

    Processors do not build the final response: they return content plus what they
    know about it, and the pipeline owns truncation, caching and rendering.
    """

    model_config = ConfigDict(extra="forbid")

    processor: str
    processor_version: str
    model: str | None = None
    summary: str = ""
    answer: str | None = None
    content_markdown: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fallbacks_used: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
