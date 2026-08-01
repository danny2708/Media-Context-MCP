"""Shared fixtures: generated files, fake backends, and a pipeline factory.

The default suite is fully offline and deterministic. OCR and vision run against
fakes; anything touching a real tesseract binary or a real provider carries the
``requires_tesseract`` / ``requires_vision`` marker and is skipped unless the
environment provides it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from generate import generate_all  # noqa: E402

from media_context_mcp.cache import CacheStore  # noqa: E402
from media_context_mcp.config import Settings  # noqa: E402
from media_context_mcp.errors import OcrFailedError  # noqa: E402
from media_context_mcp.pipeline import Pipeline  # noqa: E402
from media_context_mcp.processors.image import ImageProcessor  # noqa: E402
from media_context_mcp.processors.markitdown_adapter import MarkItDownProcessor  # noqa: E402
from media_context_mcp.processors.pdf import PdfProcessor  # noqa: E402
from media_context_mcp.processors.text import TextProcessor  # noqa: E402
from media_context_mcp.providers.base import (  # noqa: E402
    OcrResult,
    VisionImage,
    VisionProviderResult,
)

# --------------------------------------------------------------------- files --


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generated fixture files, one directory for the whole test session."""
    directory = tmp_path_factory.mktemp("media-fixtures")
    return generate_all(directory)


@pytest.fixture(scope="session")
def fixtures_root(fixtures: dict[str, Path]) -> Path:
    return next(iter(fixtures.values())).parent


# ------------------------------------------------------------------ settings --


def make_settings(root: Path, cache_dir: Path, **overrides) -> Settings:
    """Settings with the sandbox pointed at ``root`` and everything else default."""
    base = dict(
        allowed_roots=[root],
        cache_dir=cache_dir,
        cache_enabled=True,
        log_level="WARNING",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def settings(fixtures_root: Path, tmp_path: Path) -> Settings:
    return make_settings(fixtures_root, tmp_path / "cache")


# --------------------------------------------------------------------- fakes --


@dataclass
class FakeOcrBackend:
    """Deterministic OCR: returns canned text, records every call."""

    text: str = "npm ERR! code ELIFECYCLE\nerror TS2345: Argument of type"
    confidence: float | None = 0.91
    fail: bool = False
    calls: list[VisionImage] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake-ocr"

    @property
    def version(self) -> str:
        return "1.0"

    def availability(self) -> tuple[bool, str | None]:
        return (not self.fail), ("forced failure" if self.fail else None)

    def installed_languages(self) -> list[str]:
        return ["eng", "vie"]

    async def recognise(self, image: VisionImage, languages: str) -> OcrResult:
        self.calls.append(image)
        if self.fail:
            raise OcrFailedError("fake OCR failure")
        return OcrResult(
            text=self.text,
            engine=self.name,
            engine_version=self.version,
            languages=languages,
            mean_confidence=self.confidence,
        )


_STRUCTURED_REPLY = """\
## ANSWER
The build failed with error TS2345 in src/services/report.ts line 84.

## DIRECT OBSERVATIONS
- A dark terminal window showing an npm build.
- One error line rendered in red.

## EXACT TEXT
```text
error TS2345: Argument of type 'string | undefined'
```

## VISUAL INTERPRETATION
- The error line is visually emphasised against the dark background.

## INFERENCE
- A nullable value is being passed where a non-null string is required.

## UNCERTAINTY
- The bottom of the terminal is cut off.
"""


@dataclass
class FakeVisionProvider:
    """Deterministic vision: returns a structured reply, records every call.

    ``last_prompt``/``last_system``/``last_images`` let tests assert exactly what
    would have been sent to a real provider (question inclusion, OCR candidate
    inclusion, image order) without any network.
    """

    reply: str = _STRUCTURED_REPLY
    model: str = "fake/fake-vlm-1"
    calls: int = 0
    last_prompt: str | None = None
    last_system: str | None = None
    last_images: list[VisionImage] = field(default_factory=list)
    last_max_output_tokens: int | None = None
    raise_error: Exception | None = None
    closed: bool = False

    @property
    def provider_name(self) -> str:
        return "fake-provider"

    @property
    def requested_model(self) -> str:
        return self.model

    async def analyze(
        self,
        *,
        images: list[VisionImage],
        prompt: str,
        system: str | None = None,
        max_output_tokens: int,
        request_id: str,
    ) -> VisionProviderResult:
        self.calls += 1
        self.last_prompt = prompt
        self.last_system = system
        self.last_images = list(images)
        self.last_max_output_tokens = max_output_tokens
        if self.raise_error is not None:
            raise self.raise_error
        return VisionProviderResult(
            content=self.reply,
            provider=self.provider_name,
            requested_model=self.model,
            actual_model=self.model,
            duration_ms=5,
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_ocr() -> FakeOcrBackend:
    return FakeOcrBackend()


@pytest.fixture()
def fake_vision() -> FakeVisionProvider:
    return FakeVisionProvider()


# ------------------------------------------------------------------ pipeline --


def make_pipeline(
    settings: Settings,
    *,
    ocr=None,
    vision=None,
) -> Pipeline:
    """A pipeline with injected fakes -- the DI seam the whole suite runs through."""
    return Pipeline(
        settings=settings,
        cache=CacheStore(
            settings.resolved_cache_dir(),
            enabled=settings.cache_enabled,
            max_bytes=settings.cache_max_bytes,
            ttl_days=settings.cache_ttl_days,
        ),
        processors={
            "text": TextProcessor(),
            "markitdown": MarkItDownProcessor(),
            "pdf": PdfProcessor(),
            "image": ImageProcessor(),
        },
        ocr=ocr,
        vision=vision,
    )
