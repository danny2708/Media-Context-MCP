"""Tesseract OCR backend.

Tesseract was chosen over PaddleOCR and EasyOCR for one reason: it is the only
one of the three that does not drag in a deep-learning runtime and a model download
as a hard dependency. ``pytesseract`` is a thin subprocess wrapper, so the Python
side stays light and the heavy part is an OS package the operator installs once.
The trade-off is accuracy on skewed or low-contrast scans, where the neural engines
are better; :class:`OcrBackend` exists so another backend can be added without
touching any processor.

Two invocations are made per image, deliberately:

* ``image_to_string`` produces the text, with ``preserve_interword_spaces`` on so
  that indentation in a code screenshot survives;
* ``image_to_data`` produces per-word confidences, which is the only honest source
  for the ``confidence`` field in the response.

Recognised text is returned exactly as the engine produced it. It is never spell
corrected, reflowed or "fixed" -- an OCR error in an error message or a code
identifier is information the caller needs to see.
"""

from __future__ import annotations

import asyncio
import io
import shutil

from ..errors import OcrFailedError
from .base import ImageInput, OcrResult

# psm 3 = fully automatic page segmentation, no orientation detection. It handles
# both dense paragraphs and terminal output acceptably; psm 6 ("uniform block")
# does better on pure code screenshots but badly on multi-column scans, and we
# cannot tell them apart without looking.
_CONFIG = "--psm 3 -c preserve_interword_spaces=1"


class TesseractBackend:
    """Local OCR via the tesseract binary."""

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        self._configured_cmd = tesseract_cmd
        self._version: str | None = None

    @property
    def name(self) -> str:
        return "tesseract"

    @property
    def version(self) -> str:
        return self._version or "unknown"

    def _bind(self) -> object | None:
        """Import pytesseract and point it at the binary. Returns the module or None."""
        try:
            import pytesseract
        except ImportError:
            return None
        if self._configured_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._configured_cmd
        return pytesseract

    def availability(self) -> tuple[bool, str | None]:
        """Check the Python wrapper and the binary. Never raises."""
        pytesseract = self._bind()
        if pytesseract is None:
            return False, (
                "the 'pytesseract' package is not installed "
                "(pip install 'media-context-mcp[ocr]')"
            )
        if not self._configured_cmd and shutil.which("tesseract") is None:
            return False, (
                "the 'tesseract' binary is not on PATH; install it "
                "(Windows: winget install UB-Mannheim.TesseractOCR; "
                "macOS: brew install tesseract; Debian/Ubuntu: apt install tesseract-ocr) "
                "or set MEDIA_MCP_TESSERACT_CMD to its full path"
            )
        try:
            self._version = str(pytesseract.get_tesseract_version())
        except Exception as exc:  # noqa: BLE001 - wrapper raises several unrelated types
            return False, f"the tesseract binary could not be executed: {exc}"
        return True, None

    def installed_languages(self) -> list[str]:
        """Language packs tesseract reports. Empty list when it cannot be asked."""
        pytesseract = self._bind()
        if pytesseract is None:
            return []
        try:
            return list(pytesseract.get_languages(config=""))
        except Exception:  # noqa: BLE001
            return []

    async def recognise(self, image: ImageInput, languages: str) -> OcrResult:
        return await asyncio.to_thread(self._recognise_sync, image, languages)

    def _recognise_sync(self, image: ImageInput, languages: str) -> OcrResult:
        available, reason = self.availability()
        if not available:
            raise OcrFailedError(
                f"OCR backend unavailable: {reason}.",
                hint="Run 'media-context-mcp doctor' to check the OCR installation.",
            )

        pytesseract = self._bind()
        assert pytesseract is not None  # availability() already proved this
        from PIL import Image as PILImage

        warnings: list[str] = []
        installed = self.installed_languages()
        requested = [lang for lang in languages.split("+") if lang]
        missing = [lang for lang in requested if installed and lang not in installed]
        if missing:
            usable = [lang for lang in requested if lang not in missing] or ["eng"]
            warnings.append(
                f"Tesseract language pack(s) {', '.join(missing)} are not installed; "
                f"used {'+'.join(usable)} instead. Install the pack (e.g. "
                f"'apt install tesseract-ocr-{missing[0]}') for better accuracy on that "
                "language."
            )
            languages = "+".join(usable)

        pil_image = PILImage.open(io.BytesIO(image.data))
        try:
            text = pytesseract.image_to_string(pil_image, lang=languages, config=_CONFIG)
            data = pytesseract.image_to_data(
                pil_image,
                lang=languages,
                config=_CONFIG,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001 - subprocess and parsing errors alike
            raise OcrFailedError(
                f"Tesseract failed on {image.label or 'the image'}: {exc}",
                hint=(
                    "Check that the language packs named in MEDIA_MCP_OCR_LANGUAGES are "
                    "installed, and that the image is not corrupt."
                ),
            ) from exc
        finally:
            pil_image.close()

        confidences = [
            float(value)
            for value in data.get("conf", [])
            if str(value).lstrip("-").replace(".", "", 1).isdigit() and float(value) >= 0
        ]
        mean_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else None

        if mean_confidence is not None and mean_confidence < 0.55:
            warnings.append(
                f"Mean OCR confidence is low ({mean_confidence:.0%}) for "
                f"{image.label or 'this image'}. Characters, and especially digits and "
                "identifiers, may be misread. Verify anything you act on."
            )

        return OcrResult(
            text=text,
            engine=self.name,
            engine_version=self.version,
            languages=languages,
            mean_confidence=mean_confidence,
            warnings=warnings,
        )


class NullOcrBackend:
    """Stand-in used when OCR is switched off, so callers get one clear reason."""

    @property
    def name(self) -> str:
        return "none"

    @property
    def version(self) -> str:
        return "0"

    def availability(self) -> tuple[bool, str | None]:
        return False, "OCR is disabled (MEDIA_MCP_OCR_BACKEND=none)"

    def installed_languages(self) -> list[str]:
        return []

    async def recognise(self, image: ImageInput, languages: str) -> OcrResult:
        raise OcrFailedError(
            "OCR is disabled by configuration.",
            hint="Set MEDIA_MCP_OCR_BACKEND=tesseract to enable it.",
        )


def build_ocr_backend(backend: str, tesseract_cmd: str | None) -> TesseractBackend | NullOcrBackend:
    if backend == "tesseract":
        return TesseractBackend(tesseract_cmd)
    return NullOcrBackend()
