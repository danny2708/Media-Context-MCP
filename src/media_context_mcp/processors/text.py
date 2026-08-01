"""Plain-text reader.

Sending a ``.txt`` or ``.md`` file through a document converter would be pure
overhead and would rewrite the content on the way through. This processor reads the
bytes, decodes them, and hands them back unmodified.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import DocumentConversionFailedError
from ..models import (
    AnalyzeMediaRequest,
    EvidenceItem,
    EvidenceType,
    MediaInfo,
    MediaCategory,
    ProcessorResult,
)
from .base import ProcessingContext

# Tried in order. utf-8-sig strips a BOM that would otherwise appear as a stray
# character at the start of the first line.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

_FENCE_LANGUAGE = {
    ".json": "json",
    ".log": "text",
    ".txt": "text",
    ".rst": "rst",
    ".tsv": "text",
}


def decode_text(raw: bytes) -> tuple[str, str, list[str]]:
    """Decode bytes, returning ``(text, encoding, warnings)``.

    ``latin-1`` never fails, so it is the terminal fallback; when it is used we say
    so, because the result may contain mojibake the caller should not trust.
    """
    warnings: list[str] = []
    for encoding in _ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding in {"cp1252", "latin-1"}:
            warnings.append(
                f"File is not valid UTF-8; decoded as {encoding}. Non-ASCII characters "
                "may be wrong. Re-save the file as UTF-8 for a reliable reading."
            )
        return text, encoding, warnings
    raise DocumentConversionFailedError(
        "Could not decode the file as text with any supported encoding.",
        hint="The file is probably binary despite its extension.",
    )


class TextProcessor:
    """Reads text files verbatim."""

    name = "text"
    version = "1.0.0"

    def supports(self, info: MediaInfo) -> bool:
        return info.category is MediaCategory.TEXT

    async def process(
        self,
        request: AnalyzeMediaRequest,
        context: ProcessingContext,
    ) -> ProcessorResult:
        path: Path = context.info.path
        raw = path.read_bytes()
        text, encoding, warnings = decode_text(raw)

        if request.pages:
            context.warn(
                "The 'pages' parameter does not apply to plain-text files and was ignored."
            )

        line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        extension = context.info.extension

        if extension in {".md", ".markdown"}:
            # Already Markdown: pass it through so headings and tables survive.
            content = text
        else:
            language = _FENCE_LANGUAGE.get(extension, "text")
            fence = "```"
            # A file that itself contains ``` would break a 3-backtick fence.
            while fence in text:
                fence += "`"
            content = f"{fence}{language}\n{text}\n{fence}"

        summary = (
            f"Plain-text file `{context.info.name}` "
            f"({line_count} line{'s' if line_count != 1 else ''}, "
            f"{len(text):,} characters, {encoding})."
        )

        evidence = [
            EvidenceItem(
                type=EvidenceType.TEXT,
                location=f"lines 1-{line_count}" if line_count else "empty file",
                content=_first_lines(text, 12),
            )
        ]

        return ProcessorResult(
            processor=self.name,
            processor_version=self.version,
            summary=summary,
            content_markdown=content,
            evidence=evidence,
            warnings=warnings,
            extra={"encoding": encoding, "line_count": line_count},
        )


def _first_lines(text: str, count: int) -> str:
    lines = text.splitlines()
    head = lines[:count]
    if len(lines) > count:
        head.append(f"... ({len(lines) - count} more lines)")
    return "\n".join(head)
