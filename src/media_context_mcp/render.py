"""Final response assembly: truncation, detail shaping, and Markdown rendering.

The pipeline caches the *full* ProcessorResult and applies truncation here, on the
way out -- so callers with different ``max_chars`` budgets share one cache entry,
and truncation is always reported rather than silent.
"""

from __future__ import annotations

from .models import (
    AnalyzeMediaResult,
    EvidenceItem,
    ProcessingInfo,
    ProcessorResult,
    RequestInfo,
    SourceInfo,
    TruncationInfo,
)

# Truncation cuts at a line boundary within this many characters back from the
# limit, so we do not slice a table row or code line in half mid-character.
_BOUNDARY_SEARCH = 200


def truncate_markdown(text: str, max_chars: int) -> tuple[str, TruncationInfo | None]:
    """Cut ``text`` to ``max_chars``, reporting exactly what happened."""
    if len(text) <= max_chars:
        return text, None
    cut = text.rfind("\n", max_chars - _BOUNDARY_SEARCH, max_chars)
    if cut == -1:
        cut = max_chars
    notice = "\n\n---\n**[Output truncated -- see the `truncation` field.]**"
    truncated = text[:cut].rstrip() + notice
    info = TruncationInfo(
        truncated=True,
        original_chars=len(text),
        returned_chars=len(truncated),
        recovery_hint=(
            "Request a narrower slice: use the 'pages' parameter for a page or slide "
            "range, ask a more specific question, or raise max_chars (server cap: "
            "MEDIA_MCP_MAX_OUTPUT_CHARS)."
        ),
    )
    return truncated, info


def _shape_for_detail(result: ProcessorResult, detail: str) -> tuple[str, list[EvidenceItem]]:
    """Apply the detail level to content and evidence.

    ``compact`` keeps the answer and the evidence needed to verify it; ``normal``
    keeps everything but trims evidence excerpts; ``full`` passes through.
    """
    if detail == "full":
        return result.content_markdown, result.evidence

    if detail == "compact":
        parts: list[str] = []
        if result.answer:
            parts.append(result.answer)
        elif result.summary:
            parts.append(result.summary)
        # Keep at most three evidence items, clipped hard.
        evidence = [
            item.model_copy(update={"content": item.content[:400]})
            for item in result.evidence[:3]
        ]
        if not parts:
            parts.append(result.content_markdown[:1500])
        return "\n\n".join(parts), evidence

    # normal
    evidence = [
        item.model_copy(update={"content": item.content[:800]}) for item in result.evidence
    ]
    return result.content_markdown, evidence


def render_result_markdown(
    summary: str,
    answer: str | None,
    content: str,
    evidence: list[EvidenceItem],
    warnings: list[str],
    question: str | None,
) -> str:
    """The human/agent-facing Markdown body of the response."""
    sections: list[str] = [f"## Summary\n\n{summary}"]

    if question:
        if answer:
            sections.append(f"## Answer to the requested question\n\n{answer}")
        else:
            sections.append(
                "## Answer to the requested question\n\n"
                "_No direct answer was produced; consult the extracted content and "
                "evidence below._"
            )

    if content.strip():
        sections.append(f"## Extracted content\n\n{content}")

    if evidence:
        lines = []
        for item in evidence:
            location = f" ({item.location})" if item.location else ""
            confidence = (
                f" [confidence {item.confidence:.0%}]" if item.confidence is not None else ""
            )
            body = item.content.strip()
            if "\n" in body:
                lines.append(f"- **{item.type.value}**{location}{confidence}:\n\n  {body}")
            else:
                lines.append(f"- **{item.type.value}**{location}{confidence}: {body}")
        sections.append("## Evidence\n\n" + "\n".join(lines))

    if warnings:
        sections.append("## Warnings\n\n" + "\n".join(f"- {w}" for w in warnings))

    return "\n\n".join(sections)


def assemble_result(
    *,
    processor_result: ProcessorResult,
    source: SourceInfo,
    request_info: RequestInfo,
    cached: bool,
    duration_ms: int,
    cache_key: str,
    extra_warnings: list[str],
) -> AnalyzeMediaResult:
    """Combine a (possibly cached) ProcessorResult into the final response."""
    warnings = list(dict.fromkeys([*extra_warnings, *processor_result.warnings]))

    content, evidence = _shape_for_detail(processor_result, request_info.detail)
    markdown = render_result_markdown(
        processor_result.summary,
        processor_result.answer,
        content,
        evidence,
        warnings,
        request_info.question,
    )
    markdown, truncation = truncate_markdown(markdown, request_info.max_chars)
    if truncation:
        warnings = [*warnings, "Output was truncated; see the truncation field."]

    return AnalyzeMediaResult(
        success=True,
        source=source,
        request=request_info,
        processing=ProcessingInfo(
            processor=processor_result.processor,
            processor_version=processor_result.processor_version,
            model=processor_result.model,
            cached=cached,
            duration_ms=duration_ms,
            fallbacks_used=processor_result.fallbacks_used,
        ),
        summary=processor_result.summary,
        markdown=markdown,
        evidence=evidence,
        warnings=warnings,
        truncation=truncation,
        cache_key=cache_key,
    )
