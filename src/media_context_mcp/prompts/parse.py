"""Tolerant parsing of a vision model's reply.

Not every VLM follows formatting instructions. The boundary here is:

    provider raw text
        -> try to split on the section headings we asked for
        -> fall back to a safe representation of the whole text
        -> never discard a successful response because parsing failed

A reply that yields no recognisable sections is still a usable answer; it just
lands in ``raw_text``/``summary`` instead of the structured fields.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .base import (
    SECTION_ANSWER,
    SECTION_EXACT_TEXT,
    SECTION_INFERENCE,
    SECTION_INTERPRETATION,
    SECTION_OBSERVATIONS,
    SECTION_UNCERTAINTY,
)


class VisionAnalysis(BaseModel):
    """A vision reply normalised into labelled parts.

    Empty lists mean "the model reported nothing under that heading", which is
    itself information -- e.g. no uncertainties declared.
    """

    summary: str = ""
    answer: str | None = None
    direct_observations: list[str] = Field(default_factory=list)
    exact_text: list[str] = Field(default_factory=list)
    visual_interpretation: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    raw_text: str | None = None
    structured: bool = False
    """True when the reply actually followed the requested section layout."""


# Accepts "## ANSWER", "**ANSWER**", "ANSWER:" and case variants -- models are
# inconsistent about heading syntax even when they honour the section names.
def _heading_pattern(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    return re.compile(
        rf"^(?:#{{1,4}}\s*|\*\*)?{escaped}(?:\*\*)?\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )


_SECTIONS: list[tuple[str, re.Pattern[str]]] = [
    (SECTION_ANSWER, _heading_pattern(SECTION_ANSWER)),
    (SECTION_OBSERVATIONS, _heading_pattern(SECTION_OBSERVATIONS)),
    (SECTION_EXACT_TEXT, _heading_pattern(SECTION_EXACT_TEXT)),
    (SECTION_INTERPRETATION, _heading_pattern(SECTION_INTERPRETATION)),
    (SECTION_INFERENCE, _heading_pattern(SECTION_INFERENCE)),
    (SECTION_UNCERTAINTY, _heading_pattern(SECTION_UNCERTAINTY)),
]


def _to_items(body: str) -> list[str]:
    """Split a section body into items: list bullets if present, else paragraphs."""
    body = body.strip()
    if not body:
        return []
    bullet_lines = re.findall(r"^[-*+]\s+(.+)$", body, re.MULTILINE)
    if bullet_lines and len(bullet_lines) >= body.count("\n") / 2:
        return [line.strip() for line in bullet_lines if line.strip()]
    return [paragraph.strip() for paragraph in re.split(r"\n{2,}", body) if paragraph.strip()]


def parse_vision_reply(text: str) -> VisionAnalysis:
    """Split ``text`` on the requested section headings; degrade gracefully."""
    hits: list[tuple[int, int, str]] = []  # (start_of_heading, end_of_heading, name)
    for name, pattern in _SECTIONS:
        match = pattern.search(text)
        if match:
            hits.append((match.start(), match.end(), name))
    hits.sort()

    if not hits:
        cleaned = text.strip()
        first_paragraph = re.split(r"\n{2,}", cleaned)[0] if cleaned else ""
        return VisionAnalysis(
            summary=first_paragraph[:600],
            raw_text=cleaned,
            structured=False,
        )

    bodies: dict[str, str] = {}
    # Anything before the first heading is treated as a lead-in summary.
    preamble = text[: hits[0][0]].strip()
    for index, (_, heading_end, name) in enumerate(hits):
        section_end = hits[index + 1][0] if index + 1 < len(hits) else len(text)
        bodies[name] = text[heading_end:section_end].strip()

    answer_body = bodies.get(SECTION_ANSWER, "").strip()
    exact_body = bodies.get(SECTION_EXACT_TEXT, "")

    # EXACT TEXT often contains a fenced transcription block; keep fences whole
    # rather than shredding them into paragraph items.
    if "```" in exact_body:
        exact_items = [exact_body.strip()] if exact_body.strip() else []
    else:
        exact_items = _to_items(exact_body)

    summary = answer_body or preamble
    if not summary:
        observations = _to_items(bodies.get(SECTION_OBSERVATIONS, ""))
        summary = observations[0] if observations else text.strip()[:300]

    return VisionAnalysis(
        summary=summary[:600],
        answer=answer_body or None,
        direct_observations=_to_items(bodies.get(SECTION_OBSERVATIONS, "")),
        exact_text=exact_items,
        visual_interpretation=_to_items(bodies.get(SECTION_INTERPRETATION, "")),
        inferences=_to_items(bodies.get(SECTION_INFERENCE, "")),
        uncertainties=_to_items(bodies.get(SECTION_UNCERTAINTY, "")),
        raw_text=None,  # structured replies do not need the duplicate
        structured=True,
    )
