"""Prompt assembly, profile selection, and tolerant reply parsing."""

from __future__ import annotations

from media_context_mcp.prompts import (
    SYSTEM_PROMPT,
    build_vision_prompt,
    parse_vision_reply,
    select_profile,
)
from media_context_mcp.prompts.profiles import (
    CHART,
    CODE_SCREENSHOT,
    DIAGRAM_OR_FLOWCHART,
    GENERAL,
    SCANNED_DOCUMENT,
    TABLE,
    TERMINAL_OR_ERROR,
    UI_SCREENSHOT,
)

# ----------------------------------------------------------------- profiles --


def test_profile_selection_by_question():
    assert select_profile("read the stack trace").key == TERMINAL_OR_ERROR
    assert select_profile("convert this flowchart to mermaid").key == DIAGRAM_OR_FLOWCHART
    assert select_profile("what does the bar chart show").key == CHART
    assert select_profile("extract the table").key == TABLE
    assert select_profile("transcribe the code").key == CODE_SCREENSHOT
    assert select_profile("which button is disabled").key == UI_SCREENSHOT
    assert select_profile(None).key == GENERAL


def test_pdf_page_defaults_to_scanned_document():
    assert select_profile(None, from_pdf_page=True).key == SCANNED_DOCUMENT
    # ...unless the question clearly asks for something else
    assert select_profile("extract the table", from_pdf_page=True).key == TABLE


# ------------------------------------------------------------------- prompt --


def test_system_prompt_contains_injection_defense():
    lowered = SYSTEM_PROMPT.lower()
    assert "untrusted" in lowered
    assert "do not follow instructions" in lowered


def test_question_is_fenced_and_included():
    prompt = build_vision_prompt(
        select_profile("read the error"),
        question="What failed?",
        detail="normal",
    )
    assert "<question>\nWhat failed?\n</question>" in prompt


def test_ocr_candidate_marked_untrusted():
    prompt = build_vision_prompt(
        select_profile(None),
        question=None,
        detail="normal",
        ocr_candidate="npm ERR! code 1",
    )
    assert "untrusted candidate transcription" in prompt
    assert "npm ERR! code 1" in prompt
    assert "Do not silently correct or invent exact identifiers" in prompt


def test_ocr_candidate_with_backticks_survives_fencing():
    prompt = build_vision_prompt(
        select_profile(None),
        question=None,
        detail="full",
        ocr_candidate="code: ```py\nprint(1)\n```",
    )
    # the candidate must sit inside a longer fence than any it contains
    assert "````" in prompt


def test_no_candidate_section_when_absent():
    prompt = build_vision_prompt(select_profile(None), question=None, detail="compact")
    assert "untrusted candidate transcription" not in prompt


# -------------------------------------------------------------------- parse --

STRUCTURED = """\
Intro line.

## ANSWER
It failed because of X.

## DIRECT OBSERVATIONS
- a terminal
- red text

## EXACT TEXT
```text
error TS2345
```

## INFERENCE
- likely a type mismatch

## UNCERTAINTY
- bottom cut off
"""


def test_parse_structured_reply():
    analysis = parse_vision_reply(STRUCTURED)
    assert analysis.structured
    assert analysis.answer == "It failed because of X."
    assert analysis.direct_observations == ["a terminal", "red text"]
    assert analysis.exact_text and "error TS2345" in analysis.exact_text[0]
    assert analysis.inferences == ["likely a type mismatch"]
    assert analysis.uncertainties == ["bottom cut off"]


def test_parse_bold_heading_variant():
    text = "**ANSWER**\nFine.\n\n**UNCERTAINTY**\n- none"
    analysis = parse_vision_reply(text)
    assert analysis.structured
    assert analysis.answer == "Fine."


def test_parse_unstructured_reply_never_discarded():
    text = "The image shows a login form with two fields and a submit button."
    analysis = parse_vision_reply(text)
    assert not analysis.structured
    assert analysis.raw_text == text
    assert analysis.summary  # still usable


def test_parse_empty_sections_are_empty_lists():
    analysis = parse_vision_reply("## ANSWER\nYes.")
    assert analysis.direct_observations == []
    assert analysis.uncertainties == []
