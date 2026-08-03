"""Purpose-specific vision prompt profiles.

A single "describe this image" prompt produces a travel-brochure paragraph, which is
useless for debugging. Each profile below asks for the specific facts that matter for
one kind of image, and refuses the ones that do not.

Profile selection is a keyword heuristic over the caller's question plus the image's
origin. It is transparent and testable, and a wrong guess degrades to a slightly
less targeted prompt -- never to a wrong answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

GENERAL = "general"
UI_SCREENSHOT = "ui_screenshot"
UI_STRUCTURE = "ui_structure"
UI_ALIGNMENT = "ui_alignment"
UI_GROUNDING = "ui_grounding"
TERMINAL_OR_ERROR = "terminal_or_error"
CODE_SCREENSHOT = "code_screenshot"
TABLE = "table"
CHART = "chart"
DIAGRAM_OR_FLOWCHART = "diagram_or_flowchart"
SCANNED_DOCUMENT = "scanned_document"


@dataclass(frozen=True)
class PromptProfile:
    key: str
    title: str
    instructions: str


PROFILES: dict[str, PromptProfile] = {
    GENERAL: PromptProfile(
        key=GENERAL,
        title="General image reading",
        instructions="""
Read the image for whatever information it carries. Cover, in this order:
- every piece of legible text, transcribed exactly;
- what kind of image this is (screenshot, photo, scan, diagram, chart, ...);
- the main objects or regions and how they are arranged;
- anything that looks like an error, a warning, or an anomaly.
If the image is mostly text, prioritise transcription over description.
""",
    ),
    UI_SCREENSHOT: PromptProfile(
        key=UI_SCREENSHOT,
        title="User-interface screenshot",
        instructions="""
Report the interface, not the aesthetics:
- **Visual Component Hierarchy**: Represent containers and nested components in a logical visual tree (e.g. App -> Header / Sidebar / Main Panel -> Card -> ButtonGroup).
- **Visible text**: every label, heading, field value, placeholder, tooltip, menu item
  and button caption, transcribed exactly.
- **Components & Controls**: identify each control precisely (button, text input, checkbox, radio, toggle,
  dropdown, tab, modal, toast, table, card, navigation link, badge, spinner, ...).
- **Spatial Layout & Likely CSS Mechanics**: describe containers and direction (flex-row vs flex-column, grid columns),
  what contains what, and relative spatial positions (above/below/left/right of). Name regions the way a frontend developer would.
- **State**: which controls are enabled, disabled, focused, selected, checked,
  expanded, loading, or showing an error. Say how you can tell (greyed out, blue fill,
  red border, spinner).
- **Colour and iconography** only where it carries semantic meaning (a red border for error, a green check for status, warning badge).
- **Anomalies & Misalignments**: overlapping or clipped text, horizontal/vertical misalignment, improper padding/margin spacing,
  missing images, untranslated strings, placeholder text left in.
Do not claim actual DOM, CSS source, or framework implementation. Infer only plausible visual structure and mark inferences clearly.
""",
    ),
    UI_STRUCTURE: PromptProfile(
        key=UI_STRUCTURE,
        title="UI Component Structure & Hierarchy",
        instructions="""
Deconstruct the interface into a structured Visual Component Hierarchy:
1. **Container Tree**: Group components into nested containers (Header, Sidebar, Main Panel, Card, Form, Table, Modal).
2. **Layout Types**: Identify container layout behavior (flex-row, flex-column, grid with N columns, stack).
3. **Control Identification**: List all inputs, buttons, labels, and widgets within each container.
4. **Disambiguation**:
   - **Observed**: Visually confirmed elements and parent-child relationships.
   - **Inferred**: Plausible container boundaries and layout models.
Do not guess unseen code or actual DOM properties. Label all inferences explicitly.
""",
    ),
    UI_ALIGNMENT: PromptProfile(
        key=UI_ALIGNMENT,
        title="UI Spatial Alignment & CSS Diagnosis",
        instructions="""
Perform a precision spatial alignment analysis across three explicit tiers:
1. **Observed Anomalies**:
   - Report exact spatial misalignments (e.g. "Button X top edge sits ~10px lower than Date Input Y").
   - Report clipped text, truncated labels, or uneven padding/margin gaps.
2. **Likely Causes (Inferred)**:
   - Provide candidate CSS layout causes (e.g. mismatched wrapper heights, `align-items: center` vs `end`, line-height mismatch).
3. **Recommended Checks**:
   - Concrete inspection steps for frontend developers (e.g. "Check computed height on .form-group", "Inspect flex alignment on header container").
Never claim the actual CSS rules or framework source; label all cause hypotheses as inferred.
""",
    ),
    UI_GROUNDING: PromptProfile(
        key=UI_GROUNDING,
        title="UI Element Grounding & Bounding Boxes",
        instructions="""
Extract bounding boxes for all prominent UI controls and containers.
Formatting Rules:
- All bounding boxes MUST use [x_min, y_min, x_max, y_max] in a 0-1000 coordinate space relative to the complete image.
- List each component with its ID, label, control type, and bbox:
  `c1: [type: button, label: "Save", bbox: [850, 920, 960, 970]]`
- Group children under parent container IDs.
""",
    ),
    TERMINAL_OR_ERROR: PromptProfile(
        key=TERMINAL_OR_ERROR,
        title="Terminal output or error dialog",
        instructions="""
Fidelity is the whole job here. In priority order:
1. **Exact error text**, character for character, including punctuation and casing.
   Do not fix typos, do not reflow lines, do not translate.
2. **Error codes and identifiers**: exit codes, HTTP status, errno, TS/CS/PY error
   numbers, exception class names.
3. **Stack trace**, in the original order, with file paths and line numbers exactly as
   shown. Keep indentation.
4. **File paths, line and column numbers, function and symbol names.**
5. **The command that was run**, and any flags visible.
6. **Surrounding context**: what succeeded before the failure, warnings, timestamps.
Then, and only then, add a section beginning "Inference:" with the likely cause.
Finish with "Not visible in this image:" listing what a developer would need next that
the screenshot does not show (e.g. the rest of the trace, the config file, earlier
output).
If any character is genuinely ambiguous (l/1/I, 0/O, rn/m), say so at that spot rather
than picking one silently.
""",
    ),
    CODE_SCREENSHOT: PromptProfile(
        key=CODE_SCREENSHOT,
        title="Source-code screenshot",
        instructions="""
Transcribe the code into a fenced code block, preserving:
- exact indentation (count the leading spaces; do not normalise tabs to spaces or back);
- line breaks and blank lines;
- every character of every identifier, string literal and operator;
- line numbers, if the editor shows them -- put them in a separate list, not inside the
  code block.
Never complete, correct, reformat, or "improve" the code. If a line runs off the edge
of the image, transcribe what is visible and mark the cut with `/* ...cut off... */`.
Name the language only if you are confident, and say what the evidence was.
Report any visible editor markers: squiggly underlines, gutter icons, breakpoints,
diff markers, selection highlights, and what they appear to point at.
""",
    ),
    TABLE: PromptProfile(
        key=TABLE,
        title="Table extraction",
        instructions="""
Reproduce the table as a Markdown table. Rules:
- Preserve every row and column, in the original order.
- Keep values exactly as printed, including units, currency symbols, thousands
  separators, percent signs, and trailing zeros. Do not round, reformat or convert.
- Use an empty cell for a blank; write the literal text for anything else ("N/A", "-").
- If cells are merged, reproduce the value in each covered cell and note the merge below
  the table.
- If a column is cut off or a row is partially hidden, transcribe what is visible and
  say which parts are missing.
Below the table, note the row and column counts, and any totals or footnotes shown.
""",
    ),
    CHART: PromptProfile(
        key=CHART,
        title="Chart or graph",
        instructions="""
Extract the data, not the impression:
- Chart type, title, and subtitle.
- Axis labels, units, scale (linear/log), and the min and max of each axis.
- The legend, and which colour or marker maps to which series.
- Per series: the data points you can read, as a Markdown table. Read printed data
  labels exactly. Where there are no labels and you are reading a value off the axis,
  mark it as approximate with "~" and say so.
- Gridline interval, if it helps interpret the values.
- Annotations, reference lines, error bars, callouts.
Then a short "Inference:" section on what the chart shows -- trend, outliers,
crossings. Never state an unlabelled value as exact.
""",
    ),
    DIAGRAM_OR_FLOWCHART: PromptProfile(
        key=DIAGRAM_OR_FLOWCHART,
        title="Diagram or flowchart",
        instructions="""
Recover the graph:
- **Nodes**: every box, circle, cylinder or actor, with its exact label and its shape
  (shape usually encodes meaning: diamond = decision, cylinder = datastore).
- **Edges**: every connector, as `source -> target`, with its label and arrow direction.
  Note bidirectional and dashed/dotted edges and what the style appears to signify
  (dashed is often async or optional).
- **Conditions**: the branch labels on decision nodes (Yes/No, success/failure).
- **Loops** and cycles, stated explicitly.
- **Grouping**: swimlanes, boxes-around-boxes, colour groups, and their labels.
- **Start and end points.**
Then give a Mermaid representation in a ```mermaid fence (flowchart TD, or
sequenceDiagram if it is clearly a sequence diagram). The Mermaid must contain only
nodes and edges you actually saw. If a connector's endpoint is ambiguous, leave it out
of the Mermaid and list it under Uncertain.
""",
    ),
    SCANNED_DOCUMENT: PromptProfile(
        key=SCANNED_DOCUMENT,
        title="Scanned or photographed document page",
        instructions="""
Transcribe the page into structured Markdown, following the original reading order:
- Headings as headings, lists as lists, tables as Markdown tables.
- Preserve the exact wording, numbers, dates, reference codes and amounts.
- Keep the document's own numbering (clause numbers, item numbers, page numbers).
- Note headers, footers, page numbers, stamps, signatures, handwriting and marginalia
  separately from the body text -- and say which is which.
- Mark unreadable regions as `[illegible]` rather than guessing.
Do not summarise in place of transcribing. If the page is rotated or skewed, transcribe
it anyway and mention the orientation.
""",
    ),
}


# Ordered: the first pattern that matches wins, so the most specific go first.
_SELECTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        TERMINAL_OR_ERROR,
        re.compile(
            r"\b(error|exception|stack ?trace|traceback|crash|fail(ed|ure)?|"
            r"terminal|console|shell|stderr|log output|exit code|errno)\b",
            re.IGNORECASE,
        ),
    ),
    (
        DIAGRAM_OR_FLOWCHART,
        re.compile(
            r"\b(diagram|flow ?chart|flow ?diagram|mermaid|sequence diagram|"
            r"architecture|graph of|nodes?|edges?|swimlane)\b",
            re.IGNORECASE,
        ),
    ),
    (
        CHART,
        re.compile(
            r"\b(chart|graph|plot|axis|axes|bar chart|line chart|pie chart|"
            r"histogram|scatter|trend|series)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TABLE,
        re.compile(r"\b(table|spreadsheet|rows? and columns?|tabular|csv)\b", re.IGNORECASE),
    ),
    (
        CODE_SCREENSHOT,
        re.compile(
            r"\b(code|source|snippet|function|class|method|syntax|indentation|"
            r"transcribe the code)\b",
            re.IGNORECASE,
        ),
    ),
    (
        UI_SCREENSHOT,
        re.compile(
            r"\b(ui|u\.i\.|interface|screenshot|screen|layout|button|modal|dialog|"
            r"form|menu|tab|toolbar|sidebar|component|design|figma|mockup|"
            r"disabled|enabled|checkbox|dropdown)\b",
            re.IGNORECASE,
        ),
    ),
]


def select_profile(
    question: str | None,
    *,
    from_pdf_page: bool = False,
    default: str = GENERAL,
    explicit_profile: str | None = None,
) -> PromptProfile:
    """Pick a profile from explicit selection, caller's question, or origin.

    An explicit profile choice overrides heuristic selection.
    """
    if explicit_profile and explicit_profile in PROFILES:
        return PROFILES[explicit_profile]
    if question:
        for key, pattern in _SELECTION_RULES:
            if pattern.search(question):
                return PROFILES[key]
    if from_pdf_page:
        return PROFILES[SCANNED_DOCUMENT]
    return PROFILES[default]
