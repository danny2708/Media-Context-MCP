# Media Context MCP — instructions for your AGENTS.md

Copy the block below into your project's `AGENTS.md` (OpenCode) or equivalent
agent-instruction file, after registering the MCP server in `opencode.json`.

---

## Reading images, PDFs and Office documents

When the user references an image, screenshot, PDF, presentation, spreadsheet,
or other non-plain-text file, use the Media Context MCP (`analyze_media`)
instead of claiming that the file cannot be inspected.

Call `analyze_media` with a focused `question` derived from the user's task —
for example "Read the visible error message and identify the likely cause",
"Convert the flowchart into Mermaid syntax", or "Extract the table and preserve
all values" — rather than asking for a generic description.

Rules of thumb:

- For text-only models: **always** use this tool for images.
- For multimodal models: prefer this tool when the file is large, reused across
  turns, or needs structured extraction and caching; a small one-off image may
  be read directly.
- Use `mode="ocr"` when you need exact text and nothing else — it is fully
  local and never calls a cloud provider.
- Trust the result's typed evidence: `ocr`/`text`/`page` evidence is exact
  extraction; `visual` is a model's interpretation; `inference` is explicitly
  beyond what is visible. When an exact string matters (an error code, an
  identifier), prefer the OCR/text evidence over visual claims.
- Use `pages` ("1", "2-3", "1,4") to narrow large PDFs and slide decks, and
  `detail="compact"` when you only need an answer.
- If the tool returns an error code like `VISION_NOT_CONFIGURED` or
  `CLOUD_VISION_DISABLED`, relay its hint to the user — do not pretend the
  visual analysis happened.
