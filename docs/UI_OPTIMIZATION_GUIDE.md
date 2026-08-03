# UI Screenshot Optimization Guide — Media Context MCP

This guide explains how to get the highest precision spatial layout, component hierarchy, and CSS alignment analysis when inspecting UI screenshots using **Media Context MCP**.

---

## 1. How UI Screenshot Processing Works

When a UI screenshot is processed with `mode="vision"` (or `mode="auto"` with a visual question):

1. **Intent & Profile Selection**:
   Questions mentioning terms like `UI screenshot`, `layout`, `spatial alignment`, `component hierarchy`, `flex`, `grid`, or `misalignment` automatically trigger the **`ui_screenshot` prompt profile**.

2. **Native Overlapping Tiling**:
   Long scroll screenshots (aspect ratio $\ge 2.5$) are automatically cut into up to 4 overlapping strips (`96px` overlap) at 100% native resolution rather than downscaling. This preserves small text (11px/12px fonts), 1px border lines, and precise spacing.

3. **Structured Output**:
   The output returns structured sections:
   - **Component Hierarchy & Tree**: DOM-like parent/child nesting (e.g., `App -> Sidebar -> NavLink`).
   - **Spatial Layout & CSS Mechanics**: `flex-row`, `flex-column`, `grid`, and relative spatial positions (`left-of`, `above`, `below`).
   - **Anomalies & Misalignments**: Clipped text, improper padding/margin, overlapping elements, or state errors.

---

## 2. Recommended Question Format for AI Agents

AI Agents should format the `question` argument using canonical English keywords to guarantee accurate intent classification:

```json
{
  "path": "shots/ui-layout-issue.png",
  "mode": "vision",
  "question": "UI screenshot spatial layout and component hierarchy: Analyze container nesting, relative positions of buttons and inputs, flex/grid directions, and report any visual misalignments or spacing anomalies."
}
```

---

## 3. Vision Provider Options

### Option A: Cloud Vision (Hugging Face / OpenAI / OpenRouter)
Configured via `.env`:
```dotenv
MEDIA_MCP_VISION_PROVIDER=huggingface
MEDIA_MCP_VISION_API_KEY=hf_...
MEDIA_MCP_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
MEDIA_MCP_ALLOW_CLOUD_VISION=true
```

### Option B: Local Vision (Ollama / vLLM)
For 100% offline, zero-cloud visual analysis:
```dotenv
MEDIA_MCP_VISION_BASE_URL=http://localhost:11434/v1
MEDIA_MCP_VISION_MODEL=qwen2.5-vl:7b
MEDIA_MCP_ALLOW_CLOUD_VISION=true
```

---

## 4. Integration with Coding Agents (OpenCode / Antigravity / Claude Code)

Add this rule to your project's `AGENTS.md` or `GEMINI.md`:

```markdown
### UI Screenshot Handling
When analyzing UI screenshots for styling or layout bugs:
- Call `analyze_media` with `mode="vision"` and prefix the question with `"UI screenshot spatial layout and component hierarchy: ..."`.
- Use the returned Component Hierarchy and Spatial Layout sections to identify exact CSS flex/grid or spacing fixes.
```
