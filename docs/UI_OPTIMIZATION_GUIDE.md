# UI Screenshot Optimization Guide — Media Context MCP

This guide explains how to get the highest precision spatial layout, component hierarchy, and CSS alignment analysis when inspecting UI screenshots using **Media Context MCP**.

---

## 1. How UI Screenshot Processing Works

When a UI screenshot is processed with `mode="vision"` (or `mode="auto"`):

1. **Explicit `vision_profile` Contract**:
   Instead of relying on heuristic keyword guessing, pass an explicit `vision_profile` parameter (`ui_structure`, `ui_alignment`, `ui_grounding`). Question keyword matching serves as a fallback heuristic only when `vision_profile="auto"`.

2. **Multi-Pass Long Screenshot Processing**:
   Long scroll screenshots (aspect ratio $\ge 2.5$) execute a **2-pass payload**:
   - **Pass 1 (Overview)**: Global downscaled overview pass (`overview`) to capture overall layout hierarchy and macro structure.
   - **Pass 2 (Native Tiles)**: Overlapping native-resolution detail tiles (`detail`, `96px` overlap) to preserve 11px text and pixel-precise alignments.

3. **VLM Interleaved Tile Manifests**:
   Each image payload sent to the VLM is prepended with explicit manifest metadata (`[Image N of M: Native-resolution detail tile covering x=0-1440, y=0-4096...]`). This gives the VLM exact spatial coordinates and prevents double-counting elements in overlapping regions.

---

## 2. Recommended Usage for AI Agents

AI Agents should pass the explicit `vision_profile` parameter in tool calls:

```json
{
  "path": "shots/ui-layout-issue.png",
  "mode": "vision",
  "vision_profile": "ui_alignment",
  "question": "Compare the vertical alignment of the date inputs and filter button."
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
