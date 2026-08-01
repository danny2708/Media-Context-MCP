# Media Context MCP

A local [MCP](https://modelcontextprotocol.io) server that lets **text-only coding
models** read images, screenshots, PDFs, presentations, spreadsheets and other
non-plain-text files. Built for [OpenCode](https://opencode.ai) first (e.g.
DeepSeek V4 Flash, which cannot see images), but client-agnostic: it works with
any MCP-compatible agent (Claude Code, Codex, Antigravity, ...).

One tool: **`analyze_media`** — give it a file path and a focused question, get
back concise, structured, *evidence-backed* Markdown.

```text
Media file → detection → routing → MarkItDown / PyMuPDF / OCR / vision model
          → structured Markdown + typed evidence → content-addressed cache
```

## What it is (and is not)

- **MarkItDown is not a vision model.** It converts document *structure*
  (DOCX/PPTX/XLSX/HTML/...) to Markdown.
- **OCR extracts characters.** It is the preferred source for *exact* text —
  error messages, code, identifiers — and knows nothing about layout or state.
- **A vision-language model provides visual-semantic understanding** — layout,
  relationships, UI state, charts, diagrams — and may hallucinate exact strings.
- This server combines all three and **labels which is which** in every result,
  so a calling agent knows what to trust.
- **Cloud vision sends your media off-machine** and is therefore **off by
  default** (`MEDIA_MCP_ALLOW_CLOUD_VISION=false`). Document extraction and OCR
  are fully local.
- A multimodal model can read a small one-off image directly; this MCP is
  mandatory mainly for **text-only models**, and useful for anyone as a
  cache/extraction layer for large or frequently reused media.

## Requirements

- Python **3.11+**
- Optional: [Tesseract](https://github.com/tesseract-ocr/tesseract) for local OCR
  - Windows: `winget install UB-Mannheim.TesseractOCR`
  - macOS: `brew install tesseract tesseract-lang`
  - Debian/Ubuntu: `apt install tesseract-ocr tesseract-ocr-vie`
- Optional: an OpenAI-compatible vision endpoint (Hugging Face router, OpenAI,
  OpenRouter, a local vLLM/Ollama shim, ...) for visual-semantic analysis

## Install

```bash
git clone <this-repo> && cd media-context-mcp
uv venv && uv pip install -e ".[ocr]"     # or: pip install -e ".[ocr]"
```

Verify:

```bash
media-context-mcp doctor            # installation + configuration checks
media-context-mcp doctor --vision   # detailed vision checks (no network)
media-context-mcp doctor --vision --network  # one tiny real provider call
```

## Configure

Copy `.env.example` and set at minimum the workspace sandbox:

```dotenv
MEDIA_MCP_ALLOWED_ROOTS=D:\Work\my-project        # ';'-separated on Windows, ':' elsewhere
```

### Vision via Hugging Face Inference Providers (example preset)

```dotenv
MEDIA_MCP_VISION_PROVIDER=huggingface     # presets base URL https://router.huggingface.co/v1
MEDIA_MCP_VISION_API_KEY=hf_...           # token with inference permissions
MEDIA_MCP_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct   # example; verify current availability
MEDIA_MCP_ALLOW_CLOUD_VISION=true         # explicit opt-in: images leave your machine
```

Notes on Hugging Face: it is an *initial configurable provider*, not a permanent
dependency — any OpenAI-compatible endpoint works via
`MEDIA_MCP_VISION_BASE_URL`. Free/promotional model availability changes; credits
are trial capacity, not unlimited inference. The server never silently switches
models — fallbacks must be listed explicitly in
`MEDIA_MCP_VISION_FALLBACK_MODELS`. A local VLM is a planned future provider.

## OpenCode integration

Add to `opencode.json` in your project root (or the global config). Format
verified against the [OpenCode MCP docs](https://opencode.ai/docs/mcp-servers/):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "media-context": {
      "type": "local",
      "command": [
        "C:\\path\\to\\media-context-mcp\\.venv\\Scripts\\python.exe",
        "-m", "media_context_mcp"
      ],
      "enabled": true,
      "environment": {
        "MEDIA_MCP_ALLOWED_ROOTS": "C:\\path\\to\\your\\workspace",
        "MEDIA_MCP_VISION_PROVIDER": "huggingface",
        "MEDIA_MCP_VISION_API_KEY": "{env:HF_TOKEN}",
        "MEDIA_MCP_VISION_MODEL": "Qwen/Qwen2.5-VL-7B-Instruct",
        "MEDIA_MCP_ALLOW_CLOUD_VISION": "true"
      }
    }
  }
}
```

Then restart OpenCode (or run `/mcp` to check connection status). Add an
instruction like the one in [AGENTS.example.md](AGENTS.example.md) to your
project's `AGENTS.md` so the model actually uses the tool.

> Status: the configuration format above matches OpenCode's current docs; see
> `docs/IMPLEMENTATION_REPORT.md` for exactly what has and has not been verified
> end-to-end in this environment.

## Example calls

```jsonc
// exact error text from a screenshot (local OCR, no cloud)
{ "path": "shots/build-fail.png", "mode": "ocr",
  "question": "Read the exact error message." }

// visual reading of a UI (needs vision provider + cloud opt-in)
{ "path": "shots/modal.png",
  "question": "Which component is misaligned relative to the design?" }

// pages 2-3 of a PDF, compact
{ "path": "docs/spec.pdf", "pages": "2-3", "detail": "compact",
  "question": "What are the deployment requirements?" }
```

## CLI

```bash
media-context-mcp doctor [--vision] [--network]
media-context-mcp inspect <path> [-q QUESTION] [--mode ...] [--pages ...] [--json]
media-context-mcp show-config      # secrets redacted
media-context-mcp clear-cache
media-context-mcp benchmark [--live]
```

`inspect` runs the identical pipeline outside MCP — use it to separate media
processing failures from MCP transport failures.

## Security model (summary)

- File access is restricted to `MEDIA_MCP_ALLOWED_ROOTS` (canonical-path ancestry
  checks, symlinks resolved first; UNC paths, URLs and non-regular files refused).
- File-size, pixel-count, page-count, output-size and processing-time limits.
- Remote URL fetching is disabled; archives are refused (decompression bombs).
- Media content is treated as untrusted: instructions inside screenshots or
  documents are analysed as content, never followed.
- Cloud vision requires explicit opt-in; OCR-text forwarding is separately
  gated (`MEDIA_MCP_SEND_OCR_TO_CLOUD`). Screenshots may contain secrets or
  proprietary code — opt in deliberately. No automatic redaction is claimed.
- API keys never appear in logs, cache files, or tool output.

Details: `docs/SECURITY.md`. Architecture: `docs/ARCHITECTURE.md`.
Troubleshooting: `docs/TROUBLESHOOTING.md`.

## Tests

```bash
uv pip install -e ".[dev]"
pytest                      # offline, no credentials, fakes for OCR/vision
pytest -m vision_live       # optional real-provider smoke test (needs env vars)
```

## License

MIT.
