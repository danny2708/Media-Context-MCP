# Troubleshooting

First move, always:

```bash
media-context-mcp doctor              # blocking problems?
media-context-mcp inspect <path> -q "test"   # same pipeline, no MCP in the way
```

If `inspect` works but the MCP client fails, the problem is transport/config on
the client side. If `inspect` fails too, the error payload tells you which
component broke.

## Server won't connect in OpenCode

- Run the exact `command` array from your `opencode.json` in a terminal. The
  server should start silently and wait on stdin (logs go to stderr).
- Windows: use the **full path** to the venv's `python.exe` — OpenCode does not
  activate virtualenvs.
- The server is designed to start even when misconfigured (errors surface on
  the first tool call), so a spawn failure is almost always a wrong path or a
  broken Python environment, not a config value.

## `PATH_NOT_ALLOWED`

- `MEDIA_MCP_ALLOWED_ROOTS` must be set **in the MCP server's environment**
  (the `environment` block of `opencode.json`), not just in your shell.
- Windows roots use `;` between entries, POSIX uses `:` — or use a JSON array.
- Symlinked files must *resolve* inside a root; the target is what is checked.
- `/mnt/d/...` vs `D:\...`: paths are never translated across the WSL boundary;
  the error message shows the spelling the server needs.

## `OCR_NOT_CONFIGURED` / OCR unavailable

- The `tesseract` binary must be installed and on PATH, or pointed to via
  `MEDIA_MCP_TESSERACT_CMD`.
  - Windows: `winget install UB-Mannheim.TesseractOCR` then
    `MEDIA_MCP_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`
  - macOS: `brew install tesseract tesseract-lang`
  - Debian/Ubuntu: `apt install tesseract-ocr tesseract-ocr-vie`
- `doctor` reports missing language packs for `MEDIA_MCP_OCR_LANGUAGES`
  (e.g. `vie` needs `tesseract-ocr-vie`). Missing packs degrade with a warning,
  they do not fail the request.

## `VISION_NOT_CONFIGURED` / `CLOUD_VISION_DISABLED`

- All three of base URL (or the `huggingface` preset), model, and API key must
  be set — partial configuration is reported by `doctor`.
- `CLOUD_VISION_DISABLED` means vision *is* configured but
  `MEDIA_MCP_ALLOW_CLOUD_VISION` is not `true`. That gate is deliberate: images
  may contain secrets. Decide, then set it.
- `media-context-mcp doctor --vision --network` sends one tiny generated image
  and reports latency, the serving model, and auth/quota/rate-limit errors
  distinctly.

## Vision provider errors

| Code | Meaning | Fix |
|---|---|---|
| `VISION_AUTHENTICATION_FAILED` | 401 — bad key | check the key; HF tokens need inference permission |
| `VISION_PERMISSION_DENIED` | 403 — key valid, model gated | accept the model license / upgrade tier |
| `VISION_QUOTA_EXCEEDED` | 402/429-quota — credits gone | add credits or change model; free credits are trial capacity |
| `VISION_RATE_LIMITED` | 429 — too fast | wait (the error carries `retry_after_seconds` when known) |
| `VISION_MODEL_UNAVAILABLE` | 404/410 or persistent 5xx | model name wrong/retired, or base URL not the API root; the server never substitutes a model silently |
| `VISION_PROVIDER_TIMEOUT` | no answer in time | raise `MEDIA_MCP_VISION_TIMEOUT_SECONDS` or pick a faster model |
| `VISION_EMPTY_RESPONSE` | model returned nothing | often a content filter; retry or switch models — not cached, so a retry is a real retry |
| `VISION_INVALID_RESPONSE` | non-OpenAI-compatible shape | base URL must serve `/chat/completions` |

## Scanned PDF came back with "content NOT included"

That is the honest degradation: the page has no text layer and neither OCR nor
permitted vision is available. Install tesseract (local) or enable cloud vision.

## Output truncated

The `truncation` field reports original vs returned size. Narrow the request:
`pages="2-3"`, `detail="compact"`, a more specific `question`, or raise
`max_chars` (capped by `MEDIA_MCP_MAX_OUTPUT_CHARS`).

## Cache seems stale / wrong

It can't be stale from file edits (keys hash content), but config changes that
alter output (model, prompt version, OCR languages) change the key automatically.
`force_refresh=true` bypasses; `media-context-mcp clear-cache` resets. Corrupt
entries self-delete on read.

## MarkItDown failed to initialise ("bad allocation")

MarkItDown loads the magika ONNX model at construction; on a memory-starved
host this can fail. The server reports it per-call; plain-text and PDF analysis
do not need MarkItDown. Free memory or reinstall `onnxruntime`.

## Logs

STDERR, JSON per line, `request_id` correlates one call's lines. Set
`MEDIA_MCP_LOG_LEVEL=DEBUG` and optionally `MEDIA_MCP_LOG_FILE` for a persistent
copy. If you ever see non-JSON on STDOUT, that is a bug — report it.
