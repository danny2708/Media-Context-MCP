# Implementation report — Media Context MCP v0.1.0

Date: 2026-08-01. Environment: Windows 11, Python 3.12.10, uv 0.11.21.

## Implemented

- **`analyze_media` MCP tool** over STDIO (mcp 2.0 `MCPServer`), structured
  output, full parameter set (`path`, `question`, `mode`, `pages`, `detail`,
  `max_chars`, `force_refresh`), stable error codes with agent-oriented hints.
- **Processors**: plain text (encoding-aware, verbatim); MarkItDown adapter for
  DOCX/PPTX/XLSX/XLS/CSV/HTML/EPUB/MSG/IPYNB with slide/sheet evidence verified
  against python-pptx/openpyxl ground truth; PyMuPDF PDF processor with
  page-addressed text, page selection, per-page density check, and OCR/vision
  fallback at bounded concurrency (one vision request per page); image processor
  executing three plans (ocr_only / vision / hybrid).
- **Hybrid OCR+vision**: deterministic keyword intent classifier;
  text-extraction questions answered locally when OCR quality suffices;
  otherwise vision with the OCR text attached as an explicitly untrusted
  candidate transcription. Evidence typed `ocr`/`text`/`page`/`visual`/`inference`.
- **Vision providers**: `VisionProvider` protocol (multi-image, request-id,
  token cap); OpenAI-compatible Chat Completions client with full error mapping
  (401/402/403/404/408/429/5xx → 10 stable codes), bounded jittered retry,
  Retry-After capture, cancellation respected, response-size cap, no baseline
  assumptions about usage/metadata/Responses-API. Hugging Face is a config
  preset (router base URL + optional `X-HF-Bill-To`), not a code path.
- **Cloud privacy gates**: `MEDIA_MCP_ALLOW_CLOUD_VISION` (default **false**)
  enforced at the single point where the provider is handed to processors;
  `MEDIA_MCP_SEND_OCR_TO_CLOUD` separately gates OCR forwarding.
- **Prompts**: 8 versioned profiles (general, ui_screenshot, terminal_or_error,
  code_screenshot, table, chart, diagram_or_flowchart, scanned_document), each
  request carrying the injection defence and the 5-section separation
  (ANSWER / DIRECT OBSERVATIONS / EXACT TEXT / VISUAL INTERPRETATION /
  INFERENCE / UNCERTAINTY); tolerant parser that never discards an
  unstructured-but-successful reply.
- **Image preprocessing**: in-memory only, EXIF orientation, bomb guard from
  header, PNG-first format policy, downscale only when forced, overlapping
  tiling (≤4 tiles, ordered, source regions recorded) for long screenshots,
  original+processed dimensions recorded.
- **Cache**: content-addressed JSON store, atomic writes (`os.replace`),
  corruption self-heals to a miss, TTL + LRU-by-mtime size cleanup, schema
  version, successes only; key covers hash/mode/detail/pages/question-hash/
  processor+version/prompt-version/OCR config+inclusion/provider+model+route/
  preprocessing params; excludes path, `max_chars` (truncation applied at
  render), and all secrets.
- **Security**: canonical-path sandbox with ancestry checks; size/pixel/page/
  output/time limits; URL and archive refusal; magic-byte detection that
  overrides lying extensions; secret redaction in logs/config dumps.
- **CLI**: `doctor` (+ `--vision`, `--vision --network`), `inspect`,
  `show-config`, `clear-cache`, `benchmark [--live]`.
- **Docs**: README, ARCHITECTURE, SECURITY, TROUBLESHOOTING, AGENTS.example.md,
  .env.example, this report. Lock file: `uv.lock` (82 packages).

## Not implemented (by design, per spec §27)

Embeddings/vector DB, semantic retrieval, `get_media_section`, persistent
artifact IDs, audio/video, HTTP transport, per-client auth, shared cache,
automatic interception, cost-aware native-vision selection. Extension points
documented in ARCHITECTURE.md. Also not implemented: automatic secret redaction
(explicitly out of scope as unreliable; gates instead), and the explicit
fallback-model iteration loop — `MEDIA_MCP_VISION_FALLBACK_MODELS` is parsed,
surfaced in `doctor`, and reserved, but the provider does not yet iterate over
it (documented limitation).

## Key architectural decisions

1. PDFs handled by PyMuPDF, not MarkItDown — MarkItDown's PDF output has no
   page boundaries, making `pages` and page evidence impossible.
2. MarkItDown slide/sheet structure verified against ground truth before any
   per-slide/per-sheet claim; degrade-with-warning on mismatch.
3. Hybrid image plan with the quality decision at runtime and the routing
   decision in pure functions.
4. One cloud gate in the pipeline rather than checks scattered in processors.
5. Cache stores pre-truncation results keyed on everything output-shaping.
6. Hugging Face as a preset over one OpenAI-compatible client.

## Supported formats

Images: png, jpg/jpeg, webp, bmp, tif/tiff, gif. Documents: pdf, docx, pptx,
xlsx, xls, csv, tsv, html/htm, txt, md, rst, log, json, ipynb, epub, msg.
Refused with actionable errors: zip (amplification), audio (future milestone),
unknown types.

## External dependencies (locked in uv.lock)

Runtime: mcp 2.0.0, markitdown 0.1.7 (extras: docx,pptx,xlsx,xls,pdf,outlook),
pymupdf 1.28.0, pillow 12.x, httpx 0.28.x, pydantic 2.x + pydantic-settings,
python-dotenv. Optional: pytesseract (+ OS tesseract binary). Dev: pytest,
pytest-asyncio, python-docx, ruff.

## How to run

```bash
uv venv && uv pip install -e ".[ocr,dev]"
media-context-mcp doctor
media-context-mcp inspect path/to/file.pdf -q "question"
media-context-mcp-server            # MCP STDIO (or: python -m media_context_mcp)
```

## How to configure OpenCode

See README.md §OpenCode integration — `opencode.json` `mcp` block with
`"type": "local"`, the venv's python + `-m media_context_mcp`, and the
`MEDIA_MCP_*` variables in `environment`. Add the AGENTS.example.md instruction.

## How to test

```bash
pytest                    # 163 tests, offline, no credentials
pytest -m requires_vision # live provider smoke test (needs env credentials)
media-context-mcp benchmark [--live]
```

## Verification status — what was and was not verified

**Verified on this machine (Windows 11, Python 3.12.10):**
- Full offline suite: **163 passed, 1 skipped** (symlink test — needs Windows
  developer mode), 1 deselected (live vision). `ruff check` clean.
- Real MCP STDIO round-trips (subprocess): initialize, tool discovery, valid
  call, structured PATH_NOT_ALLOWED error, STDOUT purity, honest vision error.
- Headless E2E: text PDF with page evidence + cache hit on second call; DOCX;
  PPTX slide selection; scanned PDF honest degradation; benchmark 3/3 runnable
  cases passed.
- Spec §24 phase-6 scenarios: #3,#4,#6,#7,#8,#9,#10 verified directly; #1,#2,#5
  (a real text-only model choosing to call the tool) verified at the protocol
  level only — they additionally depend on the client model's tool-use behaviour.

**NOT verified on this machine:**
- **Real tesseract OCR** — the binary is not installed here. All OCR logic is
  covered by deterministic fakes; run `media-context-mcp doctor` then
  `pytest` after `winget install UB-Mannheim.TesseractOCR` to verify live.
- **Real vision provider / Hugging Face** — no credentials were used. The HTTP
  client is covered by MockTransport tests for every error class. To verify
  live: set the vision env vars + `MEDIA_MCP_ALLOW_CLOUD_VISION=true`, then
  `media-context-mcp doctor --vision --network` and `pytest -m requires_vision`.
  **Hugging Face vision is therefore implemented but not live-tested.**
- **OpenCode end-to-end** — OpenCode is not installed in this environment. The
  config format was verified against current OpenCode documentation; the MCP
  handshake was verified against the real server over STDIO with the same
  message flow a client uses. Treat the OpenCode integration as
  documentation-verified, not runtime-verified.

## Known limitations

- Intent classification is keyword-based English; unusual phrasing falls back
  to vision-when-permitted (safe, but may cost a provider call).
- PDF density threshold misclassifies legitimately sparse pages as scanned
  (cost: a wasted OCR/vision pass, reported in `fallbacks_used`).
- OCR quality gate trusts tesseract's confidence; engines reporting none are
  assumed adequate at ≥40 chars.
- `pages` applies to PDFs and PPTX slides only (XLSX sheets are all returned;
  warning issued) — sheet selection is a possible future parameter.
- Vision tiling sends tiles in one request; models that ignore multi-image
  ordering may mis-attribute content across tile boundaries.
- MarkItDown marker formats are an undocumented upstream contract; an upgrade
  can silently remove per-slide/per-sheet evidence (degrades with warning,
  caught by the pinned `<0.2` range + fixture tests).

## Recommended next milestones

1. Install tesseract + run a live HF smoke test; record results here.
2. Real OpenCode session with DeepSeek: measure whether the AGENTS instruction
   reliably triggers tool use; tune the tool description if not.
3. Explicit fallback-model iteration in the provider (config already exists).
4. Local VLM provider (Ollama/vLLM) — removes the cloud gate for home use.
5. `get_media_section` tool over the existing cache entries.
6. Sheet selection for XLSX; region-of-interest crops for large screenshots.
