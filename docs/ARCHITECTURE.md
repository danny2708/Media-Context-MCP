# Architecture

## The pipeline

```text
MCP client (OpenCode, Claude Code, ...)
    │  JSON-RPC over STDIO
    ▼
server.py            thin MCP adapter (mcp 2.x MCPServer); errors → structured payloads
    ▼
pipeline.py          validate → detect → route → cache-get → process → cache-put → render
    │
    ├─ security/paths.py     canonical-path sandbox (allowed roots)
    ├─ security/limits.py    size / pixel / page / timeout budgets
    ├─ media_info.py         magic-bytes + extension detection, SHA-256
    ├─ routing/heuristics.py pure-function routing + question-intent classifier
    ├─ cache/                content-addressed JSON store, atomic writes
    ├─ render.py             detail shaping, truncation reporting, final Markdown
    │
    └─ processors/
         text.py                 .txt/.md/... verbatim (encoding-aware)
         markitdown_adapter.py   DOCX/PPTX/XLSX/XLS/CSV/HTML/EPUB/MSG/IPYNB
         pdf.py                  PyMuPDF text layer + per-page OCR/vision fallback
         image.py                OCR-only / vision / hybrid plans
              │
              ├─ providers/tesseract.py          local OCR (OcrBackend)
              └─ providers/openai_compatible.py  cloud VLM (VisionProvider)
                    ▲ prompts/                    profiles, versioned; parse.py normalises replies
```

`pipeline.py` is MCP-independent: `cli.py inspect` runs the identical path
headless, which separates media-processing failures from transport failures.

## Key decisions and why

**PDFs bypass MarkItDown.** MarkItDown's `PdfConverter` returns one flat string
with no page boundaries — the `pages` parameter and per-page evidence are
impossible through it. PyMuPDF gives page-addressed extraction *and* page
rendering for the scanned fallback. MarkItDown keeps everything Office-shaped.

**MarkItDown structure is verified, not trusted.** PPTX slide markers
(`<!-- Slide number: N -->`) and XLSX sheet headings are an empirical, not
documented, format — and slide *content* can contain a literal marker string
(verified). The adapter cross-checks parsed markers against python-pptx's real
slide count and openpyxl's sheet names; on mismatch it degrades to one
unstructured block with a warning instead of mislabeling evidence.

**OCR and vision are complementary, not alternatives.** OCR is the trusted
source for exact characters; a VLM is the only source for layout, state and
relationships. The hybrid image plan runs OCR first, answers clearly
text-extraction questions locally when OCR quality suffices
(`ocr_quality_sufficient`: ≥40 chars and ≥65% confidence when reported), and
otherwise escalates to vision with the OCR text attached as an explicitly
untrusted candidate. Evidence items are typed (`ocr` / `visual` / `inference`)
so a caller can rank trust.

**Routing is pure functions.** `decide_route(info, mode, question, caps)` takes
plain data and returns a decision with a human-readable reason (returned to the
caller). The question-intent classifier is a keyword regex, deliberately: it is
transparent, testable, and its worst failure mode is a slightly less targeted
prompt, never a wrong answer.

**Providers are injected.** Processors receive `VisionProvider`/`OcrBackend`
through `ProcessingContext`; nothing imports a vendor. The Hugging Face
"provider" is a configuration preset over the OpenAI-compatible client (base
URL + optional `X-HF-Bill-To` header), not a separate code path. A local VLM
(Ollama/vLLM) is a future `VisionProvider` implementation, no pipeline change
needed.

**The cache stores full ProcessorResults, pre-truncation.** Truncation and
detail shaping happen at render time, so callers with different `max_chars`
budgets share one entry. The key covers everything that shapes output —
content hash, mode/detail/pages, normalised question hash, processor+version,
prompt version, OCR config and cloud-inclusion flag, provider/model/route,
preprocessing parameters — and deliberately excludes the file path, `max_chars`,
and any secret. Only successes are stored; failures always re-run.

**Cloud access has one gate.** `pipeline.py` hands the vision provider to a
processor only when `vision_configured AND allow_cloud_vision`. No processor
can reach the network without passing that line.

## Structure deltas from the original spec

| Spec | Here | Why |
|---|---|---|
| `logging.py` | `logging_setup.py` | a package module named `logging` shadows the stdlib for relative imports |
| `routing/router.py` class | dict lookup in `pipeline.py` | a router class over four entries is indirection without value |
| `processors/ocr.py` + `vision.py` | `ocr.py` (helpers) + `image.py` (plans) | the hybrid plan needs one owner for the OCR→vision escalation decision |
| — | `pipeline.py`, `render.py`, `media_info.py`, `benchmark.py` | keeps `server.py` thin and the pipeline headless-runnable |

## Extension points (future milestones, not implemented)

- **Local VLM**: implement `VisionProvider` for an Ollama/vLLM endpoint; wire in
  `build_vision_provider`.
- **Audio**: add a `MediaCategory.AUDIO` processor; detection and refusal
  messaging already exist.
- **`get_media_section` / artifact IDs**: the cache key + stored ProcessorResult
  is already addressable content; a section-retrieval tool can read it.
- **Embeddings/retrieval**: chunk the stored `content_markdown` downstream; the
  MCP stays the extraction layer.
- **HTTP transport**: `MCPServer.run(transport=...)` supports it; blocked
  deliberately until per-client authorization exists.
