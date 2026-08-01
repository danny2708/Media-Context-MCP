# Security

## Threat model

The server reads local files chosen by an AI agent and may send derived images
to a cloud vision provider. The three assets protected are: (1) files outside
the workspace, (2) secrets (API keys, and secret *content* inside media), and
(3) the integrity of the calling agent's context (prompt injection).

## Path sandbox

Every file access goes through `security/paths.py::resolve_media_path`, in this
fixed order: syntactic rejection (null bytes, any URL scheme, `data:` URIs, UNC
paths) → `~` expansion (never env-var expansion) → absolute resolution
(relative paths resolve against allowed roots, never the CWD) → `os.path.realpath`
canonicalisation (collapses symlinks, `..`, junctions, 8.3 names) → ancestry
check against each resolved root using `normcase` + `commonpath` (never a string
prefix — `/data/workspace-evil` does not pass for `/data/workspace`) → regular-
file check (rejects directories, fifos, devices).

Out-of-root paths get the same error whether or not the file exists, so the
sandbox does not leak existence. WSL↔Windows paths are refused with a
translation suggestion rather than auto-converted — a silent rewrite would
validate a path the caller never asked for.

## Resource limits

| Limit | Env var | Enforced |
|---|---|---|
| File size | `MEDIA_MCP_MAX_FILE_MB` | before any content read |
| Image pixels | `MEDIA_MCP_MAX_IMAGE_PIXELS` | from the header, before rasterisation; Pillow's bomb guard is set to the same number |
| Pages/slides | `MEDIA_MCP_MAX_PAGES` | before any rendering |
| Output size | `MEDIA_MCP_MAX_OUTPUT_CHARS` | at render, with explicit truncation report |
| Processing time | `MEDIA_MCP_PROCESS_TIMEOUT_SECONDS` | `asyncio.timeout` around the whole processor call, task cancelled on expiry |
| Provider response | 8 MB hard cap | in the provider client |
| Archives | — | `.zip` refused outright (amplification vector); OOXML zips are identified by central directory only, no member extraction |
| Remote fetching | — | all URL schemes refused; MarkItDown's URL-driven converters cannot activate (no `url` in `StreamInfo`) |

## Cloud privacy

- `MEDIA_MCP_ALLOW_CLOUD_VISION=false` (default): no image and no OCR text ever
  leaves the machine. `mode=vision` fails with `CLOUD_VISION_DISABLED`;
  `mode=auto` degrades to local extraction and says so in the result.
- `MEDIA_MCP_SEND_OCR_TO_CLOUD` independently gates OCR text in cloud requests.
- The gate lives in `pipeline.py`: processors receive the provider object only
  after the gate passes, so no code path can bypass it.
- **No automatic secret redaction is claimed.** Screenshots may contain tokens,
  keys, or proprietary code; the mitigation is explicit opt-in and local-only
  operation, not unreliable pattern scrubbing.

## Secrets handling

- The API key lives in a pydantic `SecretStr`; `show-config` and logs render
  `<set>`/`<unset>`.
- The key is not part of cache-key material and appears in no cache file
  (test-asserted).
- The structured logger scrubs secret-shaped field names; provider error bodies
  are excerpted with `Bearer` tokens stripped.
- `doctor` never prints secret values.

## Prompt injection

Media content is untrusted data. Defences, in order of strength:

1. **Architecture**: extracted text is returned as *data* in a structured field;
   the server itself never interprets or acts on media content.
2. **System prompt**: every vision call carries an explicit instruction that
   text visible in the image (and the OCR candidate) is content to report,
   never instructions to follow — including text addressed to the model.
3. **Framing**: the caller's question is wrapped in `<question>` tags and the
   OCR candidate in a code fence sized to survive embedded backticks, so
   neither can be confused with the instructions around them.
4. **Evidence typing**: `inference` and `visual` items are labelled, so a
   downstream agent can discount claims that exceed the pixels.

This is mitigation, not proof: a sufficiently suggestible VLM can still be
influenced by image content. The calling agent's own instructions (see
`AGENTS.example.md`) should treat tool output as data too.

## Error-code contract

Stable codes (`PATH_NOT_ALLOWED`, `FILE_TOO_LARGE`, `VISION_NOT_CONFIGURED`,
`CLOUD_VISION_DISABLED`, `VISION_AUTHENTICATION_FAILED`, `VISION_QUOTA_EXCEEDED`,
`VISION_RATE_LIMITED`, `VISION_MODEL_UNAVAILABLE`, `VISION_PROVIDER_TIMEOUT`,
`VISION_EMPTY_RESPONSE`, `VISION_INVALID_RESPONSE`, ...) are part of the public
API; each carries a `hint` written for an AI agent deciding what to do next.
Failures are never cached and never presented as results.

## Logging

STDERR only (STDOUT is the MCP protocol stream), JSON-per-line, request-id
correlated. Media content is not logged; image bytes and Base64 payloads never
reach a log sink.
