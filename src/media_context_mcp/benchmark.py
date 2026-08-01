"""Expected-facts regression benchmark.

Not a scientific evaluation: an engineering regression suite. Each fixture has a
set of checkable facts (substrings that must appear, plus routing expectations);
the benchmark runs the real pipeline over them and reports per-fixture accuracy,
latency, output size, provider/model, cache state, and failure type.

Without ``--live`` no vision-provider call is ever made: fixtures whose facts
require vision are skipped with an explicit note, and OCR fixtures run only if a
local OCR backend exists. That keeps the default run free and offline.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .errors import MediaContextError
from .models import AnalyzeMediaRequest


@dataclass
class BenchmarkCase:
    name: str
    fixture: str  # key in the generated-fixture map
    question: str | None
    mode: str = "auto"
    # Substrings that must appear somewhere in the returned markdown.
    must_contain: list[str] = field(default_factory=list)
    needs_ocr: bool = False
    needs_vision: bool = False


CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        name="text-pdf-extraction",
        fixture="text_pdf",
        question="Identify the deployment requirements in this PDF.",
        must_contain=["Node.js 20", "healthz", "TLS"],
    ),
    BenchmarkCase(
        name="docx-structure",
        fixture="docx",
        question=None,
        must_contain=["Release Checklist", "Export crash", "Mai"],
    ),
    BenchmarkCase(
        name="xlsx-sheets",
        fixture="xlsx",
        question="Extract the incidents table.",
        must_contain=["report-export", "auth-gateway", "Incidents"],
    ),
    BenchmarkCase(
        name="terminal-error-ocr",
        fixture="terminal_png",
        question="Read the exact error message from this terminal screenshot.",
        mode="ocr",
        must_contain=["TS2345", "report.ts"],
        needs_ocr=True,
    ),
    BenchmarkCase(
        name="terminal-error-vision",
        fixture="terminal_png",
        question="Read the visible error message and identify the likely cause.",
        must_contain=["TS2345"],
        needs_vision=True,
    ),
    BenchmarkCase(
        name="ui-screenshot-vision",
        fixture="ui_png",
        question="Which button is disabled, and what validation error is shown?",
        must_contain=["Save", "email"],
        needs_vision=True,
    ),
    BenchmarkCase(
        name="scanned-pdf-fallback",
        fixture="scanned_pdf",
        question="Extract the invoice number and the total due.",
        must_contain=["2026-0417", "1650"],
        needs_ocr=True,  # satisfiable by OCR alone; vision also acceptable
    ),
]


def run_benchmark(settings: Settings, *, live: bool) -> int:
    from .pipeline import build_pipeline

    fixtures_dir = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    sys.path.insert(0, str(fixtures_dir))
    try:
        from generate import generate_all  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    generated = generate_all()
    fixture_root = next(iter(generated.values())).parent

    # The benchmark points the sandbox at the fixture directory explicitly, so it
    # works regardless of the operator's MEDIA_MCP_ALLOWED_ROOTS.
    settings = settings.model_copy(update={"allowed_roots": [fixture_root]})
    if not live:
        settings = settings.model_copy(update={"allow_cloud_vision": False})

    pipeline = build_pipeline(settings)
    ocr_available = pipeline.ocr is not None and pipeline.ocr.availability()[0]
    vision_available = settings.cloud_vision_usable

    print(f"benchmark: ocr={'yes' if ocr_available else 'NO'}, "
          f"vision={'LIVE' if vision_available and live else 'off'}, "
          f"fixtures at {fixture_root}")
    print()

    async def _run() -> int:
        failures = 0
        for case in CASES:
            if case.needs_ocr and not ocr_available and not (vision_available and live):
                print(f"  SKIP {case.name}: needs OCR (tesseract not available)")
                continue
            if case.needs_vision and not (vision_available and live):
                print(f"  SKIP {case.name}: needs a live vision provider (--live + "
                      "MEDIA_MCP_ALLOW_CLOUD_VISION=true)")
                continue

            request = AnalyzeMediaRequest(
                path=str(generated[case.fixture]),
                question=case.question,
                mode=case.mode,  # type: ignore[arg-type]
                max_chars=settings.max_output_chars,
            )
            started = time.perf_counter()
            try:
                result = await pipeline.analyze(request)
            except MediaContextError as error:
                print(f"  FAIL {case.name}: [{error.code.value}] {error.message}")
                failures += 1
                continue
            elapsed = (time.perf_counter() - started) * 1000

            haystack = result.markdown
            hits = [fact for fact in case.must_contain if fact in haystack]
            missed = [fact for fact in case.must_contain if fact not in haystack]
            ok = not missed
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures += 1
            model = result.processing.model or "-"
            print(f"  {status} {case.name}: {len(hits)}/{len(case.must_contain)} facts, "
                  f"{elapsed:.0f} ms, {len(result.markdown)} chars, "
                  f"processor={result.processing.processor}, model={model}, "
                  f"cached={result.processing.cached}")
            if missed:
                print(f"        missing facts: {missed}")
        await pipeline.aclose()
        return failures

    failures = asyncio.run(_run())
    print()
    if failures:
        print(f"benchmark: {failures} case(s) failed.")
        return 1
    print("benchmark: all runnable cases passed.")
    return 0
