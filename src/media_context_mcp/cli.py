"""Diagnostic CLI.

``inspect`` runs the exact same pipeline as the MCP tool but outside the protocol,
which is what separates "media processing is broken" from "MCP transport is
broken" when integrating with a client.

This module prints to STDOUT freely -- it is a CLI, not the MCP server. The MCP
entry point is ``media-context-mcp-server`` / ``python -m media_context_mcp``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from pathlib import Path

from . import __version__
from .config import Settings, get_settings
from .errors import MediaContextError
from .logging_setup import configure_logging
from .models import AnalyzeMediaRequest

_OK = "  [ok]  "
_BAD = "  [FAIL]"
_WARN = "  [warn]"


def _print(status: str, message: str) -> None:
    print(f"{status} {message}")


# ------------------------------------------------------------------- doctor --


def cmd_doctor(settings: Settings, *, vision: bool, network: bool) -> int:
    failures = 0

    print(f"media-context-mcp {__version__} on Python {platform.python_version()} "
          f"({platform.system()} {platform.release()})")
    print()

    # Python version
    if sys.version_info >= (3, 11):
        _print(_OK, f"Python {platform.python_version()}")
    else:
        _print(_BAD, f"Python {platform.python_version()} -- 3.11+ required")
        failures += 1

    # MCP package
    try:
        import mcp

        _print(_OK, f"mcp package {getattr(mcp, '__version__', '(version unknown)')}")
    except ImportError:
        _print(_BAD, "mcp package not installed")
        failures += 1

    # MarkItDown
    try:
        from .processors.markitdown_adapter import get_markitdown, upstream_version

        get_markitdown()
        _print(_OK, f"MarkItDown {upstream_version()} initialised")
    except Exception as exc:  # noqa: BLE001
        _print(_BAD, f"MarkItDown failed to initialise: {exc}")
        failures += 1

    # PyMuPDF + basic PDF conversion
    try:
        import fitz

        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "doctor")
        text = page.get_text()
        document.close()
        assert "doctor" in text
        _print(_OK, f"PyMuPDF {fitz.pymupdf_version} (PDF create/extract round-trip)")
    except Exception as exc:  # noqa: BLE001
        _print(_BAD, f"PyMuPDF check failed: {exc}")
        failures += 1

    # Pillow + basic image decoding
    try:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
        buffer.seek(0)
        Image.open(buffer).load()
        _print(_OK, "Pillow image encode/decode round-trip")
    except Exception as exc:  # noqa: BLE001
        _print(_BAD, f"Pillow check failed: {exc}")
        failures += 1

    # OCR
    from .providers.tesseract import build_ocr_backend

    backend = build_ocr_backend(settings.ocr_backend, settings.tesseract_cmd)
    available, reason = backend.availability()
    if available:
        languages = backend.installed_languages()
        wanted = [lang for lang in settings.ocr_languages.split("+") if lang]
        missing = [lang for lang in wanted if languages and lang not in languages]
        _print(_OK, f"OCR: tesseract {backend.version}")
        if missing:
            _print(_WARN, f"OCR language pack(s) missing: {', '.join(missing)} "
                          f"(requested: {settings.ocr_languages})")
    else:
        _print(_WARN, f"OCR unavailable: {reason}")

    # Allowed roots
    roots = settings.resolved_roots()
    if not roots:
        _print(_BAD, "MEDIA_MCP_ALLOWED_ROOTS is not set -- every request will be rejected")
        failures += 1
    else:
        for root in roots:
            if root.is_dir():
                _print(_OK, f"allowed root: {root}")
            else:
                _print(_WARN, f"allowed root missing or not a directory: {root}")

    # Cache dir
    cache_dir = settings.resolved_cache_dir()
    if settings.cache_enabled:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            probe = cache_dir / ".doctor-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            _print(_OK, f"cache directory writable: {cache_dir}")
        except OSError as exc:
            _print(_WARN, f"cache directory not writable ({exc}); caching will be disabled")
    else:
        _print(_WARN, "cache disabled (MEDIA_MCP_CACHE_ENABLED=false)")

    # Config problems
    for problem in settings.problems():
        _print(_BAD if problem.fatal else _WARN, f"{problem.field}: {problem.message}")
        if problem.fatal:
            failures += 1

    # Vision (local checks; secrets never printed)
    if vision or network:
        failures += _doctor_vision(settings, network=network)
    else:
        if settings.vision_configured:
            state = "enabled" if settings.allow_cloud_vision else \
                "configured but BLOCKED (MEDIA_MCP_ALLOW_CLOUD_VISION=false)"
            _print(_OK, f"vision: {settings.vision_provider} / "
                        f"{settings.effective_vision_model} -- {state} "
                        "(run with --vision for detailed checks)")
        else:
            _print(_WARN, "vision not configured -- document and OCR analysis only")

    print()
    if failures:
        print(f"doctor: {failures} blocking problem(s) found.")
        return 1
    print("doctor: no blocking problems.")
    return 0


def _doctor_vision(settings: Settings, *, network: bool) -> int:
    failures = 0
    print()
    print("Vision provider checks:")

    if not settings.vision_configured:
        missing = []
        if not settings.effective_vision_base_url:
            missing.append("MEDIA_MCP_VISION_BASE_URL (or MEDIA_MCP_VISION_PROVIDER=huggingface)")
        if not settings.vision_model:
            missing.append("MEDIA_MCP_VISION_MODEL")
        if not settings.vision_api_key.get_secret_value():
            missing.append("MEDIA_MCP_VISION_API_KEY")
        _print(_BAD, "vision not configured; missing: " + ", ".join(missing))
        return 1

    _print(_OK, f"provider: {settings.vision_provider}")
    _print(_OK, f"base URL: {settings.effective_vision_base_url}")
    _print(_OK, f"model: {settings.effective_vision_model}")
    _print(_OK, "API key: <set>")  # value intentionally never shown
    _print(_OK, f"timeout: {settings.vision_timeout_seconds:.0f}s, "
                f"retries: {settings.vision_max_retries}, "
                f"max output tokens: {settings.vision_max_output_tokens}")
    if settings.vision_fallback_model_list:
        _print(_OK, f"explicit fallback models: {settings.vision_fallback_model_list}")
    if not settings.allow_cloud_vision:
        _print(_WARN, "cloud vision is DISABLED (MEDIA_MCP_ALLOW_CLOUD_VISION=false); "
                      "no image will be sent to the provider until it is set to true")

    # Client initialisation + image encoding, no network.
    try:
        from .providers.openai_compatible import build_vision_provider, data_url
        from .processors.imaging import PreprocessConfig, prepare_for_vision
        from PIL import Image

        provider = build_vision_provider(settings)
        assert provider is not None
        image = Image.new("RGB", (64, 32), (10, 20, 30))
        payloads, _ = prepare_for_vision(
            image,
            PreprocessConfig(
                max_pixels=settings.max_image_pixels,
                max_dimension=settings.vision_max_dimension,
                max_bytes=settings.vision_max_image_bytes,
                image_format=settings.vision_image_format,
            ),
            label="doctor probe",
        )
        url = data_url(payloads[0])
        assert url.startswith("data:image/")
        _print(_OK, "client initialised; test image encodes to a data URL")
    except Exception as exc:  # noqa: BLE001
        _print(_BAD, f"client/encoding check failed: {exc}")
        failures += 1

    if network:
        failures += _doctor_vision_network(settings)
    else:
        print("  (no network call made; add --network to send one tiny test image)")
    return failures


def _doctor_vision_network(settings: Settings) -> int:
    """One tiny real request. Reports latency and the model/provider that answered."""
    if not settings.allow_cloud_vision:
        _print(_BAD, "--network requested but MEDIA_MCP_ALLOW_CLOUD_VISION=false; "
                     "refusing to send anything")
        return 1

    from PIL import Image, ImageDraw

    from .processors.imaging import PreprocessConfig, prepare_for_vision
    from .providers.openai_compatible import build_vision_provider

    image = Image.new("RGB", (220, 60), (255, 255, 255))
    ImageDraw.Draw(image).text((10, 20), "MCP TEST 42", fill=(0, 0, 0))
    payloads, _ = prepare_for_vision(
        image,
        PreprocessConfig(
            max_pixels=settings.max_image_pixels,
            max_dimension=settings.vision_max_dimension,
            max_bytes=settings.vision_max_image_bytes,
            image_format=settings.vision_image_format,
        ),
        label="doctor network probe",
    )

    provider = build_vision_provider(settings)
    assert provider is not None

    async def _probe() -> int:
        started = time.perf_counter()
        try:
            result = await provider.analyze(
                images=payloads,
                prompt="Reply with exactly the text visible in this image and nothing else.",
                system=None,
                max_output_tokens=50,
                request_id="doctor",
            )
        except MediaContextError as error:
            _print(_BAD, f"network test failed: [{error.code.value}] {error.message}")
            if error.hint:
                print(f"         hint: {error.hint}")
            return 1
        finally:
            await provider.aclose()
        elapsed = (time.perf_counter() - started) * 1000
        served = result.actual_model or result.requested_model
        route = f" via {result.provider_route}" if result.provider_route else ""
        _print(_OK, f"network test: {elapsed:.0f} ms, served by {served}{route}")
        _print(_OK, f"response excerpt: {result.content.strip()[:80]!r}")
        if "42" not in result.content:
            _print(_WARN, "the model did not clearly read the test text; "
                          "check that the model accepts image input")
        return 0

    return asyncio.run(_probe())


# ------------------------------------------------------------------ inspect --


def cmd_inspect(settings: Settings, args: argparse.Namespace) -> int:
    from .pipeline import build_pipeline

    pipeline = build_pipeline(settings)
    request = AnalyzeMediaRequest(
        path=args.path,
        question=args.question,
        mode=args.mode,
        pages=args.pages,
        detail=args.detail,
        max_chars=args.max_chars or settings.max_output_chars,
        force_refresh=args.force_refresh,
    )

    async def _run() -> int:
        try:
            result = await pipeline.analyze(request)
        except MediaContextError as error:
            print(json.dumps(error.to_dict(), indent=2, ensure_ascii=False))
            return 1
        finally:
            await pipeline.aclose()
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(result.markdown)
            print()
            print(f"-- processor: {result.processing.processor} "
                  f"{result.processing.processor_version}; "
                  f"cached: {result.processing.cached}; "
                  f"{result.processing.duration_ms} ms; "
                  f"cache key: {result.cache_key[:16]}...")
        return 0

    return asyncio.run(_run())


# -------------------------------------------------------------------- misc --


def cmd_show_config(settings: Settings) -> int:
    print(json.dumps(settings.redacted_dump(), indent=2, ensure_ascii=False, default=str))
    problems = settings.problems()
    if problems:
        print()
        for problem in problems:
            marker = "FATAL" if problem.fatal else "warn"
            print(f"[{marker}] {problem.field}: {problem.message}")
    return 0


def cmd_clear_cache(settings: Settings) -> int:
    from .cache import CacheStore

    store = CacheStore(settings.resolved_cache_dir(), enabled=True)
    removed = store.clear()
    print(f"Removed {removed} cache entr{'y' if removed == 1 else 'ies'} from "
          f"{settings.resolved_cache_dir()}")
    return 0


def cmd_benchmark(settings: Settings, args: argparse.Namespace) -> int:
    """Run the expected-facts benchmark over the generated fixture set."""
    from .benchmark import run_benchmark

    return run_benchmark(settings, live=args.live)


# -------------------------------------------------------------------- main --


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="media-context-mcp",
        description="Diagnostics for the Media Context MCP server. "
        "(The MCP server itself is 'media-context-mcp-server'.)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check the installation and configuration")
    doctor.add_argument("--vision", action="store_true",
                        help="run detailed local vision-provider checks")
    doctor.add_argument("--network", action="store_true",
                        help="also send one tiny test image to the vision provider")

    inspect = subparsers.add_parser(
        "inspect", help="run the analysis pipeline on a file, outside MCP"
    )
    inspect.add_argument("path")
    inspect.add_argument("-q", "--question", default=None)
    inspect.add_argument("--mode", default="auto",
                         choices=["auto", "document", "ocr", "vision"])
    inspect.add_argument("--pages", default=None)
    inspect.add_argument("--detail", default="normal",
                         choices=["compact", "normal", "full"])
    inspect.add_argument("--max-chars", type=int, default=None)
    inspect.add_argument("--force-refresh", action="store_true")
    inspect.add_argument("--json", action="store_true", help="print the full JSON result")

    subparsers.add_parser("show-config", help="print effective configuration (secrets redacted)")
    subparsers.add_parser("clear-cache", help="delete all cache entries")

    benchmark = subparsers.add_parser(
        "benchmark", help="run the expected-facts regression benchmark on the fixture set"
    )
    benchmark.add_argument("--live", action="store_true",
                           help="allow real vision-provider calls (costs quota)")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_file)

    if args.command == "doctor":
        return cmd_doctor(settings, vision=args.vision, network=args.network)
    if args.command == "inspect":
        return cmd_inspect(settings, args)
    if args.command == "show-config":
        return cmd_show_config(settings)
    if args.command == "clear-cache":
        return cmd_clear_cache(settings)
    if args.command == "benchmark":
        return cmd_benchmark(settings, args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
