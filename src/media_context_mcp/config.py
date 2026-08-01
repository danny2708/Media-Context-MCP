"""Configuration loading and validation.

All settings come from ``MEDIA_MCP_*`` environment variables, optionally seeded
from a ``.env`` file. Loading is deliberately lenient: the process must be able to
start even when it is misconfigured, because an MCP server that dies during spawn
shows the user nothing but "failed to connect". Problems are surfaced through
:meth:`Settings.problems` -- fatal ones become a structured ``CONFIGURATION_ERROR``
on the first tool call, where an agent can actually read them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AutoImageStrategy = Literal["ocr_first", "vision_first", "hybrid"]
OcrBackendName = Literal["tesseract", "none"]
VisionProviderKind = Literal["openai-compatible", "huggingface"]
VisionImageFormat = Literal["auto", "png", "jpeg"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass(frozen=True)
class ConfigProblem:
    """A configuration defect found at load time."""

    field: str
    message: str
    fatal: bool
    hint: str


class Settings(BaseSettings):
    """Runtime configuration. See ``.env.example`` for documentation of each field."""

    model_config = SettingsConfigDict(
        env_prefix="MEDIA_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- security -----------------------------------------------------------
    # NoDecode: pydantic-settings would otherwise json.loads a list-typed env
    # value before our validator can handle the pathsep-separated form.
    allowed_roots: Annotated[list[Path], NoDecode] = Field(default_factory=list)

    # --- limits -------------------------------------------------------------
    max_file_mb: int = 50
    max_pages: int = 30
    max_image_pixels: int = 40_000_000
    max_output_chars: int = 30_000
    process_timeout_seconds: int = 120

    # --- cache --------------------------------------------------------------
    cache_enabled: bool = True
    cache_dir: Path = Path(".media-context-mcp/cache")
    cache_max_mb: int = 1024
    cache_ttl_days: int = 30

    # --- OCR ----------------------------------------------------------------
    ocr_backend: OcrBackendName = "tesseract"
    ocr_languages: str = "eng"
    tesseract_cmd: str | None = None

    # --- vision -------------------------------------------------------------
    vision_provider: VisionProviderKind = "openai-compatible"
    vision_base_url: str = ""
    vision_api_key: SecretStr = SecretStr("")
    vision_model: str = ""
    vision_route: str = ""
    vision_fallback_models: str = ""
    vision_timeout_seconds: float = 60.0
    vision_max_retries: int = 1
    vision_max_output_tokens: int = 1400
    # Hugging Face router extras; only meaningful when vision_provider=huggingface.
    hf_provider_policy: str = ""
    hf_bill_to: str = ""

    # --- cloud privacy gates ------------------------------------------------
    # Both gates are enforced in the pipeline, not in the provider: nothing can
    # reach a provider object unless the gate has already said yes.
    allow_cloud_vision: bool = False
    send_ocr_to_cloud: bool = True

    # --- vision image preprocessing ------------------------------------------
    vision_max_dimension: int = 4096
    vision_max_image_bytes: int = 10_485_760
    vision_image_format: VisionImageFormat = "auto"

    # --- routing ------------------------------------------------------------
    pdf_min_chars_per_page: int = 120
    pdf_render_dpi: int = 160
    auto_image_strategy: AutoImageStrategy = "hybrid"

    # --- logging ------------------------------------------------------------
    log_level: LogLevel = "INFO"
    log_file: Path | None = None

    @field_validator("allowed_roots", mode="before")
    @classmethod
    def _parse_roots(cls, value: object) -> object:
        """Accept a JSON array, an os.pathsep-separated list, or a real list."""
        if value is None or isinstance(value, list):
            return value or []
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"MEDIA_MCP_ALLOWED_ROOTS looks like JSON but does not parse: {exc}"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError("MEDIA_MCP_ALLOWED_ROOTS JSON must be an array of paths")
            return [str(item) for item in parsed]
        # os.pathsep is ';' on Windows and ':' elsewhere. Splitting a Windows path on
        # ':' would shred drive letters, which is exactly why we use os.pathsep and
        # not a hard-coded separator.
        return [part for part in (p.strip() for p in text.split(os.pathsep)) if part]

    @field_validator("ocr_languages")
    @classmethod
    def _clean_languages(cls, value: str) -> str:
        cleaned = "+".join(part for part in (p.strip() for p in value.split("+")) if part)
        return cleaned or "eng"

    @field_validator("vision_base_url")
    @classmethod
    def _strip_base_url(cls, value: str) -> str:
        return value.strip().rstrip("/")

    # --- derived state ------------------------------------------------------

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def cache_max_bytes(self) -> int:
        return self.cache_max_mb * 1024 * 1024

    @property
    def vision_configured(self) -> bool:
        """Vision needs all three of base URL, model and key to be usable.

        An API key alone is not enough, and we never fall back to a default vendor
        endpoint: a hidden network call to an unexpected host is worse than an
        honest VISION_NOT_CONFIGURED. The huggingface preset supplies a default
        base URL, so for it only model + key are required.
        """
        return bool(
            self.effective_vision_base_url
            and self.vision_model
            and self.vision_api_key.get_secret_value()
        )

    @property
    def effective_vision_base_url(self) -> str:
        """Base URL after applying the provider preset.

        ``huggingface`` is a *preset*, not a separate implementation: it fills in
        the router endpoint so the OpenAI-compatible client can talk to it. An
        explicit MEDIA_MCP_VISION_BASE_URL always wins.
        """
        if self.vision_base_url:
            return self.vision_base_url
        if self.vision_provider == "huggingface":
            return "https://router.huggingface.co/v1"
        return ""

    @property
    def effective_vision_model(self) -> str:
        """Model id with any explicit HF provider-policy suffix applied.

        HF routes with ``model:policy`` suffixes (e.g. ``:fastest``). We only apply
        one when the operator asked for it -- dynamic 'fastest' routing is never a
        default because it destroys reproducibility and cache validity.
        """
        model = self.vision_model
        if (
            self.vision_provider == "huggingface"
            and self.hf_provider_policy
            and ":" not in model
        ):
            model = f"{model}:{self.hf_provider_policy}"
        return model

    @property
    def vision_fallback_model_list(self) -> list[str]:
        return [m for m in (part.strip() for part in self.vision_fallback_models.split(",")) if m]

    @property
    def cloud_vision_usable(self) -> bool:
        """True only when vision is configured AND the operator opted in to cloud."""
        return self.vision_configured and self.allow_cloud_vision

    @property
    def ocr_configured(self) -> bool:
        return self.ocr_backend != "none"

    def resolved_roots(self) -> list[Path]:
        """Allowed roots as canonical, symlink-resolved absolute paths."""
        roots: list[Path] = []
        for root in self.allowed_roots:
            try:
                resolved = Path(os.path.expanduser(str(root))).resolve(strict=False)
            except (OSError, RuntimeError):
                continue
            if resolved not in roots:
                roots.append(resolved)
        return roots

    def resolved_cache_dir(self) -> Path:
        return Path(os.path.expanduser(str(self.cache_dir))).resolve(strict=False)

    def problems(self) -> list[ConfigProblem]:
        """Validate settings that pydantic cannot express as types.

        Only ``fatal`` problems block tool calls. Everything else is a warning the
        caller sees in ``doctor`` output but which does not stop document work.
        """
        found: list[ConfigProblem] = []

        if not self.allowed_roots:
            found.append(
                ConfigProblem(
                    field="MEDIA_MCP_ALLOWED_ROOTS",
                    message="No allowed roots configured; every file access will be rejected.",
                    fatal=True,
                    hint=(
                        "Set MEDIA_MCP_ALLOWED_ROOTS to the absolute path of the workspace "
                        "the server may read, e.g. MEDIA_MCP_ALLOWED_ROOTS=/home/me/project "
                        "(use ';' between roots on Windows, ':' elsewhere)."
                    ),
                )
            )
        for root, resolved in zip(self.allowed_roots, self.resolved_roots(), strict=False):
            if not resolved.exists():
                found.append(
                    ConfigProblem(
                        field="MEDIA_MCP_ALLOWED_ROOTS",
                        message=f"Allowed root does not exist: {root}",
                        fatal=False,
                        hint="Create the directory or remove it from MEDIA_MCP_ALLOWED_ROOTS.",
                    )
                )
            elif not resolved.is_dir():
                found.append(
                    ConfigProblem(
                        field="MEDIA_MCP_ALLOWED_ROOTS",
                        message=f"Allowed root is not a directory: {root}",
                        fatal=False,
                        hint="Allowed roots must be directories, not files.",
                    )
                )

        if self.max_output_chars < 500:
            found.append(
                ConfigProblem(
                    field="MEDIA_MCP_MAX_OUTPUT_CHARS",
                    message="max_output_chars below 500 makes almost every result truncated.",
                    fatal=False,
                    hint="Use at least 2000; the default of 30000 suits most agents.",
                )
            )
        if self.max_pages < 1:
            found.append(
                ConfigProblem(
                    field="MEDIA_MCP_MAX_PAGES",
                    message="max_pages must be at least 1.",
                    fatal=True,
                    hint="Set MEDIA_MCP_MAX_PAGES to a positive integer.",
                )
            )
        if self.max_file_mb < 1:
            found.append(
                ConfigProblem(
                    field="MEDIA_MCP_MAX_FILE_MB",
                    message="max_file_mb must be at least 1.",
                    fatal=True,
                    hint="Set MEDIA_MCP_MAX_FILE_MB to a positive integer.",
                )
            )

        partial_vision = any(
            [self.vision_base_url, self.vision_model, self.vision_api_key.get_secret_value()]
        )
        if partial_vision and not self.vision_configured:
            missing = [
                name
                for name, present in (
                    (
                        "MEDIA_MCP_VISION_BASE_URL",
                        bool(self.effective_vision_base_url),
                    ),
                    ("MEDIA_MCP_VISION_MODEL", bool(self.vision_model)),
                    ("MEDIA_MCP_VISION_API_KEY", bool(self.vision_api_key.get_secret_value())),
                )
                if not present
            ]
            found.append(
                ConfigProblem(
                    field="MEDIA_MCP_VISION_*",
                    message=f"Vision is partially configured; missing: {', '.join(missing)}.",
                    fatal=False,
                    hint=(
                        "Set the missing variables, or clear them all to disable vision. "
                        "With MEDIA_MCP_VISION_PROVIDER=huggingface the base URL is "
                        "preset; only the model and API key are needed."
                    ),
                )
            )

        if self.vision_configured and not self.allow_cloud_vision:
            found.append(
                ConfigProblem(
                    field="MEDIA_MCP_ALLOW_CLOUD_VISION",
                    message=(
                        "A vision provider is configured, but cloud vision is disabled "
                        "(MEDIA_MCP_ALLOW_CLOUD_VISION=false). No image or OCR text "
                        "will be sent to the provider."
                    ),
                    fatal=False,
                    hint=(
                        "Set MEDIA_MCP_ALLOW_CLOUD_VISION=true to permit sending media "
                        "to the configured provider. Screenshots may contain secrets or "
                        "proprietary code -- opt in deliberately."
                    ),
                )
            )

        return found

    def fatal_problems(self) -> list[ConfigProblem]:
        return [problem for problem in self.problems() if problem.fatal]

    def redacted_dump(self) -> dict[str, object]:
        """Settings safe to print or log: secrets are replaced, never masked-in-part."""
        data = self.model_dump(mode="json")
        key = self.vision_api_key.get_secret_value()
        data["vision_api_key"] = "<set>" if key else "<unset>"
        data["allowed_roots"] = [str(root) for root in self.resolved_roots()]
        data["cache_dir"] = str(self.resolved_cache_dir())
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached singleton. Used by tests that manipulate the environment."""
    get_settings.cache_clear()
