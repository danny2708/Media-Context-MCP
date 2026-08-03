"""Vision backend speaking the OpenAI ``/chat/completions`` dialect.

Any endpoint that accepts an OpenAI-style multimodal chat request works here:
OpenAI, OpenRouter, Together, the Hugging Face Inference Providers router, a local
vLLM or Ollama shim. Nothing about a specific vendor is hard-coded -- the base URL,
model name and key all come from configuration, and there is no default endpoint.

Chat Completions is used as the baseline deliberately: the Responses API, JSON
schema output, and usage/provider metadata are all optional extras that many
"OpenAI-compatible" implementations lack. This adapter assumes none of them --
metadata is captured when present and left ``None`` when not.

Only the base URL the operator configured is ever contacted, and only after the
pipeline's cloud opt-in gate has passed. Authorization headers, API keys and image
bytes never appear in logs or error messages.
"""

from __future__ import annotations

import asyncio
import base64
import random
import time
from typing import Any

import httpx

from ..errors import (
    MediaContextError,
    VisionAuthenticationFailedError,
    VisionEmptyResponseError,
    VisionInvalidResponseError,
    VisionModelUnavailableError,
    VisionPermissionDeniedError,
    VisionProviderError,
    VisionProviderTimeoutError,
    VisionQuotaExceededError,
    VisionRateLimitedError,
)
from .base import VisionImage, VisionProviderResult

# Response bodies larger than this are refused outright; a vision reply is text
# and has no business being tens of megabytes.
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def data_url(image: VisionImage) -> str:
    """Base64 data URL with the MIME type of the *encoded bytes*."""
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value and value.isdigit():
        return float(value)
    return None


def image_manifest_line(index: int, total: int, image: VisionImage) -> str:
    """Explicit textual manifest description for an image input in a multi-image payload."""
    if image.role == "overview":
        orig_w = image.original_width or image.width
        orig_h = image.original_height or image.height
        return (
            f"[Image {index} of {total}: Downscaled global overview pass of the full "
            f"{orig_w}x{orig_h} screenshot]."
        )

    if image.source_x is not None and image.source_y is not None:
        sx = image.source_x
        sy = image.source_y
        sw = image.source_width or image.width
        sh = image.source_height or image.height
        return (
            f"[Image {index} of {total}: Native-resolution detail tile covering "
            f"x={sx}-{sx + sw}, y={sy}-{sy + sh} of the original image. "
            f"Adjacent detail tiles overlap by 96px; do not count duplicated components in overlapping regions twice]."
        )

    if image.label:
        return f"[Image {index} of {total}: {image.label}]"

    return f"[Image {index} of {total}]"


class OpenAICompatibleVisionProvider:
    """Async client for an OpenAI-compatible multimodal chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        provider_label: str = "openai-compatible",
        route: str = "",
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        extra_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._provider_label = provider_label
        self._route = route
        self._timeout = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._extra_headers = extra_headers or {}
        self._client = client
        self._owns_client = client is None

    @property
    def requested_model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        # Label plus host: enough to identify where data went, no credentials.
        try:
            host = httpx.URL(self._base_url).host or self._base_url
        except (httpx.InvalidURL, ValueError):  # pragma: no cover - defensive
            host = "?"
        return f"{self._provider_label} ({host})"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=min(10.0, self._timeout))
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def build_payload(
        self,
        images: list[VisionImage],
        prompt: str,
        system: str | None,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        """Chat Completions multimodal payload: text part first, then interleaved
        manifest text lines and image_url blocks in order."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        has_multiple_or_spatial = len(images) > 1 or any(
            img.role == "overview" or img.source_y is not None for img in images
        )
        for idx, image in enumerate(images, start=1):
            if has_multiple_or_spatial:
                manifest_text = image_manifest_line(idx, len(images), image)
                content.append({"type": "text", "text": manifest_text})
            content.append(
                {"type": "image_url", "image_url": {"url": data_url(image)}}
            )
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": 0,
        }

    async def analyze(
        self,
        *,
        images: list[VisionImage],
        prompt: str,
        system: str | None = None,
        max_output_tokens: int,
        request_id: str,
    ) -> VisionProviderResult:
        if not images:
            raise VisionInvalidResponseError(
                "analyze() called with no images.",
                hint="This is an internal routing bug; report it.",
            )

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        payload = self.build_payload(images, prompt, system, max_output_tokens)
        started = time.perf_counter()

        attempt = 0
        while True:
            retryable, error = None, None
            try:
                response = await self._get_client().post(
                    url, json=payload, headers=headers, timeout=self._timeout
                )
            except asyncio.CancelledError:
                # The caller gave up; stop immediately, no retry.
                raise
            except httpx.TimeoutException:
                error = VisionProviderTimeoutError(
                    f"Vision provider did not answer within {self._timeout:.0f}s.",
                    hint=(
                        "Retry once, raise MEDIA_MCP_VISION_TIMEOUT_SECONDS, or pick a "
                        "faster model. Nothing was analysed."
                    ),
                    details={"provider": self.provider_name, "request_id": request_id},
                )
                retryable = True
            except httpx.HTTPError as exc:
                error = VisionProviderError(
                    f"Network error talking to the vision provider: "
                    f"{exc.__class__.__name__}.",
                    hint="Check MEDIA_MCP_VISION_BASE_URL and connectivity.",
                    details={"provider": self.provider_name, "request_id": request_id},
                )
                retryable = True
            else:
                if response.status_code == 200:
                    if len(response.content) > _MAX_RESPONSE_BYTES:
                        raise VisionInvalidResponseError(
                            "Vision provider response exceeds the size limit.",
                            details={"bytes": len(response.content)},
                        )
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    return self._parse(response.json(), duration_ms)
                error, retryable = self._map_http_error(response, request_id)

            assert error is not None
            if not retryable or attempt >= self._max_retries:
                raise error
            attempt += 1
            # Bounded retry with full jitter; honours Retry-After when one is known.
            base_delay = error.details.get("retry_after_seconds") or min(
                4.0, 0.5 * (2**attempt)
            )
            await asyncio.sleep(float(base_delay) * (0.5 + random.random() / 2))

    def _map_http_error(
        self, response: httpx.Response, request_id: str
    ) -> tuple[MediaContextError, bool]:
        """Map an HTTP failure to a stable error code and a retryability flag.

        Policy (from the addendum): 401/403 never retried; 429 never retried
        immediately (the error carries retry metadata instead); timeouts and
        502/503/504 retried at most ``max_retries`` times; other 4xx never.
        """
        status = response.status_code
        body = _safe_body(response)
        details: dict[str, Any] = {
            "status": status,
            "provider": self.provider_name,
            "request_id": request_id,
        }

        if status == 401:
            return (
                VisionAuthenticationFailedError(
                    "Vision provider rejected the API key (401).",
                    hint=(
                        "Check MEDIA_MCP_VISION_API_KEY. For Hugging Face, the token "
                        "needs the 'inference.serverless' permission."
                    ),
                    details=details,
                ),
                False,
            )
        if status == 403:
            return (
                VisionPermissionDeniedError(
                    "Vision provider denied access to this model (403).",
                    hint=(
                        "The key is valid but not allowed to use "
                        f"'{self._model}'. Some models require accepting a license on "
                        "the provider's site, or a paid tier."
                    ),
                    details=details,
                ),
                False,
            )
        if status == 402:
            return (
                VisionQuotaExceededError(
                    "Vision provider reports exhausted credits (402).",
                    hint=(
                        "The account's inference credits are used up. Add credits, or "
                        "switch MEDIA_MCP_VISION_MODEL/provider. Treat free credits as "
                        "trial capacity, not unlimited inference."
                    ),
                    details=details,
                ),
                False,
            )
        if status == 429:
            retry_after = _retry_after_seconds(response)
            if retry_after is not None:
                details["retry_after_seconds"] = retry_after
            quota_like = "quota" in body.lower() or "credit" in body.lower()
            error_cls = VisionQuotaExceededError if quota_like else VisionRateLimitedError
            return (
                error_cls(
                    "Vision provider rate-limited or out of quota (429).",
                    hint=(
                        f"Wait{f' {retry_after:.0f}s' if retry_after else ''} and retry, "
                        "or reduce request rate. Repeated immediate retries would make "
                        "it worse, so none were attempted."
                    ),
                    details=details,
                ),
                False,
            )
        if status in {404, 410}:
            return (
                VisionModelUnavailableError(
                    f"Model or endpoint not found ({status}).",
                    hint=(
                        f"'{self._model}' may not exist on this provider, may have been "
                        "retired, or MEDIA_MCP_VISION_BASE_URL may point at the wrong "
                        "API root (it should be the part before /chat/completions). "
                        "The configured model was NOT silently replaced."
                    ),
                    details=details,
                ),
                False,
            )
        if status in {502, 503, 504}:
            return (
                VisionModelUnavailableError(
                    f"Vision provider temporarily unavailable ({status}).",
                    hint="Transient upstream failure; one bounded retry is attempted.",
                    details=details,
                ),
                True,
            )
        if status in {408, 409, 425}:
            return (
                VisionProviderTimeoutError(
                    f"Vision provider timed out upstream ({status}).",
                    details=details,
                ),
                True,
            )
        return (
            VisionProviderError(
                f"Vision provider rejected the request (HTTP {status}): {body[:200]}",
                hint=(
                    "A 4xx other than auth/rate-limit usually means the model does not "
                    "accept image input, or the payload shape is unsupported. Check "
                    "that MEDIA_MCP_VISION_MODEL is a multimodal model."
                ),
                details=details,
            ),
            False,
        )

    def _parse(self, body: dict[str, Any], duration_ms: int) -> VisionProviderResult:
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionInvalidResponseError(
                "Vision provider returned a response in an unexpected shape.",
                hint=(
                    "The endpoint may not be OpenAI-compatible. Verify the base URL "
                    "serves /chat/completions."
                ),
            ) from exc

        # Some providers return content as a list of parts rather than a string.
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise VisionEmptyResponseError(
                "Vision provider returned an empty response.",
                hint=(
                    "The model may have refused the image or hit a content filter. Try "
                    "a different model, or mode='ocr' for local text extraction. This "
                    "failure is not cached; a retry will re-ask the provider."
                ),
            )

        warnings: list[str] = []
        finish = choice.get("finish_reason")
        if finish == "length":
            warnings.append(
                "The vision model stopped at its output-token limit "
                "(MEDIA_MCP_VISION_MAX_OUTPUT_TOKENS); the reading is incomplete."
            )

        usage = body.get("usage") or {}
        return VisionProviderResult(
            content=content,
            provider=self.provider_name,
            requested_model=self._model,
            # Only trust reported metadata; never echo the request back as fact.
            actual_model=body.get("model") if body.get("model") else None,
            provider_route=body.get("provider") or body.get("system_fingerprint"),
            finish_reason=finish,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            duration_ms=duration_ms,
            warnings=warnings,
        )


def _safe_body(response: httpx.Response) -> str:
    """A short, credential-free excerpt of an error body."""
    try:
        text = response.text[:400]
    except Exception:  # noqa: BLE001 - a broken body must not mask the real error
        return "<unreadable body>"
    return text.replace("Bearer ", "Bearer <redacted> ")


def build_vision_provider(settings: Any) -> OpenAICompatibleVisionProvider | None:
    """Construct the provider from settings, or ``None`` when not configured.

    The ``huggingface`` value of MEDIA_MCP_VISION_PROVIDER is a configuration
    preset over this same class: it fills the router base URL and forwards the
    optional billing header. There is no HF-specific code path beyond that.
    """
    if not settings.vision_configured:
        return None
    extra_headers: dict[str, str] = {}
    if settings.vision_provider == "huggingface" and settings.hf_bill_to:
        extra_headers["X-HF-Bill-To"] = settings.hf_bill_to
    return OpenAICompatibleVisionProvider(
        base_url=settings.effective_vision_base_url,
        api_key=settings.vision_api_key.get_secret_value(),
        model=settings.effective_vision_model,
        provider_label=settings.vision_provider,
        route=settings.vision_route,
        timeout_seconds=settings.vision_timeout_seconds,
        max_retries=settings.vision_max_retries,
        extra_headers=extra_headers,
    )
