"""Vision backend speaking the OpenAI ``/chat/completions`` dialect.

Any endpoint that accepts an OpenAI-style multimodal chat request works here:
OpenAI, OpenRouter, Together, Groq, a local vLLM or Ollama shim. Nothing about a
specific vendor is hard-coded -- the base URL, model name and key all come from
configuration, and there is no default endpoint. If they are not set, the server
says so instead of quietly calling somebody's API.

Only the base URL the operator configured is ever contacted, and only when a
processor explicitly asks for visual analysis.
"""

from __future__ import annotations

import asyncio
import base64
import random
from typing import Any

import httpx

from ..errors import VisionProviderError
from .base import ImageInput, VisionAnalysis

# Retried: transient by definition. Everything else fails immediately, because
# retrying a 400 just burns the caller's time and quota.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

_MAX_TOKENS = 2048


def _data_uri(image: ImageInput) -> str:
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


class OpenAICompatibleVisionProvider:
    """Async client for an OpenAI-compatible multimodal chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._client = client
        self._owns_client = client is None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        # The host, not the key or the full URL: enough to identify where data went,
        # with no credentials in it.
        try:
            return httpx.URL(self._base_url).host or self._base_url
        except (httpx.InvalidURL, ValueError):  # pragma: no cover - defensive
            return "vision-provider"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _build_payload(
        self, image: ImageInput, prompt: str, system: str | None
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_uri(image)}},
                ],
            }
        )
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": _MAX_TOKENS,
            # Deterministic-as-possible: the same screenshot should not produce a
            # different reading on a cache miss than it did on the first call.
            "temperature": 0,
        }

    async def analyze(
        self,
        image: ImageInput,
        prompt: str,
        *,
        system: str | None = None,
    ) -> VisionAnalysis:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(image, prompt, system)

        last_error: str = "unknown error"
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._get_client().post(
                    url, json=payload, headers=headers, timeout=self._timeout
                )
            except httpx.TimeoutException as exc:
                last_error = f"request timed out after {self._timeout:.0f}s ({exc.__class__.__name__})"
            except httpx.HTTPError as exc:
                last_error = f"network error: {exc.__class__.__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return self._parse(response.json())
                body = _safe_body(response)
                last_error = f"HTTP {response.status_code}: {body}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise VisionProviderError(
                        f"Vision provider rejected the request ({last_error}).",
                        hint=_hint_for_status(response.status_code),
                        details={"status": response.status_code, "provider": self.provider_name},
                    )

            if attempt < self._max_retries:
                # Full jitter: retries from several concurrent calls do not line up.
                delay = min(8.0, 0.5 * (2**attempt)) * (0.5 + random.random() / 2)
                await asyncio.sleep(delay)

        raise VisionProviderError(
            f"Vision provider failed after {self._max_retries + 1} attempt(s): {last_error}",
            hint=(
                "Check MEDIA_MCP_VISION_BASE_URL and that the host is reachable, then "
                "retry. Nothing was analysed visually; no result is being guessed."
            ),
            details={"provider": self.provider_name, "attempts": self._max_retries + 1},
        )

    def _parse(self, body: dict[str, Any]) -> VisionAnalysis:
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionProviderError(
                "Vision provider returned a response in an unexpected shape.",
                hint=(
                    "The endpoint may not be OpenAI-compatible. Verify "
                    "MEDIA_MCP_VISION_BASE_URL points at the API root that serves "
                    "/chat/completions."
                ),
            ) from exc

        # Some providers return content as a list of parts rather than a string.
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise VisionProviderError(
                "Vision provider returned an empty response.",
                hint=(
                    "The model may have refused the image or hit a content filter. "
                    "Try a different model, or use mode='ocr' for text-only extraction."
                ),
            )

        warnings: list[str] = []
        finish = choice.get("finish_reason")
        if finish == "length":
            warnings.append(
                "The vision model stopped because it reached its output token limit; "
                "its description of this image is incomplete."
            )

        return VisionAnalysis(
            text=content,
            model=body.get("model") or self._model,
            finish_reason=finish,
            usage=body.get("usage") or {},
            warnings=warnings,
        )


def _safe_body(response: httpx.Response) -> str:
    """A short, credential-free excerpt of an error body.

    Provider error bodies sometimes echo the request, so the excerpt is capped and
    any bearer token that appears is stripped.
    """
    try:
        text = response.text[:400]
    except Exception:  # noqa: BLE001 - a broken body must not mask the real error
        return "<unreadable body>"
    return text.replace("Bearer ", "Bearer <redacted> ")


def _hint_for_status(status: int) -> str:
    if status == 401 or status == 403:
        return (
            "Authentication failed. Check MEDIA_MCP_VISION_API_KEY, and that the key "
            "is valid for MEDIA_MCP_VISION_BASE_URL."
        )
    if status == 404:
        return (
            "Endpoint not found. MEDIA_MCP_VISION_BASE_URL should be the API root "
            "(the part before /chat/completions), e.g. https://api.openai.com/v1."
        )
    if status == 413:
        return "The image was too large for this provider. Lower MEDIA_MCP_MAX_IMAGE_PIXELS."
    if status == 400:
        return (
            "The provider rejected the request. The configured model may not accept "
            "images; check MEDIA_MCP_VISION_MODEL is a multimodal model."
        )
    return "See the provider's status page or logs for details."
