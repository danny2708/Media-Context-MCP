"""OpenAI-compatible provider: request construction, error mapping, retries.

All HTTP goes through httpx.MockTransport -- no network, no credentials.
"""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from media_context_mcp.errors import (
    VisionAuthenticationFailedError,
    VisionEmptyResponseError,
    VisionInvalidResponseError,
    VisionModelUnavailableError,
    VisionPermissionDeniedError,
    VisionProviderTimeoutError,
    VisionQuotaExceededError,
    VisionRateLimitedError,
)
from media_context_mcp.providers.base import VisionImage
from media_context_mcp.providers.openai_compatible import (
    OpenAICompatibleVisionProvider,
    data_url,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)

IMAGE = VisionImage(data=PNG_BYTES, mime_type="image/png", width=1, height=1, label="t")


def ok_body(content: str = "## ANSWER\nfine") -> dict:
    return {
        "model": "served/model-v2",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def make_provider(handler, *, retries: int = 1, **kwargs) -> OpenAICompatibleVisionProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return OpenAICompatibleVisionProvider(
        base_url="https://fake.example/v1",
        api_key="sk-TESTKEY-abc",
        model="req/model-v1",
        max_retries=retries,
        client=client,
        **kwargs,
    )


def run(coroutine):
    return asyncio.run(coroutine)


# -------------------------------------------------------- request construction


def test_data_url_format():
    url = data_url(IMAGE)
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG_BYTES


def test_payload_structure_and_image_order():
    provider = make_provider(lambda request: httpx.Response(200, json=ok_body()))
    tile2 = VisionImage(
        data=PNG_BYTES, mime_type="image/jpeg", width=1, height=1, tile_index=2
    )
    payload = provider.build_payload([IMAGE, tile2], "PROMPT", "SYSTEM", 999)
    assert payload["model"] == "req/model-v1"
    assert payload["max_tokens"] == 999
    assert payload["temperature"] == 0
    assert payload["messages"][0] == {"role": "system", "content": "SYSTEM"}
    user_content = payload["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "PROMPT"}
    assert user_content[1]["type"] == "text"
    assert user_content[2]["image_url"]["url"].startswith("data:image/png")
    assert user_content[3]["type"] == "text"
    assert user_content[4]["image_url"]["url"].startswith("data:image/jpeg")


def test_api_key_only_in_auth_header_never_in_body():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json=ok_body())

    provider = make_provider(handler)
    result = run(provider.analyze(
        images=[IMAGE], prompt="p", system="s", max_output_tokens=100, request_id="r1"
    ))
    assert captured["auth"] == "Bearer sk-TESTKEY-abc"
    assert "sk-TESTKEY-abc" not in captured["body"]
    assert result.requested_model == "req/model-v1"
    assert result.actual_model == "served/model-v2"
    assert result.input_tokens == 10


def test_key_never_in_raised_errors():
    provider = make_provider(lambda request: httpx.Response(401, json={"error": "no"}))
    with pytest.raises(VisionAuthenticationFailedError) as excinfo:
        run(provider.analyze(
            images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
        ))
    assert "sk-TESTKEY-abc" not in str(excinfo.value) + json.dumps(excinfo.value.details)


# ------------------------------------------------------------- error mapping


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, VisionAuthenticationFailedError),
        (403, VisionPermissionDeniedError),
        (402, VisionQuotaExceededError),
        (404, VisionModelUnavailableError),
        (400, None),  # generic provider error, checked separately
    ],
)
def test_non_retryable_statuses_fail_after_one_request(status, expected):
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(status, json={"error": "x"})

    provider = make_provider(handler, retries=3)
    with pytest.raises(Exception) as excinfo:
        run(provider.analyze(
            images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
        ))
    assert counter["n"] == 1, "non-retryable status must not be retried"
    if expected is not None:
        assert isinstance(excinfo.value, expected)


def test_429_maps_to_rate_limit_with_retry_metadata_and_no_retry():
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(429, json={"error": "slow down"},
                              headers={"retry-after": "17"})

    provider = make_provider(handler, retries=3)
    with pytest.raises(VisionRateLimitedError) as excinfo:
        run(provider.analyze(
            images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
        ))
    assert counter["n"] == 1
    assert excinfo.value.details.get("retry_after_seconds") == 17.0


def test_429_quota_wording_maps_to_quota_error():
    provider = make_provider(
        lambda request: httpx.Response(429, json={"error": "monthly quota exceeded"})
    )
    with pytest.raises(VisionQuotaExceededError):
        run(provider.analyze(
            images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
        ))


def test_503_retried_once_then_succeeds():
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] == 1:
            return httpx.Response(503, json={"error": "warming up"})
        return httpx.Response(200, json=ok_body())

    provider = make_provider(handler, retries=1)
    result = run(provider.analyze(
        images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
    ))
    assert counter["n"] == 2
    assert result.content


def test_503_exhausted_retries_maps_to_model_unavailable():
    provider = make_provider(
        lambda request: httpx.Response(503, json={"error": "down"}), retries=1
    )
    with pytest.raises(VisionModelUnavailableError):
        run(provider.analyze(
            images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
        ))


def test_timeout_maps_and_is_bounded():
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        raise httpx.ReadTimeout("slow")

    provider = make_provider(handler, retries=1)
    with pytest.raises(VisionProviderTimeoutError):
        run(provider.analyze(
            images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
        ))
    assert counter["n"] == 2  # initial + exactly one retry


def test_empty_response_maps():
    body = {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]}
    provider = make_provider(lambda request: httpx.Response(200, json=body))
    with pytest.raises(VisionEmptyResponseError):
        run(provider.analyze(
            images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
        ))


def test_malformed_response_maps():
    provider = make_provider(
        lambda request: httpx.Response(200, json={"totally": "unexpected"})
    )
    with pytest.raises(VisionInvalidResponseError):
        run(provider.analyze(
            images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
        ))


def test_content_parts_list_supported():
    body = {
        "choices": [
            {"message": {"content": [{"type": "text", "text": "hello "},
                                     {"type": "text", "text": "world"}]},
             "finish_reason": "stop"}
        ]
    }
    provider = make_provider(lambda request: httpx.Response(200, json=body))
    result = run(provider.analyze(
        images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
    ))
    assert result.content == "hello world"


def test_finish_reason_length_warns():
    body = ok_body()
    body["choices"][0]["finish_reason"] = "length"
    provider = make_provider(lambda request: httpx.Response(200, json=body))
    result = run(provider.analyze(
        images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
    ))
    assert any("token limit" in warning for warning in result.warnings)


def test_cancellation_stops_immediately():
    counter = {"n": 0}

    async def scenario():
        def handler(request: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            raise httpx.ReadTimeout("slow")  # would normally trigger a retry

        provider = make_provider(handler, retries=5)
        task = asyncio.ensure_future(provider.analyze(
            images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"
        ))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    # at most the in-flight attempt happened; the retry loop must not continue
    assert counter["n"] <= 1


def test_hf_bill_to_header_forwarded():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["bill"] = request.headers.get("x-hf-bill-to")
        return httpx.Response(200, json=ok_body())

    transport = httpx.MockTransport(handler)
    provider = OpenAICompatibleVisionProvider(
        base_url="https://router.huggingface.co/v1",
        api_key="hf_x",
        model="org/vlm",
        extra_headers={"X-HF-Bill-To": "my-org"},
        client=httpx.AsyncClient(transport=transport),
    )
    run(provider.analyze(images=[IMAGE], prompt="p", max_output_tokens=10, request_id="r"))
    assert captured["bill"] == "my-org"
