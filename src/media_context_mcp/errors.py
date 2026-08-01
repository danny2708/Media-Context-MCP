"""Stable error taxonomy.

Error codes are part of the tool's public contract: a calling agent may branch on
them, so they must not be renamed casually. Every error carries a ``hint`` whose
audience is an AI agent deciding what to do next -- not a human reading a stack
trace.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Stable, machine-readable failure identifiers."""

    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PATH_NOT_ALLOWED = "PATH_NOT_ALLOWED"
    NOT_A_REGULAR_FILE = "NOT_A_REGULAR_FILE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_PAGE_SELECTION = "INVALID_PAGE_SELECTION"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    MODE_NOT_APPLICABLE = "MODE_NOT_APPLICABLE"
    OCR_NOT_CONFIGURED = "OCR_NOT_CONFIGURED"
    OCR_FAILED = "OCR_FAILED"
    VISION_NOT_CONFIGURED = "VISION_NOT_CONFIGURED"
    CLOUD_VISION_DISABLED = "CLOUD_VISION_DISABLED"
    VISION_AUTHENTICATION_FAILED = "VISION_AUTHENTICATION_FAILED"
    VISION_PERMISSION_DENIED = "VISION_PERMISSION_DENIED"
    VISION_QUOTA_EXCEEDED = "VISION_QUOTA_EXCEEDED"
    VISION_RATE_LIMITED = "VISION_RATE_LIMITED"
    VISION_MODEL_UNAVAILABLE = "VISION_MODEL_UNAVAILABLE"
    VISION_PROVIDER_TIMEOUT = "VISION_PROVIDER_TIMEOUT"
    VISION_EMPTY_RESPONSE = "VISION_EMPTY_RESPONSE"
    VISION_INVALID_RESPONSE = "VISION_INVALID_RESPONSE"
    VISION_PROVIDER_ERROR = "VISION_PROVIDER_ERROR"
    DOCUMENT_CONVERSION_FAILED = "DOCUMENT_CONVERSION_FAILED"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    CACHE_ERROR = "CACHE_ERROR"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class MediaContextError(Exception):
    """Base class for every failure the pipeline reports deliberately.

    Anything that escapes as a bare ``Exception`` is a bug; the server converts
    those to INTERNAL_ERROR and logs them at ERROR level with a traceback.
    """

    code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code.value}] {self.message}"


def _named(code: ErrorCode) -> type[MediaContextError]:
    """Build a MediaContextError subclass bound to a single code."""

    class _Error(MediaContextError):
        pass

    _Error.code = code
    _Error.__name__ = "".join(part.capitalize() for part in code.value.split("_")) + "Error"
    _Error.__qualname__ = _Error.__name__
    return _Error


FileNotFoundError_ = _named(ErrorCode.FILE_NOT_FOUND)
PathNotAllowedError = _named(ErrorCode.PATH_NOT_ALLOWED)
NotARegularFileError = _named(ErrorCode.NOT_A_REGULAR_FILE)
UnsupportedMediaTypeError = _named(ErrorCode.UNSUPPORTED_MEDIA_TYPE)
FileTooLargeError = _named(ErrorCode.FILE_TOO_LARGE)
InvalidPageSelectionError = _named(ErrorCode.INVALID_PAGE_SELECTION)
InvalidArgumentError = _named(ErrorCode.INVALID_ARGUMENT)
ModeNotApplicableError = _named(ErrorCode.MODE_NOT_APPLICABLE)
OcrNotConfiguredError = _named(ErrorCode.OCR_NOT_CONFIGURED)
OcrFailedError = _named(ErrorCode.OCR_FAILED)
VisionNotConfiguredError = _named(ErrorCode.VISION_NOT_CONFIGURED)
CloudVisionDisabledError = _named(ErrorCode.CLOUD_VISION_DISABLED)
VisionAuthenticationFailedError = _named(ErrorCode.VISION_AUTHENTICATION_FAILED)
VisionPermissionDeniedError = _named(ErrorCode.VISION_PERMISSION_DENIED)
VisionQuotaExceededError = _named(ErrorCode.VISION_QUOTA_EXCEEDED)
VisionRateLimitedError = _named(ErrorCode.VISION_RATE_LIMITED)
VisionModelUnavailableError = _named(ErrorCode.VISION_MODEL_UNAVAILABLE)
VisionProviderTimeoutError = _named(ErrorCode.VISION_PROVIDER_TIMEOUT)
VisionEmptyResponseError = _named(ErrorCode.VISION_EMPTY_RESPONSE)
VisionInvalidResponseError = _named(ErrorCode.VISION_INVALID_RESPONSE)
VisionProviderError = _named(ErrorCode.VISION_PROVIDER_ERROR)
DocumentConversionFailedError = _named(ErrorCode.DOCUMENT_CONVERSION_FAILED)
ImageDecodeFailedError = _named(ErrorCode.IMAGE_DECODE_FAILED)
ProcessingTimeoutError = _named(ErrorCode.PROCESSING_TIMEOUT)
CacheError = _named(ErrorCode.CACHE_ERROR)
ConfigurationError = _named(ErrorCode.CONFIGURATION_ERROR)

__all__ = [
    "CacheError",
    "CloudVisionDisabledError",
    "ConfigurationError",
    "DocumentConversionFailedError",
    "ErrorCode",
    "FileNotFoundError_",
    "FileTooLargeError",
    "ImageDecodeFailedError",
    "InvalidArgumentError",
    "InvalidPageSelectionError",
    "MediaContextError",
    "ModeNotApplicableError",
    "NotARegularFileError",
    "OcrFailedError",
    "OcrNotConfiguredError",
    "PathNotAllowedError",
    "ProcessingTimeoutError",
    "UnsupportedMediaTypeError",
    "VisionAuthenticationFailedError",
    "VisionEmptyResponseError",
    "VisionInvalidResponseError",
    "VisionModelUnavailableError",
    "VisionNotConfiguredError",
    "VisionPermissionDeniedError",
    "VisionProviderError",
    "VisionProviderTimeoutError",
    "VisionQuotaExceededError",
    "VisionRateLimitedError",
]
