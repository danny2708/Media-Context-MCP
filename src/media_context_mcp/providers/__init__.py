"""Pluggable backends: vision models and OCR engines."""

from .base import (
    OcrBackend,
    OcrResult,
    VisionImage,
    VisionProvider,
    VisionProviderResult,
)

__all__ = [
    "OcrBackend",
    "OcrResult",
    "VisionImage",
    "VisionProvider",
    "VisionProviderResult",
]
