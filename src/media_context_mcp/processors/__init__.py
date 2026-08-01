"""Media processors. Each one owns exactly one extraction strategy."""

from .base import MediaProcessor, ProcessingContext

__all__ = ["MediaProcessor", "ProcessingContext"]
