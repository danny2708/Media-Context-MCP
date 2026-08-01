"""Content-addressed result cache."""

from .key import build_cache_key
from .store import CacheStore

__all__ = ["CacheStore", "build_cache_key"]
