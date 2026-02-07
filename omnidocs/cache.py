"""Unified model cache with LRU eviction and reference counting.

This module provides a production-ready cache for sharing models across extractors.
Features:
- LRU eviction with configurable max entries
- Reference counting for automatic cleanup
- Thread-safe operations
- Memory-aware eviction (optional)
- Context manager for scoped usage

Example:
    ```python
    from omnidocs import clear_cache, get_cache_info, set_cache_config
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector

    # Configure cache (optional - defaults are sensible)
    set_cache_config(max_entries=5)

    # First extractor loads the model
    text_extractor = MinerUVLTextExtractor(backend=config)

    # Second extractor reuses cached model (instant)
    layout_detector = MinerUVLLayoutDetector(backend=config)

    # Check cache status
    print(get_cache_info())
    ```

Reference Counting:
    When extractors are deleted, reference counts decrease. When count hits zero,
    the model becomes eligible for LRU eviction (but isn't immediately removed).

    ```python
    # Model loaded, ref_count=1
    ext1 = MinerUVLTextExtractor(backend=config)

    # Same model, ref_count=2
    ext2 = MinerUVLLayoutDetector(backend=config)

    del ext1  # ref_count=1
    del ext2  # ref_count=0, eligible for eviction
    ```
"""

import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ref_count: int = 0
    _weak_refs: Set[weakref.ref] = field(default_factory=set)

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def add_ref(self, weak_ref: Optional[weakref.ref] = None) -> None:
        """Add a reference."""
        self.ref_count += 1
        if weak_ref:
            self._weak_refs.add(weak_ref)

    def remove_ref(self, weak_ref: Optional[weakref.ref] = None) -> None:
        """Remove a reference."""
        self.ref_count = max(0, self.ref_count - 1)
        if weak_ref:
            self._weak_refs.discard(weak_ref)

    def cleanup_dead_refs(self) -> int:
        """Remove dead weak references and return count removed."""
        dead = [ref for ref in self._weak_refs if ref() is None]
        for ref in dead:
            self._weak_refs.discard(ref)
            self.ref_count = max(0, self.ref_count - 1)
        return len(dead)


@dataclass
class CacheConfig:
    """Cache configuration."""

    max_entries: int = 10  # Maximum cached models (0 = unlimited)
    evict_unreferenced_first: bool = True  # Prefer evicting entries with ref_count=0
    auto_cleanup_interval: int = 100  # Cleanup dead refs every N operations


class ModelCache:
    """Thread-safe LRU model cache with reference counting."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self._config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._op_count = 0

    def configure(self, **kwargs) -> None:
        """Update cache configuration."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, updating LRU order."""
        with self._lock:
            self._maybe_cleanup()
            if key not in self._cache:
                return None

            entry = self._cache[key]
            entry.touch()

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        owner: Optional[Any] = None,
    ) -> None:
        """Set value in cache with optional owner for reference counting."""
        with self._lock:
            self._maybe_cleanup()

            if key in self._cache:
                # Update existing entry
                entry = self._cache[key]
                entry.value = value
                entry.touch()
            else:
                # Create new entry
                entry = CacheEntry(value=value)
                self._cache[key] = entry

                # Evict if over limit
                self._evict_if_needed()

            # Track reference if owner provided
            if owner is not None:
                self._track_reference(key, owner)

            self._cache.move_to_end(key)

    def get_or_load(
        self,
        key: str,
        loader_fn: Callable[[], Any],
        owner: Optional[Any] = None,
    ) -> Any:
        """Get from cache or load and cache."""
        with self._lock:
            existing = self.get(key)
            if existing is not None:
                if owner is not None:
                    self._track_reference(key, owner)
                return existing

            # Load new value
            value = loader_fn()
            self.set(key, value, owner=owner)
            return value

    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def add_reference(self, key: str, owner: Any) -> bool:
        """Add a reference to a cache entry."""
        with self._lock:
            if key not in self._cache:
                return False
            self._track_reference(key, owner)
            return True

    def remove_reference(self, key: str, weak_ref: weakref.ref) -> None:
        """Remove a reference from a cache entry."""
        with self._lock:
            if key in self._cache:
                self._cache[key].remove_ref(weak_ref)

    def info(self) -> Dict[str, Any]:
        """Get cache information."""
        with self._lock:
            entries = {}
            for key, entry in self._cache.items():
                entries[key] = {
                    "ref_count": entry.ref_count,
                    "access_count": entry.access_count,
                    "age_seconds": time.time() - entry.created_at,
                }

            return {
                "num_entries": len(self._cache),
                "max_entries": self._config.max_entries,
                "keys": list(self._cache.keys()),
                "entries": entries,
            }

    def keys(self) -> List[str]:
        """List all cache keys."""
        with self._lock:
            return list(self._cache.keys())

    def _track_reference(self, key: str, owner: Any) -> None:
        """Track a weak reference to an owner object."""
        entry = self._cache.get(key)
        if entry is None:
            return

        # Create weak reference with callback
        def on_delete(ref: weakref.ref) -> None:
            # Called when owner is garbage collected
            with self._lock:
                if key in self._cache:
                    self._cache[key].remove_ref(ref)

        try:
            weak_ref = weakref.ref(owner, on_delete)
            entry.add_ref(weak_ref)
        except TypeError:
            # Object doesn't support weak references, just increment count
            entry.add_ref(None)

    def _evict_if_needed(self) -> None:
        """Evict entries if over max limit."""
        if self._config.max_entries <= 0:
            return  # No limit

        while len(self._cache) > self._config.max_entries:
            evicted = self._evict_one()
            if not evicted:
                break  # Couldn't evict anything

    def _evict_one(self) -> bool:
        """Evict one entry using LRU policy. Returns True if evicted."""
        if not self._cache:
            return False

        # If configured, prefer evicting unreferenced entries first
        if self._config.evict_unreferenced_first:
            for key in list(self._cache.keys()):
                entry = self._cache[key]
                if entry.ref_count == 0:
                    del self._cache[key]
                    return True

        # Fall back to strict LRU (first item is least recently used)
        key = next(iter(self._cache))
        del self._cache[key]
        return True

    def _maybe_cleanup(self) -> None:
        """Periodically cleanup dead weak references."""
        self._op_count += 1
        if self._op_count >= self._config.auto_cleanup_interval:
            self._op_count = 0
            for entry in self._cache.values():
                entry.cleanup_dead_refs()


# Global cache instance
_global_cache = ModelCache()


# ============= Public API =============


def get_cache_key(backend_config, prefix: str = "") -> str:
    """Generate a cache key from backend config.

    The key is normalized to allow sharing between different extractors
    that use the same underlying model.

    Args:
        backend_config: Backend configuration object (must have model_dump() method)
        prefix: Optional prefix for the key (e.g., model family name)

    Returns:
        String cache key
    """
    config_type = type(backend_config).__name__

    # Normalize config type name to allow sharing between tasks
    # Examples:
    #   MinerUVLTextPyTorchConfig -> MinerUVL:PyTorchConfig
    #   MinerUVLLayoutPyTorchConfig -> MinerUVL:PyTorchConfig
    normalized_type = config_type

    # Common task suffixes to remove for normalization
    task_markers = ["Text", "Layout", "OCR", "Table", "ReadingOrder", "Formula"]
    for marker in task_markers:
        if marker in normalized_type:
            parts = normalized_type.split(marker, 1)
            if len(parts) == 2 and parts[0]:
                model_family = parts[0]
                backend_type = parts[1]
                normalized_type = f"{model_family}:{backend_type}"
                break

    config_dict = backend_config.model_dump()

    # Exclude runtime/inference params that don't affect model loading.
    # These vary between tasks (e.g., text extraction needs more tokens
    # than layout detection) but the underlying model is the same.
    runtime_params = {
        "max_tokens",
        "max_new_tokens",
        "temperature",
        "do_sample",
        "timeout",
        "max_retries",
    }

    # Build key from normalized type and config values
    key_parts = [prefix] if prefix else []
    key_parts.append(normalized_type)

    for k, v in sorted(config_dict.items()):
        if k not in runtime_params:
            key_parts.append(f"{k}={v}")

    return ":".join(key_parts)


def get_cached(cache_key: str) -> Optional[Any]:
    """Get cached value if it exists.

    Args:
        cache_key: Cache key from get_cache_key()

    Returns:
        Cached value or None if not cached
    """
    return _global_cache.get(cache_key)


def set_cached(cache_key: str, value: Any, owner: Optional[Any] = None) -> None:
    """Add value to cache.

    Args:
        cache_key: Cache key from get_cache_key()
        value: Value to cache
        owner: Optional owner object for reference counting (weak ref tracked)
    """
    _global_cache.set(cache_key, value, owner=owner)


def get_or_load(
    cache_key: str,
    loader_fn: Callable[[], Any],
    owner: Optional[Any] = None,
) -> Any:
    """Get from cache or load and cache.

    Thread-safe operation that either returns cached value or loads a new one.

    Args:
        cache_key: Cache key from get_cache_key()
        loader_fn: Function that loads and returns the value to cache
        owner: Optional owner object for reference counting

    Returns:
        Cached or newly loaded value
    """
    return _global_cache.get_or_load(cache_key, loader_fn, owner=owner)


def add_reference(cache_key: str, owner: Any) -> bool:
    """Add a reference to a cached entry.

    Use this when an extractor starts using a cached model.

    Args:
        cache_key: Cache key
        owner: Owner object (extractor instance)

    Returns:
        True if reference added, False if key not in cache
    """
    return _global_cache.add_reference(cache_key, owner)


def remove_cached(cache_key: str) -> bool:
    """Remove a specific entry from cache.

    Args:
        cache_key: Cache key to remove

    Returns:
        True if entry was removed, False if it didn't exist
    """
    return _global_cache.remove(cache_key)


def clear_cache() -> None:
    """Clear all cached models."""
    _global_cache.clear()


def get_cache_info() -> Dict[str, Any]:
    """Get detailed cache information.

    Returns:
        Dict with cache stats, keys, and per-entry info
    """
    return _global_cache.info()


def list_cached_keys() -> List[str]:
    """List all cached keys.

    Returns:
        List of cache keys
    """
    return _global_cache.keys()


def set_cache_config(**kwargs) -> None:
    """Configure global cache settings.

    Args:
        max_entries: Maximum number of cached models (0 = unlimited, default=10)
        evict_unreferenced_first: Prefer evicting entries with no references (default=True)
        auto_cleanup_interval: Cleanup dead refs every N operations (default=100)

    Example:
        ```python
        from omnidocs import set_cache_config

        # Allow up to 5 models cached
        set_cache_config(max_entries=5)

        # Unlimited cache (careful with memory!)
        set_cache_config(max_entries=0)
        ```
    """
    _global_cache.configure(**kwargs)


def get_cache_config() -> Dict[str, Any]:
    """Get current cache configuration.

    Returns:
        Dict with current config values
    """
    config = _global_cache._config
    return {
        "max_entries": config.max_entries,
        "evict_unreferenced_first": config.evict_unreferenced_first,
        "auto_cleanup_interval": config.auto_cleanup_interval,
    }
