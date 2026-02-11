"""Shared In-Memory DataFrame Store — Nexus LLM Analytics
==========================================================

Thread-safe, TTL-keyed, LRU-bounded in-memory cache for
pandas DataFrames.  Eliminates the redundant disk I/O caused
by calling ``read_dataframe()`` 19+ times across 6 modules.

Usage
-----
::

    from backend.core.dataframe_store import get_dataframe_store

    store = get_dataframe_store()

    # Load (or return cached) — replaces raw ``read_dataframe()`` calls
    df = store.get_or_load(filepath, loader=lambda: read_dataframe(filepath))

    # Invalidate on re-upload
    store.invalidate(filepath)

Design
------
* **LRU eviction** — oldest-accessed entry is dropped when
  ``max_entries`` is exceeded.
* **TTL expiry** — entries older than ``ttl_seconds`` are lazily
  evicted on the next access.
* **Memory cap** — total memory tracked via
  ``DataFrame.memory_usage(deep=True)``.
* **Thread safety** — ``threading.RLock`` protects all mutations.

v2.0 Enterprise Addition
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class _CacheEntry:
    """Internal wrapper storing a DataFrame alongside its metadata."""

    __slots__ = ("df", "created_at", "last_accessed", "memory_bytes")

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.created_at = time.monotonic()
        self.last_accessed = self.created_at
        self.memory_bytes: int = int(df.memory_usage(deep=True).sum())

    def touch(self) -> None:
        self.last_accessed = time.monotonic()


class DataFrameStore:
    """TTL + LRU in-memory DataFrame cache.

    Parameters
    ----------
    max_entries:
        Maximum number of DataFrames to keep in cache (default 20).
    max_memory_bytes:
        Approximate memory ceiling in bytes (default 500 MB).
    ttl_seconds:
        Time-to-live per entry in seconds (default 1800 = 30 min).
    """

    def __init__(
        self,
        max_entries: int = 20,
        max_memory_bytes: int = 500 * 1024 * 1024,
        ttl_seconds: float = 1800.0,
    ) -> None:
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._max_memory_bytes = max_memory_bytes
        self._ttl = ttl_seconds
        self._total_memory: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """Return cached DataFrame or load it via *loader* and cache.

        Args:
            key:    Cache key — typically the filepath string.
            loader: Zero-argument callable that returns a DataFrame
                    (e.g. ``lambda: read_dataframe(path)``).

        Returns:
            The (possibly cached) DataFrame.
        """
        normalised = self._normalise_key(key)

        with self._lock:
            entry = self._cache.get(normalised)
            if entry is not None:
                # Check TTL
                if (time.monotonic() - entry.created_at) > self._ttl:
                    self._evict(normalised)
                    logger.debug("DataFrame cache TTL expired for %s", key)
                else:
                    entry.touch()
                    self._cache.move_to_end(normalised)
                    logger.debug("DataFrame cache HIT for %s", key)
                    return entry.df

        # Load outside lock to avoid blocking other threads during I/O
        df = loader()

        with self._lock:
            # Double-check: another thread might have loaded while we were reading
            if normalised in self._cache:
                self._cache[normalised].touch()
                self._cache.move_to_end(normalised)
                return self._cache[normalised].df

            new_entry = _CacheEntry(df)

            # Memory / count eviction
            while (
                self._total_memory + new_entry.memory_bytes > self._max_memory_bytes
                or len(self._cache) >= self._max_entries
            ) and self._cache:
                oldest_key = next(iter(self._cache))
                self._evict(oldest_key)

            self._cache[normalised] = new_entry
            self._total_memory += new_entry.memory_bytes
            logger.debug(
                "DataFrame cache STORE for %s (%d bytes, %d entries total)",
                key,
                new_entry.memory_bytes,
                len(self._cache),
            )

        return df

    def invalidate(self, key: str) -> None:
        """Remove a specific entry (e.g. on file re-upload)."""
        normalised = self._normalise_key(key)
        with self._lock:
            if normalised in self._cache:
                self._evict(normalised)
                logger.debug("DataFrame cache INVALIDATED for %s", key)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._total_memory = 0
            logger.debug("DataFrame cache CLEARED")

    def status(self) -> dict[str, Any]:
        """Return diagnostic snapshot for health monitoring."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "total_memory_bytes": self._total_memory,
                "max_memory_bytes": self._max_memory_bytes,
                "ttl_seconds": self._ttl,
                "keys": list(self._cache.keys()),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict(self, key: str) -> None:
        """Remove *key* from cache and update memory tracking (caller holds lock)."""
        entry = self._cache.pop(key, None)
        if entry is not None:
            self._total_memory = max(0, self._total_memory - entry.memory_bytes)

    @staticmethod
    def _normalise_key(key: str) -> str:
        """Normalise path separators so the same file always maps to one key."""
        return key.replace("\\", "/").lower()


# =====================================================================
# Singleton accessor
# =====================================================================

_instance: Optional[DataFrameStore] = None
_instance_lock = threading.Lock()


def get_dataframe_store() -> DataFrameStore:
    """Return the process-wide :class:`DataFrameStore` singleton.

    Thread Safety:
        Uses double-checked locking.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DataFrameStore()
    return _instance
