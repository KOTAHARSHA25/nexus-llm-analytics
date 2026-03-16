"""Advanced Cache — Compatibility Bridge Module
===============================================

Re-exports convenience functions for cache status and bulk invalidation
that are referenced by ``main.py``, ``api/health.py``, and other
callers via ``from backend.core.advanced_cache import …``.

Internally all work is delegated to :mod:`backend.core.enhanced_cache_integration`
which houses the full multi-tier caching system.

Exported Symbols
----------------
``get_cache_status()``
    Return a JSON-serialisable dict summarising cache health.
``clear_all_caches()``
    Purge every tier (L1/L2/L3) and reset statistics.
``_query_cache``
    Direct reference to the global query-result cache instance
    (used by ``api/upload.py`` for tag-based invalidation).
``_file_analysis_cache``
    Direct reference to the file-analysis cache instance.

Backward Compatibility
----------------------
Code that imports from ``backend.core.advanced_cache`` continues
to work without changes.

Author: Nexus Analytics Research Team
Since: v2.0 — February 2026
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Delegate to the enhanced cache integration layer
# ---------------------------------------------------------------------------

def get_cache_status() -> Dict[str, Any]:
    """Return a JSON-serialisable snapshot of cache-system health.

    Delegates to :func:`enhanced_cache_integration.get_enhanced_cache_manager`
    and returns its comprehensive statistics.  Falls back gracefully if the
    cache system is not yet initialised.

    Returns:
        Dict with keys like ``overview``, ``tier_performance``, and
        ``pattern_analysis``.  Returns a safe error dict on failure.
    """
    try:
        from backend.core.enhanced_cache_integration import get_enhanced_cache_manager

        manager = get_enhanced_cache_manager()
        stats = manager.get_comprehensive_stats()
        return {
            "status": "healthy",
            "backend": "EnhancedCacheManager",
            **stats,
        }
    except Exception as exc:
        logger.debug("Cache status unavailable: %s", exc)
        return {
            "status": "unavailable",
            "error": str(exc),
            "backend": "EnhancedCacheManager",
        }


def clear_all_caches() -> Dict[str, Any]:
    """Purge all cache tiers (L1 memory, L2 optimised, L3 persistent).

    Returns:
        Dict with ``cleared`` (bool) and optional ``error`` on failure.
    """
    try:
        from backend.core.enhanced_cache_integration import get_enhanced_cache_manager

        manager = get_enhanced_cache_manager()
        manager.clear_all()
        
        logger.info("All caches cleared via EnhancedCacheManager")
        return {"cleared": True, "tiers_cleared": "all"}

    except Exception as exc:
        logger.warning("Failed to clear caches: %s", exc)
        return {"cleared": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Direct cache references for tag-based invalidation (used by upload.py)
# ---------------------------------------------------------------------------

class _StubCache:
    """No-op cache stub returned when the real cache is unavailable.

    Prevents ``AttributeError`` on callers that do
    ``_query_cache.invalidate_by_tags(…)``.
    """

    def invalidate_by_tags(self, tags: Any) -> None:  # noqa: D401
        """No-op — cache not available."""

    def clear(self) -> None:  # noqa: D401
        """No-op — cache not available."""


def _get_cache_instance(attr_name: str) -> Any:
    """Resolve a named cache attribute from the global cache manager."""
    try:
        from backend.core.enhanced_cache_integration import get_enhanced_cache_manager
        manager = get_enhanced_cache_manager()
        return getattr(manager, attr_name, _StubCache())
    except Exception:
        return _StubCache()


# Lazy-initialised references — callers import these directly
_query_cache: Any = None
_file_analysis_cache: Any = None


def __getattr__(name: str) -> Any:
    """Module-level ``__getattr__`` for lazy resolution of cache instances.

    On first access of ``_query_cache`` or ``_file_analysis_cache`` the
    real cache object is resolved from the global manager.
    """
    global _query_cache, _file_analysis_cache

    if name == "_query_cache":
        if _query_cache is None:
            _query_cache = _get_cache_instance("query_cache")
        return _query_cache

    if name == "_file_analysis_cache":
        if _file_analysis_cache is None:
            _file_analysis_cache = _get_cache_instance("file_analysis_cache")
        return _file_analysis_cache

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
