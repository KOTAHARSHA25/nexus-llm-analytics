"""
Backend test fixtures — singleton cleanup and environment isolation.

Ensures that module-level singletons (caches, model managers, etc.)
don't leak state between test files when running the full suite.
"""
import os
import pytest

# Prevent EnhancedCacheManager from spawning asyncio background tasks
# during test collection / import (causes RuntimeError when no loop is running).
os.environ.setdefault("DISABLE_CACHE_BACKGROUND", "1")


@pytest.fixture(autouse=True)
def _reset_cache_singleton():
    """Reset the global EnhancedCacheManager singleton between tests."""
    yield
    try:
        import src.backend.core.enhanced_cache_integration as ci
        ci._enhanced_cache_manager = None
    except Exception:
        pass
