"""
Resource Lifecycle Registry — Nexus LLM Analytics v2.0
======================================================

Centralized registry for managing the lifecycle of system resources
(threads, pools, temporary files, connections). Ensures proper
cleanup on application shutdown.

Features:
* Thread-safe resource registration
* Priority-based cleanup
* Async and sync cleanup support
* Singleton accessor

Usage::

    from backend.core.resource_registry import get_resource_registry
    
    registry = get_resource_registry()
    registry.register("db_pool", db.close, priority=10)
    
    # At shutdown
    await registry.cleanup_all()

Author: Nexus Analytics Research Team
Date: February 2026
"""

import logging
import inspect
import asyncio
import threading
from typing import Dict, Any, Callable, List, Optional, Union, Coroutine
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass(order=True)
class ResourceEntry:
    priority: int
    name: str = field(compare=False)
    cleanup_func: Callable = field(compare=False)
    is_async: bool = field(compare=False, default=False)

class ResourceRegistry:
    """
    Central registry for system resources requiring explicit cleanup.
    """
    
    def __init__(self):
        self._resources: List[ResourceEntry] = []
        self._lock = threading.Lock()
        self._is_shutting_down = False
        
    def register(self, name: str, cleanup_func: Callable, priority: int = 50) -> None:
        """
        Register a resource for cleanup.
        
        Args:
            name: Unique identifier for the resource
            cleanup_func: Callable to execute during cleanup (sync or async)
            priority: Cleanup priority (0=first, 100=last). Lower executes first.
        """
        if self._is_shutting_down:
            logger.warning(f"Attempted to register resource '{name}' during shutdown")
            return
            
        is_async = inspect.iscoroutinefunction(cleanup_func)
        
        with self._lock:
            # Remove existing if active (allows re-registration)
            self._resources = [r for r in self._resources if r.name != name]
            
            entry = ResourceEntry(
                priority=priority,
                name=name,
                cleanup_func=cleanup_func,
                is_async=is_async
            )
            self._resources.append(entry)
            # persistent sort is fine for low volume
            self._resources.sort()
            
        logger.debug(f"Registered resource '{name}' (priority={priority}, async={is_async})")
        
    def unregister(self, name: str) -> None:
        """Remove a resource from the registry (e.g. manually closed)."""
        with self._lock:
            self._resources = [r for r in self._resources if r.name != name]
            
    async def cleanup_all(self) -> None:
        """
        Execute cleanup for all registered resources in priority order.
        """
        self._is_shutting_down = True
        logger.info("Starting system resource cleanup...")
        
        # Get copy to avoid modification during iteration
        with self._lock:
            resources = list(self._resources)
            self._resources.clear()
            
        total = len(resources)
        failed = 0
        
        for i, res in enumerate(resources):
            try:
                logger.info(f"Cleaning up [{i+1}/{total}]: {res.name}")
                if res.is_async:
                    await res.cleanup_func()
                else:
                    res.cleanup_func()
            except Exception as e:
                logger.error(f"Failed to cleanup '{res.name}': {e}", exc_info=True)
                failed += 1
                
        logger.info(f"Cleanup completed. {total - failed}/{total} successful.")

# Singleton instance
_registry: Optional[ResourceRegistry] = None
_registry_lock = threading.Lock()

def get_resource_registry() -> ResourceRegistry:
    """Get global ResourceRegistry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ResourceRegistry()
    return _registry
