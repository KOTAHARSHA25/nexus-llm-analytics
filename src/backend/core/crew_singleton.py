"""
Centralized CrewManager singleton access point.
Ensures only one instance across entire application for optimal performance.

This module prevents the creation of multiple CrewManager instances which was
causing severe performance degradation (198s startup vs 33s with singleton).

Usage:
    from backend.core.crew_singleton import get_crew_manager
    
    crew_manager = get_crew_manager()
    result = crew_manager.handle_query(...)
"""

from typing import Optional
import threading
import logging

# Delayed import to avoid circular dependencies
_CrewManager = None
_crew_instance: Optional[any] = None
_crew_lock = threading.Lock()

logger = logging.getLogger(__name__)


def get_crew_manager():
    """
    Get the global CrewManager singleton instance.
    Thread-safe with double-checked locking pattern.
    
    Returns:
        CrewManager: The singleton instance
        
    Note:
        This function uses lazy initialization. The CrewManager is only
        created on first call, which happens when the first API request
        requiring AI processing is made.
    """
    global _crew_instance, _CrewManager
    
    # First check (no lock for performance)
    if _crew_instance is None:
        with _crew_lock:
            # Double check with lock (prevents race conditions)
            if _crew_instance is None:
                logger.debug("Creating singleton CrewManager instance")
                
                # Import here to avoid circular dependencies and import-time overhead
                if _CrewManager is None:
                    from backend.agents.crew_manager import CrewManager
                    _CrewManager = CrewManager
                
                _crew_instance = _CrewManager()
                logger.debug("CrewManager singleton created successfully")
    
    return _crew_instance


def reset_crew_manager():
    """
    Reset the singleton instance.
    
    WARNING: This is for testing purposes ONLY!
    DO NOT use this in production code as it will break the singleton pattern
    and potentially cause issues with ongoing operations.
    """
    global _crew_instance
    with _crew_lock:
        if _crew_instance is not None:
            logger.warning("⚠️ CrewManager singleton being reset (testing only!)")
            _crew_instance = None
        else:
            logger.debug("CrewManager singleton already None, nothing to reset")


def is_initialized() -> bool:
    """
    Check if the CrewManager singleton has been initialized.
    
    Returns:
        bool: True if initialized, False otherwise
    """
    return _crew_instance is not None
