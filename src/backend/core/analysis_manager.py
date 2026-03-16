"""
Analysis Manager — Nexus LLM Analytics v2.0
============================================

Provides cancellable analysis lifecycle management with thread-safe
state tracking.  Every running analysis receives a unique identifier
that can be used to query progress, cancel execution, or clean up
stale records.

Enterprise v2.0 Additions
-------------------------
* **AnalysisObserver** — Observer interface for pluggable analysis
  lifecycle hooks (start / update / cancel / complete / error).
* **LoggingAnalysisObserver** — Built-in observer that logs every
  lifecycle transition at INFO level.
* **AnalysisLimiter** — Concurrency guard that caps the maximum
  number of simultaneously running analyses.
* ``get_analysis_manager()`` — Thread-safe singleton accessor.

Backward Compatibility
----------------------
The module-level ``analysis_manager`` singleton and the free function
``check_cancellation()`` remain available at their original import
paths.  All v1.x public signatures are unchanged.

.. versionchanged:: 2.0
   Added observer hooks, concurrency limiter, and singleton accessor.

Author: Nexus Analytics Research Team
Date: February 2026
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Set
from uuid import uuid4

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class AnalysisManager:
    """Manage running analyses and provide cancellation capabilities.

    Thread-safe analysis lifecycle manager supporting concurrent
    analyses with cancel-by-ID semantics and automatic stale-record
    cleanup.

    Attributes:
        _running_analyses: Mapping of analysis IDs to their metadata.
        _cancelled_analyses: Set of IDs that have been cancelled.
        _lock: Reentrant lock protecting internal state.
        _observers: Registered :class:`AnalysisObserver` instances.

    Example::

        mgr = AnalysisManager()
        aid = mgr.start_analysis(user_session="u-1")
        mgr.update_analysis_stage(aid, "routing")
        mgr.complete_analysis(aid)

    .. versionchanged:: 2.0
       Added observer notification on every lifecycle transition.
    """

    def __init__(self) -> None:
        self._running_analyses: Dict[str, Dict] = {}
        self._cancelled_analyses: Set[str] = set()
        self._lock = threading.Lock()
        self._observers: List[AnalysisObserver] = []

    # -- Observer management (v2.0) ------------------------------------------

    def add_observer(self, observer: "AnalysisObserver") -> None:
        """Register an observer for lifecycle events.

        Args:
            observer: An :class:`AnalysisObserver` implementation.
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: "AnalysisObserver") -> None:
        """Unregister a previously registered observer.

        Args:
            observer: The observer to remove.
        """
        self._observers = [o for o in self._observers if o is not observer]

    def _notify(self, event: str, analysis_id: str, **kwargs) -> None:
        """Notify all registered observers about a lifecycle event."""
        for obs in self._observers:
            try:
                handler = getattr(obs, f"on_{event}", None)
                if handler:
                    handler(analysis_id, **kwargs)
            except Exception as exc:  # pragma: no cover — observers must not break core
                logger.warning("Observer %s raised %s on %s", obs, exc, event)
        
    def start_analysis(self, user_session: str | None = None) -> str:
        """Start a new analysis and return its unique identifier.

        Args:
            user_session: Optional session token of the requesting user.

        Returns:
            Newly generated analysis ID.
        """
        analysis_id = str(uuid4())
        
        with self._lock:
            self._running_analyses[analysis_id] = {
                'id': analysis_id,
                'user_session': user_session,
                'start_time': time.time(),
                'status': 'running',
                'stage': 'initializing'
            }
        
        logger.info("[ANALYSIS_MANAGER] Started analysis %s for session %s", analysis_id, user_session)
        self._notify("start", analysis_id, user_session=user_session)
        return analysis_id
    
    def update_analysis_stage(self, analysis_id: str, stage: str) -> None:
        """Update the current stage of an analysis.

        Args:
            analysis_id: The unique analysis identifier.
            stage: Human-readable label for the new stage.
        """
        with self._lock:
            if analysis_id in self._running_analyses:
                self._running_analyses[analysis_id]['stage'] = stage
                self._running_analyses[analysis_id]['last_update'] = time.time()
                logger.info("[ANALYSIS_MANAGER] Analysis %s stage: %s", analysis_id, stage)
        self._notify("update", analysis_id, stage=stage)
    
    def is_cancelled(self, analysis_id: str) -> bool:
        """Check whether *analysis_id* has been cancelled.

        Args:
            analysis_id: The analysis identifier to check.

        Returns:
            ``True`` if the analysis was cancelled.
        """
        with self._lock:
            return analysis_id in self._cancelled_analyses
    
    def cancel_analysis(self, analysis_id: str) -> bool:
        """Cancel a running analysis.

        Args:
            analysis_id: The analysis identifier to cancel.

        Returns:
            ``True`` if the analysis existed and was cancelled,
            ``False`` if it was not found.
        """
        with self._lock:
            if analysis_id in self._running_analyses:
                self._cancelled_analyses.add(analysis_id)
                self._running_analyses[analysis_id]['status'] = 'cancelled'
                self._running_analyses[analysis_id]['cancelled_time'] = time.time()
                logger.info("[ANALYSIS_MANAGER] Cancelled analysis %s", analysis_id)
                self._notify("cancel", analysis_id)
                return True
            return False
    
    def complete_analysis(self, analysis_id: str) -> None:
        """Mark an analysis as completed.

        Args:
            analysis_id: The unique analysis identifier.
        """
        with self._lock:
            if analysis_id in self._running_analyses:
                self._running_analyses[analysis_id]['status'] = 'completed'
                self._running_analyses[analysis_id]['end_time'] = time.time()
                # Remove from cancelled set if it was there
                self._cancelled_analyses.discard(analysis_id)
                logger.info("[ANALYSIS_MANAGER] Completed analysis %s", analysis_id)
        self._notify("complete", analysis_id)

    def fail_analysis(self, analysis_id: str, error: str) -> None:
        """Mark an analysis as failed.

        Args:
            analysis_id: The unique analysis identifier.
            error: Error message describing the failure.
        """
        with self._lock:
            if analysis_id in self._running_analyses:
                self._running_analyses[analysis_id]['status'] = 'failed'
                self._running_analyses[analysis_id]['error'] = str(error)
                self._running_analyses[analysis_id]['end_time'] = time.time()
                self._cancelled_analyses.discard(analysis_id)
                logger.error("[ANALYSIS_MANAGER] Analysis %s failed: %s", analysis_id, error)
        self._notify("error", analysis_id, error=error)
    
    def get_analysis_status(self, analysis_id: str) -> Optional[Dict]:
        """Return the metadata dict for *analysis_id*, or ``None``.

        Args:
            analysis_id: The analysis identifier.

        Returns:
            A dictionary with keys ``id``, ``status``, ``stage``, etc.
        """
        with self._lock:
            return self._running_analyses.get(analysis_id)
    
    def get_running_analyses(self) -> Dict[str, Dict]:
        """Get all currently running analyses"""
        with self._lock:
            return {
                aid: info for aid, info in self._running_analyses.items() 
                if info['status'] == 'running'
            }
    
    def cleanup_old_analyses(self, max_age_hours: int = 24) -> None:
        """Remove completed/cancelled analysis records older than *max_age_hours*.

        Args:
            max_age_hours: Age threshold in hours for record removal.
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self._lock:
            to_remove = []
            for analysis_id, info in self._running_analyses.items():
                age = current_time - info['start_time']
                if age > max_age_seconds and info['status'] in ['completed', 'cancelled']:
                    to_remove.append(analysis_id)
            
            for analysis_id in to_remove:
                del self._running_analyses[analysis_id]
                self._cancelled_analyses.discard(analysis_id)
            
            if to_remove:
                logger.info("[ANALYSIS_MANAGER] Cleaned up %s old analyses", len(to_remove))

# Global analysis manager instance
analysis_manager = AnalysisManager()

def check_cancellation(analysis_id: str) -> None:
    """Raise :class:`HTTPException` (499) if *analysis_id* has been cancelled.

    Args:
        analysis_id: The analysis identifier to check.

    Raises:
        HTTPException: With status 499 when the analysis was cancelled.
    """
    if analysis_manager.is_cancelled(analysis_id):
        raise HTTPException(status_code=499, detail="Analysis was cancelled by user")


# ============================================================================
# Enterprise v2.0 — Observer Pattern & Concurrency Limiter
# ============================================================================

from abc import ABC, abstractmethod


class AnalysisObserver(ABC):
    """Observer interface for analysis lifecycle events.

    Implement any subset of the ``on_*`` hooks to receive notifications
    when an analysis transitions between states.

    Example::

        class SlackNotifier(AnalysisObserver):
            def on_complete(self, analysis_id: str, **kw) -> None:
                send_slack(f"Analysis {analysis_id} finished!")

        analysis_manager.add_observer(SlackNotifier())

    .. versionadded:: 2.0
    """

    def on_start(self, analysis_id: str, **kwargs) -> None:
        """Called when a new analysis begins.

        Args:
            analysis_id: Unique identifier of the analysis.
            **kwargs: Extra context (e.g. ``user_session``).
        """

    def on_update(self, analysis_id: str, **kwargs) -> None:
        """Called when the analysis stage changes.

        Args:
            analysis_id: Unique identifier of the analysis.
            **kwargs: Extra context (e.g. ``stage``).
        """

    def on_cancel(self, analysis_id: str, **kwargs) -> None:
        """Called when an analysis is cancelled.

        Args:
            analysis_id: Unique identifier of the analysis.
        """

    def on_complete(self, analysis_id: str, **kwargs) -> None:
        """Called when an analysis completes successfully.

        Args:
            analysis_id: Unique identifier of the analysis.
        """

    def on_error(self, analysis_id: str, **kwargs) -> None:
        """Called when an analysis encounters an error.

        Args:
            analysis_id: Unique identifier of the analysis.
            **kwargs: Extra context (e.g. ``error``).
        """


class LoggingAnalysisObserver(AnalysisObserver):
    """Built-in observer that emits structured log messages.

    Logs every lifecycle event at INFO level using the module logger.

    .. versionadded:: 2.0
    """

    def on_start(self, analysis_id: str, **kwargs) -> None:
        logger.info("[OBSERVER] Analysis %s started — %s", analysis_id, kwargs)

    def on_update(self, analysis_id: str, **kwargs) -> None:
        logger.info("[OBSERVER] Analysis %s updated — %s", analysis_id, kwargs)

    def on_cancel(self, analysis_id: str, **kwargs) -> None:
        logger.info("[OBSERVER] Analysis %s cancelled", analysis_id)

    def on_complete(self, analysis_id: str, **kwargs) -> None:
        logger.info("[OBSERVER] Analysis %s completed", analysis_id)

    def on_error(self, analysis_id: str, **kwargs) -> None:
        logger.warning("[OBSERVER] Analysis %s error — %s", analysis_id, kwargs)


class AnalysisLimiter:
    """Concurrency guard that caps simultaneous running analyses.

    Use as a context manager around the analysis body to prevent
    resource exhaustion.

    Args:
        max_concurrent: Maximum allowed concurrent analyses.

    Example::

        limiter = AnalysisLimiter(max_concurrent=3)
        with limiter.acquire("a-1"):
            run_heavy_analysis()

    Raises:
        RuntimeError: When the concurrency limit is reached.

    .. versionadded:: 2.0
    """

    def __init__(self, max_concurrent: int = 5) -> None:
        self._max = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active: Set[str] = set()
        self._lock = threading.Lock()

    # -- Context manager helpers ------------------------------------------------

    class _Guard:
        """RAII-style guard returned by :meth:`acquire`."""

        def __init__(self, limiter: "AnalysisLimiter", analysis_id: str) -> None:
            self._limiter = limiter
            self._aid = analysis_id

        def __enter__(self) -> "AnalysisLimiter._Guard":
            acquired = self._limiter._semaphore.acquire(blocking=False)
            if not acquired:
                raise RuntimeError(
                    f"Concurrency limit ({self._limiter._max}) reached — "
                    f"cannot start analysis {self._aid}"
                )
            with self._limiter._lock:
                self._limiter._active.add(self._aid)
            return self

        def __exit__(self, *exc_info) -> None:
            with self._limiter._lock:
                self._limiter._active.discard(self._aid)
            self._limiter._semaphore.release()

    def acquire(self, analysis_id: str) -> "_Guard":
        """Return a context-manager guard for *analysis_id*.

        Args:
            analysis_id: The analysis to guard.

        Returns:
            A context manager that releases the slot on exit.
        """
        return self._Guard(self, analysis_id)

    @property
    def active_count(self) -> int:
        """Number of analyses currently running."""
        with self._lock:
            return len(self._active)

    @property
    def available_slots(self) -> int:
        """Number of free concurrency slots."""
        return self._max - self.active_count


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor (v2.0)
# ---------------------------------------------------------------------------

_analysis_manager_lock = threading.Lock()
_analysis_manager_instance: Optional[AnalysisManager] = None


def get_analysis_manager() -> AnalysisManager:
    """Return the global :class:`AnalysisManager` singleton.

    Thread-safe with double-checked locking.

    Returns:
        The shared ``AnalysisManager`` instance.

    .. versionadded:: 2.0
    """
    global _analysis_manager_instance
    if _analysis_manager_instance is None:
        with _analysis_manager_lock:
            if _analysis_manager_instance is None:
                _analysis_manager_instance = AnalysisManager()
    return _analysis_manager_instance