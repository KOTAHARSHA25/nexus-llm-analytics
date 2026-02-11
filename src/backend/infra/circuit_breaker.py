"""Circuit Breaker Pattern Implementation for Nexus LLM Analytics
==============================================================

Provides fault tolerance and resilience for external service calls
(LLMs, databases, network I/O).  Implements the classic three-state
machine — **CLOSED** → **OPEN** → **HALF_OPEN** — with configurable
failure/success thresholds and a global registry.

v2.0 Enterprise Additions
-------------------------
* :class:`CircuitBreakerEvent` — structured state-transition audit record.
* :class:`CircuitBreakerDashboard` — real-time event log and aggregate
  health summary across all registered circuits.
* :func:`get_circuit_breaker_dashboard` — thread-safe singleton accessor.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Three states of the circuit breaker finite-state machine."""

    CLOSED = "CLOSED"     # Normal operation (calls allowed)
    OPEN = "OPEN"         # Failing (calls blocked)
    HALF_OPEN = "HALF_OPEN" # Testing recovery (limited calls allowed)

@dataclass
class CircuitBreakerConfig:
    """Tuneable knobs for a single :class:`CircuitBreaker`.

    Attributes:
        failure_threshold: Consecutive failures before opening the circuit.
        recovery_timeout:  Seconds to wait before probing recovery.
        success_threshold: Successes in HALF_OPEN before closing.
        timeout:           Per-call execution timeout (seconds).
        enabled:           Set ``False`` to bypass the breaker entirely.
    """
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    success_threshold: int = 1
    timeout: float = 60.0  # Execution timeout
    enabled: bool = True

class CircuitBreaker:
    """Three-state circuit breaker guarding an external call.

    Thread-safe via an internal ``RLock``.  Tracks per-circuit
    success/failure counters for health reporting.

    Attributes:
        name: Human-readable identifier for the protected service.
        config: Tuneable thresholds and timeouts.

    Thread Safety:
        All public methods acquire ``_lock`` before mutating state,
        making this class safe for concurrent use from multiple threads.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        """Initialise breaker *name* with optional *config* overrides."""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._total_calls = 0
        self._total_success = 0
        self._total_failures = 0
        self._lock = threading.RLock()
        
    @property
    def state(self) -> CircuitState:
        with self._lock:
            # Check if recovery timeout has passed to move from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_time
                if elapsed > self.config.recovery_timeout:
                    logger.info("Circuit '%s' attempting recovery (HALF_OPEN)", self.name)
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
            return self._state

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through the breaker, returning a fallback on failure."""
        # Check circuit state
        if self.state == CircuitState.OPEN:
            return self._get_fallback_response("Circuit is OPEN due to repeated failures")
            
        with self._lock:
            self._total_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._handle_success()
            return result
        except Exception as e:
            self._handle_failure(e)
            return self._get_fallback_response(str(e))

    async def async_call(self, coro_func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Async version of :meth:`call` for coroutine functions."""
        if self.state == CircuitState.OPEN:
            return self._get_fallback_response("Circuit is OPEN due to repeated failures")

        with self._lock:
            self._total_calls += 1

        try:
            result = await coro_func(*args, **kwargs)
            self._handle_success()
            return result
        except Exception as e:
            self._handle_failure(e)
            return self._get_fallback_response(str(e))
            
    def _handle_success(self) -> None:
        """Record a successful call and potentially close the circuit."""
        with self._lock:
            self._total_success += 1
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit '%s' recovered (CLOSED)", self.name)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _handle_failure(self, error: Exception) -> None:
        """Record a failed call; open the circuit when threshold is reached."""
        with self._lock:
            self._total_failures += 1
            self._last_failure_time = time.time()
            logger.warning("Circuit '%s' call failed: %s", self.name, error)
            
            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error("Circuit '%s' failure threshold reached! Opening circuit.", self.name, exc_info=True)
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.error("Circuit '%s' recovery failed. Re-opening circuit.", self.name, exc_info=True)

    def _get_fallback_response(self, error_msg: str) -> Dict[str, Any]:
        """Return a safe, JSON-serialisable fallback when the circuit is open."""
        return {
            "fallback_used": True,
            "error": error_msg,
            "result": f"[!] Service temporarily unavailable: {error_msg}. Please try again later.",
            "success": False
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Snapshot of circuit state and cumulative call statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "health": "healthy" if self._state == CircuitState.CLOSED else "unhealthy",
                "statistics": {
                    "total_calls": self._total_calls,
                    "success_count": self._total_success,
                    "failure_count": self._total_failures,
                    "success_rate": (self._total_success / self._total_calls * 100) if self._total_calls > 0 else 0
                }
            }
            
    def reset(self) -> None:
        """Manually reset the circuit to CLOSED."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0
            logger.info("Circuit '%s' manually reset to CLOSED", self.name)

# Registry for all circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()

def get_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """Return the named breaker, creating one with *config* if it doesn't exist."""
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]

def get_all_circuit_breaker_status() -> Dict[str, Any]:
    """Aggregate health across every registered circuit breaker."""
    with _registry_lock:
        circuits = [cb.get_health_status() for cb in _circuit_breakers.values()]
        
    return {
        "overall_health": "degraded" if any(c['state'] == 'OPEN' for c in circuits) else "healthy",
        "circuit_breakers": circuits
    }

def circuit_breaker_protected(circuit_name: str = "default") -> Callable:
    """Decorator to guard a sync function with a named circuit breaker."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cb = get_circuit_breaker(circuit_name)
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# v2.0 Enterprise Additions
# ---------------------------------------------------------------------------

from collections import Counter  # noqa: E402 (import after existing code)


@dataclass
class CircuitBreakerEvent:
    """Structured audit record for a circuit-breaker state transition.

    Attributes:
        circuit_name:  Name of the circuit that changed state.
        from_state:    Previous :class:`CircuitState`.
        to_state:      New :class:`CircuitState`.
        timestamp:     Unix epoch of the transition (defaults to *now*).
        reason:        Optional human-readable cause.
        failure_count: Failure counter at the moment of transition.
    """

    circuit_name: str
    from_state: CircuitState
    to_state: CircuitState
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    failure_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "circuit_name": self.circuit_name,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "failure_count": self.failure_count,
        }


class CircuitBreakerDashboard:
    """Enterprise monitoring dashboard for all circuit breakers.

    Maintains a bounded, thread-safe event log and exposes aggregate
    health summaries suitable for admin UIs and alerting pipelines.

    Attributes:
        max_events: Upper bound on retained events (oldest are trimmed).

    Thread Safety:
        All public methods acquire an internal ``Lock``, making this
        class safe for concurrent use from multiple threads.
    """

    def __init__(self, max_events: int = 500) -> None:
        self.max_events = max_events
        self._events: List[CircuitBreakerEvent] = []
        self._lock = threading.Lock()

    def record_event(self, event: CircuitBreakerEvent) -> None:
        """Append *event* to the log, trimming oldest entries if needed."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events :]

    def get_events(
        self,
        circuit_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return recent events, optionally filtered by *circuit_name*."""
        with self._lock:
            events = self._events
            if circuit_name is not None:
                events = [e for e in events if e.circuit_name == circuit_name]
            return [e.to_dict() for e in events[-limit:]]

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate health summary across every registered circuit."""
        with self._lock:
            events_by_circuit: Dict[str, int] = dict(Counter(
                e.circuit_name for e in self._events
            ))
            last_event_time = self._events[-1].timestamp if self._events else None

        overall_status = get_all_circuit_breaker_status()

        return {
            "total_circuits": len(_circuit_breakers),
            "overall_health": overall_status.get("overall_health", "unknown"),
            "total_events": len(self._events),
            "events_by_circuit": events_by_circuit,
            "last_event_time": last_event_time,
        }


# Singleton dashboard instance (lazy, double-checked locking)
_dashboard_instance: Optional[CircuitBreakerDashboard] = None
_dashboard_lock = threading.Lock()


def get_circuit_breaker_dashboard() -> CircuitBreakerDashboard:
    """Return the global :class:`CircuitBreakerDashboard` singleton.

    Uses double-checked locking to avoid acquiring the lock on every
    call after the instance has been created.
    """
    global _dashboard_instance
    if _dashboard_instance is None:
        with _dashboard_lock:
            if _dashboard_instance is None:
                _dashboard_instance = CircuitBreakerDashboard()
    return _dashboard_instance
