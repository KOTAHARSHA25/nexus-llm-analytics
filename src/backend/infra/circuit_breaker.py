"""
Circuit Breaker Pattern Implementation for Nexus LLM Analytics
==============================================================
Provides fault tolerance and resilience for external service calls (LLMs).
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"     # Normal operation (calls allowed)
    OPEN = "OPEN"         # Failing (calls blocked)
    HALF_OPEN = "HALF_OPEN" # Testing recovery (limited calls allowed)

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    success_threshold: int = 1
    timeout: float = 60.0  # Execution timeout
    enabled: bool = True

class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
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
                    logger.info(f"Circuit '{self.name}' attempting recovery (HALF_OPEN)")
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
            return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        # Check circuit state
        if self.state == CircuitState.OPEN:
            return self._get_fallback_response("Circuit is OPEN due to repeated failures")
            
        self._total_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._handle_success()
            return result
        except Exception as e:
            self._handle_failure(e)
            return self._get_fallback_response(str(e))
            
    def _handle_success(self):
        with self._lock:
            self._total_success += 1
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit '{self.name}' recovered (CLOSED)")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _handle_failure(self, error: Exception):
        with self._lock:
            self._total_failures += 1
            self._last_failure_time = time.time()
            logger.warning(f"Circuit '{self.name}' call failed: {error}")
            
            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(f"Circuit '{self.name}' failure threshold reached! Opening circuit.")
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.error(f"Circuit '{self.name}' recovery failed. Re-opening circuit.")

    def _get_fallback_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate a safe fallback response"""
        return {
            "fallback_used": True,
            "error": error_msg,
            "result": f"[!] Service temporarily unavailable: {error_msg}. Please try again later.",
            "success": False
        }

    def get_health_status(self) -> Dict[str, Any]:
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
            
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED")

# Registry for all circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()

def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]

def get_all_circuit_breaker_status() -> Dict[str, Any]:
    with _registry_lock:
        circuits = [cb.get_health_status() for cb in _circuit_breakers.values()]
        
    return {
        "overall_health": "degraded" if any(c['state'] == 'OPEN' for c in circuits) else "healthy",
        "circuit_breakers": circuits
    }

def circuit_breaker_protected(circuit_name: str = "default"):
    """Decorator to protect functions with circuit breaker"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cb = get_circuit_breaker(circuit_name)
            return cb.call(func, *args, **kwargs)
        return wrapper
    return decorator
