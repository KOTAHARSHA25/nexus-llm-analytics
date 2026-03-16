"""Phase 1 Integration Module.

Provides unified access to all Phase 1 resilience components and
ensures they work together through the :class:`Phase1Coordinator`.

Single entry point for:

* **Smart Fallback Management** — automatic degradation on failures.
* **Dynamic Model Discovery** — runtime detection of available models.
* **RAM-Aware Selection** — memory-pressure-driven model choice.
* **Circuit Breaker Protection** — fault isolation per service.
* **Enhanced Query Orchestration** — complexity-aware execution plans.

Design Principles:

* Zero Hardcoding — all configuration is dynamic.
* Domain Agnostic — works with any data domain.
* Fail-Safe — process never stops completely.
* Observable — full visibility into system state.

Enterprise v2.0 Additions
-------------------------
* **Phase1HealthMonitor** — periodic health-check runner that
  captures :class:`Phase1Status` snapshots and triggers alerts
  when the health score drops below a configurable threshold.

All v1.x APIs (``Phase1Coordinator``, ``get_phase1_coordinator``,
``resilient_llm_call``, re-exports) remain unchanged.

Author: Nexus Team
Since: v1.0 (Enterprise enhancements v2.0 — February 2026)
"""

import logging
from typing import Dict, Any, Optional, Callable, TypeVar
from dataclasses import dataclass
from functools import wraps
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Import all Phase 1 components
from .engine.smart_fallback import (
    SmartFallbackManager,
    get_fallback_manager,
    FallbackReason,
    GracefulDegradation
)

# Model discovery and RAM-aware selection now consolidated in model_selector
from .engine.model_selector import (
    DynamicModelDiscovery,
    get_model_discovery,
    ModelInfo,
    ModelCapability,
    RAMAwareSelector,
    get_ram_selector,
    MemoryPressureLevel,
    MemorySnapshot,
    ModelSelectionResult
)

from ..infra.circuit_breaker import (
    CircuitBreaker,
    get_circuit_breaker,
    circuit_breaker_protected,
    CircuitState,
    get_all_circuit_breaker_status
)

from .engine.query_orchestrator import (
    QueryOrchestrator,
    ExecutionPlan,
    ExecutionMethod,
    ReviewLevel
)


@dataclass
class Phase1Status:
    """Comprehensive status of all Phase 1 components"""
    healthy: bool
    fallback_stats: Dict[str, Any]
    model_discovery_stats: Dict[str, Any]
    ram_stats: Dict[str, Any]
    circuit_breaker_stats: Dict[str, Any]
    overall_health_score: float
    
    def to_dict(self) -> dict:
        return {
            'healthy': self.healthy,
            'overall_health_score': f"{self.overall_health_score:.2%}",
            'fallback': self.fallback_stats,
            'models': self.model_discovery_stats,
            'memory': self.ram_stats,
            'circuits': self.circuit_breaker_stats
        }


class Phase1Coordinator:
    """
    Coordinates all Phase 1 components for unified resilient operation.
    
    This is the main class that agents and the application should use
    to access Phase 1 functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Phase 1 coordinator.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        
        # Initialize all components via singletons
        self._fallback_manager = get_fallback_manager()
        self._model_discovery = get_model_discovery()
        self._ram_selector = get_ram_selector()
        
        # Start background monitoring
        self._ram_selector.start_background_monitoring()
        
        # Track initialization time
        self._init_time = time.time()
        
        logger.info("Phase1Coordinator initialized - all components ready")
    
    def get_best_model_for_query(self, 
                                  query: str,
                                  complexity: float = 0.5) -> str:
        """
        Get the best available model for a query.
        
        Considers:
        - Query complexity
        - Available RAM
        - Discovered models
        - Current system state
        
        Args:
            query: The user query
            complexity: Pre-computed complexity score (0-1)
        
        Returns:
            Model name string
        """
        try:
            # Get available RAM
            available_ram = self._ram_selector.get_available_ram_for_model()
            
            # Get discovered models
            model_info = self._model_discovery.get_model_for_complexity(
                complexity=complexity,
                available_ram_gb=available_ram
            )
            
            if model_info:
                logger.debug(f"Selected model: {model_info.name} for complexity {complexity:.2f}")
                return model_info.name
            
            # Fallback if no models discovered
            logger.warning("No discovered models, using fallback")
            return "tinyllama"
            
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            return "tinyllama"
    
    def get_model_fallback_chain(self) -> list:
        """
        Get ordered fallback chain of available models.
        
        Returns:
            List of model names from most capable to least
        """
        try:
            return self._model_discovery.get_model_chain()
        except Exception as e:
            logger.error(f"Failed to get fallback chain: {e}")
            return ["tinyllama"]
    
    def execute_with_resilience(self,
                               operation: Callable[..., T],
                               circuit_name: str = "default",
                               fallback: Optional[Callable[..., T]] = None,
                               *args,
                               **kwargs) -> T:
        """
        Execute an operation with full Phase 1 protection.
        
        Includes:
        - Circuit breaker protection
        - Automatic retry with fallback
        - Graceful degradation
        
        Args:
            operation: Function to execute
            circuit_name: Name for circuit breaker
            fallback: Optional fallback function
            *args, **kwargs: Arguments for operation
        
        Returns:
            Result from operation or fallback
        """
        circuit = get_circuit_breaker(circuit_name)
        
        try:
            # Execute through circuit breaker
            result = circuit.call(operation, *args, **kwargs)
            return result
            
        except Exception as e:
            logger.warning(f"Operation failed: {e}")
            
            if fallback:
                try:
                    return fallback(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            # Generate graceful degradation response
            return GracefulDegradation.generate_degraded_response(
                query=str(args) if args else "unknown",
                error=str(e)
            )
    
    def get_status(self) -> Phase1Status:
        """
        Get comprehensive status of all Phase 1 components.
        """
        try:
            fallback_stats = self._fallback_manager.get_stats()
            model_stats = self._model_discovery.get_statistics()
            ram_stats = self._ram_selector.get_statistics()
            circuit_stats = get_all_circuit_breaker_status()
            
            # Calculate overall health score
            health_score = 1.0
            
            # Deduct for fallback usage
            if fallback_stats.get('total_fallbacks', 0) > 0:
                recovery_rate = fallback_stats.get('recovered', 0) / fallback_stats['total_fallbacks']
                health_score *= recovery_rate
            
            # Deduct for open circuits
            open_circuits = len(circuit_stats.get('circuit_breakers', []))
            closed_circuits = sum(
                1 for cb in circuit_stats.get('circuit_breakers', [])
                if cb.get('state') == 'closed'
            )
            if open_circuits > 0:
                health_score *= closed_circuits / open_circuits
            
            # Deduct for memory pressure
            pressure = ram_stats.get('current_state', {}).get('pressure_level', 'low')
            if pressure == 'critical':
                health_score *= 0.5
            elif pressure == 'high':
                health_score *= 0.75
            elif pressure == 'moderate':
                health_score *= 0.9
            
            healthy = health_score >= 0.7
            
            return Phase1Status(
                healthy=healthy,
                fallback_stats=fallback_stats,
                model_discovery_stats=model_stats,
                ram_stats=ram_stats,
                circuit_breaker_stats=circuit_stats,
                overall_health_score=health_score
            )
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return Phase1Status(
                healthy=False,
                fallback_stats={'error': str(e)},
                model_discovery_stats={},
                ram_stats={},
                circuit_breaker_stats={},
                overall_health_score=0.0
            )
    
    def refresh_models(self) -> int:
        """
        Force refresh of discovered models.
        
        Returns:
            Number of models discovered
        """
        try:
            models = self._model_discovery.discover_models_sync(force_refresh=True)
            return len(models)
        except Exception as e:
            logger.error(f"Model refresh failed: {e}")
            return 0
    
    def reset_all_circuits(self):
        """Reset all circuit breakers to closed state"""
        from .circuit_breaker import _circuit_breakers
        for cb in _circuit_breakers.values():
            cb.reset()
        logger.info("All circuit breakers reset")
    
    def shutdown(self):
        """Clean shutdown of Phase 1 components"""
        try:
            self._ram_selector.stop_background_monitoring()
            logger.info("Phase1Coordinator shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# Global coordinator instance
_coordinator: Optional[Phase1Coordinator] = None


def get_phase1_coordinator() -> Phase1Coordinator:
    """Get or create the global Phase 1 coordinator"""
    global _coordinator
    if _coordinator is None:
        _coordinator = Phase1Coordinator()
    return _coordinator


def resilient_llm_call(circuit_name: str = "llm"):
    """
    Decorator for resilient LLM calls with full Phase 1 protection.
    
    Usage:
        @resilient_llm_call("my_agent")
        def generate_response(query: str) -> str:
            return llm.generate(query)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            coordinator = get_phase1_coordinator()
            return coordinator.execute_with_resilience(
                func,
                circuit_name=circuit_name,
                *args,
                **kwargs
            )
        return wrapper
    return decorator


# Export all public components
__all__ = [
    # Coordinator
    'Phase1Coordinator',
    'get_phase1_coordinator',
    'Phase1Status',
    
    # Decorator
    'resilient_llm_call',
    
    # Fallback
    'SmartFallbackManager',
    'get_fallback_manager',
    'FallbackReason',
    'GracefulDegradation',
    
    # Model Discovery
    'DynamicModelDiscovery',
    'get_model_discovery',
    'ModelInfo',
    'ModelCapability',
    
    # RAM Selection
    'RAMAwareSelector',
    'get_ram_selector',
    'MemoryPressureLevel',
    'MemorySnapshot',
    
    # Circuit Breaker
    'CircuitBreaker',
    'get_circuit_breaker',
    'circuit_breaker_protected',
    'CircuitState',
    'get_all_circuit_breaker_status',
    
    # Orchestrator
    'QueryOrchestrator',
    'ExecutionPlan',
    'ExecutionMethod',
    'ReviewLevel',
]


# ============================================================================
# Enterprise v2.0 — Phase1HealthMonitor
# ============================================================================

import threading as _threading


class Phase1HealthMonitor:
    """Periodic health-check runner for Phase 1 components.

    Captures :class:`Phase1Status` snapshots at a configurable
    interval and invokes a callback when the health score drops
    below a threshold.

    Args:
        coordinator: The :class:`Phase1Coordinator` to monitor.
        interval_seconds: Seconds between health checks.
        alert_threshold: Score below which the alert callback fires.
        on_alert: Optional callback ``(Phase1Status) -> None``.

    Example::

        coord = get_phase1_coordinator()
        monitor = Phase1HealthMonitor(coord, interval_seconds=60)
        monitor.start()

    .. versionadded:: 2.0
    """

    def __init__(
        self,
        coordinator: Phase1Coordinator | None = None,
        interval_seconds: float = 120.0,
        alert_threshold: float = 0.7,
        on_alert=None,
    ) -> None:
        self._coordinator = coordinator
        self._interval = interval_seconds
        self._threshold = alert_threshold
        self._on_alert = on_alert
        self._running = False
        self._thread: _threading.Thread | None = None
        self._history: list[dict] = []

    def start(self) -> None:
        """Start the background health-check loop."""
        if self._running:
            return
        self._running = True
        self._thread = _threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Phase1HealthMonitor started (interval=%ss)", self._interval)

    def stop(self) -> None:
        """Signal the monitor to stop after the current iteration."""
        self._running = False
        logger.info("Phase1HealthMonitor stopping")

    def _loop(self) -> None:
        import time as _time
        coord = self._coordinator or get_phase1_coordinator()
        while self._running:
            try:
                status = coord.get_status()
                entry = {
                    "healthy": status.healthy,
                    "score": status.overall_health_score,
                    "timestamp": _time.time(),
                }
                self._history.append(entry)
                # Keep last 500 entries
                if len(self._history) > 500:
                    self._history = self._history[-250:]
                if status.overall_health_score < self._threshold and self._on_alert:
                    self._on_alert(status)
            except Exception as exc:
                logger.warning("Phase1HealthMonitor check failed: %s", exc)
            _time.sleep(self._interval)

    @property
    def history(self) -> list[dict]:
        """Return the recorded health-check history."""
        return list(self._history)
