"""
Smart Fallback Manager
======================
Provides intelligent fallback mechanisms to ensure processing never stops.

Design Principles:
- Domain Agnostic: No hardcoded domain-specific logic
- Data Agnostic: Works with any data structure
- Self-Healing: Automatic recovery from failures
- Observable: Comprehensive logging and metrics

.. versionadded:: 2.0.0
   Added :class:`FallbackCircuitBreaker`, :class:`FallbackAnalytics`,
   :class:`FallbackPolicy`, and :class:`EnterpriseFallbackManager`.

Fallback Chains:
1. Model Fallback: complex → medium → simple → minimal
2. Method Fallback: code_gen → direct_llm → cached → template
3. Review Fallback: mandatory → optional → skip
4. Timeout Fallback: extend → retry → downgrade → fail gracefully
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

import psutil

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar('T')

__all__ = [
    # v1.x (backward compatible)
    "SmartFallbackManager",
    "FallbackReason",
    "FallbackEvent",
    "FallbackChain",
    "GracefulDegradation",
    "get_fallback_manager",
    # v2.0 Enterprise additions
    "FallbackCircuitBreaker",
    "FallbackAnalytics",
    "FallbackPolicy",
    "EnterpriseFallbackManager",
    "get_enterprise_fallback_manager",
]


class FallbackReason(Enum):
    """Reasons for fallback activation"""
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    MODEL_UNAVAILABLE = "model_unavailable"
    EXECUTION_ERROR = "execution_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    QUALITY_THRESHOLD = "quality_threshold"
    UNKNOWN = "unknown"


@dataclass
class FallbackEvent:
    """Record of a single fallback event.

    Captures the transition from one strategy to another, including
    the reason for the fallback and whether recovery eventually succeeded.
    """
    timestamp: float
    original_strategy: str
    fallback_strategy: str
    reason: FallbackReason
    error_message: Optional[str] = None
    recovered: bool = False
    
    def to_dict(self) -> dict:
        """Serialize the fallback event to a plain dictionary.

        Returns:
            Dictionary with timestamp, strategies, reason, and recovery status.
        """
        return {
            "timestamp": self.timestamp,
            "original": self.original_strategy,
            "fallback": self.fallback_strategy,
            "reason": self.reason.value,
            "error": self.error_message,
            "recovered": self.recovered
        }


@dataclass
class FallbackChain:
    """Ordered chain of fallback options.

    Maintains an ordered list of strategies and advances through them
    when failures occur, recording each fallback event for observability.
    """
    name: str
    strategies: List[str]
    current_index: int = 0
    events: List[FallbackEvent] = field(default_factory=list)
    
    def current(self) -> str:
        """Get current strategy.

        Returns:
            Name of the active strategy in this chain.
        """
        if self.current_index < len(self.strategies):
            return self.strategies[self.current_index]
        return self.strategies[-1]  # Return last as final fallback
    
    def next(self, reason: FallbackReason, error: str = None) -> Optional[str]:
        """Move to next fallback strategy.

        Args:
            reason: Categorised reason for the current failure.
            error: Optional human-readable error description.

        Returns:
            Name of the next strategy, or ``None`` if the chain is exhausted.
        """
        if self.current_index < len(self.strategies) - 1:
            original = self.current()
            self.current_index += 1
            new_strategy = self.current()
            
            event = FallbackEvent(
                timestamp=time.time(),
                original_strategy=original,
                fallback_strategy=new_strategy,
                reason=reason,
                error_message=error
            )
            self.events.append(event)
            
            logger.warning("Fallback [%s]: %s → %s (reason: %s)", self.name, original, new_strategy, reason.value)
            return new_strategy
        
        logger.error("Fallback chain [%s] exhausted, no more options", self.name)
        return None
    
    def reset(self) -> None:
        """Reset to first strategy."""
        self.current_index = 0
    
    def has_fallback(self) -> bool:
        """Check if more fallbacks available.

        Returns:
            ``True`` if at least one fallback strategy remains.
        """
        return self.current_index < len(self.strategies) - 1


class SmartFallbackManager:
    """
    Centralized fallback management for the entire system.
    
    Ensures that no operation fails completely - always provides
    a degraded but functional response.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.fallback_history: List[FallbackEvent] = []
        self.stats = {
            "total_fallbacks": 0,
            "recovered": 0,
            "exhausted": 0,
            "by_reason": {}
        }
        
        # Initialize fallback chains
        self._init_fallback_chains()
        
        logger.info("SmartFallbackManager initialized")
    
    def _get_installed_model_names(self) -> List[str]:
        """Fetch installed model names from Ollama dynamically.

        Returns:
            Sorted list of model names (largest first), excluding embedding models.
        """
        if requests is None:
            logger.warning("requests library not installed, cannot discover models")
            return []
        try:
            settings = get_settings()
            ollama_url = settings.ollama_base_url
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models_data = response.json().get("models", [])
                # Extract model names, filter out embedding models
                model_names = []
                for model in models_data:
                    name = model.get("name", "")
                    # Skip embedding models
                    if "embed" not in name.lower() and "nomic" not in name.lower():
                        model_names.append(name)
                
                # Sort by size (larger = more capable, should be tried first)
                model_names.sort(
                    key=lambda m: next(
                        (model.get("size", 0) for model in models_data if model.get("name") == m),
                        0
                    ),
                    reverse=True
                )
                
                logger.debug("Discovered models for fallback: %s", model_names)
                return model_names
            
            return []
        except Exception as e:
            logger.warning("Could not fetch installed models: %s", e)
            return []
    
    def _init_fallback_chains(self) -> None:
        """Initialize default fallback chains based on installed models."""
        
        # Dynamically build model fallback chain from installed models
        installed_models = self._get_installed_model_names()
        
        if installed_models:
            # Sort by estimated capability (larger models first)
            model_strategies = installed_models[:4]  # Top 4 models
        else:
            # Absolute fallback if no models detected
            model_strategies = ["llama3.1:8b", "phi3:mini", "tinyllama"]
            logger.warning("No installed models detected, using default fallback chain")
        
        self.model_chain = FallbackChain(
            name="model",
            strategies=model_strategies + ["echo"]  # 'echo' as last resort
        )
        
        # Execution method fallback chain
        self.method_chain = FallbackChain(
            name="method",
            strategies=["code_generation", "direct_llm", "template", "cached"]
        )
        
        # Review level fallback chain
        self.review_chain = FallbackChain(
            name="review",
            strategies=["mandatory", "optional", "skip"]
        )
        
        # Timeout chain (seconds)
        self.timeout_chain = FallbackChain(
            name="timeout",
            strategies=["300", "180", "60", "30"]
        )
    
    def get_model_fallback(self, current_model: str, reason: FallbackReason, error: str = None) -> str:
        """Get fallback model when current model fails.

        Args:
            current_model: Name of the model that just failed.
            reason: Categorised reason for the failure.
            error: Optional human-readable error description.

        Returns:
            Name of the next model to try.
        """
        
        # Find current position in chain
        try:
            idx = self.model_chain.strategies.index(current_model)
            self.model_chain.current_index = idx
        except ValueError:
            # Model not in chain, start from beginning
            self.model_chain.current_index = 0
        
        fallback = self.model_chain.next(reason, error)
        self._record_fallback(reason)
        
        if fallback:
            return fallback
        
        # Last resort: return tinyllama as absolute minimum
        return "tinyllama"
    
    def get_method_fallback(self, current_method: str, reason: FallbackReason, error: str = None) -> str:
        """Get fallback execution method.

        Args:
            current_method: Name of the execution method that failed.
            reason: Categorised reason for the failure.
            error: Optional human-readable error description.

        Returns:
            Name of the next execution method to try.
        """
        
        try:
            idx = self.method_chain.strategies.index(current_method)
            self.method_chain.current_index = idx
        except ValueError:
            self.method_chain.current_index = 0
        
        fallback = self.method_chain.next(reason, error)
        self._record_fallback(reason)
        
        return fallback or "direct_llm"
    
    def get_review_fallback(self, current_level: str, reason: FallbackReason, error: str = None) -> str:
        """Get fallback review level.

        Args:
            current_level: Current review stringency level.
            reason: Categorised reason for the failure.
            error: Optional human-readable error description.

        Returns:
            Name of the next review level to try.
        """
        
        try:
            idx = self.review_chain.strategies.index(current_level)
            self.review_chain.current_index = idx
        except ValueError:
            self.review_chain.current_index = 0
        
        fallback = self.review_chain.next(reason, error)
        self._record_fallback(reason)
        
        return fallback or "skip"
    
    def get_adaptive_timeout(self, base_timeout: int, model: str) -> int:
        """Get adaptive timeout based on system resources and model.

        Dynamically adjusts the base timeout using current memory
        availability, CPU load, and estimated model size.

        Args:
            base_timeout: Starting timeout in seconds before adjustment.
            model: Model name used to estimate a size multiplier.

        Returns:
            Adjusted timeout in seconds, clamped to ``[30, 900]``.
        """
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Dynamic timeout calculation based on resources
        # Lower memory → longer timeout (model loads slower)
        # Higher CPU load → longer timeout
        
        memory_factor = 1.0
        if available_gb < 4:
            memory_factor = 2.0  # Double timeout for low memory
        elif available_gb < 8:
            memory_factor = 1.5
        
        cpu_factor = 1.0
        if cpu_percent > 80:
            cpu_factor = 1.5
        elif cpu_percent > 60:
            cpu_factor = 1.25
        
        # Model size factor (estimated, not hardcoded to specific models)
        # Larger model names often indicate larger sizes
        model_factor = 1.0
        if "8b" in model.lower() or "7b" in model.lower():
            model_factor = 1.5
        elif "mini" in model.lower() or "tiny" in model.lower():
            model_factor = 0.5
        
        adaptive_timeout = int(base_timeout * memory_factor * cpu_factor * model_factor)
        
        # Bounds
        min_timeout = 30
        max_timeout = 900
        
        return max(min_timeout, min(max_timeout, adaptive_timeout))
    
    def _record_fallback(self, reason: FallbackReason) -> None:
        """Record fallback statistics.

        Args:
            reason: The categorised reason for this fallback event.
        """
        self.stats["total_fallbacks"] += 1
        reason_key = reason.value
        self.stats["by_reason"][reason_key] = self.stats["by_reason"].get(reason_key, 0) + 1
    
    def mark_recovered(self) -> None:
        """Mark that recovery was successful.

        Increments the recovered counter used to compute the overall
        recovery rate reported by :meth:`get_stats`.
        """
        self.stats["recovered"] += 1
    
    def mark_exhausted(self) -> None:
        """Mark that all fallbacks were exhausted.

        Increments the exhausted counter, indicating no recovery
        path remained for a particular operation.
        """
        self.stats["exhausted"] += 1
    
    def reset_chains(self) -> None:
        """Reset all fallback chains to initial state.

        Rewinds every chain (model, method, review, timeout) back
        to its first strategy so they can be traversed again.
        """
        self.model_chain.reset()
        self.method_chain.reset()
        self.review_chain.reset()
        self.timeout_chain.reset()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics.

        Returns:
            Dictionary with totals, recovery rate, and per-chain status.
        """
        recovery_rate = (
            self.stats["recovered"] / self.stats["total_fallbacks"]
            if self.stats["total_fallbacks"] > 0 else 1.0
        )
        
        return {
            **self.stats,
            "recovery_rate": f"{recovery_rate:.2%}",
            "chains": {
                "model": {"current": self.model_chain.current(), "events": len(self.model_chain.events)},
                "method": {"current": self.method_chain.current(), "events": len(self.method_chain.events)},
                "review": {"current": self.review_chain.current(), "events": len(self.review_chain.events)},
            }
        }
    
    def with_fallback(
        self,
        primary_func: Callable[..., T],
        fallback_func: Callable[..., T],
        max_retries: int = 2,
    ) -> Callable[..., T]:
        """Wrap a function with automatic fallback.

        After *max_retries* failures of *primary_func*, the returned
        wrapper transparently delegates to *fallback_func*.

        Args:
            primary_func: The preferred callable to execute.
            fallback_func: Callable used when all retries of *primary_func* fail.
            max_retries: Number of retry attempts before falling back.

        Returns:
            A wrapper callable with the same signature as *primary_func*.

        Raises:
            RuntimeError: When both primary and fallback functions fail.
        """
        @wraps(primary_func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return primary_func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning("Attempt %s failed: %s", attempt + 1, e)
                    
                    if attempt < max_retries:
                        # Try to determine fallback reason
                        error_str = str(e).lower()
                        if "timeout" in error_str or "timed out" in error_str:
                            reason = FallbackReason.TIMEOUT
                        elif "memory" in error_str or "oom" in error_str:
                            reason = FallbackReason.MEMORY_LIMIT
                        elif "not found" in error_str or "unavailable" in error_str:
                            reason = FallbackReason.MODEL_UNAVAILABLE
                        else:
                            reason = FallbackReason.EXECUTION_ERROR
                        
                        self._record_fallback(reason)
            
            # All retries failed, use fallback function
            logger.info("Primary function failed after %s attempts, using fallback", max_retries + 1)
            try:
                result = fallback_func(*args, **kwargs)
                self.mark_recovered()
                return result
            except Exception as fallback_error:
                self.mark_exhausted()
                raise RuntimeError(
                    f"Both primary and fallback failed. "
                    f"Primary error: {last_error}, Fallback error: {fallback_error}"
                )
        
        return wrapper


class GracefulDegradation:
    """
    Provides graceful degradation responses when all fallbacks fail.
    Ensures the system NEVER returns an empty or crash response.
    """
    
    @staticmethod
    def generate_degraded_response(
        query: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a meaningful response even when processing fails.

        Args:
            query: The user's original natural-language query.
            context: Optional metadata or environmental context.
            error: Optional error message from the failed operation.

        Returns:
            Dictionary containing a degraded but user-friendly response.
        """
        
        # Analyze query to provide relevant degraded response
        query_lower = query.lower()
        
        # Detect query intent (domain agnostic)
        if any(word in query_lower for word in ["average", "mean", "sum", "total", "count"]):
            response_type = "aggregation"
            message = "I understand you're asking for a calculation. Due to system constraints, I couldn't complete the full analysis. Please try a simpler query or retry in a moment."
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            response_type = "comparison"
            message = "I understand you're asking for a comparison. The system is currently under heavy load. Please try again or simplify your query."
        elif any(word in query_lower for word in ["trend", "over time", "pattern", "forecast"]):
            response_type = "trend_analysis"
            message = "I understand you're asking about trends or patterns. This requires additional processing that couldn't be completed. Please try again."
        elif any(word in query_lower for word in ["why", "explain", "reason", "cause"]):
            response_type = "explanation"
            message = "I understand you're asking for an explanation. I couldn't fully process this request, but I'm here to help when you try again."
        else:
            response_type = "general"
            message = "I received your query but couldn't complete the analysis due to temporary constraints. Please try again or rephrase your question."
        
        return {
            "success": False,
            "degraded": True,
            "response_type": response_type,
            "message": message,
            "query": query,
            "error": error,
            "suggestion": "Try simplifying your query, reducing data size, or waiting a moment before retrying.",
            "metadata": {
                "fallback_activated": True,
                "timestamp": time.time(),
                "context_provided": context is not None
            }
        }
    
    @staticmethod
    def get_minimal_analysis(data_preview: str, query: str) -> str:
        """Provide minimal analysis based purely on data structure.

        Args:
            data_preview: Raw text preview of the uploaded data.
            query: The user's original natural-language query.

        Returns:
            Plain-text summary derived from structural pattern matching.
        """
        
        lines = data_preview.split('\n')
        response_parts = []
        
        # Extract basic info from preview
        if lines:
            # Try to identify columns
            first_line = lines[0]
            if ',' in first_line or '\t' in first_line:
                delimiter = ',' if ',' in first_line else '\t'
                columns = first_line.split(delimiter)
                response_parts.append(f"Data contains {len(columns)} columns: {', '.join(columns[:5])}")
                if len(columns) > 5:
                    response_parts.append(f"...and {len(columns) - 5} more")
        
        # Count rows (approximate from preview)
        data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
        response_parts.append(f"Preview shows approximately {len(data_lines)} rows")
        
        # Acknowledge query
        response_parts.append(f"\nRegarding your query: '{query[:100]}...'")
        response_parts.append("Full analysis could not be completed, but here's what I can see from the data structure.")
        
        return "\n".join(response_parts)


# Singleton instance
_fallback_manager: Optional[SmartFallbackManager] = None
_fallback_lock = threading.Lock()  # Thread-safe singleton lock


def get_fallback_manager() -> SmartFallbackManager:
    """Get or create the singleton fallback manager (thread-safe).

    Returns:
        The shared :class:`SmartFallbackManager` singleton instance.
    """
    global _fallback_manager
    if _fallback_manager is None:
        with _fallback_lock:
            if _fallback_manager is None:  # Double-check locking
                _fallback_manager = SmartFallbackManager()
    return _fallback_manager


# =============================================================================
# ENTERPRISE: FALLBACK CIRCUIT BREAKER
# =============================================================================

class FallbackCircuitBreaker:
    """Circuit breaker for fallback chains to prevent cascading failures.

    Tracks per-chain failure rates and temporarily disables chains
    that are consistently failing, directing traffic to alternative
    chains until the original recovers.

    Args:
        failure_threshold: Number of failures to trigger open state.
        recovery_timeout: Seconds before attempting half-open probe.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self._lock = threading.Lock()
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._chain_states: Dict[str, Dict[str, Any]] = {}

    def _get_state(self, chain_name: str) -> Dict[str, Any]:
        if chain_name not in self._chain_states:
            self._chain_states[chain_name] = {
                "failures": 0,
                "state": "closed",
                "last_failure": 0.0,
            }
        return self._chain_states[chain_name]

    def is_available(self, chain_name: str) -> bool:
        """Check if a fallback chain is available."""
        with self._lock:
            s = self._get_state(chain_name)
            if s["state"] == "closed":
                return True
            if s["state"] == "open":
                if time.time() - s["last_failure"] >= self._recovery_timeout:
                    s["state"] = "half_open"
                    return True
                return False
            return True  # half_open

    def record_success(self, chain_name: str) -> None:
        """Record a successful fallback chain execution."""
        with self._lock:
            s = self._get_state(chain_name)
            s["failures"] = 0
            s["state"] = "closed"

    def record_failure(self, chain_name: str) -> None:
        """Record a fallback chain failure."""
        with self._lock:
            s = self._get_state(chain_name)
            s["failures"] += 1
            s["last_failure"] = time.time()
            if s["failures"] >= self._failure_threshold:
                s["state"] = "open"
                logger.warning(
                    "Fallback circuit OPEN for chain '%s' after %d failures",
                    chain_name, s["failures"],
                )

    def get_status(self) -> Dict[str, Any]:
        """Return circuit breaker status for all chains."""
        with self._lock:
            return {name: dict(s) for name, s in self._chain_states.items()}


# =============================================================================
# ENTERPRISE: FALLBACK ANALYTICS
# =============================================================================

class FallbackAnalytics:
    """Collects and reports fallback metrics for observability.

    Tracks which chains are used, how often fallbacks trigger,
    recovery times, and root-cause distribution.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_fallbacks = 0
        self._reason_counts: Dict[str, int] = {}
        self._chain_usage: Dict[str, int] = {}
        self._recovery_times: List[float] = []
        self._events: deque = deque(maxlen=200)

    def record_event(
        self,
        reason: str,
        chain: str,
        recovery_ms: float = 0.0,
        original: str = "",
        fallback: str = "",
    ) -> None:
        """Record a fallback event.

        Args:
            reason: FallbackReason value string.
            chain: Name of the fallback chain used.
            recovery_ms: Time to complete fallback in ms.
            original: Original intended value.
            fallback: Actual fallback value used.
        """
        with self._lock:
            self._total_fallbacks += 1
            self._reason_counts[reason] = self._reason_counts.get(reason, 0) + 1
            self._chain_usage[chain] = self._chain_usage.get(chain, 0) + 1
            if recovery_ms > 0:
                self._recovery_times.append(recovery_ms)
            self._events.append({
                "time": time.time(),
                "reason": reason,
                "chain": chain,
                "original": original,
                "fallback": fallback,
                "recovery_ms": recovery_ms,
            })

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive fallback analytics."""
        with self._lock:
            return {
                "total_fallbacks": self._total_fallbacks,
                "reason_distribution": dict(self._reason_counts),
                "chain_usage": dict(self._chain_usage),
                "avg_recovery_ms": round(
                    sum(self._recovery_times) / len(self._recovery_times), 2
                ) if self._recovery_times else 0,
                "recent_events": list(self._events)[-10:],
            }


# =============================================================================
# ENTERPRISE: FALLBACK POLICY
# =============================================================================

@dataclass
class FallbackPolicy:
    """Configurable policy for fallback behaviour.

    Attributes:
        name: Policy name.
        max_fallback_depth: Maximum chain hops before giving up.
        enable_circuit_breaker: Whether the circuit breaker is active.
        circuit_failure_threshold: Failures to trip the circuit.
        circuit_recovery_timeout: Recovery timeout in seconds.
        prefer_cached: Whether to prefer cached responses.
        graceful_message: Default message for graceful degradation.
    """
    name: str = "default"
    max_fallback_depth: int = 5
    enable_circuit_breaker: bool = True
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0
    prefer_cached: bool = False
    graceful_message: str = "Analysis temporarily unavailable. Please try again."


# =============================================================================
# ENTERPRISE: ENTERPRISE FALLBACK MANAGER
# =============================================================================

class EnterpriseFallbackManager(SmartFallbackManager):
    """Enterprise-grade fallback manager with circuit breakers and analytics.

    Extends :class:`SmartFallbackManager` with:

    * :class:`FallbackCircuitBreaker` to prevent cascading failures.
    * :class:`FallbackAnalytics` for observability.
    * Configurable :class:`FallbackPolicy`.

    Fully backward compatible with the base class.

    Args:
        policy: Fallback policy to enforce.
    """

    def __init__(self, policy: Optional[FallbackPolicy] = None) -> None:
        super().__init__()
        self.policy = policy or FallbackPolicy()
        self.circuit_breaker = FallbackCircuitBreaker(
            failure_threshold=self.policy.circuit_failure_threshold,
            recovery_timeout=self.policy.circuit_recovery_timeout,
        )
        self.analytics = FallbackAnalytics()
        logger.info("EnterpriseFallbackManager initialized (policy=%s)", self.policy.name)

    def get_enterprise_statistics(self) -> Dict[str, Any]:
        """Extended statistics with circuit breaker and analytics."""
        stats = self.get_statistics()
        stats["circuit_breaker"] = self.circuit_breaker.get_status()
        stats["analytics"] = self.analytics.get_statistics()
        stats["policy"] = {
            "name": self.policy.name,
            "max_depth": self.policy.max_fallback_depth,
            "circuit_breaker_enabled": self.policy.enable_circuit_breaker,
        }
        return stats


# =============================================================================
# ENTERPRISE SINGLETON
# =============================================================================

_enterprise_fallback: Optional[EnterpriseFallbackManager] = None
_enterprise_fallback_lock = threading.Lock()


def get_enterprise_fallback_manager(
    policy: Optional[FallbackPolicy] = None,
) -> EnterpriseFallbackManager:
    """Get or create singleton enterprise fallback manager (thread-safe)."""
    global _enterprise_fallback
    if _enterprise_fallback is None:
        with _enterprise_fallback_lock:
            if _enterprise_fallback is None:
                _enterprise_fallback = EnterpriseFallbackManager(policy=policy)
    return _enterprise_fallback
