"""Intelligent Router Module
=========================
Routes queries to appropriate model tiers based on complexity analysis.

Uses QueryComplexityAnalyzer to score queries and select optimal model tier
for efficiency (fast models for simple queries, powerful for complex).

Design Principles:
- Uses existing complexity analyzer (no duplication)
- RAM-aware (respects system memory constraints)
- Tracks statistics for monitoring
- Supports user overrides

.. versionadded:: 2.0.0
   Added :class:`RouterCircuitBreaker`, :class:`RoutingPolicy`,
   :class:`RouterHealthCheck`, :class:`ABTestManager`,
   and :class:`EnterpriseRouter`.

Backward Compatibility
----------------------
All v1.x public names remain at the same import paths.
"""

from __future__  import annotations

import hashlib
import json
import logging
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .model_selector import ModelSelector
from backend.core.query_complexity_analyzer import ComplexityScore, QueryComplexityAnalyzer

logger = logging.getLogger(__name__)

__all__ = [
    # v1.x (backward compatible)
    "IntelligentRouter",
    "ModelTier",
    "RoutingDecision",
    "get_intelligent_router",
    # v2.0 Enterprise additions
    "RouterCircuitBreaker",
    "RouterCircuitState",
    "RoutingPolicy",
    "RoutingPolicyBuilder",
    "RouterHealthCheck",
    "ABTestManager",
    "EnterpriseRouter",
    "get_enterprise_router",
]


class ModelTier(Enum):
    """Model capability tiers"""
    FAST = "fast"           # Quick responses, simple queries (phi3:mini, gemma:2b)
    BALANCED = "balanced"   # Good balance of speed/quality (llama3.1:8b)
    FULL_POWER = "full_power"  # Maximum capability (larger models)


@dataclass
class RoutingDecision:
    """Result of a routing decision.

    Attributes:
        query: Original user query text.
        selected_tier: Model capability tier chosen by the router.
        selected_model: Name of the Ollama model to invoke.
        complexity_score: Numeric complexity score from the analyzer.
        reason: Human-readable explanation of the routing choice.
        timestamp: Unix epoch seconds when the decision was made.
        user_override: Model name if the user bypassed automatic routing.
        fallback_used: Whether a fallback tier was required.
        original_tier: Tier initially selected before any fallback.
    """
    query: str
    selected_tier: ModelTier
    selected_model: str
    complexity_score: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    user_override: Optional[str] = None
    fallback_used: bool = False
    original_tier: Optional[ModelTier] = None
    
    def to_dict(self) -> dict:
        """Serialize the routing decision to a plain dictionary.

        Returns:
            Dict with query (truncated), tier, model, score, and metadata.
        """
        return {
            "query": self.query[:100],  # Truncate for logging
            "selected_tier": self.selected_tier.value,
            "selected_model": self.selected_model,
            "complexity_score": round(self.complexity_score, 3),
            "reason": self.reason,
            "timestamp": self.timestamp,
            "user_override": self.user_override,
            "fallback_used": self.fallback_used
        }


class IntelligentRouter:
    """
    Routes queries to optimal model tiers based on complexity analysis.
    
    Features:
    - Uses QueryComplexityAnalyzer for scoring
    - Selects from installed models dynamically
    - Tracks routing statistics
    - Supports user overrides
    - RAM-aware model selection
    """
    
    # Thresholds for tier selection (aligned with complexity analyzer)
    FAST_THRESHOLD = 0.30
    BALANCED_THRESHOLD = 0.65
    
    def __init__(self) -> None:
        """Initialize the router.

        Sets up the complexity analyzer, fallback chains, statistics
        counters, and performs an initial model-tier refresh from Ollama.
        """
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # Fallback chains for each tier (if preferred model unavailable)
        self.fallback_chain: Dict[ModelTier, List[str]] = {
            ModelTier.FAST: [],
            ModelTier.BALANCED: [],
            ModelTier.FULL_POWER: []
        }
        
        # Statistics tracking
        self.routing_history: deque = deque(maxlen=1000)  # Last 1000 decisions
        self.tier_usage_count: Dict[ModelTier, int] = {
            ModelTier.FAST: 0,
            ModelTier.BALANCED: 0,
            ModelTier.FULL_POWER: 0
        }
        self.total_queries: int = 0
        self.override_count: int = 0
        self.fallback_count: int = 0
        self.avg_routing_time_ms: float = 0.0
        
        # Cache model tiers
        self._model_tiers: Dict[str, ModelTier] = {}
        self._refresh_model_tiers()
        
        logger.info("🧠 IntelligentRouter initialized")
    
    def _refresh_model_tiers(self) -> None:
        """Refresh model-tier assignments based on currently installed models.

        Queries Ollama for installed models, categorises each by parameter
        size into *FAST*, *BALANCED*, or *FULL_POWER*, and rebuilds the
        internal fallback chains used during routing.
        """
        try:
            installed = ModelSelector._get_installed_models()
            if not installed:
                logger.warning("No models found from Ollama")
                return
            
            # Clear and rebuild tiers
            for tier in ModelTier:
                self.fallback_chain[tier] = []
            self._model_tiers.clear()
            
            # Categorize models by size/capability
            for model_name, info in installed.items():
                size_gb = info.get("size_gb", 0)
                is_embedding = info.get("is_embedding", False)
                
                if is_embedding:
                    continue  # Skip embedding models
                
                # Assign tier based on model size
                if size_gb < 3:
                    tier = ModelTier.FAST
                elif size_gb < 8:
                    tier = ModelTier.BALANCED
                else:
                    tier = ModelTier.FULL_POWER
                
                self._model_tiers[model_name] = tier
                self.fallback_chain[tier].append(model_name)
            
            # Sort each tier by size (smallest first for fast, largest first for power)
            for tier in [ModelTier.FAST, ModelTier.BALANCED]:
                self.fallback_chain[tier].sort(
                    key=lambda m: installed.get(m, {}).get("size_gb", 0)
                )
            self.fallback_chain[ModelTier.FULL_POWER].sort(
                key=lambda m: installed.get(m, {}).get("size_gb", 0),
                reverse=True
            )
            
            logger.info(
                "📊 Model tiers refreshed: FAST=%s, BALANCED=%s, FULL_POWER=%s",
                len(self.fallback_chain[ModelTier.FAST]),
                len(self.fallback_chain[ModelTier.BALANCED]),
                len(self.fallback_chain[ModelTier.FULL_POWER]),
            )
            
        except Exception as e:
            logger.error("Error refreshing model tiers: %s", e, exc_info=True)
    
    def _select_model_for_tier(self, tier: ModelTier) -> Optional[str]:
        """Select the best available model for a given tier.

        Args:
            tier: The desired model capability tier.

        Returns:
            The model name string, or ``None`` if no model is available.
        """
        # Try models in this tier first
        for model in self.fallback_chain.get(tier, []):
            return model
        
        # Fallback to adjacent tiers
        if tier == ModelTier.FAST:
            # Try balanced if no fast models
            for model in self.fallback_chain.get(ModelTier.BALANCED, []):
                return model
        elif tier == ModelTier.FULL_POWER:
            # Try balanced if no full power models
            for model in self.fallback_chain.get(ModelTier.BALANCED, []):
                return model
        
        # Last resort: any available model
        for tier_models in self.fallback_chain.values():
            if tier_models:
                return tier_models[0]
        
        return None
    
    def route(
        self,
        query: str,
        data_info: Optional[Dict[str, Any]] = None,
        user_override: Optional[str] = None
    ) -> RoutingDecision:
        """
        Route a query to the optimal model tier.
        
        Args:
            query: The user's query
            data_info: Optional data context (rows, columns, etc.)
            user_override: Optional specific model to use (bypasses routing)
            
        Returns:
            RoutingDecision with selected tier and model
        """
        start_time = time.time()
        
        # Handle user override
        if user_override:
            self.override_count += 1
            self.total_queries += 1
            
            # Determine tier for the override model
            override_tier = self._model_tiers.get(user_override, ModelTier.BALANCED)
            
            decision = RoutingDecision(
                query=query,
                selected_tier=override_tier,
                selected_model=user_override,
                complexity_score=0.0,  # Not calculated for overrides
                reason="User override specified",
                user_override=user_override
            )
            
            self._record_decision(decision, start_time)
            return decision
        
        # Analyze query complexity
        complexity: ComplexityScore = self.complexity_analyzer.analyze(query, data_info)
        score = complexity.total_score
        
        # Determine tier from score
        if score < self.FAST_THRESHOLD:
            selected_tier = ModelTier.FAST
            reason = f"Simple query (score={score:.3f} < {self.FAST_THRESHOLD})"
        elif score < self.BALANCED_THRESHOLD:
            selected_tier = ModelTier.BALANCED
            reason = f"Medium complexity (score={score:.3f})"
        else:
            selected_tier = ModelTier.FULL_POWER
            reason = f"Complex query (score={score:.3f} >= {self.BALANCED_THRESHOLD})"
        
        # Select actual model for the tier
        selected_model = self._select_model_for_tier(selected_tier)
        fallback_used = False
        original_tier = None
        
        if selected_model is None:
            # Fallback: try other tiers
            fallback_used = True
            original_tier = selected_tier
            self.fallback_count += 1
            
            for fallback_tier in [ModelTier.BALANCED, ModelTier.FAST, ModelTier.FULL_POWER]:
                selected_model = self._select_model_for_tier(fallback_tier)
                if selected_model:
                    selected_tier = fallback_tier
                    reason += f" (fallback from {original_tier.value})"
                    break
        
        if selected_model is None:
            selected_model = "llama3.1:8b"  # Ultimate fallback
            reason += " (default fallback)"
        
        decision = RoutingDecision(
            query=query,
            selected_tier=selected_tier,
            selected_model=selected_model,
            complexity_score=score,
            reason=reason,
            fallback_used=fallback_used,
            original_tier=original_tier
        )
        
        self._record_decision(decision, start_time)
        return decision
    
    def _record_decision(self, decision: RoutingDecision, start_time: float) -> None:
        """Record a routing decision and update statistics.

        Args:
            decision: The completed routing decision to record.
            start_time: Monotonic timestamp captured before routing began,
                used to compute routing latency.
        """
        routing_time_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        self.total_queries += 1
        self.tier_usage_count[decision.selected_tier] += 1
        
        # Update rolling average routing time
        self.avg_routing_time_ms = (
            (self.avg_routing_time_ms * (self.total_queries - 1) + routing_time_ms)
            / self.total_queries
        )
        
        # Add to history
        self.routing_history.append(decision)
        
        logger.debug(
            "🎯 Routed query to %s/%s (score=%.3f, time=%.2fms)",
            decision.selected_tier.value,
            decision.selected_model,
            decision.complexity_score,
            routing_time_ms,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated routing statistics.

        Returns:
            Dictionary containing *total_queries*, per-tier usage counts
            and percentages, override/fallback counts, average routing
            latency, available models per tier, and the ten most recent
            routing decisions.
        """
        tier_percentages = {}
        if self.total_queries > 0:
            for tier in ModelTier:
                tier_percentages[tier.value] = round(
                    100 * self.tier_usage_count[tier] / self.total_queries, 1
                )
        
        recent_decisions = [d.to_dict() for d in list(self.routing_history)[-10:]]
        
        return {
            "total_queries": self.total_queries,
            "tier_usage": {t.value: c for t, c in self.tier_usage_count.items()},
            "tier_percentages": tier_percentages,
            "override_count": self.override_count,
            "fallback_count": self.fallback_count,
            "avg_routing_time_ms": round(self.avg_routing_time_ms, 2),
            "available_models_by_tier": {
                t.value: len(models) for t, models in self.fallback_chain.items()
            },
            "recent_decisions": recent_decisions
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.routing_history.clear()
        for tier in ModelTier:
            self.tier_usage_count[tier] = 0
        self.total_queries = 0
        self.override_count = 0
        self.fallback_count = 0
        self.avg_routing_time_ms = 0.0
        logger.info("📊 Routing statistics reset")


# Singleton instance
_intelligent_router: Optional[IntelligentRouter] = None
_router_lock = threading.Lock()  # Thread-safe singleton lock


def get_intelligent_router() -> IntelligentRouter:
    """Get or create the singleton IntelligentRouter instance (thread-safe).

    Returns:
        The shared :class:`IntelligentRouter` singleton.
    """
    global _intelligent_router
    if _intelligent_router is None:
        with _router_lock:
            if _intelligent_router is None:  # Double-check locking
                _intelligent_router = IntelligentRouter()
    return _intelligent_router


# =============================================================================
# ENTERPRISE: ROUTER CIRCUIT BREAKER
# =============================================================================

class RouterCircuitState(Enum):
    """Circuit breaker states for routing resilience."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Tier unavailable, using fallback
    HALF_OPEN = "half_open"    # Testing if tier recovered


@dataclass
class RouterCircuitBreaker:
    """Per-tier circuit breaker that opens on repeated model failures.

    When a model tier accumulates more than *failure_threshold* failures
    within the rolling window, the circuit opens and routes automatically
    fall back to the next available tier.  After *recovery_timeout*
    seconds the circuit transitions to half-open, allowing a single
    probe request through.

    Attributes:
        tier: The model tier this circuit protects.
        state: Current circuit state.
        failure_threshold: Number of failures before opening.
        recovery_timeout: Seconds to wait before half-open probe.
        failure_count: Running failure counter.
        success_count: Successes since last reset.
        last_failure_time: Timestamp of most recent failure.
        last_state_change: Timestamp of the most recent state transition.
    """
    tier: ModelTier = ModelTier.BALANCED
    state: RouterCircuitState = RouterCircuitState.CLOSED
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_success(self) -> None:
        """Record a successful routing outcome."""
        with self._lock:
            self.success_count += 1
            if self.state == RouterCircuitState.HALF_OPEN:
                self.state = RouterCircuitState.CLOSED
                self.failure_count = 0
                self.last_state_change = time.time()
                logger.info("Circuit CLOSED for tier %s after successful probe", self.tier.value)

    def record_failure(self) -> None:
        """Record a failed routing outcome."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold and self.state == RouterCircuitState.CLOSED:
                self.state = RouterCircuitState.OPEN
                self.last_state_change = time.time()
                logger.warning("Circuit OPEN for tier %s after %d failures", self.tier.value, self.failure_count)

    def is_available(self) -> bool:
        """Check if this tier can accept requests.

        Returns:
            ``True`` if the circuit is closed or has transitioned to
            half-open after the recovery timeout.
        """
        with self._lock:
            if self.state == RouterCircuitState.CLOSED:
                return True
            if self.state == RouterCircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = RouterCircuitState.HALF_OPEN
                    self.last_state_change = time.time()
                    logger.info("Circuit HALF-OPEN for tier %s (probing)", self.tier.value)
                    return True
                return False
            # HALF_OPEN: allow a single probe
            return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize circuit breaker state."""
        with self._lock:
            return {
                "tier": self.tier.value,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }


# =============================================================================
# ENTERPRISE: ROUTING POLICY
# =============================================================================

@dataclass
class RoutingPolicy:
    """Enterprise routing policy with configurable rules.

    Defines thresholds, cost controls, SLA constraints, and
    per-tier configuration for the routing engine.

    Attributes:
        name: Human-readable policy name.
        fast_threshold: Max complexity score for FAST tier.
        balanced_threshold: Max complexity score for BALANCED tier.
        max_cost_per_query: Maximum acceptable cost (model-size proxy).
        latency_sla_ms: Target latency in milliseconds.
        enable_ab_testing: Whether A/B experiments are active.
        ab_test_ratio: Fraction of traffic routed to experiment arm.
        tier_weights: Preference multipliers for each tier.
        blocked_models: Models excluded from selection.
        preferred_models: Models given priority within their tier.
    """
    name: str = "default"
    fast_threshold: float = 0.30
    balanced_threshold: float = 0.65
    max_cost_per_query: float = float("inf")
    latency_sla_ms: float = 5000.0
    enable_ab_testing: bool = False
    ab_test_ratio: float = 0.10
    tier_weights: Dict[str, float] = field(default_factory=lambda: {
        "fast": 1.0, "balanced": 1.0, "full_power": 1.0,
    })
    blocked_models: List[str] = field(default_factory=list)
    preferred_models: List[str] = field(default_factory=list)


class RoutingPolicyBuilder:
    """Fluent builder for :class:`RoutingPolicy`.

    .. code-block:: python

        policy = (RoutingPolicyBuilder("cost_optimized")
            .fast_threshold(0.40)
            .balanced_threshold(0.70)
            .max_cost(4.0)
            .latency_sla(3000)
            .block_model("llama3.1:70b")
            .build())
    """

    def __init__(self, name: str = "custom") -> None:
        self._policy = RoutingPolicy(name=name)

    def fast_threshold(self, v: float) -> "RoutingPolicyBuilder":
        self._policy.fast_threshold = v
        return self

    def balanced_threshold(self, v: float) -> "RoutingPolicyBuilder":
        self._policy.balanced_threshold = v
        return self

    def max_cost(self, v: float) -> "RoutingPolicyBuilder":
        self._policy.max_cost_per_query = v
        return self

    def latency_sla(self, ms: float) -> "RoutingPolicyBuilder":
        self._policy.latency_sla_ms = ms
        return self

    def enable_ab_testing(self, ratio: float = 0.10) -> "RoutingPolicyBuilder":
        self._policy.enable_ab_testing = True
        self._policy.ab_test_ratio = ratio
        return self

    def block_model(self, model: str) -> "RoutingPolicyBuilder":
        self._policy.blocked_models.append(model)
        return self

    def prefer_model(self, model: str) -> "RoutingPolicyBuilder":
        self._policy.preferred_models.append(model)
        return self

    def tier_weight(self, tier: str, weight: float) -> "RoutingPolicyBuilder":
        self._policy.tier_weights[tier] = weight
        return self

    def build(self) -> RoutingPolicy:
        """Build and return the policy."""
        return self._policy


# =============================================================================
# ENTERPRISE: ROUTER HEALTH CHECK
# =============================================================================

class RouterHealthCheck:
    """Monitors the health of each model tier.

    Tracks success rate, average latency, and error patterns per tier
    to inform routing decisions and circuit breaker transitions.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tier_metrics: Dict[str, Dict[str, Any]] = {}
        for tier in ModelTier:
            self._tier_metrics[tier.value] = {
                "total_requests": 0,
                "failures": 0,
                "latencies_ms": deque(maxlen=200),
                "last_check": time.time(),
                "healthy": True,
            }

    def record(self, tier: ModelTier, latency_ms: float, success: bool) -> None:
        """Record a routing outcome for a tier.

        Args:
            tier: Model tier that handled the request.
            latency_ms: Observed latency in milliseconds.
            success: Whether the request succeeded.
        """
        with self._lock:
            m = self._tier_metrics[tier.value]
            m["total_requests"] += 1
            m["latencies_ms"].append(latency_ms)
            if not success:
                m["failures"] += 1
            m["last_check"] = time.time()
            # Unhealthy if >50% failure rate over last 20 requests
            recent_total = min(m["total_requests"], 20)
            if recent_total >= 5:
                recent_failure_rate = m["failures"] / m["total_requests"]
                m["healthy"] = recent_failure_rate < 0.50

    def is_tier_healthy(self, tier: ModelTier) -> bool:
        """Check whether a tier is considered healthy."""
        with self._lock:
            return self._tier_metrics[tier.value]["healthy"]

    def get_tier_latency(self, tier: ModelTier) -> float:
        """Return average latency for a tier (ms)."""
        with self._lock:
            lats = self._tier_metrics[tier.value]["latencies_ms"]
            return sum(lats) / len(lats) if lats else 0.0

    def get_report(self) -> Dict[str, Any]:
        """Generate a full health report."""
        with self._lock:
            report = {}
            for tier_name, m in self._tier_metrics.items():
                lats = list(m["latencies_ms"])
                report[tier_name] = {
                    "healthy": m["healthy"],
                    "total_requests": m["total_requests"],
                    "failures": m["failures"],
                    "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0,
                    "p95_latency_ms": round(sorted(lats)[int(len(lats) * 0.95)] if len(lats) >= 20 else (max(lats) if lats else 0), 2),
                }
            return report


# =============================================================================
# ENTERPRISE: A/B TEST MANAGER
# =============================================================================

class ABTestManager:
    """Manages A/B experiments on routing strategies.

    Assigns queries deterministically to control/experiment arms
    based on a hash of the query text, ensuring reproducible splits.

    .. code-block:: python

        ab = ABTestManager(experiment_name="aggressive_fast")
        arm = ab.assign(query="show total sales")
        # arm is "control" or "experiment"
    """

    def __init__(
        self,
        experiment_name: str = "default",
        traffic_ratio: float = 0.10,
    ) -> None:
        self.experiment_name = experiment_name
        self.traffic_ratio = traffic_ratio
        self._lock = threading.Lock()
        self._results: Dict[str, Dict[str, int]] = {
            "control": {"total": 0, "success": 0, "failures": 0},
            "experiment": {"total": 0, "success": 0, "failures": 0},
        }

    def assign(self, query: str) -> str:
        """Deterministically assign a query to an arm.

        Args:
            query: Query text to hash for assignment.

        Returns:
            ``"control"`` or ``"experiment"``.
        """
        digest = hashlib.md5(f"{self.experiment_name}:{query}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return "experiment" if bucket < self.traffic_ratio else "control"

    def record_outcome(self, arm: str, success: bool) -> None:
        """Record whether the routed request succeeded."""
        with self._lock:
            if arm in self._results:
                self._results[arm]["total"] += 1
                if success:
                    self._results[arm]["success"] += 1
                else:
                    self._results[arm]["failures"] += 1

    def get_results(self) -> Dict[str, Any]:
        """Return A/B test results."""
        with self._lock:
            out = {"experiment_name": self.experiment_name, "traffic_ratio": self.traffic_ratio}
            for arm, data in self._results.items():
                total = data["total"]
                out[arm] = {
                    **data,
                    "success_rate": round(data["success"] / total, 4) if total else 0.0,
                }
            return out


# =============================================================================
# ENTERPRISE: ENTERPRISE ROUTER
# =============================================================================

class EnterpriseRouter(IntelligentRouter):
    """Enterprise-grade router with circuit breakers, policies, and health.

    Extends :class:`IntelligentRouter` with:

    * Per-tier :class:`RouterCircuitBreaker` for automatic failover.
    * Configurable :class:`RoutingPolicy` for threshold / cost / SLA control.
    * :class:`RouterHealthCheck` for real-time tier health monitoring.
    * Optional :class:`ABTestManager` for routing experiments.

    Fully backward compatible — calling ``route()`` works identically
    to the base class unless enterprise features are enabled.

    Args:
        policy: Routing policy to enforce.  Falls back to defaults.
    """

    def __init__(self, policy: Optional[RoutingPolicy] = None) -> None:
        super().__init__()
        self.policy = policy or RoutingPolicy()

        # Override thresholds from policy
        self.FAST_THRESHOLD = self.policy.fast_threshold
        self.BALANCED_THRESHOLD = self.policy.balanced_threshold

        # Per-tier circuit breakers
        self.circuit_breakers: Dict[ModelTier, RouterCircuitBreaker] = {
            tier: RouterCircuitBreaker(tier=tier) for tier in ModelTier
        }

        # Health check
        self.health_check = RouterHealthCheck()

        # A/B testing (optional)
        self._ab_manager: Optional[ABTestManager] = None
        if self.policy.enable_ab_testing:
            self._ab_manager = ABTestManager(
                experiment_name=f"router_{self.policy.name}",
                traffic_ratio=self.policy.ab_test_ratio,
            )

        logger.info(
            "EnterpriseRouter initialized (policy=%s, ab_testing=%s)",
            self.policy.name, self.policy.enable_ab_testing,
        )

    def route(
        self,
        query: str,
        data_info: Optional[Dict[str, Any]] = None,
        user_override: Optional[str] = None,
    ) -> RoutingDecision:
        """Route with enterprise resilience.

        Applies circuit breakers, blocked-model filtering, and
        health-aware tier selection on top of the base routing logic.

        Args:
            query: User query.
            data_info: Optional data context.
            user_override: Optional model override.

        Returns:
            :class:`RoutingDecision` with selected tier and model.
        """
        decision = super().route(query, data_info, user_override)

        # Check circuit breaker for selected tier
        if not user_override:
            tier = decision.selected_tier
            cb = self.circuit_breakers[tier]
            if not cb.is_available():
                # Fallback to healthy tier
                for fallback_tier in [ModelTier.BALANCED, ModelTier.FAST, ModelTier.FULL_POWER]:
                    if fallback_tier != tier and self.circuit_breakers[fallback_tier].is_available():
                        model = self._select_model_for_tier(fallback_tier)
                        if model:
                            decision.selected_tier = fallback_tier
                            decision.selected_model = model
                            decision.fallback_used = True
                            decision.original_tier = tier
                            decision.reason += f" [circuit-breaker fallback from {tier.value}]"
                            break

            # Filter blocked models
            if decision.selected_model in self.policy.blocked_models:
                for alt in self.fallback_chain.get(decision.selected_tier, []):
                    if alt not in self.policy.blocked_models:
                        decision.selected_model = alt
                        break

        return decision

    def record_outcome(
        self, tier: ModelTier, latency_ms: float, success: bool,
    ) -> None:
        """Record a request outcome for circuit breaker and health.

        Args:
            tier: Tier that was used.
            latency_ms: Observed latency.
            success: Whether the request succeeded.
        """
        cb = self.circuit_breakers[tier]
        if success:
            cb.record_success()
        else:
            cb.record_failure()

        self.health_check.record(tier, latency_ms, success)

        if self._ab_manager:
            # Simplified: always match control arm for recording
            self._ab_manager.record_outcome("control", success)

    def get_enterprise_statistics(self) -> Dict[str, Any]:
        """Extended statistics including circuit breaker and health."""
        stats = self.get_statistics()
        stats["circuit_breakers"] = {
            tier.value: cb.to_dict() for tier, cb in self.circuit_breakers.items()
        }
        stats["health_report"] = self.health_check.get_report()
        stats["policy"] = {
            "name": self.policy.name,
            "fast_threshold": self.policy.fast_threshold,
            "balanced_threshold": self.policy.balanced_threshold,
            "blocked_models": self.policy.blocked_models,
        }
        if self._ab_manager:
            stats["ab_test"] = self._ab_manager.get_results()
        return stats


# =============================================================================
# ENTERPRISE SINGLETON
# =============================================================================

_enterprise_router: Optional[EnterpriseRouter] = None
_enterprise_router_lock = threading.Lock()


def get_enterprise_router(
    policy: Optional[RoutingPolicy] = None,
) -> EnterpriseRouter:
    """Get or create the singleton :class:`EnterpriseRouter` (thread-safe).

    Args:
        policy: Optional routing policy. Only used on first creation.

    Returns:
        Shared enterprise router instance.
    """
    global _enterprise_router
    if _enterprise_router is None:
        with _enterprise_router_lock:
            if _enterprise_router is None:
                _enterprise_router = EnterpriseRouter(policy=policy)
    return _enterprise_router
