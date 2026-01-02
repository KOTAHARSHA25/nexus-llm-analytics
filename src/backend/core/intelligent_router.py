"""
Intelligent Router Module
=========================
Routes queries to appropriate model tiers based on complexity analysis.

Uses QueryComplexityAnalyzer to score queries and select optimal model tier
for efficiency (fast models for simple queries, powerful for complex).

Design Principles:
- Uses existing complexity analyzer (no duplication)
- RAM-aware (respects system memory constraints)
- Tracks statistics for monitoring
- Supports user overrides
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from collections import deque

from .query_complexity_analyzer import QueryComplexityAnalyzer, ComplexityScore
from .model_selector import ModelSelector

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model capability tiers"""
    FAST = "fast"           # Quick responses, simple queries (phi3:mini, gemma:2b)
    BALANCED = "balanced"   # Good balance of speed/quality (llama3.1:8b)
    FULL_POWER = "full_power"  # Maximum capability (larger models)


@dataclass
class RoutingDecision:
    """Result of a routing decision"""
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
    
    def __init__(self):
        """Initialize the router"""
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
    
    def _refresh_model_tiers(self):
        """Refresh the model tier assignments based on installed models"""
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
            
            logger.info(f"📊 Model tiers refreshed: "
                       f"FAST={len(self.fallback_chain[ModelTier.FAST])}, "
                       f"BALANCED={len(self.fallback_chain[ModelTier.BALANCED])}, "
                       f"FULL_POWER={len(self.fallback_chain[ModelTier.FULL_POWER])}")
            
        except Exception as e:
            logger.error(f"Error refreshing model tiers: {e}")
    
    def _select_model_for_tier(self, tier: ModelTier) -> Optional[str]:
        """Select best available model for a given tier"""
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
    
    def _record_decision(self, decision: RoutingDecision, start_time: float):
        """Record a routing decision for statistics"""
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
            f"🎯 Routed query to {decision.selected_tier.value}/{decision.selected_model} "
            f"(score={decision.complexity_score:.3f}, time={routing_time_ms:.2f}ms)"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
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
    
    def reset_statistics(self):
        """Reset all statistics"""
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


def get_intelligent_router() -> IntelligentRouter:
    """Get or create the singleton IntelligentRouter instance"""
    global _intelligent_router
    if _intelligent_router is None:
        _intelligent_router = IntelligentRouter()
    return _intelligent_router
