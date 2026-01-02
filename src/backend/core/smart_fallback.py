"""
Smart Fallback Manager
======================
Provides intelligent fallback mechanisms to ensure processing never stops.

Design Principles:
- Domain Agnostic: No hardcoded domain-specific logic
- Data Agnostic: Works with any data structure
- Self-Healing: Automatic recovery from failures
- Observable: Comprehensive logging and metrics

Fallback Chains:
1. Model Fallback: complex → medium → simple → minimal
2. Method Fallback: code_gen → direct_llm → cached → template
3. Review Fallback: mandatory → optional → skip
4. Timeout Fallback: extend → retry → downgrade → fail gracefully
"""

import logging
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, TypeVar
from enum import Enum
from functools import wraps
import traceback

logger = logging.getLogger(__name__)

T = TypeVar('T')


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
    """Record of a fallback event"""
    timestamp: float
    original_strategy: str
    fallback_strategy: str
    reason: FallbackReason
    error_message: Optional[str] = None
    recovered: bool = False
    
    def to_dict(self) -> dict:
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
    """Ordered chain of fallback options"""
    name: str
    strategies: List[str]
    current_index: int = 0
    events: List[FallbackEvent] = field(default_factory=list)
    
    def current(self) -> str:
        """Get current strategy"""
        if self.current_index < len(self.strategies):
            return self.strategies[self.current_index]
        return self.strategies[-1]  # Return last as final fallback
    
    def next(self, reason: FallbackReason, error: str = None) -> Optional[str]:
        """Move to next fallback strategy"""
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
            
            logger.warning(f"Fallback [{self.name}]: {original} → {new_strategy} (reason: {reason.value})")
            return new_strategy
        
        logger.error(f"Fallback chain [{self.name}] exhausted, no more options")
        return None
    
    def reset(self):
        """Reset to first strategy"""
        self.current_index = 0
    
    def has_fallback(self) -> bool:
        """Check if more fallbacks available"""
        return self.current_index < len(self.strategies) - 1


class SmartFallbackManager:
    """
    Centralized fallback management for the entire system.
    
    Ensures that no operation fails completely - always provides
    a degraded but functional response.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        """Fetch installed model names from Ollama dynamically"""
        try:
            import requests
            import os
            
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                
                logger.debug(f"Discovered models for fallback: {model_names}")
                return model_names
            
            return []
        except Exception as e:
            logger.warning(f"Could not fetch installed models: {e}")
            return []
    
    def _init_fallback_chains(self):
        """Initialize default fallback chains based on installed models"""
        
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
        """Get fallback model when current model fails"""
        
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
        """Get fallback execution method"""
        
        try:
            idx = self.method_chain.strategies.index(current_method)
            self.method_chain.current_index = idx
        except ValueError:
            self.method_chain.current_index = 0
        
        fallback = self.method_chain.next(reason, error)
        self._record_fallback(reason)
        
        return fallback or "direct_llm"
    
    def get_review_fallback(self, current_level: str, reason: FallbackReason, error: str = None) -> str:
        """Get fallback review level"""
        
        try:
            idx = self.review_chain.strategies.index(current_level)
            self.review_chain.current_index = idx
        except ValueError:
            self.review_chain.current_index = 0
        
        fallback = self.review_chain.next(reason, error)
        self._record_fallback(reason)
        
        return fallback or "skip"
    
    def get_adaptive_timeout(self, base_timeout: int, model: str) -> int:
        """
        Get adaptive timeout based on system resources and model.
        No hardcoding - uses dynamic resource detection.
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
    
    def _record_fallback(self, reason: FallbackReason):
        """Record fallback statistics"""
        self.stats["total_fallbacks"] += 1
        reason_key = reason.value
        self.stats["by_reason"][reason_key] = self.stats["by_reason"].get(reason_key, 0) + 1
    
    def mark_recovered(self):
        """Mark that recovery was successful"""
        self.stats["recovered"] += 1
    
    def mark_exhausted(self):
        """Mark that all fallbacks were exhausted"""
        self.stats["exhausted"] += 1
    
    def reset_chains(self):
        """Reset all fallback chains to initial state"""
        self.model_chain.reset()
        self.method_chain.reset()
        self.review_chain.reset()
        self.timeout_chain.reset()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics"""
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
    
    def with_fallback(self, 
                      primary_func: Callable[..., T],
                      fallback_func: Callable[..., T],
                      max_retries: int = 2) -> Callable[..., T]:
        """
        Decorator to wrap a function with automatic fallback.
        
        Usage:
            @manager.with_fallback(primary=primary_fn, fallback=fallback_fn)
            def my_function(...): ...
        """
        @wraps(primary_func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return primary_func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    
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
            logger.info(f"Primary function failed after {max_retries + 1} attempts, using fallback")
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
        """
        Generate a meaningful response even when processing fails.
        Domain and data agnostic.
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
        """
        Provide minimal analysis based purely on data structure.
        No LLM required - pure pattern matching.
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


def get_fallback_manager() -> SmartFallbackManager:
    """Get or create the singleton fallback manager"""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = SmartFallbackManager()
    return _fallback_manager
