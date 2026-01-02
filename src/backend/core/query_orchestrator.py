"""
Query Orchestrator - Unified Decision Maker (Enhanced Phase 1)
==============================================================
Integrates three innovation tracks into one coherent system:
1. Complexity-based model selection (RAM-aware, Dynamic Discovery)
2. Execution method selection (code gen vs direct LLM, with fallbacks)
3. Two Friends Model activation (when to apply review)

Phase 1 Enhancements:
- Smart Fallback integration (process never stops)
- Dynamic Model Discovery (no hardcoding)
- RAM-Aware Selection (actual memory checking)
- Circuit Breaker protection (resilient LLM calls)
- Domain and Data Agnostic design
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Import Phase 1 components
try:
    from .smart_fallback import get_fallback_manager, FallbackReason, GracefulDegradation
    from .model_selector import get_model_discovery, ModelInfo, get_ram_selector, MemoryPressureLevel
    from .circuit_breaker import get_circuit_breaker, CircuitState
    from .user_preferences import get_preferences_manager
    PHASE1_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Phase 1 components not all available: {e}")
    PHASE1_COMPONENTS_AVAILABLE = False


class ExecutionMethod(Enum):
    """How to execute the query"""
    DIRECT_LLM = "direct_llm"           # LLM analyzes directly
    CODE_GENERATION = "code_generation"  # Generate Python code for execution


class ReviewLevel(Enum):
    """When to apply Two Friends Model (Generator + Critic)"""
    NONE = "none"          # Skip review (fast path for simple queries)
    OPTIONAL = "optional"   # Apply if enabled in config
    MANDATORY = "mandatory" # Always apply (complex queries, code generation)


@dataclass
class ExecutionPlan:
    """
    Unified execution plan combining all three innovation tracks
    """
    model: str                          # Track 1: Which model to use
    execution_method: ExecutionMethod   # Track 2: Code gen or direct
    review_level: ReviewLevel           # Track 3: Two Friends activation
    complexity_score: float             # Computed complexity
    reasoning: str                      # Human-readable explanation
    fallback_chain: List[str] = field(default_factory=list)  # Phase 1: Available fallbacks
    memory_available_gb: float = 0.0    # Phase 1: RAM at plan time
    adaptive_timeout: int = 60          # Phase 1: Computed timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/APIs"""
        return {
            'model': self.model,
            'execution_method': self.execution_method.value,
            'review_level': self.review_level.value,
            'complexity_score': self.complexity_score,
            'reasoning': self.reasoning,
            'fallback_chain': self.fallback_chain,
            'memory_available_gb': round(self.memory_available_gb, 2),
            'adaptive_timeout': self.adaptive_timeout
        }


class QueryOrchestrator:
    """
    Master decision maker that integrates all three innovation tracks:
    
    Track 1: Complexity → Model Selection (RAM-aware, Dynamic Discovery)
        - Simple queries (< 0.3) → smallest viable model
        - Medium queries (0.3-0.7) → medium capability model
        - Complex queries (> 0.7) → most capable model available
    
    Track 2: Query Type → Execution Method
        - Computational queries → Code Generation (accurate)
        - Conversational queries → Direct LLM (natural language)
    
    Track 3: Complexity + Method → Review Decision
        - Simple queries → Skip review (fast path)
        - Medium queries → Optional review
        - Complex queries → Mandatory review
        - Code generation → Mandatory review (validate code)
    
    Phase 1 Enhancements:
        - Dynamic model discovery (no hardcoded model names)
        - RAM-aware selection (actual memory checking)
        - Smart fallback chains (process never stops)
        - Circuit breaker protection (resilient)
    """
    
    def __init__(self, 
                 complexity_analyzer,
                 config: Dict[str, Any]):
        """
        Initialize orchestrator with Phase 1 enhancements
        
        Args:
            complexity_analyzer: QueryComplexityAnalyzer instance
            config: Configuration dictionary with model_selection and cot_review settings
        """
        self.complexity_analyzer = complexity_analyzer
        self.config = config
        
        # Load thresholds from config
        model_config = config.get('model_selection', {})
        self.simple_threshold = model_config.get('thresholds', {}).get('simple_max', 0.3)
        self.medium_threshold = model_config.get('thresholds', {}).get('medium_max', 0.7)
        
        # Model names from config (used as fallback if discovery fails)
        self.model_simple = model_config.get('simple', 'tinyllama')
        self.model_medium = model_config.get('medium', 'phi3:mini')
        self.model_complex = model_config.get('complex', 'llama3.1:8b')
        
        # Review activation rules
        review_config = config.get('cot_review', {}).get('activation_rules', {})
        self.review_always_on_complexity = review_config.get('always_on_complexity', 0.7)
        self.review_optional_range = review_config.get('optional_range', [0.3, 0.7])
        self.review_always_on_code_gen = review_config.get('always_on_code_gen', True)
        
        # Code generation keywords (domain-agnostic patterns)
        self.code_gen_keywords = [
            'calculate', 'compute', 'sum', 'average', 'mean', 'total',
            'count', 'group by', 'aggregate', 'filter', 'sort', 'rank',
            'correlation', 'percentage', 'ratio', 'compare', 'maximum',
            'minimum', 'median', 'standard deviation', 'variance',
            'top', 'bottom', 'highest', 'lowest', 'by region', 'by category',
            'per', 'each', 'distribution', 'breakdown'
        ]
        
        # Phase 1: Initialize enhanced components
        self._init_phase1_components()
        
        logger.info("QueryOrchestrator initialized with Phase 1 enhancements")
    
    def _init_phase1_components(self):
        """Initialize Phase 1 smart fallback components"""
        self._discovered_models: List[Tuple[str, float]] = []  # (name, ram_estimate)
        self._last_discovery_time: float = 0
        self._discovery_cache_ttl: float = 300  # 5 minutes
        
        if PHASE1_COMPONENTS_AVAILABLE:
            try:
                # Discover available models dynamically
                self._refresh_model_discovery()
                logger.info(f"Phase 1: Discovered {len(self._discovered_models)} models")
            except Exception as e:
                logger.warning(f"Model discovery failed, using config defaults: {e}")
    
    def _refresh_model_discovery(self):
        """Refresh the list of available models from Ollama"""
        if not PHASE1_COMPONENTS_AVAILABLE:
            return
        
        current_time = time.time()
        if current_time - self._last_discovery_time < self._discovery_cache_ttl:
            return  # Use cache
        
        try:
            discovery = get_model_discovery()
            models = discovery.discover_models_sync()
            
            # CRITICAL: Filter out embedding models from discovered models
            llm_models = [m for m in models if not self._is_embedding_model(m.name)]
            
            # Build model list with RAM estimates (only LLM models)
            self._discovered_models = [
                (m.name, m.estimated_ram_gb) for m in llm_models
            ]
            self._last_discovery_time = current_time
            
            # Update model names based on what's available (also filters embedding models)
            self._update_model_assignments(models)
            
        except Exception as e:
            logger.warning(f"Model discovery refresh failed: {e}")
    
    def _is_embedding_model(self, model_name: str) -> bool:
        """Check if a model is an embedding model (not for text generation)"""
        embedding_patterns = [
            'embed', 'embedding', 'nomic', 'mxbai', 'all-minilm', 
            'sentence-transformers', 'bge-', 'gte-', 'e5-'
        ]
        name_lower = model_name.lower()
        return any(pattern in name_lower for pattern in embedding_patterns)
    
    def _update_model_assignments(self, models: List):
        """Update model assignments based on discovered models"""
        if not models:
            return
        
        # CRITICAL: Filter out embedding models - they can't generate text
        llm_models = [m for m in models if not self._is_embedding_model(m.name)]
        
        if not llm_models:
            logger.warning("No LLM models found (only embedding models), using config defaults")
            return
        
        # Sort by complexity score (capability)
        sorted_models = sorted(llm_models, key=lambda m: m.complexity_score)
        
        # Assign to simple/medium/complex based on capability tiers
        if len(sorted_models) >= 3:
            self.model_simple = sorted_models[0].name
            self.model_medium = sorted_models[len(sorted_models)//2].name
            self.model_complex = sorted_models[-1].name
        elif len(sorted_models) == 2:
            self.model_simple = sorted_models[0].name
            self.model_medium = sorted_models[0].name
            self.model_complex = sorted_models[1].name
        elif len(sorted_models) == 1:
            # Only one model available - use for everything
            self.model_simple = sorted_models[0].name
            self.model_medium = sorted_models[0].name
            self.model_complex = sorted_models[0].name
        
        logger.info(f"Model assignments updated: simple={self.model_simple}, "
                   f"medium={self.model_medium}, complex={self.model_complex}")
    
    def _get_user_model_preference(self) -> Tuple[str, bool]:
        """
        Get user's model preference and routing settings.
        
        CRITICAL: This is the single source of truth for model selection.
        When intelligent routing is OFF, the user's selected model MUST be used.
        
        Returns:
            Tuple of (user_selected_model, enable_intelligent_routing)
        """
        try:
            if PHASE1_COMPONENTS_AVAILABLE:
                prefs = get_preferences_manager().load_preferences()
                # User's primary model is what they selected
                user_model = prefs.primary_model
                use_routing = prefs.enable_intelligent_routing
                
                logger.debug(f"User preferences: model={user_model}, intelligent_routing={use_routing}")
                return user_model, use_routing
            else:
                # Fallback to config defaults
                return self.model_medium, True
        except Exception as e:
            logger.warning(f"Failed to load user preferences: {e}")
            return self.model_medium, True
    
    def create_execution_plan(self, 
                             query: str, 
                             data: Any = None,
                             context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """
        Create unified execution plan combining all three tracks.
        
        Phase 1 Enhanced: Includes fallback chains and RAM awareness.
        
        CRITICAL: Respects user preferences!
        - If enable_intelligent_routing is OFF → Use user's selected model ONLY
        - If enable_intelligent_routing is ON → Use dynamic model selection
        
        Args:
            query: User's natural language query
            data: Optional data context (DataFrame, etc.)
            context: Additional context (agent capabilities, etc.)
        
        Returns:
            ExecutionPlan with model, method, review, and fallback decisions
        """
        # CRITICAL: Check user preferences FIRST
        user_model, use_intelligent_routing = self._get_user_model_preference()
        
        # TRACK 1: Analyze complexity → determines model AND review baseline
        complexity = self._analyze_complexity(query, data, context)
        
        if use_intelligent_routing:
            # Intelligent routing enabled - use dynamic model selection
            self._refresh_model_discovery()
            model, fallback_chain, memory_available = self._select_model_with_fallback(complexity)
        else:
            # User disabled intelligent routing - RESPECT their model choice
            model = user_model
            fallback_chain = []  # No fallbacks - user wants this specific model
            memory_available = 0.0
            logger.info(f"Using user-selected model (routing disabled): {model}")
        
        # TRACK 2: Detect query type → determines execution method
        execution_method = self._select_execution_method(query, complexity, data)
        
        # TRACK 3: Combine complexity + method → review decision
        review_level = self._select_review_level(complexity, execution_method)
        
        # PHASE 1: Calculate adaptive timeout based on model and memory
        adaptive_timeout = self._calculate_adaptive_timeout(model, complexity)
        
        # Generate human-readable explanation
        reasoning = self._explain_plan(query, model, execution_method, review_level, complexity)
        
        plan = ExecutionPlan(
            model=model,
            execution_method=execution_method,
            review_level=review_level,
            complexity_score=complexity,
            reasoning=reasoning,
            fallback_chain=fallback_chain,
            memory_available_gb=memory_available,
            adaptive_timeout=adaptive_timeout
        )
        
        logger.info(f"Execution Plan: {model} | {execution_method.value} | {review_level.value} "
                   f"(complexity={complexity:.2f}, fallbacks={len(fallback_chain)})")
        
        return plan
    
    def _select_model_with_fallback(self, complexity: float) -> Tuple[str, List[str], float]:
        """
        Phase 1 Enhanced: Select model with RAM awareness and fallback chain.
        
        Returns:
            Tuple of (selected_model, fallback_chain, available_ram_gb)
        """
        if not PHASE1_COMPONENTS_AVAILABLE:
            # Fallback to original logic
            model = self._select_model(complexity)
            return model, [], 0.0
        
        try:
            ram_selector = get_ram_selector()
            memory_available = ram_selector.get_available_ram_for_model()
            
            # Get model options sorted by capability
            if self._discovered_models:
                model_options = self._discovered_models
            else:
                # Use config defaults
                model_options = [
                    (self.model_simple, 1.0),
                    (self.model_medium, 3.0),
                    (self.model_complex, 6.0)
                ]
            
            # Determine preferred model based on complexity
            preferred = self._select_model(complexity)
            
            # Use RAM-aware selection
            result = ram_selector.select_model(
                preferred_model=preferred,
                model_options=model_options,
                complexity=complexity
            )
            
            # Build fallback chain (models smaller than selected)
            fallback_chain = []
            selected_found = False
            for name, _ in sorted(model_options, key=lambda x: x[1], reverse=True):
                if name == result.selected_model:
                    selected_found = True
                    continue
                if selected_found:
                    fallback_chain.append(name)
            
            if result.fallback_triggered:
                logger.warning(f"RAM constraint: {preferred} → {result.selected_model}")
            
            return result.selected_model, fallback_chain, memory_available
            
        except Exception as e:
            logger.warning(f"RAM-aware selection failed: {e}")
            model = self._select_model(complexity)
            return model, [], 0.0
    
    def _calculate_adaptive_timeout(self, model: str, complexity: float) -> int:
        """
        Phase 1: Calculate adaptive timeout based on model, complexity, and system state.
        """
        base_timeout = 60  # Default base
        
        # Increase timeout for complex queries
        if complexity > 0.7:
            base_timeout = 120
        elif complexity > 0.3:
            base_timeout = 90
        
        if PHASE1_COMPONENTS_AVAILABLE:
            try:
                fallback_manager = get_fallback_manager()
                return fallback_manager.get_adaptive_timeout(base_timeout, model)
            except Exception as e:
                logger.warning(f"Adaptive timeout calculation failed: {e}")
        
        return base_timeout
    
    def _analyze_complexity(self, 
                           query: str, 
                           data: Any,
                           context: Optional[Dict[str, Any]]) -> float:
        """Analyze query complexity (Track 1 foundation)"""
        if self.complexity_analyzer is None:
            # Fallback: simple heuristic based on query length and keywords
            return self._heuristic_complexity(query, data)
        
        try:
            # Build data_info dict for the analyzer (accepts query + optional data_info)
            data_info = None
            if data is not None or context is not None:
                data_info = {
                    "data": data,
                    "context": context
                }
            
            result = self.complexity_analyzer.analyze(query, data_info)
            # Result is ComplexityScore dataclass with total_score attribute
            return result.total_score if hasattr(result, 'total_score') else 0.5
        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}, using heuristic")
            return self._heuristic_complexity(query, data)
    
    def _heuristic_complexity(self, query: str, data: Any) -> float:
        """
        Simple complexity heuristic when analyzer unavailable.
        
        Scoring Guide:
        - < 0.3: Simple (definitions, short questions) → tinyllama
        - 0.3-0.7: Medium (calculations, filters) → phi3:mini
        - > 0.7: Complex (multi-step analysis, correlations) → llama3.1:8b
        """
        query_lower = query.lower()
        query_len = len(query)
        
        # Very short queries are simple regardless of data
        if query_len < 50:
            # Check for simple question patterns
            simple_patterns = ['what is', 'how many', 'show me', 'list', 'get the']
            if any(p in query_lower for p in simple_patterns):
                return 0.15  # Definitely simple
        
        complexity = 0.1  # Start low
        
        # Length indicates complexity
        if query_len > 200:
            complexity += 0.4
        elif query_len > 120:
            complexity += 0.3
        elif query_len > 80:
            complexity += 0.2
        elif query_len > 50:
            complexity += 0.1
        
        # Multi-step indicators (high complexity)
        multi_step_keywords = ['and then', 'after that', 'then predict', 'then identify',
                               'then forecast', 'then recommend', 'correlation', 'regression', 
                               'forecast', 'predict', 'over time', 'trend', 'segment by',
                               'risk assessment', 'recommend intervention', 'ABC analysis']
        multi_step_count = sum(1 for word in multi_step_keywords if word in query_lower)
        complexity += min(multi_step_count * 0.2, 0.5)
        
        # Multiple conditions (medium boost)
        condition_keywords = ['where', 'if', 'when', 'filter by', 'group by', 'for each']
        condition_count = sum(1 for word in condition_keywords if word in query_lower)
        complexity += min(condition_count * 0.1, 0.2)
        
        # Aggregations (medium complexity)
        agg_count = sum(1 for word in self.code_gen_keywords if word in query_lower)
        complexity += min(agg_count * 0.05, 0.15)
        
        return min(complexity, 1.0)
    
    def _select_model(self, complexity: float) -> str:
        """
        Track 1: Complexity → Model Selection (RAM-aware)
        
        Returns model name based on complexity score
        """
        if complexity < self.simple_threshold:
            return self.model_simple  # tinyllama (637 MB)
        elif complexity < self.medium_threshold:
            return self.model_medium  # phi3:mini (2.2 GB)
        else:
            return self.model_complex  # llama3.1:8b (4.9 GB)
    
    def _select_execution_method(self, 
                                 query: str, 
                                 complexity: float,
                                 data: Any) -> ExecutionMethod:
        """
        Track 2: Query Type → Execution Method
        
        Determines whether to use code generation or direct LLM analysis
        """
        query_lower = query.lower()
        
        # Check if query needs computation (expanded keywords)
        needs_computation = any(kw in query_lower for kw in self.code_gen_keywords)
        
        # CRITICAL: Queries asking about specific values in data ALWAYS need code execution
        # The LLM cannot know what's in the data without actually running code
        data_value_keywords = [
            'what is', 'which is', 'find the', 'show me', 'get the',
            'most', 'least', 'best', 'worst', 'popular', 'listened',
            'name of', 'value of', 'how many', 'how much', 'list all',
            'who is', 'where is', 'when is'
        ]
        asks_about_data_values = any(kw in query_lower for kw in data_value_keywords)
        
        # Must have data for code generation
        has_data = data is not None
        
        # CHANGED: Use code generation for ANY data-related query, not just complex ones
        # Because the LLM can't answer questions about data content without executing code
        if has_data and (needs_computation or asks_about_data_values):
            return ExecutionMethod.CODE_GENERATION
        
        # Default to direct LLM (natural language) - only for conversational/conceptual queries
        return ExecutionMethod.DIRECT_LLM
    
    def _select_review_level(self, 
                            complexity: float,
                            method: ExecutionMethod) -> ReviewLevel:
        """
        Track 3: Complexity + Method → Two Friends Model Activation
        
        Determines when to apply Generator-Critic review loop
        """
        # RULE 1: Mandatory for complex queries (high stakes)
        if complexity >= self.review_always_on_complexity:
            return ReviewLevel.MANDATORY
        
        # RULE 2: Mandatory for code generation (validate generated code)
        if method == ExecutionMethod.CODE_GENERATION and self.review_always_on_code_gen:
            return ReviewLevel.MANDATORY
        
        # RULE 3: Optional for medium complexity
        min_optional, max_optional = self.review_optional_range
        if min_optional <= complexity < max_optional:
            return ReviewLevel.OPTIONAL
        
        # RULE 4: Skip for simple queries (fast path)
        return ReviewLevel.NONE
    
    def _explain_plan(self,
                     query: str,
                     model: str,
                     method: ExecutionMethod,
                     review: ReviewLevel,
                     complexity: float) -> str:
        """Generate human-readable explanation of the execution plan"""
        
        # Model selection reason
        if complexity < 0.3:
            model_reason = f"lightweight model (low complexity)"
        elif complexity < 0.7:
            model_reason = f"balanced model (medium complexity)"
        else:
            model_reason = f"powerful model (high complexity)"
        
        # Execution method reason
        if method == ExecutionMethod.CODE_GENERATION:
            method_reason = "code generation (accurate computation)"
        else:
            method_reason = "direct LLM (natural language analysis)"
        
        # Review decision reason
        if review == ReviewLevel.MANDATORY:
            if complexity >= 0.7:
                review_reason = "Two Friends review (complex query)"
            else:
                review_reason = "Two Friends review (validate code)"
        elif review == ReviewLevel.OPTIONAL:
            review_reason = "optional review (if enabled)"
        else:
            review_reason = "skip review (fast path)"
        
        return f"""Complexity: {complexity:.2f}
Model: {model} ({model_reason})
Method: {method.value} ({method_reason})
Review: {review.value} ({review_reason})"""
    
    def should_retry_with_different_model(self, 
                                         plan: ExecutionPlan,
                                         error: str) -> Optional[str]:
        """
        Phase 1 Enhanced: Intelligent fallback with error categorization.
        
        Args:
            plan: Current execution plan
            error: Error message from failed execution
        
        Returns:
            Suggested fallback model name, or None if no fallback
        """
        error_lower = error.lower()
        
        # Phase 1: Use smart fallback manager if available
        if PHASE1_COMPONENTS_AVAILABLE:
            try:
                fallback_manager = get_fallback_manager()
                
                # Categorize error
                if 'memory' in error_lower or 'oom' in error_lower:
                    reason = FallbackReason.MEMORY_LIMIT
                elif 'timeout' in error_lower or 'timed out' in error_lower:
                    reason = FallbackReason.TIMEOUT
                elif 'not found' in error_lower or 'unavailable' in error_lower:
                    reason = FallbackReason.MODEL_UNAVAILABLE
                elif 'resource' in error_lower:
                    reason = FallbackReason.RESOURCE_EXHAUSTED
                else:
                    reason = FallbackReason.EXECUTION_ERROR
                
                fallback = fallback_manager.get_model_fallback(
                    plan.model, reason, error
                )
                
                if fallback and fallback != plan.model:
                    logger.info(f"Smart fallback: {plan.model} → {fallback} (reason: {reason.value})")
                    return fallback
                    
            except Exception as e:
                logger.warning(f"Smart fallback failed: {e}")
        
        # Fallback to original logic
        if 'memory' in error_lower or 'timeout' in error_lower or 'resource' in error_lower:
            if plan.model == self.model_complex:
                logger.info(f"Falling back from {plan.model} to {self.model_medium} due to: {error}")
                return self.model_medium
            elif plan.model == self.model_medium:
                logger.info(f"Falling back from {plan.model} to {self.model_simple} due to: {error}")
                return self.model_simple
        
        if 'not found' in error_lower or 'unavailable' in error_lower:
            logger.info(f"Model {plan.model} unavailable, falling back to {self.model_medium}")
            return self.model_medium
        
        return None
    
    def execute_with_fallback(self, 
                             plan: ExecutionPlan,
                             execute_func,
                             max_retries: int = 3) -> Dict[str, Any]:
        """
        Phase 1: Execute with automatic fallback through the chain.
        
        Ensures the process NEVER stops completely - always provides
        a response even if degraded.
        
        Args:
            plan: Execution plan with fallback chain
            execute_func: Function to execute (receives model name)
            max_retries: Maximum retry attempts
        
        Returns:
            Result dict, always includes 'success' key
        """
        current_model = plan.model
        fallback_chain = list(plan.fallback_chain)
        attempts = []
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Attempt {attempt + 1}: Executing with model {current_model}")
                result = execute_func(current_model)
                
                # Record success
                if PHASE1_COMPONENTS_AVAILABLE:
                    get_fallback_manager().mark_recovered()
                
                result['_execution_metadata'] = {
                    'model_used': current_model,
                    'attempt': attempt + 1,
                    'fallback_activated': attempt > 0
                }
                return result
                
            except Exception as e:
                error_str = str(e)
                attempts.append({
                    'model': current_model,
                    'error': error_str[:200],
                    'attempt': attempt + 1
                })
                
                logger.warning(f"Attempt {attempt + 1} failed with {current_model}: {error_str}")
                
                # Try next model in chain
                if fallback_chain:
                    current_model = fallback_chain.pop(0)
                    logger.info(f"Falling back to: {current_model}")
                elif attempt < max_retries:
                    # No more fallbacks but can still retry
                    continue
                else:
                    # All options exhausted - provide graceful degradation
                    logger.error(f"All fallback options exhausted after {attempt + 1} attempts")
                    
                    if PHASE1_COMPONENTS_AVAILABLE:
                        get_fallback_manager().mark_exhausted()
                        return GracefulDegradation.generate_degraded_response(
                            query=plan.reasoning,
                            context={'attempts': attempts},
                            error=error_str
                        )
                    else:
                        return {
                            'success': False,
                            'error': f'All models failed after {attempt + 1} attempts',
                            'attempts': attempts,
                            'message': 'The system could not complete your request. Please try again later.'
                        }
        
        # Should not reach here, but just in case
        return {'success': False, 'error': 'Unexpected execution path'}
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """
        Phase 1: Get comprehensive orchestrator status for monitoring.
        """
        status = {
            'models': {
                'simple': self.model_simple,
                'medium': self.model_medium,
                'complex': self.model_complex
            },
            'thresholds': {
                'simple_max': self.simple_threshold,
                'medium_max': self.medium_threshold
            },
            'discovered_models': len(self._discovered_models),
            'phase1_available': PHASE1_COMPONENTS_AVAILABLE
        }
        
        if PHASE1_COMPONENTS_AVAILABLE:
            try:
                status['fallback_stats'] = get_fallback_manager().get_stats()
                status['ram_status'] = get_ram_selector().get_statistics()
                status['model_discovery'] = get_model_discovery().get_statistics()
            except Exception as e:
                status['phase1_error'] = str(e)
        
        return status


# Example usage:
if __name__ == "__main__":
    # Mock config
    config = {
        'model_selection': {
            'simple': 'tinyllama',
            'medium': 'phi3:mini',
            'complex': 'llama3.1:8b',
            'thresholds': {
                'simple_max': 0.3,
                'medium_max': 0.7
            }
        },
        'cot_review': {
            'activation_rules': {
                'always_on_complexity': 0.7,
                'optional_range': [0.3, 0.7],
                'always_on_code_gen': True
            }
        }
    }
    
    orchestrator = QueryOrchestrator(None, config)
    
    # Test queries
    test_queries = [
        ("What is a customer?", None, "Simple definition"),
        ("Calculate average sales by region", "mock_dataframe", "Medium computation"),
        ("Analyze correlation between price and demand, then predict next quarter", "mock_dataframe", "Complex analysis")
    ]
    
    print("="*70)
    print("QUERY ORCHESTRATOR TEST")
    print("="*70)
    
    for query, data, description in test_queries:
        print(f"\n{description}")
        print(f"Query: {query}")
        plan = orchestrator.create_execution_plan(query, data)
        print(plan.reasoning)
        print(f"→ Plan: {plan.model} | {plan.execution_method.value} | {plan.review_level.value}")
