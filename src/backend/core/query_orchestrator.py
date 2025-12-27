"""
Query Orchestrator - Unified Decision Maker
Integrates three innovation tracks into one coherent system:
1. Complexity-based model selection (RAM-aware)
2. Execution method selection (code gen vs direct LLM)
3. Two Friends Model activation (when to apply review)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/APIs"""
        return {
            'model': self.model,
            'execution_method': self.execution_method.value,
            'review_level': self.review_level.value,
            'complexity_score': self.complexity_score,
            'reasoning': self.reasoning
        }


class QueryOrchestrator:
    """
    Master decision maker that integrates all three innovation tracks:
    
    Track 1: Complexity → Model Selection (RAM-aware)
        - Simple queries (< 0.3) → tinyllama (637 MB)
        - Medium queries (0.3-0.7) → phi3:mini (2.2 GB)
        - Complex queries (> 0.7) → llama3.1:8b (4.9 GB)
    
    Track 2: Query Type → Execution Method
        - Computational queries → Code Generation (accurate)
        - Conversational queries → Direct LLM (natural language)
    
    Track 3: Complexity + Method → Review Decision
        - Simple queries → Skip review (fast path)
        - Medium queries → Optional review
        - Complex queries → Mandatory review
        - Code generation → Mandatory review (validate code)
    """
    
    def __init__(self, 
                 complexity_analyzer,
                 config: Dict[str, Any]):
        """
        Initialize orchestrator
        
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
        
        # Model names from config
        self.model_simple = model_config.get('simple', 'tinyllama')
        self.model_medium = model_config.get('medium', 'phi3:mini')
        self.model_complex = model_config.get('complex', 'llama3.1:8b')
        
        # Review activation rules
        review_config = config.get('cot_review', {}).get('activation_rules', {})
        self.review_always_on_complexity = review_config.get('always_on_complexity', 0.7)
        self.review_optional_range = review_config.get('optional_range', [0.3, 0.7])
        self.review_always_on_code_gen = review_config.get('always_on_code_gen', True)
        
        # Code generation keywords
        self.code_gen_keywords = [
            'calculate', 'compute', 'sum', 'average', 'mean', 'total',
            'count', 'group by', 'aggregate', 'filter', 'sort', 'rank',
            'correlation', 'percentage', 'ratio', 'compare', 'maximum',
            'minimum', 'median', 'standard deviation', 'variance'
        ]
        
        logger.info("QueryOrchestrator initialized with unified decision logic")
    
    def create_execution_plan(self, 
                             query: str, 
                             data: Any = None,
                             context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """
        Create unified execution plan combining all three tracks
        
        Args:
            query: User's natural language query
            data: Optional data context (DataFrame, etc.)
            context: Additional context (agent capabilities, etc.)
        
        Returns:
            ExecutionPlan with model, method, and review decisions
        """
        # TRACK 1: Analyze complexity → determines model AND review baseline
        complexity = self._analyze_complexity(query, data, context)
        model = self._select_model(complexity)
        
        # TRACK 2: Detect query type → determines execution method
        execution_method = self._select_execution_method(query, complexity, data)
        
        # TRACK 3: Combine complexity + method → review decision
        review_level = self._select_review_level(complexity, execution_method)
        
        # Generate human-readable explanation
        reasoning = self._explain_plan(query, model, execution_method, review_level, complexity)
        
        plan = ExecutionPlan(
            model=model,
            execution_method=execution_method,
            review_level=review_level,
            complexity_score=complexity,
            reasoning=reasoning
        )
        
        logger.info(f"Execution Plan: {model} | {execution_method.value} | {review_level.value} (complexity={complexity:.2f})")
        
        return plan
    
    def _analyze_complexity(self, 
                           query: str, 
                           data: Any,
                           context: Optional[Dict[str, Any]]) -> float:
        """Analyze query complexity (Track 1 foundation)"""
        if self.complexity_analyzer is None:
            # Fallback: simple heuristic based on query length and keywords
            return self._heuristic_complexity(query, data)
        
        try:
            result = self.complexity_analyzer.analyze(query, data, context)
            return result.get('complexity_score', 0.5)
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
        
        # Check if query needs computation
        needs_computation = any(kw in query_lower for kw in self.code_gen_keywords)
        
        # Must have data for code generation
        has_data = data is not None
        
        # Code generation for computational tasks with data and sufficient complexity
        if needs_computation and has_data and complexity >= 0.3:
            return ExecutionMethod.CODE_GENERATION
        
        # Default to direct LLM (natural language)
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
        Adaptive model selection: If current model fails, suggest fallback
        
        Args:
            plan: Current execution plan
            error: Error message from failed execution
        
        Returns:
            Suggested fallback model name, or None if no fallback
        """
        error_lower = error.lower()
        
        # OOM or timeout → try smaller model
        if 'memory' in error_lower or 'timeout' in error_lower or 'resource' in error_lower:
            if plan.model == self.model_complex:
                logger.info(f"Falling back from {plan.model} to {self.model_medium} due to: {error}")
                return self.model_medium
            elif plan.model == self.model_medium:
                logger.info(f"Falling back from {plan.model} to {self.model_simple} due to: {error}")
                return self.model_simple
        
        # Model not found → try default
        if 'not found' in error_lower or 'unavailable' in error_lower:
            logger.info(f"Model {plan.model} unavailable, falling back to {self.model_medium}")
            return self.model_medium
        
        return None


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
