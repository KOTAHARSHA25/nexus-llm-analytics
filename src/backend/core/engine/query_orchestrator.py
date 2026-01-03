"""
Query Orchestrator - The Brain (Streamlined & Fixed)
====================================================
CORE PURPOSE: Make ONE intelligent decision about how to execute a query

Three Tracks Unified:
1. Complexity → Model (simple/medium/complex)
2. Query Type → Method (code_generation vs direct_llm)  
3. Complexity + Method → Review (none/optional/mandatory)

FIXES FROM V1:
- Removed over-engineered Phase1 optional dependencies (causes import errors)
- Simplified model selection (discovery is separate concern)
- User preferences properly integrated (respects intelligent_routing toggle)
- Config unified (single source of truth from cot_review_config.json)
- Heuristic complexity is the default (QueryComplexityAnalyzer optional)
- Fallback logic simplified (fewer moving parts)

DESIGN PHILOSOPHY:
- Core intelligence should work standalone
- Optional enhancements don't break core
- User control over system vs intelligent routing
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ExecutionMethod(Enum):
    """How to execute the query"""
    DIRECT_LLM = "direct_llm"           # LLM analyzes directly
    CODE_GENERATION = "code_generation"  # Generate Python code


class ReviewLevel(Enum):
    """When to apply Two Friends Model (Generator + Critic)"""
    NONE = "none"          # Skip review (fast path)
    OPTIONAL = "optional"   # Apply if enabled in config
    MANDATORY = "mandatory" # Always apply (complex/code)


@dataclass
class ExecutionPlan:
    """Unified execution plan combining all three tracks"""
    model: str                          # Which model to use
    execution_method: ExecutionMethod   # Code gen or direct
    review_level: ReviewLevel           # Two Friends activation
    complexity_score: float             # Computed complexity
    reasoning: str                      # Human-readable explanation
    user_override: bool = False         # True if user disabled intelligent routing
    fallback_models: List[str] = field(default_factory=list)  # Fallback chain
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model,
            'execution_method': self.execution_method.value,
            'review_level': self.review_level.value,
            'complexity_score': round(self.complexity_score, 3),
            'reasoning': self.reasoning,
            'user_override': self.user_override,
            'fallback_models': self.fallback_models
        }


class QueryOrchestrator:
    """
    Master decision maker integrating three innovation tracks.
    
    Streamlined design:
    - Heuristic complexity by default (fast, no dependencies)
    - User preferences respected (intelligent_routing toggle)
    - Config from single source (cot_review_config.json)
    - Optional enhancements don't break core
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize orchestrator with unified config.
        
        Args:
            config_path: Path to cot_review_config.json (auto-detected if None)
        """
        # Load unified config
        self.config = self._load_config(config_path)
        
        # Extract model selection settings
        model_config = self.config.get('model_selection', {})
        self.simple_threshold = model_config.get('thresholds', {}).get('simple_max', 0.3)
        self.medium_threshold = model_config.get('thresholds', {}).get('medium_max', 0.7)
        
        # Model tiers (will be updated by discovery if available)
        self.model_simple = model_config.get('simple', 'tinyllama')
        self.model_medium = model_config.get('medium', 'phi3:mini')
        self.model_complex = model_config.get('complex', 'llama3.1:8b')
        
        # Review activation rules
        review_config = self.config.get('cot_review', {}).get('activation_rules', {})
        self.review_always_on_complexity = review_config.get('always_on_complexity', 0.7)
        self.review_optional_range = review_config.get('optional_range', [0.3, 0.7])
        self.review_always_on_code_gen = review_config.get('always_on_code_gen', True)
        
        # Code generation keywords - load from config (NO hardcoding)
        keyword_config = self.config.get('query_analysis', {})
        self.code_gen_keywords = keyword_config.get('code_generation_keywords', [])
        self.multi_step_keywords = keyword_config.get('multi_step_keywords', [])
        self.condition_keywords = keyword_config.get('condition_keywords', [])
        self.simple_query_patterns = keyword_config.get('simple_query_patterns', [])
        
        if not self.code_gen_keywords:
            logger.warning("No code_generation_keywords in config - code detection may not work")
        
        # Optional: Try to load advanced components
        self._try_load_advanced_components()
        
        logger.info(f"QueryOrchestrator initialized: {self.model_simple}/{self.model_medium}/{self.model_complex}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load unified configuration from cot_review_config.json"""
        if config_path is None:
            # Auto-detect: look in config/ directory
            possible_paths = [
                Path(__file__).parent.parent.parent.parent / 'config' / 'cot_review_config.json',
                Path.cwd() / 'config' / 'cot_review_config.json'
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    logger.info(f"Loaded config from {config_path}")
                    return loaded_config
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                raise RuntimeError(f"Cannot load config from {config_path}: {e}")
        
        # No hardcoded defaults - config MUST exist
        logger.error(f"Config file not found at {config_path}. Cannot initialize without config.")
        raise FileNotFoundError(f"Config file not found. Checked paths: {config_path}")
    
    def _try_load_advanced_components(self):
        """Try to load optional advanced components (Phase 1)"""
        self.user_prefs_manager = None
        self.model_discovery = None
        self.ram_selector = None
        
        try:
            from .user_preferences import get_preferences_manager
            self.user_prefs_manager = get_preferences_manager()
            logger.debug("User preferences manager loaded")
        except ImportError:
            pass
        
        try:
            from .model_selector import get_model_discovery, get_ram_selector
            self.model_discovery = get_model_discovery()
            self.ram_selector = get_ram_selector()
            
            # Update model tiers from discovery
            models = self.model_discovery.discover_models_sync()
            llm_models = [m for m in models if not self._is_embedding_model(m.name)]
            
            if len(llm_models) >= 3:
                sorted_models = sorted(llm_models, key=lambda m: m.estimated_ram_gb)
                self.model_simple = sorted_models[0].name
                self.model_medium = sorted_models[len(sorted_models)//2].name
                self.model_complex = sorted_models[-1].name
                logger.info(f"Models updated from discovery: {self.model_simple}/{self.model_medium}/{self.model_complex}")
        except ImportError:
            pass
    
    def _is_embedding_model(self, model_name: str) -> bool:
        """Check if model is for embeddings (not text generation)"""
        patterns = ['embed', 'nomic', 'mxbai', 'all-minilm', 'bge-', 'gte-', 'e5-']
        return any(p in model_name.lower() for p in patterns)
    
    def create_execution_plan(self, 
                             query: str, 
                             data: Any = None,
                             context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """
        Create unified execution plan combining all three tracks.
        
        CRITICAL: Respects user preferences!
        - enable_intelligent_routing OFF → Use user's selected model
        - enable_intelligent_routing ON → Dynamic model selection
        
        Args:
            query: User's natural language query
            data: Optional data context
            context: Additional context (columns, filepath, etc.)
        
        Returns:
            ExecutionPlan with unified decisions
        """
        # STEP 1: Check user preferences (HIGHEST PRIORITY - CANNOT BE OVERRIDDEN)
        user_prefs = self._get_user_preferences()
        
        # If user has explicitly chosen a model, use it - NO EXCEPTIONS
        if user_prefs['user_explicit_choice']:
            model = user_prefs['primary_model']
            complexity = self._analyze_complexity(query, data, context)
            execution_method = self._select_execution_method(query, data)
            review_level = self._select_review_level(complexity, execution_method)
            reasoning = self._explain_plan(query, model, execution_method, review_level, complexity, True)
            
            logger.info(f"USER CHOICE (absolute priority): {model}")
            return ExecutionPlan(
                model=model,
                execution_method=execution_method,
                review_level=review_level,
                complexity_score=complexity,
                reasoning=reasoning,
                user_override=True,
                fallback_models=[]  # User wants THIS model, no fallbacks
            )
        
        # STEP 2: Analyze complexity (only if user allows intelligent routing)
        complexity = self._analyze_complexity(query, data, context)
        
        # STEP 3: Select model intelligently (only if user allows)
        if user_prefs['enable_intelligent_routing']:
            model = self._select_model_intelligent(complexity)
            user_override = False
            fallback_models = self._build_fallback_chain(model)
        else:
            model = user_prefs['primary_model']
            user_override = True
            fallback_models = []
            logger.debug(f"User disabled intelligent routing, using {model}")
        
        # STEP 4: Select execution method
        execution_method = self._select_execution_method(query, data)
        
        # STEP 5: Select review level
        review_level = self._select_review_level(complexity, execution_method)
        
        # STEP 6: Generate reasoning
        reasoning = self._explain_plan(query, model, execution_method, review_level, complexity, user_override)
        
        plan = ExecutionPlan(
            model=model,
            execution_method=execution_method,
            review_level=review_level,
            complexity_score=complexity,
            reasoning=reasoning,
            user_override=user_override,
            fallback_models=fallback_models
        )
        
        logger.info(f"Plan: {model} | {execution_method.value} | {review_level.value} (complexity={complexity:.2f})")
        return plan
    
    def _get_user_preferences(self) -> Dict[str, Any]:
        """
        Get user preferences with ABSOLUTE priority.
        
        Returns dict with:
        - primary_model: User's chosen model
        - enable_intelligent_routing: Whether to allow system to override
        - user_explicit_choice: Whether user has explicitly locked a model
        """
        if self.user_prefs_manager:
            try:
                prefs = self.user_prefs_manager.load_preferences()
                return {
                    'primary_model': prefs.primary_model,
                    'enable_intelligent_routing': prefs.enable_intelligent_routing,
                    'user_explicit_choice': not prefs.enable_intelligent_routing  # If routing OFF, user chose explicitly
                }
            except Exception as e:
                logger.warning(f"Failed to load user preferences: {e}")
        
        # No user preferences - allow intelligent routing with first available model
        return {
            'primary_model': self.model_medium,
            'enable_intelligent_routing': True,
            'user_explicit_choice': False
        }
    
    def _analyze_complexity(self, query: str, data: Any, context: Optional[Dict]) -> float:
        """
        Analyze query complexity using fast heuristic.
        
        Returns float 0.0-1.0:
        - < 0.3: Simple (definitions, show data)
        - 0.3-0.7: Medium (calculations, filters)
        - > 0.7: Complex (multi-step, correlations)
        """
        query_lower = query.lower()
        query_len = len(query)
        
        complexity = 0.1
        
        # Simple patterns get low score immediately (from config)
        if self.simple_query_patterns and query_len < 50:
            if any(p in query_lower for p in self.simple_query_patterns):
                return 0.15
        
        # Length contribution
        if query_len > 200:
            complexity += 0.4
        elif query_len > 120:
            complexity += 0.3
        elif query_len > 80:
            complexity += 0.2
        elif query_len > 50:
            complexity += 0.1
        
        # Multi-step indicators (HIGH complexity) - from config
        if self.multi_step_keywords:
            complexity += sum(0.2 for w in self.multi_step_keywords if w in query_lower)
        
        # Conditions (MEDIUM boost) - from config
        if self.condition_keywords:
            complexity += sum(0.1 for w in self.condition_keywords if w in query_lower)
        
        # Computations (MEDIUM boost) - from config
        if self.code_gen_keywords:
            computations = sum(1 for w in self.code_gen_keywords[:10] if w in query_lower)
            complexity += min(computations * 0.05, 0.15)
        
        return min(complexity, 1.0)
    
    def _select_model_intelligent(self, complexity: float) -> str:
        """Select model based on complexity (Track 1)"""
        if complexity < self.simple_threshold:
            return self.model_simple
        elif complexity < self.medium_threshold:
            return self.model_medium
        else:
            return self.model_complex
    
    def _select_execution_method(self, query: str, data: Any) -> ExecutionMethod:
        """
        Select execution method based on query type (Track 2)
        
        FIXED LOGIC: If query asks about data values, MUST use code generation
        because LLM cannot know what's in the data without executing code.
        """
        query_lower = query.lower()
        
        # Check if we have data to analyze
        has_data = data is not None
        
        # Check if query needs computation OR asks about specific data values
        needs_code = any(kw in query_lower for kw in self.code_gen_keywords)
        
        if has_data and needs_code:
            return ExecutionMethod.CODE_GENERATION
        
        # Default: conversational/conceptual queries use direct LLM
        return ExecutionMethod.DIRECT_LLM
    
    def _select_review_level(self, complexity: float, method: ExecutionMethod) -> ReviewLevel:
        """Select review level based on complexity + method (Track 3)"""
        # RULE 1: Mandatory for complex queries
        if complexity >= self.review_always_on_complexity:
            return ReviewLevel.MANDATORY
        
        # RULE 2: Mandatory for code generation
        if method == ExecutionMethod.CODE_GENERATION and self.review_always_on_code_gen:
            return ReviewLevel.MANDATORY
        
        # RULE 3: Optional for medium complexity
        min_opt, max_opt = self.review_optional_range
        if min_opt <= complexity < max_opt:
            return ReviewLevel.OPTIONAL
        
        # RULE 4: Skip for simple queries
        return ReviewLevel.NONE
    
    def _build_fallback_chain(self, selected_model: str) -> List[str]:
        """Build simple fallback chain (larger → smaller models)"""
        all_models = [self.model_complex, self.model_medium, self.model_simple]
        
        # Find index of selected model
        try:
            idx = all_models.index(selected_model)
            # Return models after the selected one (smaller models)
            return all_models[idx+1:] if idx < len(all_models)-1 else []
        except ValueError:
            return []
    
    def _explain_plan(self, query: str, model: str, method: ExecutionMethod, 
                     review: ReviewLevel, complexity: float, user_override: bool) -> str:
        """Generate human-readable explanation"""
        parts = []
        
        # Complexity
        if complexity < 0.3:
            parts.append(f"Complexity: {complexity:.2f} (simple)")
        elif complexity < 0.7:
            parts.append(f"Complexity: {complexity:.2f} (medium)")
        else:
            parts.append(f"Complexity: {complexity:.2f} (complex)")
        
        # Model
        if user_override:
            parts.append(f"Model: {model} (user selected)")
        else:
            parts.append(f"Model: {model} (intelligent routing)")
        
        # Method
        if method == ExecutionMethod.CODE_GENERATION:
            parts.append("Method: Code generation (accurate computation)")
        else:
            parts.append("Method: Direct LLM (natural language)")
        
        # Review
        if review == ReviewLevel.MANDATORY:
            reason = "complex query" if complexity >= 0.7 else "code validation"
            parts.append(f"Review: Mandatory ({reason})")
        elif review == ReviewLevel.OPTIONAL:
            parts.append("Review: Optional (medium complexity)")
        else:
            parts.append("Review: None (fast path)")
        
        return " | ".join(parts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status for monitoring"""
        return {
            'models': {
                'simple': self.model_simple,
                'medium': self.model_medium,
                'complex': self.model_complex
            },
            'thresholds': {
                'simple_max': self.simple_threshold,
                'medium_max': self.medium_threshold
            },
            'advanced_components': {
                'user_preferences': self.user_prefs_manager is not None,
                'model_discovery': self.model_discovery is not None,
                'ram_selector': self.ram_selector is not None
            }
        }
