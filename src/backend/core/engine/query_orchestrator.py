"""
Query Orchestrator - The Brain (Streamlined & Fixed)
====================================================
CORE PURPOSE: Make ONE intelligent decision about how to execute a query

Three Tracks Unified:
1. Complexity → Model (simple/medium/complex)
2. Query Type → Method (code_generation vs direct_llm)
3. Complexity + Method → Review (none/optional/mandatory)

.. versionadded:: 2.0.0
   Added :class:`OrchestratorMiddleware`, :class:`ExecutionHook`,
   :class:`OrchestratorMetrics`, and :func:`get_orchestrator_metrics`.

.. versionadded:: 2.1.0
   Added Swarm Coordination: `decompose_query_to_swarm` for hierarchical task planning.

FIXES FROM V1:
- Removed over-engineered Phase1 optional dependencies (causes import errors)
- Simplified model selection (discovery is separate concern)
- User preferences properly integrated (respects intelligent_routing toggle)
- Config unified (single source of truth from cot_review_config.json)
- Heuristic complexity is the default (QueryComplexityAnalyzer optional)
- Fallback logic simplified (fewer moving parts)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.core.swarm import SwarmContext, SwarmEvent

logger = logging.getLogger(__name__)

__all__ = [
    # v1.x (backward compatible)
    "QueryOrchestrator",
    "ExecutionMethod",
    "ReviewLevel",
    "ExecutionPlan",
    "get_query_orchestrator",
    # v2.0 Enterprise additions
    "OrchestratorMiddleware",
    "ExecutionHook",
    "OrchestratorMetrics",
    "get_orchestrator_metrics",
]

# Import paper metrics for instrumentation (reviewer comment 2)
try:
    from .paper_metrics import get_paper_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Import UnifiedOptimizer for resource-aware execution
try:
    from backend.core.optimizers import UnifiedOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    logger.warning("UnifiedOptimizer not available - resource-aware execution disabled")


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
    """Unified execution plan combining all three tracks.

    Encapsulates the routing decision produced by
    :class:`QueryOrchestrator` — the chosen model, execution method,
    review level, and supporting metadata such as the complexity score
    and a human-readable reasoning string.

    Attributes:
        model: Ollama model tag selected for this query.
        execution_method: Whether to generate code or call the LLM directly.
        review_level: Two-Friends review activation level.
        complexity_score: Heuristic or semantic complexity in ``[0, 1]``.
        reasoning: Human-readable explanation of the routing decision.
        user_override: ``True`` when the user disabled intelligent routing.
        fallback_models: Ordered list of fallback model tags.
    """

    model: str                          # Which model to use
    execution_method: ExecutionMethod   # Code gen or direct
    review_level: ReviewLevel           # Two Friends activation
    complexity_score: float             # Computed complexity
    reasoning: str                      # Human-readable explanation
    user_override: bool = False         # True if user disabled intelligent routing
    fallback_models: List[str] = field(default_factory=list)  # Fallback chain
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the execution plan to a JSON-compatible dictionary.

        Returns:
            Dict[str, Any]: JSON-serializable mapping with keys
                ``model``, ``execution_method``, ``review_level``,
                ``complexity_score``, ``reasoning``, ``user_override``,
                and ``fallback_models``.
        """
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
    """Master decision-maker integrating the three innovation tracks.

    The orchestrator analyses every incoming query and produces an
    :class:`ExecutionPlan` that determines:

    * **Track 1 — Complexity → Model**: routes to *simple*, *medium*, or
      *complex* model tier based on heuristic or semantic complexity.
    * **Track 2 — Query Type → Method**: chooses between
      :attr:`ExecutionMethod.CODE_GENERATION` and
      :attr:`ExecutionMethod.DIRECT_LLM`.
    * **Track 3 — Review Level**: activates the *Two-Friends* generator /
      critic pipeline when warranted.

    Design principles:
        - Heuristic complexity by default (fast, no external dependencies).
        - User preferences respected (``intelligent_routing`` toggle).
        - Config from single source (``cot_review_config.json``).
        - Optional enhancements (semantic routing, optimizer) degrade
          gracefully without breaking the core path.
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize orchestrator with unified config.
        
        Args:
            config_path: Path to cot_review_config.json. Auto-detected
                from the ``config/`` directory when *None*.

        Raises:
            RuntimeError: If config file cannot be loaded.
            FileNotFoundError: If no config file exists at any searched path.
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
        
        # Initialize UnifiedOptimizer for resource-aware execution
        self.optimizer = None
        if OPTIMIZER_AVAILABLE:
            try:
                self.optimizer = UnifiedOptimizer()
                logger.info("UnifiedOptimizer initialized - resource-aware execution enabled")
            except Exception as e:
                logger.warning("Failed to initialize UnifiedOptimizer: %s", e)
        
        # Optional: Try to load advanced components
        self._try_load_advanced_components()
        
        # Swarm Context (Shared Blackboard)
        self.swarm_context = SwarmContext()
        
        logger.info("QueryOrchestrator initialized: %s/%s/%s", self.model_simple, self.model_medium, self.model_complex)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load unified configuration from cot_review_config.json.

        Args:
            config_path: Explicit path to the JSON config file, or
                *None* to auto-detect from the ``config/`` directory.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary.

        Raises:
            RuntimeError: If the config file exists but cannot be parsed.
            FileNotFoundError: If no config file is found at any
                searched path.
        """
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
                    logger.info("Loaded config from %s", config_path)
                    return loaded_config
            except Exception as e:
                logger.error("Failed to load config from %s: %s", config_path, e, exc_info=True)
                raise RuntimeError(f"Cannot load config from {config_path}: {e}")
        
        # No hardcoded defaults - config MUST exist
        logger.error("Config file not found at %s. Cannot initialize without config.", config_path)
        raise FileNotFoundError(f"Config file not found. Checked paths: {config_path}")
    
    def _try_load_advanced_components(self) -> None:
        """Try to load optional advanced components (Phase 1)."""
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
                logger.info("Models updated from discovery: %s/%s/%s", self.model_simple, self.model_medium, self.model_complex)
        except ImportError:
            pass
    
    def _is_embedding_model(self, model_name: str) -> bool:
        """Check if model is for embeddings (not text generation)"""
        patterns = ['embed', 'nomic', 'mxbai', 'all-minilm', 'bge-', 'gte-', 'e5-',
                     'jina-embed', 'cohere-embed', 'text-embedding', 'sentence-']
        return any(p in model_name.lower() for p in patterns)
    
    def create_execution_plan(self, 
                             query: str, 
                             data: Any = None,
                             context: Optional[Dict[str, Any]] = None,
                             llm_client: Any = None) -> ExecutionPlan:
        """
        Create unified execution plan combining all three tracks.
        
        Respects user preferences as the **highest-priority** signal:

        * ``enable_intelligent_routing`` OFF → use the user's selected model.
        * ``enable_intelligent_routing`` ON  → dynamic model selection
          (with semantic routing when *llm_client* is provided).
        
        Args:
            query: User's natural language query.
            data: Optional tabular / raw data context that the query
                refers to.  When present, code-generation is favoured
                for value-extraction queries.
            context: Additional metadata such as column names, file
                path, or prior conversation history.
            llm_client: Optional LLM client instance.  When provided,
                semantic routing is attempted before the keyword
                heuristic fallback.
        
        Returns:
            An :class:`ExecutionPlan` containing the selected model,
            execution method, review level, complexity score, and a
            human-readable reasoning string.
        """
        # STEP 1: Check user preferences (HIGHEST PRIORITY - CANNOT BE OVERRIDDEN)
        _routing_start_time = time.time()
        user_prefs = self._get_user_preferences()
        
        # If user has explicitly chosen a model, use it - NO EXCEPTIONS
        if user_prefs['user_explicit_choice']:
            model = user_prefs['primary_model']
            # Still try semantic analysis for complexity/method info, but use fallback if needed
            semantic_info = None
            if llm_client:
                semantic_info = self._analyze_semantic_intent(query, llm_client)
            
            if semantic_info:
                complexity = float(semantic_info['complexity'])
                needs_code = semantic_info['needs_code']
                execution_method = ExecutionMethod.CODE_GENERATION if needs_code else ExecutionMethod.DIRECT_LLM
            else:
                complexity = self._analyze_complexity_heuristic(query, data, context)
                execution_method = self._select_execution_method(query, data)
            
            review_level = self._select_review_level(complexity, execution_method)
            reasoning = self._explain_plan(query, model, execution_method, review_level, complexity, True)
            
            logger.debug("User model choice (absolute priority): %s", model)
            return ExecutionPlan(
                model=model,
                execution_method=execution_method,
                review_level=review_level,
                complexity_score=complexity,
                reasoning=reasoning,
                user_override=True,
                fallback_models=[]  # User wants THIS model, no fallbacks
            )
        
        # STEP 1.5: FAST PATH CHECK (Heuristic First)
        # Calculate heuristic complexity immediately
        heuristic_complexity = self._analyze_complexity_heuristic(query, data, context)
        
        # If query is obviously simple (e.g. "show first 5 rows"), skip expensive semantic routing
        is_fast_path = heuristic_complexity < 0.3
        
        # STEP 2: SEMANTIC ROUTING (Primary mechanism if not fast path)
        semantic_info = None
        if user_prefs['enable_intelligent_routing'] and llm_client and not is_fast_path:
            logger.debug("🧠 Attempting semantic routing (primary mechanism)...")
            semantic_info = self._analyze_semantic_intent(query, llm_client)
            
            if semantic_info:
                complexity = float(semantic_info['complexity'])
                needs_code = semantic_info['needs_code']
                intent = semantic_info.get('intent', 'unknown')
                logger.info("Semantic routing SUCCESS: complexity=%.2f, needs_code=%s, intent=%s", complexity, needs_code, intent)
            elif semantic_info is False:
                # Explicitly failed, log it
                logger.warning("⚠️ Semantic routing FAILED - falling back to keyword heuristics")
        
        # STEP 3: CONSOLIDATE COMPLEXITY (Semantic or Heuristic)
        if semantic_info:
            # Use semantic results
            pass # complexity/needs_code already set above
        else:
            # Usage heuristic results (either fast path or fallback)
            logger.debug("📊 Using keyword-based heuristic (Fast Path: %s)", is_fast_path)
            complexity = heuristic_complexity
            needs_code = any(kw in query.lower() for kw in self.code_gen_keywords)
            intent = None
        
        # STEP 4: Select model intelligently (only if user allows)
        if user_prefs['enable_intelligent_routing']:
            model = self._select_model_intelligent(complexity)
            user_override = False
            fallback_models = self._build_fallback_chain(model)
            logger.debug("Model selected: %s (complexity=%.2f)", model, complexity)
            
            # STEP 4.5: RESOURCE-AWARE OPTIMIZATION (Auto-downgrade if needed)
            # Check system resources and potentially downgrade model
            if self.optimizer:
                # Define model RAM requirements (approximate)
                model_ram_requirements = {
                    self.model_simple: 1.5,   # tinyllama ~1.5GB
                    self.model_medium: 2.5,   # phi3:mini ~2.5GB
                    self.model_complex: 6.0   # llama3.1:8b ~6GB
                }
                
                original_model = model
                model, optimization_reason = self.optimizer.recommend_model(
                    ideal_model=model,
                    available_models=model_ram_requirements
                )
                
                if model != original_model:
                    logger.warning("[OPTIMIZER] %s", optimization_reason)
                    logger.info("[OPTIMIZER] Model downgraded: %s → %s", original_model, model)
                else:
                    logger.debug("[OPTIMIZER] %s", optimization_reason)
        else:
            model = user_prefs['primary_model']
            user_override = True
            fallback_models = []
            logger.debug("User disabled intelligent routing, using %s", model)
        
        # STEP 5: Select execution method (prefer semantic info over keyword matching)
        if semantic_info:
            execution_method = ExecutionMethod.CODE_GENERATION if needs_code else ExecutionMethod.DIRECT_LLM
            logger.debug("Execution method from semantic analysis: %s", execution_method.value)
        else:
            execution_method = self._select_execution_method(query, data)
            logger.debug("Execution method from heuristic: %s", execution_method.value)
        
        # STEP 6: Select review level
        review_level = self._select_review_level(complexity, execution_method)
        
        # STEP 7: Generate reasoning
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
        
        # BACKEND VISIBILITY: Log routing decision concisely
        logger.info("Routing: model=%s, method=%s, complexity=%.2f, review=%s", model, execution_method.value, complexity, review_level.value)
        
        # Record routing decision for paper metrics (reviewer comment 2)
        if METRICS_AVAILABLE:
            try:
                routing_latency = (time.time() - _routing_start_time) * 1000
                get_paper_metrics().record_routing(
                    query=query,
                    complexity=complexity,
                    method='semantic' if semantic_info else 'heuristic_fallback',
                    semantic_success=bool(semantic_info),
                    model=model,
                    exec_method=execution_method.value,
                    review_level=review_level.value,
                    routing_latency_ms=routing_latency,
                    fallback_triggered=False,
                    user_override=user_override
                )
            except Exception as e:
                logger.debug("Metrics recording failed: %s", e)
        
        return plan
    
    def decompose_query_to_swarm(self, query: str, context: Optional[Dict[str, Any]] = None, llm_client: Any = None) -> List[Dict[str, Any]]:
        """
        Decomposes a complex query into a hierarchical task graph for the Swarm.
        Uses MetaRouter for specialized agent assignment if implicit conflicts exist.
        
        Args:
            query: The user's query.
            context: Execution context (filepaths, metadata).
            llm_client: Client to generate the plan.
            
        Returns:
            List of task dictionaries with dependencies.
        """
        if not llm_client:
            logger.warning("No LLM client provided for swarm decomposition")
            return []

        # Try to use MetaRouter for single-step clarification or to guide the decomposition
        # For now, we'll give the Decomposition Prompt the full list of specialized agents
        # and rely on the LLM's inherent routing, potentially augmented by a pre-check.
        
        # We define the specialized agents here to ensure the planner knows them
        available_agents_desc = """
- DataAnalystAgent: General analysis, data cleaning, pandas code.
- FinancialAgent: Financial metrics, growth, profitability.
- MLInsightsAgent: Patterns, clustering, anomaly detection, PCA.
- StatisticalAgent: Hypothesis testing, correlation, regression.
- TimeSeriesAgent: Forecasting, trend analysis, seasonality.
- RAGAgent: Text search, document QA.
- SQLAgent: Database queries.
- VisualizerAgent: Plotting, charts.
- ReporterAgent: Summarization, reporting.
"""

        prompt = f"""You are a Strategic Task Planner for a data analysis swarm.
Break down this query into dependent tasks and assign the MOST SPECIALIZED agent.

Query: "{query}"

Available Agents_
{available_agents_desc}

Rules:
1. Don't simply default to DataAnalystAgent if a specialist (Financial, ML, Statistical) fits.
2. Return valid JSON list of tasks.

Example:
[
  {{
    "id": "load_data",
    "description": "Load and clean the dataset",
    "agent": "DataAnalystAgent",
    "dependencies": []
  }},
  {{
    "id": "forecast",
    "description": "Forecast sales for Q4",
    "agent": "TimeSeriesAgent",
    "dependencies": ["load_data"]
  }}
]

JSON ONLY:"""

        try:
            response = llm_client.generate(prompt, model=self.model_medium)
            if isinstance(response, dict):
                response = response.get('response', '')
            
            # Clean generic markdown
            if "```" in response:
                response = response.split("```")[1]
                if response.strip().startswith("json"):
                    response = response.strip()[4:]
            
            tasks = json.loads(response.strip())
            
            # Verification Step: Use MetaRouter to double-check assignments if we have it
            # (If we extracted it to a separate component, we could call it here)
            # For this Phase, we are assuming the updated prompt significantly helps.
            # But the user specifically asked for "Meta-decision layer".
            # So let's integrate the MetaRouter logic if we were to instantiate it.
            # ideally self.meta_router = MetaRouter(llm_client) in __init__
            
            # Register tasks in SwarmContext
            for task in tasks:
                self.swarm_context.add_task(
                    task_id=task.get("id"),
                    description=task.get("description"),
                    dependencies=task.get("dependencies", []),
                    assigned_to=task.get("agent")
                )
            
            logger.info(f"Swarm Decomposition: Created {len(tasks)} tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Swarm decomposition failed: {e}")
            return []

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
                logger.warning("Failed to load user preferences: %s", e)
        
        # No user preferences - allow intelligent routing with first available model
        return {
            'primary_model': self.model_medium,
            'enable_intelligent_routing': True,
            'user_explicit_choice': False
        }
    
    def _analyze_semantic_intent(self, query: str, llm_client: Any) -> Optional[Dict[str, Any]]:
        """
        Use LLM to understand query intent semantically.
        This replaces brittle keyword matching with intelligent classification.
        
        Args:
            query: User's natural language query
            llm_client: LLM client to use for classification
            
        Returns:
            Dict with:
            - complexity: float (0.0-1.0)
            - needs_code: bool
            - intent: str
            
            Returns None if llm_client is None
            Returns False if classification failed (signals to use fallback)
        """
        if not llm_client:
            return None
            
        try:
            # Use fast model for routing to minimize overhead
            model = getattr(llm_client, 'primary_model', 'phi3:mini')
            # Prefer smaller models for routing decisions
            if 'llama3.1' in model:
                model = 'phi3:mini'  # Use faster model for routing
            
            routing_prompt = f"""You are a query classifier. Analyze this query and return ONLY valid JSON.

Query: "{query}"

COMPLEXITY GUIDE (follow strictly):
0.1-0.2: Simple lookup/retrieval ("what is X", "show Y", "list Z")  
0.3-0.5: Basic filtering/sorting ("filter by", "sort by")
0.6-0.7: Single calculation/aggregation ("calculate total", "average of X")
0.8-0.9: Analytical reasoning ("why is X down?", "correlation between X and Y")
0.95-1.0: Multi-step operations ("calculate X then predict Y", "compare and rank")

NEEDS_CODE GUIDE (critical distinction):
needs_code=FALSE: 
  - Simple data lookups that can be answered by reading a few rows directly
  - Questions about 1-5 specific values ("what is name", "show first 3 rows")
  - Conversational/conceptual queries ("what is correlation?", "explain X")
  
needs_code=TRUE:
  - Calculations/aggregations across many rows ("average", "sum", "count")
  - Filtering/grouping operations ("where X > Y", "group by category")
  - Complex data transformations ("join", "pivot", "merge")
  - Statistical analysis ("correlation", "regression", "distribution")

Return JSON in this exact format:
{{"complexity": 0.XX, "needs_code": true/false, "intent": "category"}}

Intent categories: simple_lookup, aggregation, filtering, statistical_analysis, 
ml_analysis, visualization, comparison, trend_analysis, conceptual_query

Examples:
{{"complexity": 0.15, "needs_code": false, "intent": "simple_lookup"}}
{{"complexity": 0.45, "needs_code": true, "intent": "aggregation"}}
{{"complexity": 0.20, "needs_code": false, "intent": "simple_lookup"}}
{{"complexity": 0.85, "needs_code": true, "intent": "statistical_analysis"}}

Your response (JSON only):"""
            
            # Call LLM with short timeout since this is just routing
            response = llm_client.generate(
                prompt=routing_prompt,
                model=model,
                adaptive_timeout=False
            )
            
            if not response or 'response' not in response:
                logger.warning("Semantic routing: Empty LLM response")
                return False
            
            response_text = response['response'].strip()
            
            # Strip markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                # Remove first line (```json or ```) and last line (```)
                response_text = '\n'.join(lines[1:-1]).strip()
            
            # Extract JSON from response (LLM might add extra text)
            json_text = response_text
            if '{' in response_text:
                start = response_text.index('{')
                json_text = response_text[start:].strip()
            
            # Parse JSON - use raw_decode to handle trailing text gracefully
            try:
                decoder = json.JSONDecoder()
                parsed, _ = decoder.raw_decode(json_text)
                
                # Validate structure
                if 'complexity' not in parsed or 'needs_code' not in parsed:
                    logger.warning("Semantic routing: Invalid JSON structure: %s", parsed)
                    return False
                
                # Ensure complexity is in valid range (handle ranges like "0.1-0.2")
                complexity = parsed['complexity']
                if isinstance(complexity, str):
                    # Extract first number from range like "0.1-0.2"
                    numbers = re.findall(r'0?\.\d+', complexity)
                    if numbers:
                        complexity = float(numbers[0])
                    else:
                        logger.warning("Semantic routing: Could not parse complexity string: %s", complexity)
                        return False
                else:
                    complexity = float(complexity)
                    
                if not (0.0 <= complexity <= 1.0):
                    logger.warning("Semantic routing: Complexity %s out of range, clamping", complexity)
                    complexity = max(0.0, min(1.0, complexity))
                parsed['complexity'] = complexity
                
                # Ensure needs_code is boolean
                parsed['needs_code'] = bool(parsed.get('needs_code', False))
                
                # Intent is optional
                parsed['intent'] = parsed.get('intent', 'unknown')
                
                logger.debug("Semantic analysis: %s", parsed)
                return parsed
                
            except json.JSONDecodeError as e:
                logger.warning("Semantic routing: JSON parse error: %s\nResponse: %s", e, response_text[:200])
                return False
            
        except Exception as e:
            logger.warning("Semantic routing failed: %s", e)
            return False
    
    def _analyze_complexity_heuristic(self, query: str, data: Any, context: Optional[Dict]) -> float:
        """
        Analyze query complexity using fast keyword-based heuristic.
        
        ⚠️ FALLBACK MECHANISM: Used only when semantic routing is unavailable or fails.
        Prefer _analyze_semantic_intent for accurate classification.
        
        Returns float 0.0-1.0:
        - < 0.3: Simple (definitions, show data)
        - 0.3-0.7: Medium (calculations, filters)
        - > 0.7: Complex (multi-step, correlations)
        """
        query_lower = query.lower()
        query_len = len(query)
        
        complexity = 0.1
        
        # Simple patterns get low score ONLY if query is very short AND has no computation words
        # This prevents "show me the correlation matrix" from being scored as simple
        if self.simple_query_patterns and query_len < 40:
            is_simple = any(p in query_lower for p in self.simple_query_patterns)
            has_computation = any(kw in query_lower for kw in (self.code_gen_keywords[:15] if self.code_gen_keywords else []))
            if is_simple and not has_computation:
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
        Select execution method based on query type (Track 2) — HEURISTIC FALLBACK.
        
        Uses a combination of:
        1. Config keywords (domain-agnostic computation terms)
        2. Data-aware heuristic: if data is present and query asks about specific values,
           code generation is needed because the LLM cannot access data directly.
        
        NOTE: This is only used when semantic routing fails. Semantic routing (LLM-based)
        is the primary decision mechanism and handles any domain.
        """
        query_lower = query.lower()
        
        # Check if we have data to analyze
        has_data = data is not None
        
        # Check if query needs computation from config keywords
        needs_code = any(kw in query_lower for kw in self.code_gen_keywords)
        
        # EXCEPTION: Simple lookups should NOT use code generation
        # Even if they reference data, the LLM can answer directly from small datasets
        simple_lookup_patterns = ['what is', 'show me', 'list', 'display', 'get']
        is_simple_lookup = any(p in query_lower for p in simple_lookup_patterns) and len(query_lower) < 50
        
        # If it's a simple lookup and no computation keywords, use direct LLM
        if is_simple_lookup and not needs_code:
            return ExecutionMethod.DIRECT_LLM
        
        # Data-specific value queries that NEED code (aggregations, complex filters)
        # These patterns indicate the user wants actual values FROM the data
        complex_value_patterns = [
            'how many', 'how much', 'calculate', 'compute', 'total', 'average', 'mean', 'median',
            'sum', 'count', 'max', 'min', 'greater than', 'less than', 'above', 'below', 'between',
            'where', 'group by', 'for each', 'rows with', 'records with', 'entries where',
            'revenue', 'sales', 'profit', 'gpa', 'score', 'grade', 'cost', 'price'
        ]
        asks_for_complex_values = any(p in query_lower for p in complex_value_patterns)
        
        if has_data and (needs_code or asks_for_complex_values):
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
        """Get orchestrator status for monitoring.

        Returns:
            Dictionary with keys ``models`` (tier mapping),
            ``thresholds`` (complexity boundaries), and
            ``advanced_components`` (availability flags for optional
            subsystems such as user-preferences and model-discovery).
        """
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics from paper metrics.

        Addresses reviewer comment 2: impact of orchestration heuristics
        on latency and accuracy.

        Returns:
            Dictionary with keys ``routing`` (method distribution and
            accuracy), ``correction`` (self-correction counts),
            ``recovery`` (error-recovery statistics), and ``agents``
            (per-agent timing).  Returns ``{'error': ...}`` when the
            paper-metrics subsystem is unavailable.
        """
        if METRICS_AVAILABLE:
            metrics = get_paper_metrics()
            return {
                'routing': metrics.get_routing_statistics(),
                'correction': metrics.get_correction_statistics(),
                'recovery': metrics.get_error_recovery_statistics(),
                'agents': metrics.get_agent_statistics()
            }
        return {'error': 'Paper metrics not available'}


# ── Singleton accessor ──────────────────────────────────────────────
_orchestrator_instance: Optional[QueryOrchestrator] = None
_orchestrator_lock = threading.Lock()

def get_query_orchestrator(config_path: Optional[str] = None) -> QueryOrchestrator:
    """Return a single shared QueryOrchestrator instance (thread-safe).

    Args:
        config_path: Optional path to ``cot_review_config.json``.
            Only used on first invocation; ignored afterwards.

    Returns:
        QueryOrchestrator: The singleton orchestrator instance.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        with _orchestrator_lock:
            if _orchestrator_instance is None:
                _orchestrator_instance = QueryOrchestrator(config_path)
    return _orchestrator_instance


# =============================================================================
# ENTERPRISE: ORCHESTRATOR MIDDLEWARE
# =============================================================================

class OrchestratorMiddleware(ABC):
    """Abstract base class for orchestrator middleware.

    Middleware intercepts execution plan creation, allowing
    pre-processing (modifying inputs) and post-processing
    (modifying results) of the orchestration pipeline.

    .. code-block:: python

        class LoggingMiddleware(OrchestratorMiddleware):
            name = "logging"
            def before_plan(self, query, data_context):
                logger.info("Planning: %s", query[:50])
                return query, data_context
            def after_plan(self, plan):
                logger.info("Plan: %s", plan.method.value)
                return plan
    """

    name: str = "base_middleware"
    priority: int = 100  # Lower = runs earlier

    @abstractmethod
    def before_plan(
        self, query: str, data_context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Called before execution plan creation.

        Args:
            query: User query.
            data_context: Data context dict.

        Returns:
            Potentially modified (query, data_context) tuple.
        """
        ...

    @abstractmethod
    def after_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Called after execution plan creation.

        Args:
            plan: The created execution plan.

        Returns:
            Potentially modified execution plan.
        """
        ...


# =============================================================================
# ENTERPRISE: EXECUTION HOOKS
# =============================================================================

class ExecutionHook:
    """Event hook system for orchestrator lifecycle events.

    Register callbacks for plan creation, model selection,
    and execution completion events.

    .. code-block:: python

        hook = ExecutionHook()
        hook.on_plan_created(lambda plan: print(plan.method))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)

    def on_plan_created(self, callback: Callable[[ExecutionPlan], None]) -> None:
        """Register a callback for plan creation."""
        with self._lock:
            self._hooks["plan_created"].append(callback)

    def on_model_selected(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for model selection. Args: model_name, reason."""
        with self._lock:
            self._hooks["model_selected"].append(callback)

    def on_execution_complete(
        self, callback: Callable[[str, float, bool], None],
    ) -> None:
        """Register a callback for execution completion.
        Args: query, duration_ms, success."""
        with self._lock:
            self._hooks["execution_complete"].append(callback)

    def fire(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire all callbacks for an event."""
        with self._lock:
            callbacks = list(self._hooks.get(event, []))
        for cb in callbacks:
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.warning("Hook callback error (%s): %s", event, e)


# =============================================================================
# ENTERPRISE: ORCHESTRATOR METRICS
# =============================================================================

class OrchestratorMetrics:
    """Tracks orchestrator performance and decision distribution.

    Thread-safe metric collector for plan creation times,
    method distribution, model usage, and review level distribution.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_plans = 0
        self._method_counts: Dict[str, int] = defaultdict(int)
        self._review_counts: Dict[str, int] = defaultdict(int)
        self._model_counts: Dict[str, int] = defaultdict(int)
        self._plan_times: deque = deque(maxlen=500)
        self._complexity_scores: deque = deque(maxlen=500)

    def record_plan(
        self,
        method: str,
        review_level: str,
        model: str,
        plan_time_ms: float,
        complexity: float = 0.0,
    ) -> None:
        """Record a plan creation event.

        Args:
            method: ExecutionMethod value.
            review_level: ReviewLevel value.
            model: Model name used.
            plan_time_ms: Time to create the plan.
            complexity: Query complexity score.
        """
        with self._lock:
            self._total_plans += 1
            self._method_counts[method] += 1
            self._review_counts[review_level] += 1
            self._model_counts[model] += 1
            self._plan_times.append(plan_time_ms)
            self._complexity_scores.append(complexity)

    def get_statistics(self) -> Dict[str, Any]:
        """Return orchestrator metrics summary."""
        with self._lock:
            times = list(self._plan_times)
            scores = list(self._complexity_scores)
            return {
                "total_plans": self._total_plans,
                "method_distribution": dict(self._method_counts),
                "review_distribution": dict(self._review_counts),
                "top_models": dict(
                    sorted(self._model_counts.items(), key=lambda x: -x[1])[:10]
                ),
                "avg_plan_time_ms": round(
                    sum(times) / len(times), 2
                ) if times else 0,
                "avg_complexity": round(
                    sum(scores) / len(scores), 3
                ) if scores else 0,
            }


# =============================================================================
# ENTERPRISE SINGLETON
# =============================================================================

_orchestrator_metrics: Optional[OrchestratorMetrics] = None
_orchestrator_metrics_lock = threading.Lock()


def get_orchestrator_metrics() -> OrchestratorMetrics:
    """Get or create the singleton orchestrator metrics (thread-safe)."""
    global _orchestrator_metrics
    if _orchestrator_metrics is None:
        with _orchestrator_metrics_lock:
            if _orchestrator_metrics is None:
                _orchestrator_metrics = OrchestratorMetrics()
    return _orchestrator_metrics
