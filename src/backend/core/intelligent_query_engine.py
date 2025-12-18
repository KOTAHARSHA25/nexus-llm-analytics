# Intelligent Query Optimization Engine
# BEFORE: Static query processing, no optimization, inefficient routing
# AFTER: Dynamic query optimization, intelligent routing, predictive execution

import asyncio
import time
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
from functools import lru_cache
import statistics
import re
from concurrent.futures import ThreadPoolExecutor
import weakref
import pickle

# Import our optimized components
from .optimized_data_structures import OptimizedTrie, HighPerformanceHashMap, PerformanceMonitor
from .enhanced_cache_integration import get_enhanced_cache_manager, enhanced_cached
from .optimized_llm_client import OptimizedLLMClient, ModelProvider

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels for optimization routing"""
    SIMPLE = "simple"        # Single table, basic operations
    MODERATE = "moderate"    # Multiple tables, joins, aggregations
    COMPLEX = "complex"      # Advanced analytics, ML operations
    EXPERT = "expert"        # Custom algorithms, complex transformations

class QueryType(Enum):
    """Types of queries for specialized routing"""
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    STATISTICS = "statistics" 
    MACHINE_LEARNING = "machine_learning"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    FINANCIAL_ANALYSIS = "financial_analysis"
    TEXT_ANALYSIS = "text_analysis"
    PREDICTION = "prediction"

class AgentCapability(Enum):
    """Agent capabilities for intelligent routing"""
    STATISTICAL_ANALYSIS = "statistical_analysis"
    DATA_VISUALIZATION = "data_visualization"
    MACHINE_LEARNING = "machine_learning"
    FINANCIAL_MODELING = "financial_modeling"
    TEXT_PROCESSING = "text_processing"
    BUSINESS_METRICS = "business_metrics"
    PREDICTIVE_ANALYTICS = "predictive_analytics"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"

@dataclass
class QueryProfile:
    """Comprehensive query profiling for optimization"""
    query_text: str
    query_type: QueryType
    complexity: QueryComplexity
    estimated_duration: float = 0.0
    required_capabilities: Set[AgentCapability] = field(default_factory=set)
    data_sources: List[str] = field(default_factory=list)
    expected_output_size: int = 0
    priority: int = 1  # 1-10, higher is more important
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentProfile:
    """Agent capability and performance profile"""
    agent_id: str
    name: str
    capabilities: Set[AgentCapability]
    performance_scores: Dict[QueryType, float] = field(default_factory=dict)
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    current_load: int = 0
    max_concurrent: int = 5
    cost_per_query: float = 0.0
    model_provider: ModelProvider = ModelProvider.OPENAI

@dataclass
class ExecutionPlan:
    """Optimized query execution plan"""
    query_id: str
    original_query: str
    optimized_query: str
    selected_agent: str
    execution_steps: List[Dict[str, Any]]
    estimated_cost: float
    estimated_duration: float
    cache_strategy: str
    parallel_opportunities: List[str] = field(default_factory=list)
    optimization_rules_applied: List[str] = field(default_factory=list)

class QueryPatternAnalyzer:
    """Analyze query patterns for optimization opportunities"""
    
    def __init__(self):
        self.pattern_trie = OptimizedTrie()
        self.query_templates = HighPerformanceHashMap()
        self.common_phrases = {}
        self.optimization_patterns = []
        self._build_pattern_database()
    
    def _build_pattern_database(self):
        """Build database of common query patterns"""
        # Common analysis patterns
        analysis_patterns = [
            ("analyze", "data_analysis", ["statistics", "trends", "patterns"]),
            ("visualize", "visualization", ["chart", "graph", "plot"]),
            ("predict", "prediction", ["forecast", "future", "trend"]),
            ("compare", "comparison", ["versus", "vs", "difference"]),
            ("summarize", "summary", ["overview", "summary", "key points"]),
            ("correlate", "correlation", ["relationship", "connection", "association"]),
            ("cluster", "machine_learning", ["group", "segment", "category"]),
            ("optimize", "optimization", ["improve", "enhance", "maximize"]),
            ("benchmark", "performance", ["performance", "speed", "efficiency"]),
            ("financial", "financial_analysis", ["revenue", "profit", "cost"])
        ]
        
        for pattern, query_type, keywords in analysis_patterns:
            pattern_data = {
                "type": query_type,
                "keywords": keywords,
                "complexity": "moderate",
                "priority": 5
            }
            self.pattern_trie.insert(pattern, pattern_data)
            
            # Add related keywords
            for keyword in keywords:
                self.pattern_trie.insert(keyword, {
                    "type": query_type,
                    "parent_pattern": pattern,
                    "complexity": "simple",
                    "priority": 3
                })
    
    def analyze_query(self, query: str) -> QueryProfile:
        """Analyze query and extract optimization profile"""
        query_lower = query.lower()
        
        # Extract patterns using trie
        detected_patterns = []
        words = re.findall(r'\b\w+\b', query_lower)
        
        for word in words:
            matches = self.pattern_trie.search_prefix(word, max_suggestions=3)
            for match_text, frequency, metadata in matches:
                if word.startswith(match_text.rstrip('_')):
                    detected_patterns.append((match_text, metadata))
        
        # Determine query type and complexity
        query_type = self._determine_query_type(detected_patterns, query_lower)
        complexity = self._determine_complexity(query, detected_patterns)
        capabilities = self._extract_required_capabilities(query_type, detected_patterns)
        
        # Estimate resource requirements
        estimated_duration = self._estimate_duration(query, complexity)
        expected_output_size = self._estimate_output_size(query, query_type)
        
        return QueryProfile(
            query_text=query,
            query_type=query_type,
            complexity=complexity,
            estimated_duration=estimated_duration,
            required_capabilities=capabilities,
            expected_output_size=expected_output_size,
            metadata={
                "detected_patterns": detected_patterns,
                "word_count": len(words),
                "question_type": self._classify_question_type(query)
            }
        )
    
    def _determine_query_type(self, patterns: List[Tuple], query: str) -> QueryType:
        """Determine the primary query type"""
        type_scores = defaultdict(int)
        
        # Score based on detected patterns
        for pattern_text, metadata in patterns:
            query_type = metadata.get("type", "data_analysis")
            priority = metadata.get("priority", 1)
            type_scores[query_type] += priority
        
        # Additional keyword-based scoring
        keyword_mapping = {
            "chart": QueryType.VISUALIZATION,
            "graph": QueryType.VISUALIZATION,
            "plot": QueryType.VISUALIZATION,
            "statistics": QueryType.STATISTICS,
            "correlation": QueryType.STATISTICS,
            "regression": QueryType.MACHINE_LEARNING,
            "cluster": QueryType.MACHINE_LEARNING,
            "revenue": QueryType.FINANCIAL_ANALYSIS,
            "profit": QueryType.FINANCIAL_ANALYSIS,
            "sentiment": QueryType.TEXT_ANALYSIS,
            "nlp": QueryType.TEXT_ANALYSIS,
            "forecast": QueryType.PREDICTION,
            "predict": QueryType.PREDICTION
        }
        
        query_lower = query.lower()
        for keyword, query_type in keyword_mapping.items():
            if keyword in query_lower:
                type_scores[query_type.value] += 2
        
        # Return type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])[0]
            try:
                return QueryType(best_type)
            except ValueError:
                logging.debug("Operation failed (non-critical) - continuing")
        
        return QueryType.DATA_ANALYSIS  # Default
    
    def _determine_complexity(self, query: str, patterns: List[Tuple]) -> QueryComplexity:
        """Determine query complexity"""
        complexity_indicators = {
            "simple": ["show", "list", "count", "sum", "average"],
            "moderate": ["analyze", "compare", "group", "filter", "join"],
            "complex": ["optimize", "correlate", "regression", "cluster", "model"],
            "expert": ["algorithm", "custom", "advanced", "machine learning"]
        }
        
        query_lower = query.lower()
        scores = defaultdict(int)
        
        # Score based on keywords
        for complexity, keywords in complexity_indicators.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[complexity] += 1
        
        # Additional complexity factors
        word_count = len(query.split())
        if word_count > 50:
            scores["complex"] += 2
        elif word_count > 20:
            scores["moderate"] += 1
        
        # Multiple data sources increase complexity
        if "and" in query_lower and ("file" in query_lower or "data" in query_lower):
            scores["moderate"] += 1
        
        # Statistical terms increase complexity
        stats_terms = ["correlation", "regression", "significance", "p-value", "hypothesis"]
        for term in stats_terms:
            if term in query_lower:
                scores["complex"] += 1
        
        # Return highest scoring complexity
        if scores:
            best_complexity = max(scores.items(), key=lambda x: x[1])[0]
            try:
                return QueryComplexity(best_complexity)
            except ValueError:
                logging.debug("Operation failed (non-critical) - continuing")
        
        return QueryComplexity.SIMPLE  # Default
    
    def _extract_required_capabilities(self, query_type: QueryType, 
                                     patterns: List[Tuple]) -> Set[AgentCapability]:
        """Extract required agent capabilities"""
        capabilities = set()
        
        # Map query types to capabilities
        type_capability_map = {
            QueryType.DATA_ANALYSIS: {AgentCapability.STATISTICAL_ANALYSIS},
            QueryType.VISUALIZATION: {AgentCapability.DATA_VISUALIZATION},
            QueryType.STATISTICS: {AgentCapability.STATISTICAL_ANALYSIS},
            QueryType.MACHINE_LEARNING: {AgentCapability.MACHINE_LEARNING, AgentCapability.PREDICTIVE_ANALYTICS},
            QueryType.BUSINESS_INTELLIGENCE: {AgentCapability.BUSINESS_METRICS, AgentCapability.DATA_VISUALIZATION},
            QueryType.FINANCIAL_ANALYSIS: {AgentCapability.FINANCIAL_MODELING, AgentCapability.STATISTICAL_ANALYSIS},
            QueryType.TEXT_ANALYSIS: {AgentCapability.TEXT_PROCESSING},
            QueryType.PREDICTION: {AgentCapability.PREDICTIVE_ANALYTICS, AgentCapability.MACHINE_LEARNING}
        }
        
        capabilities.update(type_capability_map.get(query_type, set()))
        
        # Add capabilities based on patterns
        for pattern_text, metadata in patterns:
            if "visualization" in metadata.get("type", ""):
                capabilities.add(AgentCapability.DATA_VISUALIZATION)
            elif "machine_learning" in metadata.get("type", ""):
                capabilities.add(AgentCapability.MACHINE_LEARNING)
            elif "financial" in metadata.get("type", ""):
                capabilities.add(AgentCapability.FINANCIAL_MODELING)
        
        return capabilities
    
    def _estimate_duration(self, query: str, complexity: QueryComplexity) -> float:
        """Estimate query execution duration"""
        base_times = {
            QueryComplexity.SIMPLE: 2.0,
            QueryComplexity.MODERATE: 5.0,
            QueryComplexity.COMPLEX: 15.0,
            QueryComplexity.EXPERT: 30.0
        }
        
        base_time = base_times[complexity]
        
        # Adjust based on query characteristics
        word_count = len(query.split())
        if word_count > 30:
            base_time *= 1.5
        
        # File processing adds time
        if "file" in query.lower():
            base_time *= 1.3
        
        return base_time
    
    def _estimate_output_size(self, query: str, query_type: QueryType) -> int:
        """Estimate expected output size in characters"""
        base_sizes = {
            QueryType.DATA_ANALYSIS: 2000,
            QueryType.VISUALIZATION: 1500,
            QueryType.STATISTICS: 1000,
            QueryType.MACHINE_LEARNING: 3000,
            QueryType.BUSINESS_INTELLIGENCE: 2500,
            QueryType.FINANCIAL_ANALYSIS: 2000,
            QueryType.TEXT_ANALYSIS: 1500,
            QueryType.PREDICTION: 1800
        }
        
        return base_sizes.get(query_type, 2000)
    
    def _classify_question_type(self, query: str) -> str:
        """Classify the type of question being asked"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "which", "who"]):
            return "factual"
        elif any(word in query_lower for word in ["how", "why"]):
            return "explanatory"
        elif any(word in query_lower for word in ["will", "predict", "forecast"]):
            return "predictive"
        elif any(word in query_lower for word in ["compare", "versus", "difference"]):
            return "comparative"
        else:
            return "analytical"

class IntelligentQueryRouter:
    """Route queries to optimal agents based on capabilities and performance"""
    
    def __init__(self):
        self.agents = {}
        self.performance_history = defaultdict(list)
        self.load_balancer = HighPerformanceHashMap()
        self.routing_rules = []
        self._initialize_agents()
        self._initialize_routing_rules()
    
    def _initialize_agents(self):
        """Initialize available agents with their profiles"""
        agents_config = [
            {
                "agent_id": "data_analyst",
                "name": "Data Analyst Agent",
                "capabilities": {
                    AgentCapability.STATISTICAL_ANALYSIS,
                    AgentCapability.DATA_VISUALIZATION
                },
                "model_provider": ModelProvider.OPENAI,
                "max_concurrent": 5,
                "cost_per_query": 0.02
            },
            {
                "agent_id": "ml_engineer", 
                "name": "ML Engineer Agent",
                "capabilities": {
                    AgentCapability.MACHINE_LEARNING,
                    AgentCapability.PREDICTIVE_ANALYTICS,
                    AgentCapability.STATISTICAL_ANALYSIS
                },
                "model_provider": ModelProvider.ANTHROPIC,
                "max_concurrent": 3,
                "cost_per_query": 0.05
            },
            {
                "agent_id": "financial_analyst",
                "name": "Financial Analyst Agent", 
                "capabilities": {
                    AgentCapability.FINANCIAL_MODELING,
                    AgentCapability.BUSINESS_METRICS,
                    AgentCapability.STATISTICAL_ANALYSIS
                },
                "model_provider": ModelProvider.OPENAI,
                "max_concurrent": 4,
                "cost_per_query": 0.03
            },
            {
                "agent_id": "business_intelligence",
                "name": "Business Intelligence Agent",
                "capabilities": {
                    AgentCapability.BUSINESS_METRICS,
                    AgentCapability.DATA_VISUALIZATION,
                    AgentCapability.PERFORMANCE_OPTIMIZATION
                },
                "model_provider": ModelProvider.GOOGLE,
                "max_concurrent": 6,
                "cost_per_query": 0.015
            },
            {
                "agent_id": "text_analyst",
                "name": "Text Analysis Agent",
                "capabilities": {
                    AgentCapability.TEXT_PROCESSING,
                    AgentCapability.STATISTICAL_ANALYSIS
                },
                "model_provider": ModelProvider.ANTHROPIC,
                "max_concurrent": 4,
                "cost_per_query": 0.025
            }
        ]
        
        for config in agents_config:
            agent = AgentProfile(**config)
            self.agents[agent.agent_id] = agent
    
    def _initialize_routing_rules(self):
        """Initialize intelligent routing rules"""
        self.routing_rules = [
            # High priority queries go to best performing agents
            {
                "condition": lambda profile: profile.priority >= 8,
                "action": "route_to_best_performer",
                "weight": 10
            },
            # Complex queries need specialized agents
            {
                "condition": lambda profile: profile.complexity == QueryComplexity.EXPERT,
                "action": "route_to_specialist",
                "weight": 9
            },
            # Financial queries to financial agent
            {
                "condition": lambda profile: profile.query_type == QueryType.FINANCIAL_ANALYSIS,
                "action": "route_to_financial",
                "weight": 8
            },
            # ML queries to ML agent
            {
                "condition": lambda profile: profile.query_type == QueryType.MACHINE_LEARNING,
                "action": "route_to_ml",
                "weight": 8
            },
            # Text analysis to text agent
            {
                "condition": lambda profile: profile.query_type == QueryType.TEXT_ANALYSIS,
                "action": "route_to_text",
                "weight": 8
            },
            # Load balancing for similar capabilities
            {
                "condition": lambda profile: profile.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE],
                "action": "load_balance",
                "weight": 5
            }
        ]
    
    def route_query(self, query_profile: QueryProfile) -> str:
        """Route query to optimal agent"""
        # Find matching routing rules
        applicable_rules = []
        for rule in self.routing_rules:
            if rule["condition"](query_profile):
                applicable_rules.append(rule)
        
        # Sort by weight and apply highest priority rule
        if applicable_rules:
            best_rule = max(applicable_rules, key=lambda x: x["weight"])
            return self._apply_routing_action(best_rule["action"], query_profile)
        
        # Fallback to capability-based routing
        return self._route_by_capabilities(query_profile)
    
    def _apply_routing_action(self, action: str, query_profile: QueryProfile) -> str:
        """Apply specific routing action"""
        if action == "route_to_best_performer":
            return self._get_best_performer(query_profile)
        elif action == "route_to_specialist":
            return self._get_specialist_agent(query_profile)
        elif action == "route_to_financial":
            return "financial_analyst"
        elif action == "route_to_ml":
            return "ml_engineer"
        elif action == "route_to_text":
            return "text_analyst"
        elif action == "load_balance":
            return self._load_balance_agents(query_profile)
        else:
            return self._route_by_capabilities(query_profile)
    
    def _get_best_performer(self, query_profile: QueryProfile) -> str:
        """Get best performing agent for query type"""
        candidates = []
        
        for agent_id, agent in self.agents.items():
            if self._agent_can_handle(agent, query_profile):
                performance_score = agent.performance_scores.get(query_profile.query_type, 0.5)
                candidates.append((agent_id, performance_score))
        
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
        
        return "data_analyst"  # Default fallback
    
    def _get_specialist_agent(self, query_profile: QueryProfile) -> str:
        """Get most specialized agent for complex queries"""
        best_match = None
        max_capability_overlap = 0
        
        for agent_id, agent in self.agents.items():
            overlap = len(agent.capabilities.intersection(query_profile.required_capabilities))
            if overlap > max_capability_overlap and agent.current_load < agent.max_concurrent:
                max_capability_overlap = overlap
                best_match = agent_id
        
        return best_match or "ml_engineer"  # ML agent for complex queries
    
    def _load_balance_agents(self, query_profile: QueryProfile) -> str:
        """Load balance among capable agents"""
        capable_agents = []
        
        for agent_id, agent in self.agents.items():
            if (self._agent_can_handle(agent, query_profile) and 
                agent.current_load < agent.max_concurrent):
                load_ratio = agent.current_load / agent.max_concurrent
                capable_agents.append((agent_id, load_ratio))
        
        if capable_agents:
            # Return agent with lowest load
            return min(capable_agents, key=lambda x: x[1])[0]
        
        return "data_analyst"  # Default fallback
    
    def _route_by_capabilities(self, query_profile: QueryProfile) -> str:
        """Route based on agent capabilities"""
        best_match = None
        max_score = 0
        
        for agent_id, agent in self.agents.items():
            if agent.current_load >= agent.max_concurrent:
                continue
            
            # Calculate compatibility score
            capability_overlap = len(agent.capabilities.intersection(query_profile.required_capabilities))
            load_factor = 1.0 - (agent.current_load / agent.max_concurrent)
            performance_factor = agent.performance_scores.get(query_profile.query_type, 0.5)
            
            score = capability_overlap * 0.5 + load_factor * 0.3 + performance_factor * 0.2
            
            if score > max_score:
                max_score = score
                best_match = agent_id
        
        return best_match or "data_analyst"
    
    def _agent_can_handle(self, agent: AgentProfile, query_profile: QueryProfile) -> bool:
        """Check if agent can handle the query"""
        # Check capability overlap
        required_caps = query_profile.required_capabilities
        agent_caps = agent.capabilities
        
        if not required_caps.intersection(agent_caps):
            return False
        
        # Check current load
        if agent.current_load >= agent.max_concurrent:
            return False
        
        return True
    
    def update_agent_performance(self, agent_id: str, query_type: QueryType, 
                                performance_score: float, duration: float):
        """Update agent performance metrics"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # Update performance score (rolling average)
        current_score = agent.performance_scores.get(query_type, 0.5)
        new_score = 0.8 * current_score + 0.2 * performance_score
        agent.performance_scores[query_type] = new_score
        
        # Update average response time
        self.performance_history[agent_id].append(duration)
        if len(self.performance_history[agent_id]) > 100:
            self.performance_history[agent_id].pop(0)
        
        agent.avg_response_time = statistics.mean(self.performance_history[agent_id])

class QueryOptimizer:
    """Optimize queries for better performance"""
    
    def __init__(self):
        self.optimization_rules = []
        self.query_cache = HighPerformanceHashMap()
        self.template_cache = {}
        self._initialize_optimization_rules()
    
    def _initialize_optimization_rules(self):
        """Initialize query optimization rules"""
        self.optimization_rules = [
            {
                "name": "simplify_redundant_phrases",
                "pattern": r"\b(please|can you|could you|would you)\b",
                "replacement": "",
                "weight": 1
            },
            {
                "name": "standardize_analysis_requests",
                "pattern": r"\b(analyze|analyse)\b",
                "replacement": "analyze",
                "weight": 2
            },
            {
                "name": "optimize_data_references",
                "pattern": r"\b(the data|this data|my data)\b",
                "replacement": "data",
                "weight": 1
            },
            {
                "name": "consolidate_visualization_terms",
                "pattern": r"\b(create|make|generate|build)\s+(a\s+)?(chart|graph|plot|visualization)\b",
                "replacement": "visualize",
                "weight": 3
            },
            {
                "name": "standardize_comparison_requests",
                "pattern": r"\b(compare|contrast)\s+(.+?)\s+(with|to|versus|vs)\s+(.+)\b",
                "replacement": r"compare \2 vs \4",
                "weight": 2
            }
        ]
    
    def optimize_query(self, query: str) -> Tuple[str, List[str]]:
        """Optimize query text for better processing"""
        optimized_query = query
        applied_rules = []
        
        # Apply optimization rules
        for rule in self.optimization_rules:
            pattern = rule["pattern"]
            replacement = rule["replacement"]
            
            if re.search(pattern, optimized_query, re.IGNORECASE):
                optimized_query = re.sub(pattern, replacement, optimized_query, flags=re.IGNORECASE)
                applied_rules.append(rule["name"])
        
        # Clean up extra whitespace
        optimized_query = re.sub(r'\s+', ' ', optimized_query).strip()
        
        # Cache optimization result
        cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]
        self.query_cache.put(cache_key, optimized_query)
        
        return optimized_query, applied_rules
    
    def extract_query_template(self, query: str) -> str:
        """Extract reusable template from query"""
        # Replace specific values with placeholders
        template = query
        
        # Replace numbers with placeholder
        template = re.sub(r'\b\d+\.?\d*\b', '[NUMBER]', template)
        
        # Replace file names with placeholder
        template = re.sub(r'\b\w+\.(csv|json|xlsx|txt|pdf)\b', '[FILE]', template, flags=re.IGNORECASE)
        
        # Replace specific dates with placeholder
        template = re.sub(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b', '[DATE]', template)
        
        # Replace column names (capitalized words in quotes)
        template = re.sub(r'"[A-Z][a-zA-Z_\s]*"', '[COLUMN]', template)
        
        return template

class ExecutionPlanOptimizer:
    """Create and optimize query execution plans"""
    
    def __init__(self):
        self.plan_cache = HighPerformanceHashMap()
        self.performance_monitor = PerformanceMonitor()
        
    def create_execution_plan(self, query_profile: QueryProfile, 
                            selected_agent: str) -> ExecutionPlan:
        """Create optimized execution plan"""
        query_id = hashlib.sha256(query_profile.query_text.encode()).hexdigest()[:16]
        
        # Check if we have a cached plan for similar queries
        template = self._extract_execution_template(query_profile)
        cached_plan = self.plan_cache.get(template)
        
        if cached_plan:
            # Adapt cached plan
            return self._adapt_cached_plan(cached_plan, query_profile, selected_agent)
        
        # Create new execution plan
        execution_steps = self._generate_execution_steps(query_profile)
        
        # Optimize execution order
        optimized_steps = self._optimize_execution_order(execution_steps)
        
        # Identify parallel opportunities
        parallel_ops = self._identify_parallel_opportunities(optimized_steps)
        
        # Estimate costs
        estimated_cost = self._estimate_execution_cost(query_profile, selected_agent)
        
        plan = ExecutionPlan(
            query_id=query_id,
            original_query=query_profile.query_text,
            optimized_query=query_profile.query_text,  # Could be optimized by QueryOptimizer
            selected_agent=selected_agent,
            execution_steps=optimized_steps,
            estimated_cost=estimated_cost,
            estimated_duration=query_profile.estimated_duration,
            cache_strategy=self._determine_cache_strategy(query_profile),
            parallel_opportunities=parallel_ops
        )
        
        # Cache the plan template
        self.plan_cache.put(template, plan)
        
        return plan
    
    def _extract_execution_template(self, query_profile: QueryProfile) -> str:
        """Extract execution template for caching"""
        template_data = {
            "query_type": query_profile.query_type.value,
            "complexity": query_profile.complexity.value,
            "capabilities": sorted([cap.value for cap in query_profile.required_capabilities])
        }
        return json.dumps(template_data, sort_keys=True)
    
    def _adapt_cached_plan(self, cached_plan: ExecutionPlan, 
                          query_profile: QueryProfile, 
                          selected_agent: str) -> ExecutionPlan:
        """Adapt cached execution plan to current query"""
        new_query_id = hashlib.sha256(query_profile.query_text.encode()).hexdigest()[:16]
        
        return ExecutionPlan(
            query_id=new_query_id,
            original_query=query_profile.query_text,
            optimized_query=query_profile.query_text,
            selected_agent=selected_agent,
            execution_steps=cached_plan.execution_steps.copy(),
            estimated_cost=cached_plan.estimated_cost,
            estimated_duration=query_profile.estimated_duration,
            cache_strategy=cached_plan.cache_strategy,
            parallel_opportunities=cached_plan.parallel_opportunities.copy(),
            optimization_rules_applied=["cached_plan_adaptation"]
        )
    
    def _generate_execution_steps(self, query_profile: QueryProfile) -> List[Dict[str, Any]]:
        """Generate execution steps based on query profile"""
        steps = []
        
        # Data loading step
        if query_profile.data_sources:
            steps.append({
                "step": "data_loading",
                "action": "load_data_sources",
                "resources": query_profile.data_sources,
                "estimated_time": 2.0,
                "parallelizable": True
            })
        
        # Preprocessing step
        if query_profile.complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            steps.append({
                "step": "preprocessing",
                "action": "clean_and_prepare_data",
                "estimated_time": 1.0,
                "parallelizable": False
            })
        
        # Analysis step
        analysis_time = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 3.0,
            QueryComplexity.COMPLEX: 8.0,
            QueryComplexity.EXPERT: 15.0
        }
        
        steps.append({
            "step": "analysis",
            "action": f"perform_{query_profile.query_type.value}",
            "estimated_time": analysis_time[query_profile.complexity],
            "parallelizable": False
        })
        
        # Visualization step (if needed)
        if (query_profile.query_type == QueryType.VISUALIZATION or
            AgentCapability.DATA_VISUALIZATION in query_profile.required_capabilities):
            steps.append({
                "step": "visualization",
                "action": "create_visualizations",
                "estimated_time": 2.0,
                "parallelizable": True
            })
        
        # Result formatting step
        steps.append({
            "step": "formatting",
            "action": "format_results",
            "estimated_time": 0.5,
            "parallelizable": False
        })
        
        return steps
    
    def _optimize_execution_order(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize the order of execution steps"""
        # Sort by dependencies and parallelization opportunities
        # This is a simplified version - real implementation would be more sophisticated
        
        parallelizable_steps = [step for step in steps if step.get("parallelizable", False)]
        sequential_steps = [step for step in steps if not step.get("parallelizable", False)]
        
        # Group parallelizable steps together
        optimized_steps = []
        
        # Add initial sequential steps
        for step in sequential_steps:
            if step["step"] in ["preprocessing", "analysis"]:
                optimized_steps.append(step)
        
        # Add parallelizable steps
        optimized_steps.extend(parallelizable_steps)
        
        # Add final sequential steps
        for step in sequential_steps:
            if step["step"] == "formatting":
                optimized_steps.append(step)
        
        return optimized_steps
    
    def _identify_parallel_opportunities(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Identify opportunities for parallel execution"""
        parallel_ops = []
        
        for step in steps:
            if step.get("parallelizable", False):
                parallel_ops.append(step["step"])
        
        # Additional parallel opportunities
        if len([s for s in steps if s["step"] == "data_loading"]) > 0:
            parallel_ops.append("parallel_data_loading")
        
        if any(s["step"] == "visualization" for s in steps):
            parallel_ops.append("parallel_chart_generation")
        
        return parallel_ops
    
    def _estimate_execution_cost(self, query_profile: QueryProfile, agent_id: str) -> float:
        """Estimate execution cost"""
        # Base cost factors
        complexity_multiplier = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 2.0,
            QueryComplexity.COMPLEX: 4.0,
            QueryComplexity.EXPERT: 8.0
        }
        
        base_cost = 0.01  # Base cost per query
        complexity_cost = base_cost * complexity_multiplier[query_profile.complexity]
        
        # Agent-specific cost (would be retrieved from agent profile)
        agent_cost_multiplier = 1.0  # Default
        
        return complexity_cost * agent_cost_multiplier
    
    def _determine_cache_strategy(self, query_profile: QueryProfile) -> str:
        """Determine optimal caching strategy"""
        if query_profile.complexity == QueryComplexity.SIMPLE:
            return "aggressive_cache"
        elif query_profile.complexity == QueryComplexity.MODERATE:
            return "moderate_cache"
        elif query_profile.expected_output_size > 5000:
            return "result_cache_only"
        else:
            return "minimal_cache"

class IntelligentQueryEngine:
    """Main query optimization engine coordinating all components"""
    
    def __init__(self):
        self.pattern_analyzer = QueryPatternAnalyzer()
        self.query_router = IntelligentQueryRouter()
        self.query_optimizer = QueryOptimizer()
        self.execution_planner = ExecutionPlanOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.cache_manager = get_enhanced_cache_manager()
        
        # Statistics
        self.query_stats = defaultdict(int)
        self.performance_history = []
        
    @enhanced_cached(ttl=3600, tags={"query_analysis"})
    async def analyze_and_optimize_query(self, query: str, 
                                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze and optimize query with comprehensive intelligence"""
        start_time = time.time()
        self.performance_monitor.start_timer("query_optimization")
        
        try:
            # Step 1: Analyze query patterns
            query_profile = self.pattern_analyzer.analyze_query(query)
            
            # Step 2: Optimize query text
            optimized_query, optimization_rules = self.query_optimizer.optimize_query(query)
            query_profile.query_text = optimized_query  # Update with optimized version
            
            # Step 3: Route to optimal agent
            selected_agent = self.query_router.route_query(query_profile)
            
            # Step 4: Create execution plan
            execution_plan = self.execution_planner.create_execution_plan(
                query_profile, selected_agent
            )
            execution_plan.optimization_rules_applied.extend(optimization_rules)
            
            # Step 5: Update statistics
            self.query_stats[query_profile.query_type] += 1
            self.query_stats["total_queries"] += 1
            
            processing_time = time.time() - start_time
            
            result = {
                "query_profile": {
                    "original_query": query,
                    "optimized_query": optimized_query,
                    "query_type": query_profile.query_type.value,
                    "complexity": query_profile.complexity.value,
                    "estimated_duration": query_profile.estimated_duration,
                    "required_capabilities": [cap.value for cap in query_profile.required_capabilities],
                    "priority": query_profile.priority
                },
                "routing": {
                    "selected_agent": selected_agent,
                    "routing_confidence": 0.85,  # Simplified
                },
                "execution_plan": {
                    "plan_id": execution_plan.query_id,
                    "steps": execution_plan.execution_steps,
                    "estimated_cost": execution_plan.estimated_cost,
                    "estimated_duration": execution_plan.estimated_duration,
                    "cache_strategy": execution_plan.cache_strategy,
                    "parallel_opportunities": execution_plan.parallel_opportunities,
                    "optimizations_applied": execution_plan.optimization_rules_applied
                },
                "performance": {
                    "analysis_time": processing_time,
                    "optimization_score": len(optimization_rules) * 0.1 + 0.5
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Query optimization error: {e}")
            raise
        finally:
            self.performance_monitor.end_timer("query_optimization")
    
    async def execute_optimized_query(self, optimization_result: Dict[str, Any], 
                                    llm_client: OptimizedLLMClient) -> Dict[str, Any]:
        """Execute query using optimization results"""
        execution_plan = optimization_result["execution_plan"]
        selected_agent = optimization_result["routing"]["selected_agent"]
        optimized_query = optimization_result["query_profile"]["optimized_query"]
        
        start_time = time.time()
        
        try:
            # Execute using optimized LLM client
            result = await llm_client.analyze_batch(
                [optimized_query], 
                model_name=self._get_model_for_agent(selected_agent)
            )
            
            execution_time = time.time() - start_time
            
            # Update agent performance
            performance_score = 1.0 if result else 0.0
            query_type = QueryType(optimization_result["query_profile"]["query_type"])
            
            self.query_router.update_agent_performance(
                selected_agent, query_type, performance_score, execution_time
            )
            
            return {
                "result": result[0] if result else None,
                "execution_metrics": {
                    "actual_duration": execution_time,
                    "estimated_duration": execution_plan["estimated_duration"],
                    "efficiency_ratio": execution_plan["estimated_duration"] / max(execution_time, 0.1),
                    "agent_used": selected_agent,
                    "cache_hit": False  # Would be determined by actual execution
                }
            }
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            # Update agent performance negatively
            self.query_router.update_agent_performance(
                selected_agent, 
                QueryType(optimization_result["query_profile"]["query_type"]), 
                0.0, 
                time.time() - start_time
            )
            raise
    
    def _get_model_for_agent(self, agent_id: str) -> str:
        """Get appropriate model for agent"""
        agent_model_map = {
            "data_analyst": "gpt-3.5-turbo",
            "ml_engineer": "claude-3",
            "financial_analyst": "gpt-4",
            "business_intelligence": "gpt-3.5-turbo",
            "text_analyst": "claude-3"
        }
        return agent_model_map.get(agent_id, "gpt-3.5-turbo")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        performance_stats = self.performance_monitor.get_performance_report()
        
        return {
            "query_distribution": dict(self.query_stats),
            "performance_metrics": performance_stats,
            "agent_performance": {
                agent_id: {
                    "avg_response_time": agent.avg_response_time,
                    "current_load": agent.current_load,
                    "success_rate": agent.success_rate
                }
                for agent_id, agent in self.query_router.agents.items()
            },
            "optimization_efficiency": {
                "avg_optimization_time": performance_stats.get("query_optimization", {}).get("average", 0.0),
                "cache_hit_rate": 0.75,  # Would be from cache manager
                "total_optimizations": self.query_stats.get("total_queries", 0)
            }
        }

# Global query engine instance
_query_engine = None
_engine_lock = threading.Lock()

def get_query_engine() -> IntelligentQueryEngine:
    """Get or create global query engine instance"""
    global _query_engine
    
    if _query_engine is None:
        with _engine_lock:
            if _query_engine is None:
                _query_engine = IntelligentQueryEngine()
    
    return _query_engine

# Utility functions
async def optimize_query(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Quick query optimization"""
    engine = get_query_engine()
    return await engine.analyze_and_optimize_query(query, context)

async def execute_intelligent_query(query: str, llm_client: OptimizedLLMClient) -> Dict[str, Any]:
    """Execute query with full intelligence pipeline"""
    engine = get_query_engine()
    
    # Optimize query
    optimization_result = await engine.analyze_and_optimize_query(query)
    
    # Execute optimized query
    execution_result = await engine.execute_optimized_query(optimization_result, llm_client)
    
    return {
        "optimization": optimization_result,
        "execution": execution_result
    }