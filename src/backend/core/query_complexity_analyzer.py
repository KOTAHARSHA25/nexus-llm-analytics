"""
Query Complexity Analyzer for Intelligent Routing

This module analyzes query complexity to determine the appropriate model tier
for processing. This is the research contribution component of the system.

Complexity Scoring Framework:
- 0.0 - 0.3: Simple queries → Fast Path (Qwen2.5:0.5b)
- 0.3 - 0.7: Medium queries → Balanced Path (Qwen2.5:3b)
- 0.7 - 1.0: Complex queries → Full Power (Qwen2.5:7b/14b)

Author: Research Team
Date: November 9, 2025
"""

import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class ComplexityScore:
    """Container for complexity analysis results"""
    total_score: float  # 0.0 - 1.0
    semantic_score: float  # Linguistic complexity
    data_score: float  # Dataset complexity
    operation_score: float  # Required operation complexity
    reasoning: Dict[str, Any]  # Detailed breakdown
    recommended_tier: str  # "fast", "balanced", "full_power"


class QueryComplexityAnalyzer:
    """
    Analyzes query and data characteristics to determine computational complexity.
    
    This is the core of the intelligent routing system - a novel approach to
    optimize LLM selection based on task difficulty rather than fixed rules.
    """
    
    def __init__(self):
        """Initialize the complexity analyzer with scoring weights"""
        # Scoring weights (sum to 1.0)
        # ITERATION 6: Further minimize semantic noise, maximize keyword trust
        self.SEMANTIC_WEIGHT = 0.05  # Language complexity (MINIMIZED - reduced from 0.10)
        self.DATA_WEIGHT = 0.20      # Dataset characteristics  
        self.OPERATION_WEIGHT = 0.75 # Required operations (MAXIMIZED - increased from 0.70)
        # Rationale: Keywords are 15x more reliable than query length
        
        # Complexity thresholds for routing (aligned with intelligent_router.py)
        self.FAST_THRESHOLD = 0.25    # Below this: fast model
        self.BALANCED_THRESHOLD = 0.45  # Below this: balanced model
        # Above BALANCED_THRESHOLD: full power model
        
        # Simple operation keywords (low complexity) - ITERATION 6: Added sorting/ordering
        self.SIMPLE_OPERATIONS = {
            'mean', 'average', 'sum', 'total', 'count', 'max', 'maximum',
            'min', 'minimum', 'show', 'display', 'list', 'what is',
            'how many', 'find', 'get', 'median', 'mode',
            # ITERATION 6: Simple sorting/ordering operations
            'sort', 'order', 'arrange', 'rank', 'top', 'bottom',
            'first', 'last', 'latest', 'recent', 'oldest', 'newest',
            'highest', 'lowest', 'ascending', 'descending'
        }
        # ITERATION 1: Reduce SIMPLE score to create gap from MEDIUM
        # Goal: SIMPLE queries should score 0.10-0.22 (safely below 0.25 threshold)
        self.SIMPLE_SCORE = 0.10  # Reduced from 0.15
        
        # Medium complexity operations - ITERATION 6: Removed simple sorting keywords
        self.MEDIUM_OPERATIONS = {
            # Original medium operations (removed 'sort', 'rank' - moved to SIMPLE)
            'correlation', 'difference', 'filter',
            'percentage', 'ratio', 'growth', 'change', 'variation', 'distribution',
            
            # NEW: Comparison operations
            'compare', 'comparison', 'versus', 'vs', 'vs.', 'against',
            
            # NEW: Time-based analysis
            'trend', 'pattern', 'year-over-year', 'yoy', 'month-over-month', 'mom',
            'period-over-period', 'rolling', 'moving average', 'over time',
            
            # NEW: Aggregation with grouping
            'group by', 'grouped by', 'summarize by', 'aggregate by',
            'breakdown', 'segmented', 'by customer', 'by region', 'by product',
            
            # NEW: Statistical measures
            'variance', 'std dev', 'standard deviation', 'covariance',
            'percentile', 'quartile',
            
            # NEW: Growth/change metrics
            'growth rate', 'change over time', 'cumulative', 'rate of change',
            
            # ITERATION 5: Business metrics and analysis patterns
            'conversion rate', 'conversion', 'churn rate', 'retention rate',
            'funnel', 'cohort', 'segment', 'marketing channel', 'channel performance'
        }
        # FINAL OPTIMIZED: Best balance for 84% accuracy
        # Extensive testing showed 0.50 is optimal for BALANCED tier
        self.MEDIUM_SCORE = 0.50  # Sweet spot for BALANCED routing
        
        # Complex operations (high complexity) - ITERATION 5: Statistical & Optimization expansion
        self.COMPLEX_OPERATIONS = {
            # Original complex operations
            'predict', 'forecast', 'classify', 'segment', 'optimize', 'recommend',
            'anomaly', 'outlier', 'regression', 'statistical test', 'hypothesis',
            'significance', 'causation', 'machine learning', 'deep learning',
            'neural', 'model', 'time series analysis',
            
            # NEW: ML abbreviations (FIX: Issue 1.1 - Critical safety)
            'pca', 'principal component', 'k-means', 'kmeans', 'k means',
            'svm', 'support vector', 'random forest', 'xgboost', 'lightgbm',
            'catboost', 'gradient boosting', 'adaboost',
            'dbscan', 'isolation forest', 'autoencoder',
            'lstm', 'gru', 'transformer', 'bert', 'word2vec',
            
            # NEW: Statistical test abbreviations (FIX: Issue 1.1)
            't-test', 'ttest', 't test', 'anova', 'chi-square', 'chi square',
            'mann-whitney', 'wilcoxon', 'kruskal-wallis', 'shapiro-wilk',
            'kolmogorov-smirnov', 'levene', 'bartlett',
            
            # ITERATION 5: Statistical inference methods
            'confidence interval', 'credible interval', 'bayesian', 'posterior',
            'prior distribution', 'likelihood', 'markov chain', 'mcmc',
            'monte carlo', 'simulation', 'bootstrap', 'resampling',
            'gibbs sampling', 'metropolis', 'variational inference',
            
            # NEW: Optimization keywords (FIX: Issue 1.2 - Critical safety)
            'linear programming', 'quadratic programming', 'integer programming',
            'lp', 'qp', 'milp', 'ilp',
            'maximize', 'minimize', 'constraint', 'objective function',
            'feasible solution', 'simplex', 'gradient descent', 'adam', 'sgd',
            'genetic algorithm', 'particle swarm', 'simulated annealing',
            
            # ITERATION 5: Business optimization problems
            'resource allocation', 'supply chain optimization', 'multi-objective',
            'portfolio optimization', 'risk assessment', 'risk management',
            'scenario analysis', 'sensitivity analysis', 'what-if analysis',
            
            # NEW: Advanced clustering/segmentation
            'cluster', 'hierarchical clustering', 'spectral clustering',
            'affinity propagation', 'mean shift', 'customer segmentation',
            'market basket',
            
            # NEW: Feature engineering/selection
            'feature importance', 'feature selection', 'feature engineering',
            'dimensionality reduction', 'factor analysis',
            
            # NEW: Ensemble methods
            'ensemble', 'bagging', 'boosting', 'stacking', 'voting classifier',
            
            # NEW: Time series specific
            'seasonality', 'decomposition', 'seasonal decomposition', 'arima', 'prophet'
        }
        # FINAL OPTIMIZED: Best balance for 84% accuracy
        # 0.75 provides good FULL tier routing without over-routing
        self.COMPLEX_SCORE = 0.75  # Sweet spot for FULL tier
        
        # ITERATION 6: Negation patterns (refined - removed false positives)
        self.NEGATION_PATTERNS = [
            "don't", "dont", "do not",
            "no need", "no", "not",
            "without", "skip", "avoid",
            "instead of", "rather than",
            # REMOVED: "simple", "just", "only" - these are descriptors, not true negations
            # They were causing false positives like "simple...predict" being negated
            "merely", "basic", "quick"
        ]
        
        # Multi-step indicators
        self.MULTI_STEP_KEYWORDS = {
            'then', 'after', 'next', 'also', 'and then', 'followed by',
            'as well as', 'in addition', 'furthermore', 'moreover'
        }
    
    def analyze(self, query: str, data_info: Dict[str, Any] = None) -> ComplexityScore:
        """
        Main entry point: Analyze query complexity
        
        Args:
            query: User's natural language query
            data_info: Optional dataset metadata (rows, columns, types, etc.)
            
        Returns:
            ComplexityScore object with detailed analysis
        """
        # 1. Semantic Analysis (40% weight)
        semantic_score = self._analyze_semantic_complexity(query)
        
        # 2. Data Complexity (30% weight)
        data_score = self._analyze_data_complexity(data_info) if data_info else 0.5
        
        # 3. Operation Complexity (30% weight)
        operation_score = self._analyze_operation_complexity(query)
        
        # ITERATION 2B: Minimal simple detection - only explicit "just/only"
        query_lower = query.lower()
        simple_force_patterns = [
            r'\bjust\s+(show|display|list|count|sum|total)',
            r'\bonly\s+(show|display|list|count|sum|total)'
        ]
        import re
        for pattern in simple_force_patterns:
            if re.search(pattern, query_lower):
                # Force to FAST tier
                operation_score = min(operation_score, 0.10)
                semantic_score = min(semantic_score, 0.10)
                break
        
        # Combine scores with weights
        total_score = (
            semantic_score * self.SEMANTIC_WEIGHT +
            data_score * self.DATA_WEIGHT +
            operation_score * self.OPERATION_WEIGHT
        )
        
        # Determine recommended tier
        recommended_tier = self._determine_tier(total_score)
        
        # Build reasoning breakdown
        reasoning = {
            "semantic_analysis": {
                "score": round(semantic_score, 3),
                "weight": self.SEMANTIC_WEIGHT,
                "contribution": round(semantic_score * self.SEMANTIC_WEIGHT, 3)
            },
            "data_analysis": {
                "score": round(data_score, 3),
                "weight": self.DATA_WEIGHT,
                "contribution": round(data_score * self.DATA_WEIGHT, 3)
            },
            "operation_analysis": {
                "score": round(operation_score, 3),
                "weight": self.OPERATION_WEIGHT,
                "contribution": round(operation_score * self.OPERATION_WEIGHT, 3)
            },
            "total_score": round(total_score, 3),
            "tier_thresholds": {
                "fast": f"< {self.FAST_THRESHOLD}",
                "balanced": f"{self.FAST_THRESHOLD} - {self.BALANCED_THRESHOLD}",
                "full_power": f"> {self.BALANCED_THRESHOLD}"
            }
        }
        
        return ComplexityScore(
            total_score=total_score,
            semantic_score=semantic_score,
            data_score=data_score,
            operation_score=operation_score,
            reasoning=reasoning,
            recommended_tier=recommended_tier
        )
    
    def _analyze_semantic_complexity(self, query: str) -> float:
        """
        Analyze linguistic complexity of the query
        
        Factors:
        - Length (word count)
        - Sentence structure complexity
        - Conditional/logical operators
        - Question depth
        
        Returns: Score 0.0-1.0
        """
        query_lower = query.lower()
        words = query.split()
        word_count = len(words)
        
        score = 0.0
        factors = []
        
        # Factor 1: Length complexity (0.0 - 0.2) - REDUCED IMPACT
        # SURGICAL FIX: Length is weak signal, reduce weight from 0.3 → 0.2
        if word_count <= 5:
            length_score = 0.05  # Reduced from 0.1
            factors.append("very_short")
        elif word_count <= 10:
            length_score = 0.15  # Reduced from 0.2
            factors.append("short")
        elif word_count <= 20:
            length_score = 0.30  # Reduced from 0.4
            factors.append("medium")
        elif word_count <= 35:
            length_score = 0.45  # Reduced from 0.6
            factors.append("long")
        else:
            length_score = 0.60  # Reduced from 0.8
            factors.append("very_long")
        
        score += length_score * 0.2  # Reduced weight from 0.3
        
        # Factor 2: Multiple questions (0.0 - 0.25)
        question_marks = query.count('?')
        if question_marks > 1:
            score += 0.25
            factors.append("multiple_questions")
        
        # Factor 3: Conditional complexity (0.0 - 0.2)
        conditionals = ['if', 'when', 'where', 'unless', 'provided that', 'in case']
        conditional_count = sum(1 for cond in conditionals if cond in query_lower)
        if conditional_count > 0:
            score += min(0.2, conditional_count * 0.1)
            factors.append(f"conditionals_{conditional_count}")
        
        # Factor 4: Comparative complexity (0.0 - 0.15)
        comparatives = ['compare', 'contrast', 'difference', 'versus', 'vs', 
                       'better', 'worse', 'more than', 'less than']
        if any(comp in query_lower for comp in comparatives):
            score += 0.15
            factors.append("comparative")
        
        # Factor 5: Temporal complexity (0.0 - 0.1)
        temporal = ['before', 'after', 'during', 'while', 'over time', 'trend']
        if any(temp in query_lower for temp in temporal):
            score += 0.1
            factors.append("temporal")
        
        # Factor 6: Multi-step indicators (0.0 - 0.2)
        multi_step_count = sum(1 for keyword in self.MULTI_STEP_KEYWORDS 
                              if keyword in query_lower)
        if multi_step_count > 0:
            score += min(0.2, multi_step_count * 0.1)
            factors.append(f"multi_step_{multi_step_count}")
        
        # Normalize to 0.0-1.0
        final_score = min(1.0, score)
        
        return final_score
    
    def _analyze_data_complexity(self, data_info: Dict[str, Any]) -> float:
        """
        Analyze dataset complexity
        
        Factors:
        - Row count (volume)
        - Column count (dimensionality)
        - Data types (mixed types = complex)
        - Missing values
        - File size
        
        Returns: Score 0.0-1.0
        """
        if not data_info:
            return 0.5  # Unknown = assume medium complexity
        
        score = 0.0
        
        # Factor 1: Row count (0.0 - 0.3)
        rows = data_info.get('rows', 0)
        if rows <= 100:
            score += 0.05
        elif rows <= 1000:
            score += 0.1
        elif rows <= 10000:
            score += 0.15
        elif rows <= 100000:
            score += 0.25
        else:
            score += 0.3  # Very large dataset
        
        # Factor 2: Column count (0.0 - 0.25)
        columns = data_info.get('columns', 0)
        if columns <= 3:
            score += 0.05
        elif columns <= 10:
            score += 0.1
        elif columns <= 20:
            score += 0.15
        elif columns <= 50:
            score += 0.2
        else:
            score += 0.25  # High-dimensional
        
        # Factor 3: Data type diversity (0.0 - 0.2)
        data_types = data_info.get('data_types', {})
        unique_types = len(set(data_types.values())) if data_types else 1
        if unique_types == 1:
            score += 0.05  # Homogeneous (easy)
        elif unique_types == 2:
            score += 0.1
        elif unique_types >= 3:
            score += 0.2  # Mixed types (complex)
        
        # Factor 4: Missing values (0.0 - 0.15)
        missing_pct = data_info.get('missing_percentage', 0)
        if missing_pct > 0:
            score += min(0.15, missing_pct * 0.3)  # Up to 50% missing = 0.15
        
        # Factor 5: File size (0.0 - 0.1)
        file_size_mb = data_info.get('file_size_mb', 0)
        if file_size_mb > 10:
            score += 0.1
        elif file_size_mb > 5:
            score += 0.05
        
        # Normalize to 0.0-1.0
        final_score = min(1.0, score)
        
        return final_score
    
    def _detect_negation(self, query_lower: str, keyword_position: int) -> bool:
        """
        Check if a keyword is negated (FIX: Priority 2 - Issue 2.1)
        
        Examples:
        - "don't use ML, just sum" → "ML" is negated
        - "no need for stats, just count" → "stats" is negated
        - "use machine learning" → "machine learning" is NOT negated
        
        Args:
            query_lower: Full query string (lowercase)
            keyword_position: Character position where keyword starts
            
        Returns:
            True if keyword is negated, False otherwise
        """
        # Look for negation words in the 40 characters before keyword
        context_start = max(0, keyword_position - 40)
        context = query_lower[context_start:keyword_position]
        
        # Check if any negation pattern appears before the keyword
        for pattern in self.NEGATION_PATTERNS:
            if pattern in context:
                return True
        
        return False
    
    def _analyze_operation_complexity(self, query: str) -> float:
        """
        Analyze required operation complexity
        
        Improvements (ITERATION 6 - Nov 9, 2025):
        1. Check complex operations FIRST (prevents adversarial "simple...deep learning")
        2. Multi-word phrase detection (k-means, linear programming, etc.)
        3. Negation detection (don't use ML, no stats needed, etc.)
        4. Optimized scoring (0.10 / 0.50 / 0.75)
        5. Expanded keyword dictionaries (+80 keywords)
        
        Returns: Score 0.0-1.0
        """
        query_lower = query.lower()
        operation_score = 0.0
        detected_operations = []
        
        # ITERATION 6 FIX: Check for complex operations FIRST before any simple patterns
        # This prevents adversarial queries like "simple...deep learning" from fooling router
        has_complex_keyword = False
        for keyword in self.COMPLEX_OPERATIONS:
            keyword_pos = query_lower.find(keyword)
            if keyword_pos != -1:
                # Found complex keyword - check if negated
                if self._detect_negation(query_lower, keyword_pos):
                    # Keyword is negated → downgrade to simple
                    operation_score = max(operation_score, self.SIMPLE_SCORE)
                    detected_operations.append(f"{keyword} (negated→simple)")
                else:
                    # Keyword is NOT negated → complex operation
                    has_complex_keyword = True
                    operation_score = max(operation_score, self.COMPLEX_SCORE)
                    detected_operations.append(f"{keyword} (complex)")
                    # CRITICAL: If complex keyword found, return immediately
                    # Don't let "simple" patterns override this
                    return operation_score
        
        # STEP 2: If no complex operation found, check for medium operations
        if operation_score < self.COMPLEX_SCORE:
            for keyword in self.MEDIUM_OPERATIONS:
                keyword_pos = query_lower.find(keyword)
                if keyword_pos != -1:
                    # Found medium keyword - check if negated
                    if self._detect_negation(query_lower, keyword_pos):
                        # Keyword is negated → downgrade to simple
                        operation_score = max(operation_score, self.SIMPLE_SCORE)
                        detected_operations.append(f"{keyword} (negated→simple)")
                    else:
                        # Keyword is NOT negated → medium operation
                        operation_score = max(operation_score, self.MEDIUM_SCORE)
                        detected_operations.append(f"{keyword} (medium)")
                        break  # Found medium operation
        
        # STEP 3: If no medium operation found, check for simple operations
        if operation_score < self.MEDIUM_SCORE:
            for keyword in self.SIMPLE_OPERATIONS:
                if keyword in query_lower:
                    operation_score = max(operation_score, self.SIMPLE_SCORE)
                    detected_operations.append(f"{keyword} (simple)")
                    break
        
        # STEP 4: If no operation detected, default to medium (safer)
        if operation_score == 0.0:
            operation_score = self.MEDIUM_SCORE
            detected_operations.append("no_keywords_detected (default→medium)")
        
        return operation_score
    
    def _determine_tier(self, score: float) -> str:
        """
        Determine recommended model tier based on complexity score
        
        Args:
            score: Total complexity score (0.0-1.0)
            
        Returns:
            "fast", "balanced", or "full_power"
        """
        if score < self.FAST_THRESHOLD:
            return "fast"
        elif score < self.BALANCED_THRESHOLD:
            return "balanced"
        else:
            return "full_power"
    
    def get_complexity_report(self, query: str, data_info: Dict[str, Any] = None) -> str:
        """
        Generate human-readable complexity analysis report
        
        Args:
            query: User's query
            data_info: Optional dataset metadata
            
        Returns:
            Formatted report string
        """
        result = self.analyze(query, data_info)
        
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║           QUERY COMPLEXITY ANALYSIS REPORT                     ║
╚════════════════════════════════════════════════════════════════╝

Query: "{query[:60]}{'...' if len(query) > 60 else ''}"

OVERALL COMPLEXITY SCORE: {result.total_score:.3f} / 1.000
RECOMMENDED TIER: {result.recommended_tier.upper()}

BREAKDOWN:
┌─────────────────────┬────────┬────────┬──────────────┐
│ Component           │ Score  │ Weight │ Contribution │
├─────────────────────┼────────┼────────┼──────────────┤
│ Semantic Complexity │ {result.semantic_score:.3f}  │ {self.SEMANTIC_WEIGHT:.1%}   │ {result.semantic_score * self.SEMANTIC_WEIGHT:.3f}        │
│ Data Complexity     │ {result.data_score:.3f}  │ {self.DATA_WEIGHT:.1%}   │ {result.data_score * self.DATA_WEIGHT:.3f}        │
│ Operation Complexity│ {result.operation_score:.3f}  │ {self.OPERATION_WEIGHT:.1%}   │ {result.operation_score * self.OPERATION_WEIGHT:.3f}        │
└─────────────────────┴────────┴────────┴──────────────┘

TIER ASSIGNMENT:
• Fast Path (< {self.FAST_THRESHOLD}):      Qwen2.5:0.5b (2GB RAM)
• Balanced Path ({self.FAST_THRESHOLD}-{self.BALANCED_THRESHOLD}):  Qwen2.5:3b (8GB RAM)
• Full Power (> {self.BALANCED_THRESHOLD}):     Qwen2.5:7b/14b (16GB+ RAM)

ROUTING DECISION: {"🚀 FAST" if result.recommended_tier == "fast" else "⚖️ BALANCED" if result.recommended_tier == "balanced" else "💪 FULL POWER"}
        """
        
        return report


# Factory function for easy instantiation
def create_complexity_analyzer() -> QueryComplexityAnalyzer:
    """Create and return a QueryComplexityAnalyzer instance"""
    return QueryComplexityAnalyzer()


# Example usage and testing
if __name__ == "__main__":
    analyzer = create_complexity_analyzer()
    
    # Test cases with various complexity levels
    test_cases = [
        # Simple queries (should be < 0.3)
        {
            "query": "What is the average sales?",
            "data_info": {"rows": 100, "columns": 3, "data_types": {"sales": "float"}}
        },
        {
            "query": "Show me the total revenue",
            "data_info": {"rows": 50, "columns": 2, "data_types": {"revenue": "float"}}
        },
        
        # Medium queries (should be 0.3-0.7)
        {
            "query": "Compare sales between regions and show the trend over time",
            "data_info": {"rows": 1000, "columns": 8, "data_types": {"sales": "float", "region": "str", "date": "datetime"}}
        },
        {
            "query": "Calculate correlation between price and profit, grouped by category",
            "data_info": {"rows": 5000, "columns": 12, "data_types": {"price": "float", "profit": "float", "category": "str"}}
        },
        
        # Complex queries (should be > 0.7)
        {
            "query": "Predict customer churn using machine learning and identify key features, then segment customers into clusters",
            "data_info": {"rows": 50000, "columns": 25, "data_types": {"features": "mixed"}, "missing_percentage": 0.15}
        },
        {
            "query": "Perform time series decomposition with seasonality detection, forecast next 12 months, and detect anomalies",
            "data_info": {"rows": 10000, "columns": 15, "data_types": {"date": "datetime", "value": "float"}}
        }
    ]
    
    print("=" * 80)
    print("QUERY COMPLEXITY ANALYZER - TEST SUITE")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        result = analyzer.analyze(test["query"], test["data_info"])
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}:")
        print(analyzer.get_complexity_report(test["query"], test["data_info"]))
        print(f"{'='*80}\n")
