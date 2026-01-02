"""
Hyperparameter Sensitivity Analysis Module
==========================================

This module provides comprehensive hyperparameter sensitivity analysis
for tuning and understanding the Nexus LLM Analytics system.

Features:
- Grid search analysis
- Random search with budget
- Sensitivity scoring
- Interaction effects analysis
- Optimal configuration discovery
"""

import random
import math
import json
import itertools
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter."""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    default_value: Any
    search_space: List[Any]  # Values to try
    description: str = ""
    

@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for one hyperparameter."""
    parameter: str
    sensitivity_score: float  # How much it affects performance
    best_value: Any
    worst_value: Any
    performance_range: Tuple[float, float]  # (min, max) performance
    variance_explained: float  # Portion of variance explained
    recommendations: List[str]


@dataclass
class GridSearchResult:
    """Result from grid search."""
    best_config: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    parameter_importance: Dict[str, float]
    search_time: float


@dataclass
class InteractionEffect:
    """Interaction effect between two hyperparameters."""
    param1: str
    param2: str
    interaction_strength: float
    best_combination: Tuple[Any, Any]
    effect_type: str  # 'synergistic', 'antagonistic', 'neutral'


class HyperparameterSpace:
    """
    Defines and manages the hyperparameter search space.
    """
    
    def __init__(self):
        """Initialize hyperparameter space with system defaults."""
        self.parameters: Dict[str, HyperparameterConfig] = {}
        self._initialize_default_space()
    
    def _initialize_default_space(self):
        """Initialize with default system hyperparameters."""
        # Semantic Cache parameters
        self.add_parameter(HyperparameterConfig(
            name="cache_similarity_threshold",
            param_type="continuous",
            default_value=0.85,
            search_space=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            description="Threshold for semantic similarity in cache lookup"
        ))
        
        self.add_parameter(HyperparameterConfig(
            name="cache_max_size",
            param_type="discrete",
            default_value=1000,
            search_space=[100, 500, 1000, 2000, 5000],
            description="Maximum number of cached responses"
        ))
        
        self.add_parameter(HyperparameterConfig(
            name="cache_ttl_hours",
            param_type="discrete",
            default_value=24,
            search_space=[1, 6, 12, 24, 48, 72],
            description="Cache time-to-live in hours"
        ))
        
        # RAG parameters
        self.add_parameter(HyperparameterConfig(
            name="rag_chunk_size",
            param_type="discrete",
            default_value=500,
            search_space=[200, 300, 500, 750, 1000],
            description="Size of document chunks for retrieval"
        ))
        
        self.add_parameter(HyperparameterConfig(
            name="rag_chunk_overlap",
            param_type="discrete",
            default_value=50,
            search_space=[0, 25, 50, 100, 150],
            description="Overlap between adjacent chunks"
        ))
        
        self.add_parameter(HyperparameterConfig(
            name="rag_top_k",
            param_type="discrete",
            default_value=5,
            search_space=[1, 3, 5, 7, 10],
            description="Number of chunks to retrieve"
        ))
        
        self.add_parameter(HyperparameterConfig(
            name="rag_rerank_enabled",
            param_type="categorical",
            default_value=True,
            search_space=[True, False],
            description="Whether to use reranking"
        ))
        
        # Routing parameters
        self.add_parameter(HyperparameterConfig(
            name="routing_confidence_threshold",
            param_type="continuous",
            default_value=0.7,
            search_space=[0.5, 0.6, 0.7, 0.8, 0.9],
            description="Confidence threshold for model routing"
        ))
        
        self.add_parameter(HyperparameterConfig(
            name="routing_complexity_weight",
            param_type="continuous",
            default_value=0.5,
            search_space=[0.2, 0.3, 0.5, 0.7, 0.8],
            description="Weight for complexity in routing decision"
        ))
        
        # Two Friends Review parameters
        self.add_parameter(HyperparameterConfig(
            name="review_enabled",
            param_type="categorical",
            default_value=True,
            search_space=[True, False],
            description="Whether to enable Two Friends review"
        ))
        
        self.add_parameter(HyperparameterConfig(
            name="review_threshold",
            param_type="continuous",
            default_value=0.8,
            search_space=[0.6, 0.7, 0.8, 0.9, 0.95],
            description="Quality threshold for review pass"
        ))
        
        # Model parameters
        self.add_parameter(HyperparameterConfig(
            name="temperature",
            param_type="continuous",
            default_value=0.7,
            search_space=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
            description="Model temperature for generation"
        ))
        
        self.add_parameter(HyperparameterConfig(
            name="max_tokens",
            param_type="discrete",
            default_value=2000,
            search_space=[500, 1000, 2000, 4000],
            description="Maximum tokens in response"
        ))
    
    def add_parameter(self, config: HyperparameterConfig):
        """Add a hyperparameter to the search space."""
        self.parameters[config.name] = config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {name: param.default_value for name, param in self.parameters.items()}
    
    def get_search_space(self) -> Dict[str, List[Any]]:
        """Get complete search space."""
        return {name: param.search_space for name, param in self.parameters.items()}
    
    def sample_random_config(self) -> Dict[str, Any]:
        """Sample a random configuration."""
        config = {}
        for name, param in self.parameters.items():
            config[name] = random.choice(param.search_space)
        return config
    
    def get_grid_configs(self, subset: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate all grid configurations.
        
        Args:
            subset: Optional subset of parameters to include
            
        Returns:
            List of all configuration combinations
        """
        if subset:
            params = {k: v for k, v in self.parameters.items() if k in subset}
        else:
            params = self.parameters
        
        # Generate all combinations
        keys = list(params.keys())
        values = [params[k].search_space for k in keys]
        
        configs = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)
        
        return configs


class SensitivityAnalyzer:
    """
    Analyzes sensitivity of performance to hyperparameter changes.
    """
    
    def __init__(self, hyperparameter_space: HyperparameterSpace):
        """
        Initialize sensitivity analyzer.
        
        Args:
            hyperparameter_space: The hyperparameter space to analyze
        """
        self.space = hyperparameter_space
        self.results_cache: Dict[str, float] = {}
    
    def analyze_sensitivity(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        n_samples: int = 50
    ) -> List[SensitivityResult]:
        """
        Analyze sensitivity of each hyperparameter.
        
        Args:
            evaluate_fn: Function that takes config and returns score
            n_samples: Number of random samples per parameter
            
        Returns:
            List of sensitivity results for each parameter
        """
        results = []
        base_config = self.space.get_default_config()
        base_score = evaluate_fn(base_config)
        
        for param_name, param_config in self.space.parameters.items():
            scores = []
            
            # Evaluate each value in search space
            for value in param_config.search_space:
                test_config = base_config.copy()
                test_config[param_name] = value
                
                score = evaluate_fn(test_config)
                scores.append((value, score))
            
            # Calculate sensitivity metrics
            score_values = [s[1] for s in scores]
            min_score = min(score_values)
            max_score = max(score_values)
            score_range = max_score - min_score
            
            # Variance explained (normalized by total variance)
            mean_score = sum(score_values) / len(score_values)
            variance = sum((s - mean_score) ** 2 for s in score_values) / len(score_values)
            
            # Sensitivity score (0-1 scale based on range)
            sensitivity = score_range  # Simple range-based sensitivity
            
            # Find best and worst values
            best_value = max(scores, key=lambda x: x[1])[0]
            worst_value = min(scores, key=lambda x: x[1])[0]
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                param_name, param_config, scores, sensitivity
            )
            
            results.append(SensitivityResult(
                parameter=param_name,
                sensitivity_score=sensitivity,
                best_value=best_value,
                worst_value=worst_value,
                performance_range=(min_score, max_score),
                variance_explained=variance,
                recommendations=recommendations
            ))
        
        # Sort by sensitivity
        results.sort(key=lambda x: x.sensitivity_score, reverse=True)
        
        return results
    
    def _generate_recommendations(
        self,
        param_name: str,
        param_config: HyperparameterConfig,
        scores: List[Tuple[Any, float]],
        sensitivity: float
    ) -> List[str]:
        """Generate recommendations based on sensitivity analysis."""
        recommendations = []
        
        if sensitivity > 0.2:
            recommendations.append(f"⚠️ High sensitivity - carefully tune {param_name}")
        elif sensitivity < 0.05:
            recommendations.append(f"✅ Low sensitivity - default value is acceptable")
        
        # Find optimal value
        best_value = max(scores, key=lambda x: x[1])[0]
        if best_value != param_config.default_value:
            recommendations.append(
                f"💡 Consider changing default from {param_config.default_value} to {best_value}"
            )
        
        return recommendations


class GridSearchAnalyzer:
    """
    Performs grid search optimization and analysis.
    """
    
    def __init__(self, hyperparameter_space: HyperparameterSpace):
        """
        Initialize grid search analyzer.
        
        Args:
            hyperparameter_space: The hyperparameter space to search
        """
        self.space = hyperparameter_space
    
    def run_grid_search(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        parameter_subset: Optional[List[str]] = None,
        max_configs: Optional[int] = None
    ) -> GridSearchResult:
        """
        Run grid search over hyperparameter space.
        
        Args:
            evaluate_fn: Function that takes config and returns score
            parameter_subset: Optional subset of parameters to search
            max_configs: Maximum number of configurations to try
            
        Returns:
            Grid search results
        """
        import time
        start_time = time.time()
        
        # Generate configurations
        configs = self.space.get_grid_configs(parameter_subset)
        
        if max_configs and len(configs) > max_configs:
            random.shuffle(configs)
            configs = configs[:max_configs]
        
        # Evaluate each configuration
        all_results = []
        best_score = float('-inf')
        best_config = None
        
        for config in configs:
            score = evaluate_fn(config)
            all_results.append({
                'config': config,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_config = config
        
        # Calculate parameter importance
        parameter_importance = self._calculate_importance(all_results)
        
        search_time = time.time() - start_time
        
        return GridSearchResult(
            best_config=best_config,
            best_score=best_score,
            all_results=all_results,
            parameter_importance=parameter_importance,
            search_time=search_time
        )
    
    def _calculate_importance(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate parameter importance from grid search results."""
        if not results:
            return {}
        
        importance = {}
        
        # Get all parameters
        params = list(results[0]['config'].keys())
        
        for param in params:
            # Group by parameter value
            value_scores: Dict[Any, List[float]] = {}
            
            for result in results:
                value = result['config'][param]
                # Convert to string for hashable key
                value_key = str(value)
                if value_key not in value_scores:
                    value_scores[value_key] = []
                value_scores[value_key].append(result['score'])
            
            # Calculate variance between groups
            group_means = []
            for scores in value_scores.values():
                if scores:
                    group_means.append(sum(scores) / len(scores))
            
            if len(group_means) > 1:
                overall_mean = sum(group_means) / len(group_means)
                variance = sum((m - overall_mean) ** 2 for m in group_means) / len(group_means)
                importance[param] = variance
            else:
                importance[param] = 0.0
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance


class InteractionAnalyzer:
    """
    Analyzes interaction effects between hyperparameters.
    """
    
    def __init__(self, hyperparameter_space: HyperparameterSpace):
        """
        Initialize interaction analyzer.
        
        Args:
            hyperparameter_space: The hyperparameter space to analyze
        """
        self.space = hyperparameter_space
    
    def analyze_interactions(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        parameter_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> List[InteractionEffect]:
        """
        Analyze interaction effects between parameter pairs.
        
        Args:
            evaluate_fn: Function that takes config and returns score
            parameter_pairs: Optional specific pairs to analyze
            
        Returns:
            List of interaction effects
        """
        base_config = self.space.get_default_config()
        params = list(self.space.parameters.keys())
        
        if parameter_pairs is None:
            # Generate all pairs
            parameter_pairs = list(itertools.combinations(params, 2))
        
        interactions = []
        
        for param1, param2 in parameter_pairs:
            if param1 not in self.space.parameters or param2 not in self.space.parameters:
                continue
            
            space1 = self.space.parameters[param1].search_space
            space2 = self.space.parameters[param2].search_space
            
            # Evaluate all combinations
            scores = {}
            for v1 in space1:
                for v2 in space2:
                    config = base_config.copy()
                    config[param1] = v1
                    config[param2] = v2
                    
                    score = evaluate_fn(config)
                    scores[(str(v1), str(v2))] = score
            
            # Analyze interaction
            interaction_strength = self._calculate_interaction_strength(scores, space1, space2)
            best_combo = max(scores, key=scores.get)
            
            # Determine effect type
            if interaction_strength > 0.1:
                effect_type = 'synergistic'
            elif interaction_strength < -0.1:
                effect_type = 'antagonistic'
            else:
                effect_type = 'neutral'
            
            interactions.append(InteractionEffect(
                param1=param1,
                param2=param2,
                interaction_strength=abs(interaction_strength),
                best_combination=best_combo,
                effect_type=effect_type
            ))
        
        # Sort by interaction strength
        interactions.sort(key=lambda x: x.interaction_strength, reverse=True)
        
        return interactions
    
    def _calculate_interaction_strength(
        self,
        scores: Dict[Tuple[str, str], float],
        space1: List[Any],
        space2: List[Any]
    ) -> float:
        """Calculate interaction strength between two parameters."""
        # Calculate expected additive score vs actual
        # Strong interactions show non-additive effects
        
        # Calculate marginal effects
        marginal1 = {}
        for v1 in space1:
            v1_key = str(v1)
            v1_scores = [s for (k1, k2), s in scores.items() if k1 == v1_key]
            if v1_scores:
                marginal1[v1_key] = sum(v1_scores) / len(v1_scores)
        
        marginal2 = {}
        for v2 in space2:
            v2_key = str(v2)
            v2_scores = [s for (k1, k2), s in scores.items() if k2 == v2_key]
            if v2_scores:
                marginal2[v2_key] = sum(v2_scores) / len(v2_scores)
        
        # Calculate interaction as deviation from additivity
        if not marginal1 or not marginal2:
            return 0.0
        
        overall_mean = sum(scores.values()) / len(scores)
        
        deviations = []
        for (v1, v2), actual in scores.items():
            expected = marginal1.get(v1, overall_mean) + marginal2.get(v2, overall_mean) - overall_mean
            deviations.append((actual - expected) ** 2)
        
        return math.sqrt(sum(deviations) / len(deviations)) if deviations else 0.0


def run_hyperparameter_analysis(
    evaluate_fn: Callable[[Dict[str, Any]], float],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive hyperparameter analysis.
    
    Args:
        evaluate_fn: Function that takes config and returns score
        output_path: Optional path to save results
        
    Returns:
        Complete hyperparameter analysis results
    """
    print("\n" + "=" * 70)
    print("  HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    space = HyperparameterSpace()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'default_config': space.get_default_config(),
        'analyses': {}
    }
    
    # 1. Sensitivity Analysis
    print("\n📊 Running sensitivity analysis...")
    sensitivity_analyzer = SensitivityAnalyzer(space)
    sensitivity_results = sensitivity_analyzer.analyze_sensitivity(evaluate_fn)
    
    results['analyses']['sensitivity'] = [
        {
            'parameter': r.parameter,
            'sensitivity_score': r.sensitivity_score,
            'best_value': r.best_value,
            'worst_value': r.worst_value,
            'performance_range': r.performance_range,
            'recommendations': r.recommendations
        }
        for r in sensitivity_results
    ]
    
    print("   Top 5 most sensitive parameters:")
    for r in sensitivity_results[:5]:
        print(f"   - {r.parameter}: {r.sensitivity_score:.4f} (best={r.best_value})")
    
    # 2. Grid Search (limited)
    print("\n🔍 Running limited grid search...")
    grid_analyzer = GridSearchAnalyzer(space)
    
    # Select top 3 sensitive parameters for focused grid search
    top_params = [r.parameter for r in sensitivity_results[:3]]
    grid_results = grid_analyzer.run_grid_search(
        evaluate_fn, 
        parameter_subset=top_params,
        max_configs=50
    )
    
    results['analyses']['grid_search'] = {
        'best_config': grid_results.best_config,
        'best_score': grid_results.best_score,
        'search_time': grid_results.search_time,
        'parameter_importance': grid_results.parameter_importance,
        'n_configs_evaluated': len(grid_results.all_results)
    }
    
    print(f"   Best score: {grid_results.best_score:.4f}")
    print(f"   Evaluated {len(grid_results.all_results)} configurations in {grid_results.search_time:.2f}s")
    
    # 3. Interaction Analysis (top pairs)
    print("\n🔗 Analyzing parameter interactions...")
    interaction_analyzer = InteractionAnalyzer(space)
    
    # Analyze interactions between top 4 parameters
    top4_pairs = list(itertools.combinations([r.parameter for r in sensitivity_results[:4]], 2))
    interactions = interaction_analyzer.analyze_interactions(evaluate_fn, top4_pairs)
    
    results['analyses']['interactions'] = [
        {
            'param1': i.param1,
            'param2': i.param2,
            'strength': i.interaction_strength,
            'best_combination': i.best_combination,
            'effect_type': i.effect_type
        }
        for i in interactions
    ]
    
    print("   Significant interactions found:")
    for i in interactions[:3]:
        if i.interaction_strength > 0.01:
            print(f"   - {i.param1} × {i.param2}: {i.interaction_strength:.4f} ({i.effect_type})")
    
    # Summary
    print("\n" + "-" * 70)
    print("HYPERPARAMETER ANALYSIS SUMMARY")
    print("-" * 70)
    print(f"  Parameters analyzed: {len(space.parameters)}")
    print(f"  Most sensitive: {sensitivity_results[0].parameter}")
    print(f"  Best configuration score: {grid_results.best_score:.4f}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    for r in sensitivity_results[:3]:
        for rec in r.recommendations:
            print(f"   {rec}")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_path}")
    
    return results


# Example usage
if __name__ == "__main__":
    import random
    
    # Mock evaluation function
    def mock_evaluate(config: Dict[str, Any]) -> float:
        """Mock evaluation - in reality would run full benchmark."""
        score = 0.7
        
        # Simulate parameter effects
        score += 0.1 * (config.get('cache_similarity_threshold', 0.85) - 0.85)
        score += 0.05 * (config.get('rag_top_k', 5) - 5) / 5
        score += 0.08 * (1 if config.get('review_enabled', True) else 0)
        score += 0.03 * (0.7 - abs(config.get('temperature', 0.7) - 0.7))
        
        # Add noise
        score += random.gauss(0, 0.02)
        
        return max(0, min(1, score))
    
    results = run_hyperparameter_analysis(mock_evaluate, 'benchmarks/results/hyperparameter_analysis.json')
    print("\n✅ Hyperparameter analysis complete!")
