"""
Cross-Validation and Rigorous Evaluation Metrics
=================================================

This module provides K-fold cross-validation and stratified evaluation
for research-grade benchmarking of the Nexus LLM Analytics system.

Features:
- K-fold cross-validation with stratification
- Leave-one-out cross-validation
- Bootstrap resampling validation
- Learning curve analysis
- Variance-bias tradeoff analysis
"""

import random
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CVFold:
    """Represents a single cross-validation fold."""
    fold_index: int
    train_indices: List[int]
    test_indices: List[int]
    train_size: int
    test_size: int


@dataclass
class CVResult:
    """Results from a single cross-validation fold."""
    fold_index: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    predictions: List[Any]
    ground_truth: List[Any]


@dataclass
class CrossValidationResults:
    """Complete cross-validation results."""
    n_folds: int
    fold_results: List[CVResult]
    aggregated_metrics: Dict[str, Dict[str, float]]
    variance_analysis: Dict[str, float]
    stability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class KFoldCrossValidator:
    """
    K-Fold Cross-Validation for LLM benchmark evaluation.
    
    Provides stratified splitting based on domain, complexity, or other
    attributes to ensure balanced evaluation across different query types.
    """
    
    def __init__(self, n_folds: int = 5, shuffle: bool = True, random_seed: int = 42):
        """
        Initialize K-fold cross-validator.
        
        Args:
            n_folds: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_seed: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def create_folds(self, data: List[Any]) -> List[CVFold]:
        """
        Create K folds from the dataset.
        
        Args:
            data: List of data points to split
            
        Returns:
            List of CVFold objects describing each fold
        """
        n = len(data)
        if n < self.n_folds:
            raise ValueError(f"Dataset size ({n}) is smaller than n_folds ({self.n_folds})")
        
        indices = list(range(n))
        if self.shuffle:
            random.shuffle(indices)
        
        fold_size = n // self.n_folds
        remainder = n % self.n_folds
        
        folds = []
        start = 0
        
        for i in range(self.n_folds):
            # Distribute remainder across first folds
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end = start + current_fold_size
            
            test_indices = indices[start:end]
            train_indices = indices[:start] + indices[end:]
            
            folds.append(CVFold(
                fold_index=i,
                train_indices=train_indices,
                test_indices=test_indices,
                train_size=len(train_indices),
                test_size=len(test_indices)
            ))
            
            start = end
        
        return folds
    
    def create_stratified_folds(
        self,
        data: List[Any],
        stratify_key: Callable[[Any], str]
    ) -> List[CVFold]:
        """
        Create stratified K folds maintaining class distribution.
        
        Args:
            data: List of data points to split
            stratify_key: Function to extract stratification key from each item
            
        Returns:
            List of CVFold objects with stratified splits
        """
        # Group indices by stratification key
        groups: Dict[str, List[int]] = {}
        for i, item in enumerate(data):
            key = stratify_key(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        # Shuffle within each group
        if self.shuffle:
            for key in groups:
                random.shuffle(groups[key])
        
        # Initialize folds
        folds_indices = [[] for _ in range(self.n_folds)]
        
        # Distribute each group across folds proportionally
        for key, indices in groups.items():
            n = len(indices)
            fold_size = n // self.n_folds
            remainder = n % self.n_folds
            
            start = 0
            for i in range(self.n_folds):
                current_size = fold_size + (1 if i < remainder else 0)
                folds_indices[i].extend(indices[start:start + current_size])
                start += current_size
        
        # Create CVFold objects
        folds = []
        all_indices = list(range(len(data)))
        
        for i, test_indices in enumerate(folds_indices):
            test_set = set(test_indices)
            train_indices = [idx for idx in all_indices if idx not in test_set]
            
            folds.append(CVFold(
                fold_index=i,
                train_indices=train_indices,
                test_indices=test_indices,
                train_size=len(train_indices),
                test_size=len(test_indices)
            ))
        
        return folds
    
    def run_cross_validation(
        self,
        data: List[Any],
        evaluate_fn: Callable[[List[Any], List[Any]], Tuple[Dict[str, float], List[Any]]],
        stratify_key: Optional[Callable[[Any], str]] = None
    ) -> CrossValidationResults:
        """
        Run complete K-fold cross-validation.
        
        Args:
            data: Complete dataset
            evaluate_fn: Function that takes (train_data, test_data) and returns
                        (metrics_dict, predictions)
            stratify_key: Optional stratification function
            
        Returns:
            CrossValidationResults with aggregated statistics
        """
        # Create folds
        if stratify_key:
            folds = self.create_stratified_folds(data, stratify_key)
        else:
            folds = self.create_folds(data)
        
        # Run each fold
        fold_results = []
        all_test_metrics: Dict[str, List[float]] = {}
        
        for fold in folds:
            train_data = [data[i] for i in fold.train_indices]
            test_data = [data[i] for i in fold.test_indices]
            
            # Evaluate
            metrics, predictions = evaluate_fn(train_data, test_data)
            
            # Extract ground truth if available
            ground_truth = []
            for item in test_data:
                if isinstance(item, dict) and 'expected' in item:
                    ground_truth.append(item['expected'])
            
            fold_results.append(CVResult(
                fold_index=fold.fold_index,
                train_metrics={},  # Could be populated if evaluate_fn returns both
                test_metrics=metrics,
                predictions=predictions,
                ground_truth=ground_truth
            ))
            
            # Collect metrics for aggregation
            for key, value in metrics.items():
                if key not in all_test_metrics:
                    all_test_metrics[key] = []
                all_test_metrics[key].append(value)
        
        # Aggregate metrics
        aggregated = {}
        variance_analysis = {}
        
        for key, values in all_test_metrics.items():
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            std_val = math.sqrt(variance)
            
            aggregated[key] = {
                'mean': mean_val,
                'std': std_val,
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values),
                'cv': std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
            }
            
            variance_analysis[key] = variance
        
        # Calculate stability score (1 - average CV across metrics)
        cvs = [stats['cv'] for stats in aggregated.values() if stats['mean'] > 0]
        stability_score = 1 - (sum(cvs) / len(cvs)) if cvs else 0
        
        return CrossValidationResults(
            n_folds=self.n_folds,
            fold_results=fold_results,
            aggregated_metrics=aggregated,
            variance_analysis=variance_analysis,
            stability_score=max(0, min(1, stability_score)),
            metadata={
                'timestamp': datetime.now().isoformat(),
                'random_seed': self.random_seed,
                'shuffle': self.shuffle,
                'stratified': stratify_key is not None
            }
        )


class LeaveOneOutValidator:
    """
    Leave-One-Out Cross-Validation for thorough evaluation.
    
    Each data point is used exactly once as test data.
    Provides the most thorough but computationally expensive evaluation.
    """
    
    def run_loo_validation(
        self,
        data: List[Any],
        evaluate_single_fn: Callable[[List[Any], Any], Tuple[float, Any]]
    ) -> Dict[str, Any]:
        """
        Run leave-one-out cross-validation.
        
        Args:
            data: Complete dataset
            evaluate_single_fn: Function that takes (train_data, test_item) 
                               and returns (score, prediction)
            
        Returns:
            Dictionary with LOO results and statistics
        """
        n = len(data)
        scores = []
        predictions = []
        
        for i in range(n):
            # Leave one out
            train_data = data[:i] + data[i+1:]
            test_item = data[i]
            
            score, prediction = evaluate_single_fn(train_data, test_item)
            scores.append(score)
            predictions.append(prediction)
        
        mean_score = sum(scores) / n
        variance = sum((s - mean_score) ** 2 for s in scores) / n
        std_score = math.sqrt(variance)
        
        return {
            'method': 'leave-one-out',
            'n_samples': n,
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': min(scores),
            'max_score': max(scores),
            'all_scores': scores,
            'predictions': predictions,
            'stability': 1 - (std_score / mean_score) if mean_score > 0 else 0
        }


class BootstrapValidator:
    """
    Bootstrap resampling for confidence interval estimation.
    
    Uses random sampling with replacement to estimate
    the uncertainty in evaluation metrics.
    """
    
    def __init__(self, n_iterations: int = 1000, sample_ratio: float = 1.0, 
                 random_seed: int = 42):
        """
        Initialize bootstrap validator.
        
        Args:
            n_iterations: Number of bootstrap iterations
            sample_ratio: Ratio of original dataset size to sample
            random_seed: Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.sample_ratio = sample_ratio
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def run_bootstrap(
        self,
        data: List[Any],
        evaluate_fn: Callable[[List[Any]], Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Run bootstrap resampling validation.
        
        Args:
            data: Complete dataset
            evaluate_fn: Function that evaluates a dataset and returns metrics
            
        Returns:
            Dictionary with bootstrap statistics and confidence intervals
        """
        n = len(data)
        sample_size = int(n * self.sample_ratio)
        
        all_metrics: Dict[str, List[float]] = {}
        
        for _ in range(self.n_iterations):
            # Sample with replacement
            sample_indices = [random.randint(0, n - 1) for _ in range(sample_size)]
            sample = [data[i] for i in sample_indices]
            
            # Evaluate
            metrics = evaluate_fn(sample)
            
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Calculate statistics for each metric
        results = {
            'method': 'bootstrap',
            'n_iterations': self.n_iterations,
            'sample_ratio': self.sample_ratio,
            'metrics': {}
        }
        
        for key, values in all_metrics.items():
            sorted_values = sorted(values)
            n_vals = len(sorted_values)
            
            mean_val = sum(values) / n_vals
            variance = sum((v - mean_val) ** 2 for v in values) / n_vals
            std_val = math.sqrt(variance)
            
            # Confidence intervals
            ci_95_low = sorted_values[int(0.025 * n_vals)]
            ci_95_high = sorted_values[int(0.975 * n_vals)]
            ci_99_low = sorted_values[int(0.005 * n_vals)]
            ci_99_high = sorted_values[int(0.995 * n_vals)]
            
            results['metrics'][key] = {
                'mean': mean_val,
                'std': std_val,
                'ci_95': [ci_95_low, ci_95_high],
                'ci_99': [ci_99_low, ci_99_high],
                'min': min(values),
                'max': max(values)
            }
        
        return results


class LearningCurveAnalyzer:
    """
    Analyzes learning curves to understand sample efficiency.
    
    Evaluates performance across different training set sizes
    to understand how the system scales with data.
    """
    
    def __init__(self, train_sizes: Optional[List[float]] = None, 
                 n_iterations: int = 5, random_seed: int = 42):
        """
        Initialize learning curve analyzer.
        
        Args:
            train_sizes: List of training set size ratios (0.0 to 1.0)
            n_iterations: Number of random iterations per size
            random_seed: Random seed for reproducibility
        """
        self.train_sizes = train_sizes or [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def analyze_learning_curve(
        self,
        data: List[Any],
        evaluate_fn: Callable[[List[Any], List[Any]], Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Analyze learning curve across different training sizes.
        
        Args:
            data: Complete dataset
            evaluate_fn: Function that evaluates (train_data, test_data) -> metrics
            
        Returns:
            Learning curve analysis results
        """
        n = len(data)
        test_size = int(n * 0.2)  # Fixed 20% test set
        
        results = {
            'train_sizes': [],
            'train_scores': {'mean': [], 'std': []},
            'test_scores': {'mean': [], 'std': []},
            'sample_efficiency': None
        }
        
        for size_ratio in self.train_sizes:
            train_size = int((n - test_size) * size_ratio)
            if train_size < 1:
                continue
            
            iteration_train_scores = []
            iteration_test_scores = []
            
            for _ in range(self.n_iterations):
                # Random split
                indices = list(range(n))
                random.shuffle(indices)
                
                test_indices = indices[:test_size]
                available_train = indices[test_size:]
                train_indices = available_train[:train_size]
                
                train_data = [data[i] for i in train_indices]
                test_data = [data[i] for i in test_indices]
                
                metrics = evaluate_fn(train_data, test_data)
                
                # Assume 'score' is the primary metric
                score = metrics.get('score', metrics.get('quality_score', 0))
                iteration_test_scores.append(score)
            
            # Aggregate results
            results['train_sizes'].append(train_size)
            
            mean_test = sum(iteration_test_scores) / len(iteration_test_scores)
            var_test = sum((s - mean_test) ** 2 for s in iteration_test_scores) / len(iteration_test_scores)
            
            results['test_scores']['mean'].append(mean_test)
            results['test_scores']['std'].append(math.sqrt(var_test))
        
        # Calculate sample efficiency (improvement per sample)
        if len(results['train_sizes']) >= 2:
            score_diff = results['test_scores']['mean'][-1] - results['test_scores']['mean'][0]
            size_diff = results['train_sizes'][-1] - results['train_sizes'][0]
            results['sample_efficiency'] = score_diff / size_diff if size_diff > 0 else 0
        
        return results


class VarianceBiasAnalyzer:
    """
    Analyzes the variance-bias tradeoff for model evaluation.
    
    Provides insights into model stability and generalization.
    """
    
    def __init__(self, n_bootstrap: int = 100, random_seed: int = 42):
        """
        Initialize variance-bias analyzer.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def analyze(
        self,
        data: List[Any],
        predict_fn: Callable[[List[Any], Any], float],
        true_values: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Perform variance-bias analysis.
        
        Args:
            data: Training dataset
            predict_fn: Function to make prediction given train data and test item
            true_values: Optional ground truth values for bias calculation
            
        Returns:
            Dictionary with variance, bias, and total error estimates
        """
        n = len(data)
        all_predictions: List[List[float]] = [[] for _ in range(n)]
        
        # Generate bootstrap predictions
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            sample_indices = [random.randint(0, n - 1) for _ in range(n)]
            sample = [data[i] for i in sample_indices]
            
            # Out-of-bag indices
            oob_indices = set(range(n)) - set(sample_indices)
            
            # Predict for out-of-bag samples
            for idx in oob_indices:
                pred = predict_fn(sample, data[idx])
                all_predictions[idx].append(pred)
        
        # Calculate variance and bias for each sample
        variances = []
        biases = []
        
        for i in range(n):
            if len(all_predictions[i]) < 2:
                continue
            
            preds = all_predictions[i]
            mean_pred = sum(preds) / len(preds)
            
            # Variance
            variance = sum((p - mean_pred) ** 2 for p in preds) / len(preds)
            variances.append(variance)
            
            # Bias (if true values available)
            if true_values and i < len(true_values):
                bias = (mean_pred - true_values[i]) ** 2
                biases.append(bias)
        
        avg_variance = sum(variances) / len(variances) if variances else 0
        avg_bias = sum(biases) / len(biases) if biases else 0
        
        return {
            'average_variance': avg_variance,
            'average_bias_squared': avg_bias,
            'total_error': avg_variance + avg_bias,
            'variance_ratio': avg_variance / (avg_variance + avg_bias) if (avg_variance + avg_bias) > 0 else 0,
            'bias_ratio': avg_bias / (avg_variance + avg_bias) if (avg_variance + avg_bias) > 0 else 0,
            'n_samples_analyzed': len(variances),
            'stability_indicator': 1 / (1 + avg_variance) if avg_variance >= 0 else 0
        }


def run_comprehensive_cv_analysis(
    dataset: List[Dict[str, Any]],
    evaluate_fn: Callable,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive cross-validation analysis on benchmark dataset.
    
    Args:
        dataset: Benchmark dataset
        evaluate_fn: Evaluation function
        output_path: Optional path to save results
        
    Returns:
        Complete cross-validation analysis results
    """
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE CROSS-VALIDATION ANALYSIS")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(dataset),
        'analyses': {}
    }
    
    # 1. K-Fold Cross-Validation
    print("\n📊 Running 5-fold cross-validation...")
    kfold = KFoldCrossValidator(n_folds=5)
    
    def kfold_evaluate(train_data, test_data):
        metrics = evaluate_fn(test_data)
        return metrics, []
    
    # Stratify by domain if available
    def get_domain(item):
        return item.get('domain', 'unknown') if isinstance(item, dict) else 'unknown'
    
    cv_results = kfold.run_cross_validation(dataset, kfold_evaluate, stratify_key=get_domain)
    
    results['analyses']['kfold'] = {
        'n_folds': cv_results.n_folds,
        'aggregated_metrics': cv_results.aggregated_metrics,
        'stability_score': cv_results.stability_score,
        'metadata': cv_results.metadata
    }
    print(f"   Stability Score: {cv_results.stability_score:.4f}")
    
    # 2. Bootstrap Validation
    print("\n🔄 Running bootstrap validation (100 iterations)...")
    bootstrap = BootstrapValidator(n_iterations=100)
    
    def bootstrap_evaluate(sample):
        return evaluate_fn(sample)
    
    bootstrap_results = bootstrap.run_bootstrap(dataset, bootstrap_evaluate)
    results['analyses']['bootstrap'] = bootstrap_results
    
    if 'score' in bootstrap_results.get('metrics', {}):
        ci = bootstrap_results['metrics']['score']['ci_95']
        print(f"   95% CI for score: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # 3. Learning Curve Analysis
    print("\n📈 Analyzing learning curve...")
    lc_analyzer = LearningCurveAnalyzer(train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0])
    
    def lc_evaluate(train_data, test_data):
        return evaluate_fn(test_data)
    
    lc_results = lc_analyzer.analyze_learning_curve(dataset, lc_evaluate)
    results['analyses']['learning_curve'] = lc_results
    
    if lc_results.get('sample_efficiency'):
        print(f"   Sample Efficiency: {lc_results['sample_efficiency']:.6f}")
    
    # Summary
    print("\n" + "-" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("-" * 70)
    print(f"  Dataset Size: {len(dataset)}")
    print(f"  K-Fold Stability: {cv_results.stability_score:.4f}")
    
    for metric, stats in cv_results.aggregated_metrics.items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_path}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Create sample dataset
    sample_data = [
        {'query': f'Query {i}', 'domain': f'domain_{i % 3}', 'score': 0.7 + 0.3 * (i % 5) / 5}
        for i in range(100)
    ]
    
    def mock_evaluate(data):
        scores = [item.get('score', 0.5) for item in data]
        return {'score': sum(scores) / len(scores) if scores else 0}
    
    results = run_comprehensive_cv_analysis(sample_data, mock_evaluate)
    print("\n✅ Cross-validation analysis complete!")
