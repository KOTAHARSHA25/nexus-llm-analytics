"""
Tests for Cross-Validation, Error Analysis, and Hyperparameter Modules
======================================================================

Comprehensive test suite for research-grade benchmark modules.
"""

import pytest
import sys
import os
import math
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Cross-Validation Tests
# ============================================================================

class TestKFoldCrossValidator:
    """Tests for K-Fold Cross-Validation."""
    
    def test_create_folds_basic(self):
        """Test basic fold creation."""
        from benchmarks.cross_validation import KFoldCrossValidator
        
        cv = KFoldCrossValidator(n_folds=5, shuffle=False)
        data = list(range(100))
        folds = cv.create_folds(data)
        
        assert len(folds) == 5
        for fold in folds:
            assert fold.test_size == 20
            assert fold.train_size == 80
    
    def test_create_folds_uneven(self):
        """Test fold creation with uneven data."""
        from benchmarks.cross_validation import KFoldCrossValidator
        
        cv = KFoldCrossValidator(n_folds=3, shuffle=False)
        data = list(range(10))
        folds = cv.create_folds(data)
        
        assert len(folds) == 3
        total_test = sum(f.test_size for f in folds)
        assert total_test == 10  # All data used in test sets
    
    def test_create_folds_no_overlap(self):
        """Test that fold test sets don't overlap."""
        from benchmarks.cross_validation import KFoldCrossValidator
        
        cv = KFoldCrossValidator(n_folds=5, shuffle=False)
        data = list(range(50))
        folds = cv.create_folds(data)
        
        # Check no overlap between test sets
        all_test_indices = []
        for fold in folds:
            all_test_indices.extend(fold.test_indices)
        
        assert len(all_test_indices) == len(set(all_test_indices))
    
    def test_stratified_folds(self):
        """Test stratified fold creation."""
        from benchmarks.cross_validation import KFoldCrossValidator
        
        cv = KFoldCrossValidator(n_folds=5, shuffle=True, random_seed=42)
        data = [
            {'id': i, 'domain': f'domain_{i % 3}'} 
            for i in range(30)
        ]
        
        folds = cv.create_stratified_folds(data, lambda x: x['domain'])
        
        assert len(folds) == 5
        # Each fold should have roughly equal domain distribution
    
    def test_run_cross_validation(self):
        """Test complete cross-validation run."""
        from benchmarks.cross_validation import KFoldCrossValidator
        
        cv = KFoldCrossValidator(n_folds=3, random_seed=42)
        data = [{'value': i / 100} for i in range(30)]
        
        def evaluate(train, test):
            score = sum(d['value'] for d in test) / len(test)
            return {'score': score}, []
        
        results = cv.run_cross_validation(data, evaluate)
        
        assert results.n_folds == 3
        assert 'score' in results.aggregated_metrics
        assert 0 <= results.stability_score <= 1
    
    def test_cv_small_dataset_error(self):
        """Test error on too small dataset."""
        from benchmarks.cross_validation import KFoldCrossValidator
        
        cv = KFoldCrossValidator(n_folds=10)
        data = list(range(5))
        
        with pytest.raises(ValueError):
            cv.create_folds(data)


class TestLeaveOneOutValidator:
    """Tests for Leave-One-Out validation."""
    
    def test_loo_validation(self):
        """Test LOO validation."""
        from benchmarks.cross_validation import LeaveOneOutValidator
        
        validator = LeaveOneOutValidator()
        data = [{'value': i} for i in range(10)]
        
        def evaluate_single(train, test_item):
            return test_item['value'] / 10, test_item['value']
        
        results = validator.run_loo_validation(data, evaluate_single)
        
        assert results['n_samples'] == 10
        assert 'mean_score' in results
        assert len(results['all_scores']) == 10


class TestBootstrapValidator:
    """Tests for Bootstrap validation."""
    
    def test_bootstrap_validation(self):
        """Test bootstrap resampling."""
        from benchmarks.cross_validation import BootstrapValidator
        
        validator = BootstrapValidator(n_iterations=50, random_seed=42)
        data = [{'score': 0.7 + 0.1 * (i % 3)} for i in range(30)]
        
        def evaluate(sample):
            scores = [d['score'] for d in sample]
            return {'mean': sum(scores) / len(scores)}
        
        results = validator.run_bootstrap(data, evaluate)
        
        assert results['n_iterations'] == 50
        assert 'mean' in results['metrics']
        assert 'ci_95' in results['metrics']['mean']
    
    def test_bootstrap_confidence_intervals(self):
        """Test that confidence intervals are sensible."""
        from benchmarks.cross_validation import BootstrapValidator
        
        validator = BootstrapValidator(n_iterations=100, random_seed=42)
        data = [{'score': 0.8} for _ in range(50)]  # Constant scores
        
        def evaluate(sample):
            return {'score': sum(d['score'] for d in sample) / len(sample)}
        
        results = validator.run_bootstrap(data, evaluate)
        
        ci_95 = results['metrics']['score']['ci_95']
        # CI should be narrow for constant data
        assert ci_95[1] - ci_95[0] < 0.1


class TestLearningCurveAnalyzer:
    """Tests for learning curve analysis."""
    
    def test_learning_curve_analysis(self):
        """Test learning curve generation."""
        from benchmarks.cross_validation import LearningCurveAnalyzer
        
        analyzer = LearningCurveAnalyzer(
            train_sizes=[0.2, 0.5, 1.0], 
            n_iterations=3,
            random_seed=42
        )
        
        data = [{'score': 0.7 + 0.2 * (i % 5) / 5} for i in range(50)]
        
        def evaluate(train, test):
            return {'score': 0.75}
        
        results = analyzer.analyze_learning_curve(data, evaluate)
        
        assert len(results['train_sizes']) > 0
        assert len(results['test_scores']['mean']) > 0


# ============================================================================
# Error Analysis Tests
# ============================================================================

class TestErrorClassifier:
    """Tests for error classification."""
    
    def test_classify_factual_error(self):
        """Test factual error classification."""
        from benchmarks.error_analysis import ErrorClassifier, ErrorCategory
        
        classifier = ErrorClassifier()
        category, severity, details = classifier.classify(
            "2+2=5",
            expected="2+2=4",
            error_details="incorrect factual information"
        )
        
        assert category == ErrorCategory.FACTUAL
    
    def test_classify_truncated(self):
        """Test truncated response classification."""
        from benchmarks.error_analysis import ErrorClassifier, ErrorCategory
        
        classifier = ErrorClassifier()
        category, severity, details = classifier.classify(
            "This response was cut off and truncated",
            expected="This should be a complete response with all details"
        )
        
        assert category == ErrorCategory.TRUNCATED
    
    def test_classify_hallucination(self):
        """Test hallucination detection."""
        from benchmarks.error_analysis import ErrorClassifier, ErrorCategory
        
        classifier = ErrorClassifier()
        category, severity, details = classifier.classify(
            "Made up fact",
            error_details="hallucinated information"
        )
        
        assert category == ErrorCategory.HALLUCINATION
    
    def test_classify_irrelevant(self):
        """Test irrelevant response classification."""
        from benchmarks.error_analysis import ErrorClassifier, ErrorCategory
        
        classifier = ErrorClassifier()
        category, severity, details = classifier.classify(
            "The weather is nice today and off-topic",
            expected="What is the capital of France? Paris.",
            error_details="not relevant to the question"
        )
        
        assert category == ErrorCategory.IRRELEVANT


class TestErrorPatternDetector:
    """Tests for error pattern detection."""
    
    def test_detect_patterns(self):
        """Test pattern detection in errors."""
        from benchmarks.error_analysis import (
            ErrorPatternDetector, ErrorInstance, 
            ErrorCategory, Severity
        )
        
        detector = ErrorPatternDetector(min_pattern_frequency=3)
        
        # Create multiple errors of same category
        errors = [
            ErrorInstance(
                error_id=f"ERR_{i}",
                query=f"Query {i}",
                response="Incomplete...",
                expected=None,
                category=ErrorCategory.INCOMPLETE,
                severity=Severity.MEDIUM,
                details="Missing info",
                context={'domain': 'general'}
            )
            for i in range(5)
        ]
        
        patterns = detector.detect_patterns(errors)
        
        assert len(patterns) >= 1
        assert patterns[0].category == ErrorCategory.INCOMPLETE
        assert patterns[0].frequency == 5
    
    def test_pattern_recommendations(self):
        """Test that patterns include recommendations."""
        from benchmarks.error_analysis import (
            ErrorPatternDetector, ErrorInstance,
            ErrorCategory, Severity
        )
        
        detector = ErrorPatternDetector()
        errors = [
            ErrorInstance(
                error_id=f"ERR_{i}",
                query=f"Query {i}",
                response="Made up info",
                expected=None,
                category=ErrorCategory.HALLUCINATION,
                severity=Severity.CRITICAL,
                details="Fabricated"
            )
            for i in range(3)
        ]
        
        patterns = detector.detect_patterns(errors)
        
        assert len(patterns) > 0
        assert patterns[0].recommendation is not None


class TestErrorAnalyzer:
    """Tests for main error analyzer."""
    
    def test_analyze_failures(self):
        """Test failure analysis."""
        from benchmarks.error_analysis import ErrorAnalyzer
        
        analyzer = ErrorAnalyzer()
        
        results = [
            {'query': 'Q1', 'response': 'Good answer', 'score': 0.9},
            {'query': 'Q2', 'response': '', 'score': 0.1},  # Empty response
            {'query': 'Q3', 'response': 'Partial...', 'score': 0.3, 'error_details': 'truncated'},
            {'query': 'Q4', 'response': 'Wrong fact', 'score': 0.2, 'error_details': 'incorrect'},
        ]
        
        report = analyzer.analyze_failures(results, threshold=0.5)
        
        assert report.total_samples == 4
        assert report.total_errors == 3
        assert report.error_rate == 0.75
    
    def test_generate_report_text(self):
        """Test report generation."""
        from benchmarks.error_analysis import ErrorAnalyzer
        
        analyzer = ErrorAnalyzer()
        results = [
            {'query': 'Q1', 'response': 'OK', 'score': 0.8},
            {'query': 'Q2', 'response': 'Bad', 'score': 0.2, 'error_details': 'incomplete'},
        ]
        
        report = analyzer.analyze_failures(results)
        text = analyzer.generate_report_text(report)
        
        assert 'ERROR ANALYSIS REPORT' in text
        assert 'Summary' in text


class TestErrorImpactAnalyzer:
    """Tests for error impact analysis."""
    
    def test_calculate_impact_score(self):
        """Test impact score calculation."""
        from benchmarks.error_analysis import (
            ErrorImpactAnalyzer, ErrorInstance,
            ErrorCategory, Severity
        )
        
        analyzer = ErrorImpactAnalyzer()
        
        critical_error = ErrorInstance(
            error_id="ERR_001",
            query="Test",
            response="Bad",
            expected=None,
            category=ErrorCategory.HALLUCINATION,
            severity=Severity.CRITICAL,
            details="Critical issue"
        )
        
        low_error = ErrorInstance(
            error_id="ERR_002",
            query="Test",
            response="OK",
            expected=None,
            category=ErrorCategory.FORMATTING,
            severity=Severity.LOW,
            details="Minor issue"
        )
        
        critical_impact = analyzer.calculate_impact_score(critical_error)
        low_impact = analyzer.calculate_impact_score(low_error)
        
        assert critical_impact > low_impact
        assert 0 <= critical_impact <= 1
        assert 0 <= low_impact <= 1


# ============================================================================
# Hyperparameter Analysis Tests
# ============================================================================

class TestHyperparameterSpace:
    """Tests for hyperparameter space."""
    
    def test_default_space_initialization(self):
        """Test default hyperparameter space."""
        from benchmarks.hyperparameter_analysis import HyperparameterSpace
        
        space = HyperparameterSpace()
        
        assert len(space.parameters) > 0
        assert 'cache_similarity_threshold' in space.parameters
        assert 'rag_top_k' in space.parameters
    
    def test_get_default_config(self):
        """Test default configuration retrieval."""
        from benchmarks.hyperparameter_analysis import HyperparameterSpace
        
        space = HyperparameterSpace()
        config = space.get_default_config()
        
        assert 'cache_similarity_threshold' in config
        assert config['cache_similarity_threshold'] == 0.85
    
    def test_sample_random_config(self):
        """Test random configuration sampling."""
        from benchmarks.hyperparameter_analysis import HyperparameterSpace
        
        space = HyperparameterSpace()
        config1 = space.sample_random_config()
        config2 = space.sample_random_config()
        
        # Both should have all parameters
        assert set(config1.keys()) == set(space.parameters.keys())
        # Values should be from search space
        for param, value in config1.items():
            assert value in space.parameters[param].search_space
    
    def test_get_grid_configs_subset(self):
        """Test grid configuration with subset."""
        from benchmarks.hyperparameter_analysis import HyperparameterSpace
        
        space = HyperparameterSpace()
        configs = space.get_grid_configs(subset=['temperature', 'max_tokens'])
        
        assert len(configs) > 0
        assert all('temperature' in c for c in configs)
        assert all('max_tokens' in c for c in configs)


class TestSensitivityAnalyzer:
    """Tests for sensitivity analysis."""
    
    def test_analyze_sensitivity(self):
        """Test sensitivity analysis."""
        from benchmarks.hyperparameter_analysis import (
            HyperparameterSpace, SensitivityAnalyzer
        )
        
        space = HyperparameterSpace()
        analyzer = SensitivityAnalyzer(space)
        
        def mock_evaluate(config):
            # Simulate temperature sensitivity
            return 0.8 - 0.2 * abs(config.get('temperature', 0.7) - 0.7)
        
        results = analyzer.analyze_sensitivity(mock_evaluate)
        
        assert len(results) > 0
        # Temperature should be among most sensitive
        temp_result = next((r for r in results if r.parameter == 'temperature'), None)
        assert temp_result is not None
        assert temp_result.sensitivity_score >= 0
    
    def test_sensitivity_recommendations(self):
        """Test that sensitivity analysis includes recommendations."""
        from benchmarks.hyperparameter_analysis import (
            HyperparameterSpace, SensitivityAnalyzer
        )
        
        space = HyperparameterSpace()
        analyzer = SensitivityAnalyzer(space)
        
        def mock_evaluate(config):
            return 0.75
        
        results = analyzer.analyze_sensitivity(mock_evaluate)
        
        # All results should have recommendations list
        for result in results:
            assert isinstance(result.recommendations, list)


class TestGridSearchAnalyzer:
    """Tests for grid search analysis."""
    
    def test_run_grid_search(self):
        """Test grid search execution."""
        from benchmarks.hyperparameter_analysis import (
            HyperparameterSpace, GridSearchAnalyzer
        )
        
        space = HyperparameterSpace()
        analyzer = GridSearchAnalyzer(space)
        
        call_count = [0]
        
        def mock_evaluate(config):
            call_count[0] += 1
            return 0.7 + 0.1 * config.get('rag_top_k', 5) / 10
        
        results = analyzer.run_grid_search(
            mock_evaluate,
            parameter_subset=['rag_top_k'],
            max_configs=5
        )
        
        assert results.best_config is not None
        assert results.best_score > 0
        assert call_count[0] <= 5
    
    def test_grid_search_importance(self):
        """Test parameter importance calculation."""
        from benchmarks.hyperparameter_analysis import (
            HyperparameterSpace, GridSearchAnalyzer
        )
        
        space = HyperparameterSpace()
        analyzer = GridSearchAnalyzer(space)
        
        def mock_evaluate(config):
            # rag_top_k has strong effect, temperature has weak effect
            return 0.5 + 0.3 * config.get('rag_top_k', 5) / 10 + 0.05 * config.get('temperature', 0.7)
        
        results = analyzer.run_grid_search(
            mock_evaluate,
            parameter_subset=['rag_top_k', 'temperature'],
            max_configs=20
        )
        
        assert 'rag_top_k' in results.parameter_importance
        assert 'temperature' in results.parameter_importance


class TestInteractionAnalyzer:
    """Tests for interaction analysis."""
    
    def test_analyze_interactions(self):
        """Test interaction analysis."""
        from benchmarks.hyperparameter_analysis import (
            HyperparameterSpace, InteractionAnalyzer
        )
        
        space = HyperparameterSpace()
        analyzer = InteractionAnalyzer(space)
        
        def mock_evaluate(config):
            # Simple additive model (no interaction)
            return 0.5 + 0.1 * config.get('rag_top_k', 5) / 10 + 0.1 * config.get('temperature', 0.7)
        
        interactions = analyzer.analyze_interactions(
            mock_evaluate,
            parameter_pairs=[('rag_top_k', 'temperature')]
        )
        
        assert len(interactions) == 1
        assert interactions[0].param1 == 'rag_top_k'
        assert interactions[0].param2 == 'temperature'
        assert interactions[0].effect_type in ['synergistic', 'antagonistic', 'neutral']
    
    def test_interaction_strength(self):
        """Test that interaction strength is calculated."""
        from benchmarks.hyperparameter_analysis import (
            HyperparameterSpace, InteractionAnalyzer
        )
        
        space = HyperparameterSpace()
        analyzer = InteractionAnalyzer(space)
        
        def mock_evaluate(config):
            return 0.7
        
        interactions = analyzer.analyze_interactions(
            mock_evaluate,
            parameter_pairs=[('rag_top_k', 'cache_max_size')]
        )
        
        assert len(interactions) == 1
        assert isinstance(interactions[0].interaction_strength, float)
        assert interactions[0].interaction_strength >= 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestModuleIntegration:
    """Integration tests for benchmark modules."""
    
    def test_cv_with_error_analysis(self):
        """Test cross-validation followed by error analysis."""
        from benchmarks.cross_validation import KFoldCrossValidator
        from benchmarks.error_analysis import ErrorAnalyzer
        
        # Create test data
        data = [
            {'query': f'Q{i}', 'response': 'OK' if i % 3 != 0 else '', 'expected_score': 0.8 if i % 3 != 0 else 0.2}
            for i in range(30)
        ]
        
        # Run CV
        cv = KFoldCrossValidator(n_folds=3)
        
        def evaluate(train, test):
            scores = [d['expected_score'] for d in test]
            return {'score': sum(scores) / len(scores)}, []
        
        cv_results = cv.run_cross_validation(data, evaluate)
        assert cv_results.n_folds == 3
        
        # Run error analysis on data
        results = [
            {'query': d['query'], 'response': d['response'], 'score': d['expected_score']}
            for d in data
        ]
        
        analyzer = ErrorAnalyzer()
        error_report = analyzer.analyze_failures(results, threshold=0.5)
        
        assert error_report.total_samples == 30
        assert error_report.total_errors == 10  # Every 3rd item has low score
    
    def test_hyperparameter_search_workflow(self):
        """Test complete hyperparameter search workflow."""
        from benchmarks.hyperparameter_analysis import (
            HyperparameterSpace, SensitivityAnalyzer, GridSearchAnalyzer
        )
        
        space = HyperparameterSpace()
        
        def evaluate(config):
            base = 0.7
            base += 0.1 * (config.get('temperature', 0.7) - 0.5)
            base += 0.05 * (config.get('rag_top_k', 5) / 10)
            return max(0, min(1, base))
        
        # Step 1: Sensitivity analysis
        sensitivity_analyzer = SensitivityAnalyzer(space)
        sensitivity_results = sensitivity_analyzer.analyze_sensitivity(evaluate)
        
        assert len(sensitivity_results) > 0
        
        # Step 2: Grid search on top parameters
        top_params = [r.parameter for r in sensitivity_results[:2]]
        grid_analyzer = GridSearchAnalyzer(space)
        grid_results = grid_analyzer.run_grid_search(
            evaluate,
            parameter_subset=top_params,
            max_configs=20
        )
        
        assert grid_results.best_score > 0
        assert grid_results.best_config is not None
    
    def test_all_modules_import(self):
        """Test that all modules can be imported together."""
        from benchmarks.cross_validation import (
            KFoldCrossValidator,
            LeaveOneOutValidator,
            BootstrapValidator,
            LearningCurveAnalyzer
        )
        from benchmarks.error_analysis import (
            ErrorClassifier,
            ErrorPatternDetector,
            ErrorAnalyzer,
            ErrorImpactAnalyzer
        )
        from benchmarks.hyperparameter_analysis import (
            HyperparameterSpace,
            SensitivityAnalyzer,
            GridSearchAnalyzer,
            InteractionAnalyzer
        )
        
        # All classes should be instantiable
        assert KFoldCrossValidator() is not None
        assert LeaveOneOutValidator() is not None
        assert BootstrapValidator() is not None
        assert LearningCurveAnalyzer() is not None
        assert ErrorClassifier() is not None
        assert ErrorPatternDetector() is not None
        assert ErrorAnalyzer() is not None
        assert ErrorImpactAnalyzer() is not None
        assert HyperparameterSpace() is not None
        
        space = HyperparameterSpace()
        assert SensitivityAnalyzer(space) is not None
        assert GridSearchAnalyzer(space) is not None
        assert InteractionAnalyzer(space) is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
