"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS LLM ANALYTICS - PHASE 4 RESEARCH READINESS TESTS
═══════════════════════════════════════════════════════════════════════════════

Tests for:
- 4.1: Benchmark Dataset
- 4.2: Evaluation Metrics
- 4.3/4.4: Baseline Comparisons & Ablation Studies

Version: 1.0.0
"""

import pytest
import json
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "benchmarks"))

from benchmarks.evaluation_metrics import (
    MetricsCalculator,
    AccuracyMetrics,
    EfficiencyMetrics,
    QualityMetrics,
    SystemMetrics,
    ResearchMetrics,
    AggregateMetrics,
    evaluate
)
from benchmarks.baseline_comparisons import (
    BaselineRunner,
    BaselineConfig,
    ComparisonResult,
    AblationStudy
)


# =============================================================================
# Test: Benchmark Dataset (4.1)
# =============================================================================

class TestBenchmarkDataset:
    """Tests for benchmark dataset structure and content"""
    
    @pytest.fixture
    def dataset(self):
        """Load benchmark dataset"""
        dataset_path = project_root / "benchmarks" / "benchmark_dataset.json"
        with open(dataset_path, encoding='utf-8') as f:
            return json.load(f)
    
    def test_dataset_exists(self):
        """Verify dataset file exists"""
        dataset_path = project_root / "benchmarks" / "benchmark_dataset.json"
        assert dataset_path.exists(), "Benchmark dataset file should exist"
    
    def test_dataset_has_required_fields(self, dataset):
        """Verify dataset has required top-level fields"""
        assert "metadata" in dataset
        assert "domains" in dataset
        assert "cross_domain_queries" in dataset
        assert "edge_case_queries" in dataset
        assert "summary" in dataset
    
    def test_minimum_query_count(self, dataset):
        """Verify minimum 150 queries"""
        metadata = dataset["metadata"]
        assert metadata["total_queries"] >= 150, "Should have at least 150 queries"
    
    def test_domain_coverage(self, dataset):
        """Verify multiple domains covered"""
        domains = dataset["domains"]
        assert len(domains) >= 5, "Should cover at least 5 domains"
    
    def test_complexity_distribution(self, dataset):
        """Verify queries across complexity levels"""
        summary = dataset["summary"]
        by_complexity = summary["queries_by_complexity"]
        assert "simple" in by_complexity
        assert "medium" in by_complexity
        assert "complex" in by_complexity
        assert by_complexity["simple"] > 0
        assert by_complexity["medium"] > 0
        assert by_complexity["complex"] > 0
    
    def test_query_structure(self, dataset):
        """Verify individual query structure"""
        # Get first query from any domain
        first_domain = list(dataset["domains"].keys())[0]
        query = dataset["domains"][first_domain]["queries"][0]
        assert "id" in query
        assert "query" in query
        assert "complexity" in query


# =============================================================================
# Test: Evaluation Metrics (4.2)
# =============================================================================

class TestEvaluationMetrics:
    """Tests for evaluation metrics calculator"""
    
    @pytest.fixture
    def calculator(self):
        """Create metrics calculator"""
        return MetricsCalculator()
    
    def test_jaccard_similarity(self, calculator):
        """Test Jaccard similarity calculation"""
        text1 = "the quick brown fox"
        text2 = "the quick brown dog"
        
        similarity = calculator.calculate_jaccard_similarity(text1, text2)
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Most words match
    
    def test_jaccard_identical_texts(self, calculator):
        """Test Jaccard with identical texts"""
        text = "hello world"
        similarity = calculator.calculate_jaccard_similarity(text, text)
        assert similarity == 1.0
    
    def test_cosine_similarity(self, calculator):
        """Test cosine similarity calculation"""
        text1 = "machine learning is great"
        text2 = "deep learning is amazing"
        
        similarity = calculator.calculate_cosine_similarity(text1, text2)
        assert 0 <= similarity <= 1
    
    def test_bleu_1(self, calculator):
        """Test BLEU-1 score"""
        reference = "the cat sat on the mat"
        candidate = "the cat is on the mat"
        
        bleu = calculator.calculate_bleu_1(reference, candidate)
        assert 0 <= bleu <= 1
    
    def test_rouge_l(self, calculator):
        """Test ROUGE-L score"""
        reference = "the cat sat on the mat"
        candidate = "the cat on the mat sat"
        
        rouge = calculator.calculate_rouge_l(reference, candidate)
        assert 0 <= rouge <= 1
    
    def test_numeric_extraction(self, calculator):
        """Test number extraction from text"""
        text = "Revenue was $1,234.56 with 25% growth"
        numbers = calculator.extract_numbers(text)
        
        assert len(numbers) >= 2
        assert 1234.56 in numbers or 123456 in numbers  # Depending on parsing
        assert 25 in numbers
    
    def test_numeric_accuracy(self, calculator):
        """Test numeric accuracy calculation"""
        response = "The total is 100 units with 25% efficiency"
        expected = [100.0, 25.0]
        
        accuracy = calculator.calculate_numeric_accuracy(response, expected)
        assert accuracy == 1.0  # Both numbers present
    
    def test_completeness(self, calculator):
        """Test completeness scoring"""
        response = "Revenue increased. Profit margins improved."
        required = ["revenue", "profit", "growth"]
        
        score, found, total = calculator.calculate_completeness(response, required)
        assert 0 <= score <= 1
        assert found == 2  # revenue and profit
        assert total == 3
    
    def test_coherence_assessment(self, calculator):
        """Test coherence scoring"""
        # Coherent response with structure
        good_response = """
        First, we analyzed the data. The results show significant trends.
        Furthermore, the correlation between variables is strong.
        In conclusion, we recommend the following actions:
        - Action 1
        - Action 2
        """
        
        score = calculator.assess_coherence(good_response)
        assert score > 0.5, "Coherent response should score well"
    
    def test_specificity_assessment(self, calculator):
        """Test specificity scoring"""
        specific = "Revenue was exactly $1,234,567, a 15.3% increase"
        vague = "Revenue was somewhat higher than before"
        
        specific_score = calculator.assess_specificity(specific)
        vague_score = calculator.assess_specificity(vague)
        
        assert specific_score > vague_score
    
    def test_full_evaluation(self, calculator):
        """Test complete evaluation pipeline"""
        result = calculator.evaluate_response(
            query_id="test-001",
            query="What is the total revenue?",
            response="The total revenue is $1,000,000 for Q4 2024",
            expected_values=[1000000.0],
            required_elements=["revenue", "Q4"],
            execution_context={
                "latency_seconds": 2.5,
                "model_calls": 1
            }
        )
        
        assert isinstance(result, ResearchMetrics)
        assert result.query_id == "test-001"
        assert 0 <= result.overall_score <= 1
        assert result.accuracy.numeric_accuracy == 1.0
    
    def test_convenience_evaluate_function(self):
        """Test convenience function"""
        result = evaluate(
            query="What is the average?",
            response="The average is 50 units",
            query_id="quick-test"
        )
        
        assert isinstance(result, ResearchMetrics)


class TestAggregateMetrics:
    """Tests for aggregate metrics"""
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics list"""
        calculator = MetricsCalculator()
        metrics = []
        
        for i in range(10):
            result = calculator.evaluate_response(
                query_id=f"test-{i}",
                query=f"Query {i}",
                response=f"Response {i} with value {i * 100}",
                execution_context={"latency_seconds": 1.0 + i * 0.1}
            )
            metrics.append(result)
        
        return metrics
    
    def test_summary_calculation(self, sample_metrics):
        """Test summary statistics"""
        aggregate = AggregateMetrics(sample_metrics)
        summary = aggregate.calculate_summary()
        
        assert "total_evaluations" in summary
        assert summary["total_evaluations"] == 10
        assert "overall_score" in summary
        assert "mean" in summary["overall_score"]
        assert "latency_seconds" in summary


# =============================================================================
# Test: Baseline Comparisons (4.3)
# =============================================================================

class TestBaselineComparisons:
    """Tests for baseline comparison framework"""
    
    @pytest.fixture
    def runner(self, tmp_path):
        """Create baseline runner with temp output"""
        return BaselineRunner(output_dir=str(tmp_path))
    
    @pytest.fixture
    def sample_queries(self):
        """Create sample queries for testing"""
        return [
            {"id": f"q{i}", "complexity": "medium", "query_type": "analytical", "requires_data": True}
            for i in range(20)
        ]
    
    def test_baseline_configs_exist(self):
        """Verify baseline configurations are defined"""
        assert len(BaselineRunner.BASELINES) >= 5
        assert "single_gpt4" in BaselineRunner.BASELINES
        assert "no_review" in BaselineRunner.BASELINES
        assert "no_rag" in BaselineRunner.BASELINES
    
    def test_full_system_simulation(self, runner):
        """Test full system simulation"""
        query = {"complexity": "medium", "query_type": "analytical", "requires_data": True}
        result = runner.simulate_full_system(query)
        
        assert "latency_seconds" in result
        assert "quality_score" in result
        assert 0 <= result["quality_score"] <= 1
    
    def test_baseline_simulation(self, runner):
        """Test baseline system simulation"""
        query = {"complexity": "medium", "query_type": "analytical", "requires_data": True}
        config = BaselineRunner.BASELINES["no_review"]
        
        result = runner.simulate_baseline(query, config)
        
        assert "latency_seconds" in result
        assert "quality_score" in result
        assert result["review_applied"] == False  # No review in this baseline
    
    def test_run_comparison(self, runner, sample_queries):
        """Test single baseline comparison"""
        result = runner.run_comparison(sample_queries, "no_review")
        
        assert isinstance(result, ComparisonResult)
        assert result.queries_evaluated == len(sample_queries)
        assert result.improvement_percent is not None
    
    def test_run_all_comparisons(self, runner, sample_queries):
        """Test all baseline comparisons"""
        results = runner.run_all_comparisons(sample_queries)
        
        assert len(results) == len(BaselineRunner.BASELINES)
    
    def test_generate_report(self, runner, sample_queries):
        """Test report generation"""
        runner.run_all_comparisons(sample_queries)
        report = runner.generate_report()
        
        assert "metadata" in report
        assert "summary" in report
        assert "results" in report
        assert "feature_impact" in report
    
    def test_save_report(self, runner, sample_queries, tmp_path):
        """Test report saving"""
        runner.run_all_comparisons(sample_queries)
        filepath = runner.save_report("test_report.json")
        
        assert Path(filepath).exists()


# =============================================================================
# Test: Ablation Studies (4.4)
# =============================================================================

class TestAblationStudy:
    """Tests for ablation study framework"""
    
    @pytest.fixture
    def sample_queries(self):
        """Create sample queries"""
        return [
            {"id": f"q{i}", "complexity": "medium", "query_type": "analytical", "requires_data": True}
            for i in range(15)
        ]
    
    def test_component_list(self):
        """Verify component list is defined"""
        assert len(AblationStudy.COMPONENTS) >= 4
    
    def test_run_ablation(self, sample_queries):
        """Test ablation study execution"""
        study = AblationStudy()
        results = study.run_ablation(sample_queries)
        
        assert "full_system" in results
        assert "without_rag" in results
        assert "without_review" in results
        
        # Full system should have best quality
        full_quality = results["full_system"]["quality"]
        assert full_quality > 0
    
    def test_component_ranking(self, sample_queries):
        """Test component importance ranking"""
        study = AblationStudy()
        study.run_ablation(sample_queries)
        ranking = study.get_component_ranking()
        
        assert len(ranking) >= 3
        # Each item is (component, importance)
        for component, importance in ranking:
            assert isinstance(component, str)
            assert isinstance(importance, (int, float))


# =============================================================================
# Test: Benchmark Runner
# =============================================================================

class TestBenchmarkRunner:
    """Tests for benchmark runner module"""
    
    def test_runner_import(self):
        """Test benchmark runner can be imported"""
        from benchmarks.benchmark_runner import BenchmarkRunner, QueryResult, BenchmarkReport
        assert BenchmarkRunner is not None
    
    def test_runner_initialization(self):
        """Test runner initialization"""
        from benchmarks.benchmark_runner import BenchmarkRunner
        
        runner = BenchmarkRunner(mode="quick", simulate=True)
        assert runner.mode == "quick"
        assert runner.simulate == True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
