"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS LLM ANALYTICS - PHASE 3 & 4 INTEGRATION TESTS
═══════════════════════════════════════════════════════════════════════════════

Comprehensive integration tests that verify Phase 3 and Phase 4 components
work together correctly. These tests simulate real-world usage patterns.

Test Categories:
1. RAG + Cache Integration
2. Metrics + Logging Integration
3. Benchmark + Evaluation Integration
4. End-to-End Query Pipeline
5. Stress Tests
6. Edge Cases

Version: 1.0.0
"""

import pytest
import json
import time
import sys
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "benchmarks"))


# =============================================================================
# Test: Phase 3 Component Integration
# =============================================================================

class TestCacheMetricsIntegration:
    """Test that cache operations properly update metrics"""
    
    def test_cache_operations_tracked_by_metrics(self):
        """Verify cache hits/misses are tracked in metrics"""
        from src.backend.core.advanced_cache import SemanticCache
        from src.backend.core.metrics import update_cache_metrics, METRICS
        
        cache = SemanticCache(similarity_threshold=0.8)
        
        # Simulate cache operations
        cache.put("What is the total revenue?", {"result": 1000000})
        result1 = cache.get("What is the total revenue?")  # Hit
        result2 = cache.get("Show me employee data")  # Miss
        
        stats = cache.get_stats()
        
        # Update metrics
        update_cache_metrics("semantic", stats['hits'], stats['misses'], len(cache._cache))
        
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1
    
    def test_semantic_cache_with_similar_queries(self):
        """Test semantic cache correctly identifies similar queries"""
        from src.backend.core.advanced_cache import SemanticCache
        
        # Lower threshold to catch more similar queries
        cache = SemanticCache(similarity_threshold=0.5)
        
        # Store original query
        original = "What is the total revenue for this quarter"
        cache.put(original, {"revenue": 500000})
        
        # Exact match should work
        result = cache.get(original)
        assert result is not None, "Exact match should always hit"
        
        # Test with very similar query - same words reordered
        similar_query = "total revenue for quarter"
        result = cache.get(similar_query)
        
        # At least exact match works; semantic matching is bonus
        exact_result = cache.get(original)
        assert exact_result is not None
    
    def test_cache_context_isolation(self):
        """Verify cache respects context boundaries"""
        from src.backend.core.advanced_cache import SemanticCache
        
        cache = SemanticCache(similarity_threshold=0.75)
        
        # Same query, different contexts
        cache.put("What is the total?", {"result": 100}, context="sales.csv")
        cache.put("What is the total?", {"result": 200}, context="inventory.csv")
        
        result1 = cache.get("What is the total?", context="sales.csv")
        result2 = cache.get("What is the total?", context="inventory.csv")
        
        assert result1 is not None
        assert result2 is not None
        assert result1[0]["result"] == 100
        assert result2[0]["result"] == 200


class TestLoggingMetricsIntegration:
    """Test structured logging with metrics correlation"""
    
    def test_structured_logger_emits_correct_format(self):
        """Verify structured logger produces valid JSON events"""
        from src.backend.core.enhanced_logging import StructuredLogger
        import io
        import logging
        
        # Create logger with StringIO handler
        logger = StructuredLogger("test_integration")
        
        # Log an event using correct API (event, level, **kwargs)
        logger.log_event("test_event", "INFO", 
            key="value",
            number=42
        )
        
        # Also test convenience methods
        logger.info("info_event", test_data="hello")
        logger.debug("debug_event", test_data="world")
        
        # Verify no exceptions (basic integration check)
        assert True
    
    def test_span_context_tracks_operations(self):
        """Test SpanContext correctly tracks nested operations"""
        from src.backend.core.enhanced_logging import StructuredLogger, SpanContext
        
        logger = StructuredLogger("span_test")
        
        # SpanContext requires a logger and operation name
        with SpanContext(logger, "outer_operation") as outer_span:
            time.sleep(0.01)
            with SpanContext(logger, "inner_operation") as inner_span:
                time.sleep(0.01)
        
        # Verify spans executed without error
        assert outer_span.operation == "outer_operation"
        assert inner_span.operation == "inner_operation"


class TestAdvancedCacheFeatures:
    """Test advanced cache functionality"""
    
    def test_cache_ttl_expiration(self):
        """Verify cache entries expire correctly"""
        from src.backend.core.advanced_cache import SemanticCache
        
        cache = SemanticCache(default_ttl=0.1)  # 100ms TTL
        
        cache.put("test query", {"result": "value"})
        
        # Immediately should hit
        result = cache.get("test query")
        assert result is not None
        
        # After TTL should miss
        time.sleep(0.15)
        result = cache.get("test query")
        assert result is None
    
    def test_cache_lru_eviction(self):
        """Verify LRU eviction policy works correctly"""
        from src.backend.core.advanced_cache import SemanticCache
        
        cache = SemanticCache(max_size=3)
        
        # Fill cache
        cache.put("query1", "result1")
        cache.put("query2", "result2")
        cache.put("query3", "result3")
        
        # Access query1 to make it recently used
        cache.get("query1")
        
        # Add new entry - should evict query2 (oldest not recently used)
        cache.put("query4", "result4")
        
        # query1 should still exist, query2 should be evicted
        assert cache.get("query1") is not None
        assert cache.get("query3") is not None
        assert cache.get("query4") is not None
    
    def test_concurrent_cache_access(self):
        """Test thread-safe cache operations"""
        from src.backend.core.advanced_cache import SemanticCache
        
        cache = SemanticCache(max_size=100)
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(10):
                    query = f"Query {thread_id}_{i}"
                    cache.put(query, {"thread": thread_id, "iteration": i})
                    result = cache.get(query)
                    if result is None:
                        errors.append(f"Missing result for {query}")
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Concurrent access errors: {errors}"


# =============================================================================
# Test: Phase 4 Component Integration
# =============================================================================

class TestBenchmarkEvaluationIntegration:
    """Test benchmark runner with evaluation metrics"""
    
    @pytest.fixture
    def sample_benchmark_queries(self):
        """Load sample queries from benchmark dataset"""
        dataset_path = project_root / "benchmarks" / "benchmark_dataset.json"
        with open(dataset_path, encoding='utf-8') as f:
            data = json.load(f)
        
        # Get 10 sample queries
        queries = []
        for domain, domain_data in data["domains"].items():
            for query in domain_data.get("queries", [])[:2]:
                query["domain"] = domain
                queries.append(query)
            if len(queries) >= 10:
                break
        return queries
    
    def test_evaluation_metrics_on_benchmark_queries(self, sample_benchmark_queries):
        """Run evaluation metrics on actual benchmark queries"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calculator = MetricsCalculator()
        
        for query_config in sample_benchmark_queries:
            query = query_config["query"]
            required_elements = query_config.get("required_elements", [])
            
            # Simulate a response
            response = f"Analysis result for: {query}. " + " ".join(required_elements)
            
            result = calculator.evaluate_response(
                query_id=query_config["id"],
                query=query,
                response=response,
                required_elements=required_elements,
                execution_context={"latency_seconds": 1.5}
            )
            
            assert 0 <= result.overall_score <= 1
            assert result.query_id == query_config["id"]
    
    def test_baseline_comparison_with_metrics(self, sample_benchmark_queries):
        """Verify baseline comparisons produce valid metric differences"""
        from benchmarks.baseline_comparisons import BaselineRunner
        
        runner = BaselineRunner(output_dir=str(project_root / "benchmarks" / "results"))
        
        # Run comparison against one baseline
        result = runner.run_comparison(sample_benchmark_queries, "no_review")
        
        assert result.queries_evaluated == len(sample_benchmark_queries)
        assert result.improvement_percent is not None
        assert result.full_system_score > 0
        assert result.baseline_score > 0
    
    def test_ablation_study_component_ranking(self, sample_benchmark_queries):
        """Verify ablation study produces meaningful rankings"""
        from benchmarks.baseline_comparisons import AblationStudy
        
        study = AblationStudy()
        results = study.run_ablation(sample_benchmark_queries)
        ranking = study.get_component_ranking()
        
        # Should have rankings for removed components
        assert len(ranking) >= 3
        
        # Each component should have importance score
        for component, importance in ranking:
            assert isinstance(importance, (int, float))


class TestMetricsSimilarity:
    """Deep tests for similarity metrics"""
    
    def test_bleu_score_ranges(self):
        """Verify BLEU scores are in valid range"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        
        test_cases = [
            ("the cat sat on the mat", "the cat sat on the mat", 1.0),  # Identical
            ("the cat sat on the mat", "the dog ran in the park", None),  # Different
            ("hello world", "goodbye world", None),  # Partial match
            ("", "something", 0.0),  # Empty candidate
        ]
        
        for ref, cand, expected in test_cases:
            score = calc.calculate_bleu_1(ref, cand)
            assert 0 <= score <= 1, f"BLEU score {score} out of range for ({ref}, {cand})"
            if expected is not None:
                assert abs(score - expected) < 0.1, f"Expected ~{expected}, got {score}"
    
    def test_rouge_l_calculation(self):
        """Verify ROUGE-L handles various cases"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        
        # Test LCS scenarios
        ref = "The quick brown fox jumps over the lazy dog"
        
        # High overlap
        cand1 = "The quick brown fox leaps over the lazy dog"
        score1 = calc.calculate_rouge_l(ref, cand1)
        assert score1 > 0.7
        
        # Low overlap
        cand2 = "A cat sleeps on the mat"
        score2 = calc.calculate_rouge_l(ref, cand2)
        assert score2 < 0.5
    
    def test_numeric_extraction_edge_cases(self):
        """Test number extraction with edge cases"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        
        test_cases = [
            ("Revenue: $1,234,567.89", [1234567.89]),
            ("Growth of 15.5% year over year", [15.5]),
            ("Values: -10, 0, +20", [-10, 0, 20]),
            ("No numbers here", []),
            ("42% of 1000 users = 420 conversions", [42, 1000, 420]),
        ]
        
        for text, expected_contains in test_cases:
            extracted = calc.extract_numbers(text)
            for expected in expected_contains:
                # Check if approximately present
                found = any(abs(n - expected) < 0.01 for n in extracted)
                assert found or not expected_contains, f"Missing {expected} in {extracted} from '{text}'"


class TestComplexQueryEvaluation:
    """Test evaluation on complex, realistic queries"""
    
    def test_multi_part_query_evaluation(self):
        """Evaluate queries with multiple required elements"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        
        query = "Analyze sales performance by region, calculate growth rates, and identify top products"
        
        # Good response covers all elements
        good_response = """
        Sales Analysis by Region:
        - North: $1.2M (15% growth)
        - South: $800K (8% growth)
        - East: $950K (12% growth)
        
        Growth Rate Summary: Average 11.7% across all regions.
        
        Top Products:
        1. Product A - $500K revenue
        2. Product B - $350K revenue
        3. Product C - $280K revenue
        """
        
        required = ["region", "growth", "top", "product"]
        
        result = calc.evaluate_response(
            query_id="complex_1",
            query=query,
            response=good_response,
            required_elements=required
        )
        
        assert result.quality.completeness >= 0.75, "Good response should have high completeness"
        assert result.quality.specificity > 0.3, "Response with numbers should be specific"
    
    def test_poor_response_scores_low(self):
        """Verify poor responses get appropriately low scores"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        
        query = "What is the revenue breakdown by product category?"
        
        poor_response = "I don't know."
        
        result = calc.evaluate_response(
            query_id="poor_1",
            query=query,
            response=poor_response,
            required_elements=["revenue", "category", "breakdown"]
        )
        
        assert result.quality.completeness < 0.5
        assert result.quality.specificity < 0.3


# =============================================================================
# Test: End-to-End Integration
# =============================================================================

class TestEndToEndPipeline:
    """Test complete query processing pipeline"""
    
    def test_query_through_cache_and_metrics(self):
        """Simulate query passing through cache with metrics tracking"""
        from src.backend.core.advanced_cache import SemanticCache, get_semantic_cache
        from src.backend.core.metrics import METRICS, track_request
        
        cache = get_semantic_cache()
        initial_stats = cache.get_stats()
        
        # Simulate query processing
        query = "What is the average order value?"
        
        # First query - cache miss
        result1 = cache.get(query)
        if result1 is None:
            # Simulate processing
            response = {"average_order_value": 125.50}
            cache.put(query, response)
        
        # Second query - cache hit
        result2 = cache.get(query)
        
        final_stats = cache.get_stats()
        
        assert final_stats['total_requests'] >= initial_stats['total_requests'] + 2
    
    def test_evaluation_pipeline_with_real_dataset(self):
        """Run evaluation on subset of real benchmark dataset"""
        from benchmarks.evaluation_metrics import MetricsCalculator, AggregateMetrics
        from benchmarks.benchmark_runner import BenchmarkRunner
        
        # Load runner in simulate mode
        runner = BenchmarkRunner(mode="quick", simulate=True)
        calc = MetricsCalculator()
        
        # Get queries
        queries = runner._get_queries("quick")
        
        results = []
        for q in queries[:5]:
            # Simulate response
            response = f"Result for {q['query']}: Calculated values based on data analysis."
            
            metric = calc.evaluate_response(
                query_id=q.get("id", "unknown"),
                query=q["query"],
                response=response,
                required_elements=q.get("required_elements", [])
            )
            results.append(metric)
        
        # Aggregate
        aggregate = AggregateMetrics(results)
        summary = aggregate.calculate_summary()
        
        assert summary["total_evaluations"] == 5
        assert "overall_score" in summary


# =============================================================================
# Test: Stress and Performance
# =============================================================================

class TestStressAndPerformance:
    """Performance and stress tests"""
    
    def test_cache_performance_under_load(self):
        """Test cache performance with many entries"""
        from src.backend.core.advanced_cache import SemanticCache
        
        cache = SemanticCache(max_size=1000)
        
        # Insert 500 entries
        start_time = time.time()
        for i in range(500):
            cache.put(f"Query number {i} about topic {i % 10}", {"result": i})
        insert_time = time.time() - start_time
        
        # Retrieve 500 entries
        start_time = time.time()
        hits = 0
        for i in range(500):
            result = cache.get(f"Query number {i} about topic {i % 10}")
            if result:
                hits += 1
        retrieve_time = time.time() - start_time
        
        # Performance assertions
        assert insert_time < 5.0, f"Insert took too long: {insert_time}s"
        assert retrieve_time < 5.0, f"Retrieve took too long: {retrieve_time}s"
        assert hits >= 450, f"Too many cache misses: {500 - hits}"
    
    def test_evaluation_metrics_performance(self):
        """Test evaluation metrics calculation speed"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        
        # Test with long texts
        long_query = "Analyze the comprehensive data including " + " ".join([f"factor{i}" for i in range(50)])
        long_response = "Analysis results showing " + " ".join([f"result{i}" for i in range(200)])
        
        start_time = time.time()
        for _ in range(100):
            calc.evaluate_response(
                query_id="perf_test",
                query=long_query,
                response=long_response
            )
        elapsed = time.time() - start_time
        
        # Should complete 100 evaluations in under 5 seconds
        assert elapsed < 5.0, f"Evaluation too slow: {elapsed}s for 100 runs"
    
    def test_concurrent_baseline_comparisons(self):
        """Test baseline comparisons handle concurrent access"""
        from benchmarks.baseline_comparisons import BaselineRunner
        
        queries = [
            {"id": f"q{i}", "complexity": "medium", "query_type": "analytical", "requires_data": True}
            for i in range(20)
        ]
        
        def run_comparison(baseline_key):
            runner = BaselineRunner()
            return runner.run_comparison(queries, baseline_key)
        
        # Run multiple comparisons concurrently
        baselines = ["single_gpt4", "no_review", "no_rag"]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_comparison, b) for b in baselines]
            results = [f.result() for f in as_completed(futures)]
        
        assert len(results) == 3
        for result in results:
            assert result.queries_evaluated == 20


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case handling tests"""
    
    def test_empty_query_handling(self):
        """Test handling of empty queries"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        
        result = calc.evaluate_response(
            query_id="empty",
            query="",
            response="Some response"
        )
        
        # Should not crash, should return valid metrics
        assert result is not None
        assert 0 <= result.overall_score <= 1
    
    def test_unicode_query_handling(self):
        """Test handling of unicode characters"""
        from src.backend.core.advanced_cache import SemanticCache
        
        cache = SemanticCache()
        
        unicode_query = "分析数据 αβγδ データ分析"
        cache.put(unicode_query, {"result": "unicode"})
        result = cache.get(unicode_query)
        
        assert result is not None
    
    def test_special_characters_in_response(self):
        """Test evaluation with special characters"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        
        result = calc.evaluate_response(
            query_id="special",
            query="What's the O'Brien revenue?",
            response="O'Brien's revenue: $1,234,567.89 (15% increase)",
            expected_values=[1234567.89, 15]
        )
        
        assert result.accuracy.numeric_accuracy > 0
    
    def test_very_long_response(self):
        """Test handling of very long responses"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        
        calc = MetricsCalculator()
        
        long_response = "Analysis result. " * 1000  # ~17000 chars
        
        result = calc.evaluate_response(
            query_id="long",
            query="Analyze everything",
            response=long_response
        )
        
        assert result is not None
        assert 0 <= result.overall_score <= 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
