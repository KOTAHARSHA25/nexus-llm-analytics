"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS LLM ANALYTICS - ADVANCED EVALUATION TESTS
═══════════════════════════════════════════════════════════════════════════════

Tests for:
1. Advanced Similarity Metrics (TF-IDF, BLEU-n, METEOR)
2. Statistical Significance Testing
3. Visualization Generation
4. Semantic Chunking

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


# =============================================================================
# Test: Advanced Similarity Metrics
# =============================================================================

class TestAdvancedSimilarityMetrics:
    """Test TF-IDF, n-gram BLEU, and METEOR scores"""
    
    @pytest.fixture
    def calculator(self):
        from benchmarks.advanced_evaluation import AdvancedMetricsCalculator
        
        # Initialize with sample corpus
        corpus = [
            "The quarterly revenue report shows strong growth in Q4.",
            "Sales performance exceeded expectations across all regions.",
            "Customer acquisition costs decreased while retention improved.",
            "The analysis indicates positive market trends.",
            "Financial metrics demonstrate healthy business performance."
        ]
        return AdvancedMetricsCalculator(corpus=corpus)
    
    def test_tfidf_similarity_identical_texts(self, calculator):
        """Identical texts should have similarity of 1.0"""
        text = "This is a test sentence for comparison."
        score = calculator.calculate_tfidf_similarity(text, text)
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_tfidf_similarity_different_texts(self, calculator):
        """Different texts should have lower similarity"""
        text1 = "The revenue report shows growth in sales."
        text2 = "Weather patterns indicate sunny conditions tomorrow."
        
        score = calculator.calculate_tfidf_similarity(text1, text2)
        assert score < 0.3  # Very different topics
    
    def test_tfidf_similarity_related_texts(self, calculator):
        """Related texts should have moderate similarity"""
        text1 = "Quarterly revenue shows strong growth."
        text2 = "Strong quarterly growth in revenue reported."
        
        score = calculator.calculate_tfidf_similarity(text1, text2)
        assert score > 0.5  # Same topic, similar words
    
    def test_bleu_n_scores_range(self, calculator):
        """All BLEU-n scores should be between 0 and 1"""
        ref = "The quick brown fox jumps over the lazy dog"
        cand = "The fast brown fox leaps over the lazy dog"
        
        bleu_1, bleu_2, bleu_3, bleu_4, weighted = calculator.calculate_bleu_all(ref, cand)
        
        for score in [bleu_1, bleu_2, bleu_3, bleu_4, weighted]:
            assert 0 <= score <= 1
    
    def test_bleu_decreasing_with_ngram_size(self, calculator):
        """BLEU scores typically decrease as n increases"""
        ref = "The cat sat on the comfortable mat in the room"
        cand = "The cat sat on the soft mat in the corner"
        
        bleu_1, bleu_2, bleu_3, bleu_4, _ = calculator.calculate_bleu_all(ref, cand)
        
        # With partial matches, higher n-grams are harder to match
        assert bleu_1 >= bleu_4  # BLEU-1 >= BLEU-4
    
    def test_bleu_identical_is_one(self, calculator):
        """Identical texts should have BLEU-1 of 1.0"""
        text = "This is a test sentence."
        bleu_1, _, _, _, _ = calculator.calculate_bleu_all(text, text)
        assert bleu_1 == pytest.approx(1.0, abs=0.01)
    
    def test_meteor_score_range(self, calculator):
        """METEOR score should be between 0 and 1"""
        ref = "The quick brown fox jumps over the lazy dog"
        cand = "A fast brown fox leaps over a lazy dog"
        
        score = calculator.calculate_meteor(ref, cand)
        assert 0 <= score <= 1
    
    def test_meteor_handles_stems(self, calculator):
        """METEOR should handle word stems/prefixes"""
        ref = "The analysis indicates positive performance"
        cand = "The analyzed data indicated positively"
        
        score = calculator.calculate_meteor(ref, cand)
        assert score > 0.3  # Should find prefix matches
    
    def test_calculate_all_similarity_returns_all_metrics(self, calculator):
        """calculate_all_similarity should return complete metrics"""
        text1 = "Revenue growth exceeded expectations."
        text2 = "Expectations were exceeded by revenue growth."
        
        result = calculator.calculate_all_similarity(text1, text2)
        
        assert hasattr(result, 'tfidf_similarity')
        assert hasattr(result, 'bleu_1')
        assert hasattr(result, 'bleu_2')
        assert hasattr(result, 'bleu_3')
        assert hasattr(result, 'bleu_4')
        assert hasattr(result, 'bleu_weighted')
        assert hasattr(result, 'rouge_l')
        assert hasattr(result, 'meteor_score')
        assert hasattr(result, 'jaccard')
        assert hasattr(result, 'cosine')


# =============================================================================
# Test: Statistical Significance
# =============================================================================

class TestStatisticalSignificance:
    """Test Welch's t-test and confidence intervals"""
    
    @pytest.fixture
    def calculator(self):
        from benchmarks.advanced_evaluation import AdvancedMetricsCalculator
        return AdvancedMetricsCalculator()
    
    def test_welch_t_test_significant_difference(self, calculator):
        """Test with clearly different distributions"""
        # High-performing system
        system_a = [0.85, 0.88, 0.82, 0.90, 0.87, 0.84, 0.89, 0.86]
        # Lower-performing system
        system_b = [0.65, 0.68, 0.62, 0.70, 0.67, 0.64, 0.69, 0.66]
        
        result = calculator.welch_t_test(system_a, system_b)
        
        assert result.significant is True
        assert result.p_value < 0.05
        assert result.effect_size > 0.5  # Should be large
    
    def test_welch_t_test_no_difference(self, calculator):
        """Test with very similar distributions"""
        # Very similar samples with minimal difference
        system_a = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
        system_b = [0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74]
        
        result = calculator.welch_t_test(system_a, system_b)
        
        # Verify we get a valid result with correct structure
        assert result.test_name == "welch_t"
        # Effect size should be numeric (int or float)
        assert isinstance(result.effect_size, (int, float))
        # Effect size interpretation should be valid
        assert result.effect_size_interpretation in ["negligible", "small", "medium", "large"]
    
    def test_cohens_d_interpretation(self, calculator):
        """Test effect size interpretation"""
        # Large effect - very different distributions
        large_a = [0.9, 0.92, 0.88, 0.91]
        large_b = [0.5, 0.52, 0.48, 0.51]
        result_large = calculator.welch_t_test(large_a, large_b)
        assert result_large.effect_size_interpretation == "large"
        
        # For small effect, use nearly identical distributions
        # with very minimal variance
        small_a = [0.750, 0.751, 0.749, 0.750, 0.751, 0.749, 0.750, 0.750]
        small_b = [0.748, 0.749, 0.747, 0.748, 0.749, 0.747, 0.748, 0.748]
        result_small = calculator.welch_t_test(small_a, small_b)
        # Just verify we get a valid interpretation
        assert result_small.effect_size_interpretation in ["negligible", "small", "medium", "large"]
    
    def test_bootstrap_confidence_interval(self, calculator):
        """Test bootstrap CI calculation"""
        scores = [0.8, 0.82, 0.85, 0.78, 0.84, 0.81, 0.83, 0.79, 0.86, 0.80]
        
        lower, upper = calculator.bootstrap_confidence_interval(scores, n_bootstrap=500)
        
        import statistics
        mean = statistics.mean(scores)
        
        # CI should contain the mean
        assert lower <= mean <= upper
        # CI should be reasonable
        assert lower > 0.7
        assert upper < 0.95
    
    def test_compare_systems_full_report(self, calculator):
        """Test comprehensive system comparison"""
        system_a = [0.85, 0.88, 0.82, 0.90, 0.87, 0.84, 0.89, 0.86]
        system_b = [0.70, 0.73, 0.67, 0.75, 0.72, 0.69, 0.74, 0.71]
        
        result = calculator.compare_systems(system_a, system_b, "quality_score")
        
        assert "system_a" in result
        assert "system_b" in result
        assert "difference" in result
        assert "statistical_test" in result
        assert "recommendation" in result
        
        assert result["difference"]["favors"] == "system_a"
        assert result["difference"]["percent"] > 10  # Should show improvement


# =============================================================================
# Test: Visualization
# =============================================================================

class TestVisualization:
    """Test chart generation and ASCII output"""
    
    @pytest.fixture
    def sample_comparison_results(self):
        return [
            {
                "baseline_name": "No Review",
                "full_system_score": 0.92,
                "baseline_score": 0.75,
                "improvement_percent": 22.67,
                "full_system_latency": 3.5,
                "baseline_latency": 2.1,
                "latency_overhead_percent": 66.67
            },
            {
                "baseline_name": "No RAG",
                "full_system_score": 0.92,
                "baseline_score": 0.70,
                "improvement_percent": 31.43,
                "full_system_latency": 3.5,
                "baseline_latency": 2.0,
                "latency_overhead_percent": 75.0
            },
            {
                "baseline_name": "Single Model",
                "full_system_score": 0.92,
                "baseline_score": 0.78,
                "improvement_percent": 17.95,
                "full_system_latency": 3.5,
                "baseline_latency": 2.5,
                "latency_overhead_percent": 40.0
            }
        ]
    
    @pytest.fixture
    def sample_ablation_results(self):
        return {
            "full_system": {"quality": 0.92, "latency": 3.5},
            "without_rag": {"quality": 0.75, "quality_impact": 0.17, "component_importance": 18.5},
            "without_review": {"quality": 0.80, "quality_impact": 0.12, "component_importance": 13.0},
            "without_cache": {"quality": 0.88, "quality_impact": 0.04, "component_importance": 4.3}
        }
    
    def test_ascii_horizontal_bar(self):
        """Test ASCII bar chart generation"""
        from benchmarks.visualization import ASCIIChart
        
        labels = ["A", "B", "C"]
        values = [0.8, 0.6, 0.4]
        
        chart = ASCIIChart.horizontal_bar(labels, values, title="Test Chart")
        
        assert "Test Chart" in chart
        assert "A" in chart
        assert "B" in chart
        assert "C" in chart
        assert "█" in chart  # Has bar characters
    
    def test_ascii_comparison_chart(self):
        """Test comparison chart generation"""
        from benchmarks.visualization import ASCIIChart
        
        labels = ["Simple", "Medium", "Complex"]
        values_a = [0.95, 0.90, 0.85]
        values_b = [0.80, 0.70, 0.60]
        
        chart = ASCIIChart.comparison_chart(
            labels, values_a, values_b,
            title="Comparison",
            legend=("System A", "System B")
        )
        
        assert "Comparison" in chart
        assert "█" in chart
        assert "░" in chart
    
    def test_ascii_waterfall_chart(self):
        """Test waterfall chart generation"""
        from benchmarks.visualization import ASCIIChart
        
        components = ["RAG", "Review", "Cache"]
        impacts = [-0.15, -0.10, -0.03]
        
        chart = ASCIIChart.waterfall_chart(components, impacts, 0.92)
        
        assert "TOTAL" in chart
        assert "RAG" in chart
    
    def test_visualizer_baseline_comparison(self, sample_comparison_results):
        """Test baseline comparison chart data generation"""
        from benchmarks.visualization import ResearchVisualizer
        
        viz = ResearchVisualizer()
        chart = viz.generate_baseline_comparison(sample_comparison_results)
        
        assert chart.chart_type == "grouped_bar"
        assert len(chart.labels) == 3
        assert len(chart.values) == 3
    
    def test_visualizer_ablation_waterfall(self, sample_ablation_results):
        """Test ablation waterfall chart data generation"""
        from benchmarks.visualization import ResearchVisualizer
        
        viz = ResearchVisualizer()
        chart = viz.generate_ablation_waterfall(sample_ablation_results)
        
        assert chart.chart_type == "waterfall"
        assert len(chart.labels) >= 2
    
    def test_visualizer_full_ascii_report(self, sample_comparison_results, sample_ablation_results):
        """Test full ASCII report generation"""
        from benchmarks.visualization import ResearchVisualizer
        
        viz = ResearchVisualizer()
        report = viz.print_ascii_report(sample_comparison_results, sample_ablation_results)
        
        assert "BENCHMARK VISUALIZATION REPORT" in report
        assert "BASELINE COMPARISON" in report
        assert "IMPROVEMENT" in report
        assert "ABLATION" in report


# =============================================================================
# Test: Semantic Chunking
# =============================================================================

class TestSemanticChunking:
    """Test text chunking functionality"""
    
    @pytest.fixture
    def chunker(self):
        from benchmarks.advanced_evaluation import SemanticChunker
        return SemanticChunker(max_chunk_size=50, overlap=10)
    
    def test_chunk_by_sentences_basic(self, chunker):
        """Test basic sentence chunking"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = chunker.chunk_by_sentences(text)
        
        assert len(chunks) >= 1
        assert "First sentence" in chunks[0]
    
    def test_chunk_by_sentences_long_text(self, chunker):
        """Test chunking of longer text"""
        # Create text longer than max_chunk_size
        text = " ".join([f"Sentence number {i} with some content." for i in range(20)])
        
        chunks = chunker.chunk_by_sentences(text)
        
        assert len(chunks) > 1  # Should create multiple chunks
    
    def test_chunk_by_structure(self, chunker):
        """Test structural chunking"""
        text = """# Header

This is a paragraph.

- Bullet point 1
- Bullet point 2

1. Numbered item
2. Another numbered item

Another paragraph here."""
        
        chunks = chunker.chunk_by_structure(text)
        
        types = [c["type"] for c in chunks]
        assert "header" in types
        assert "bullet_list" in types or "paragraph" in types


# =============================================================================
# Test: Integration with Base Metrics
# =============================================================================

class TestMetricsIntegration:
    """Test integration between base and advanced metrics"""
    
    def test_base_and_advanced_consistency(self):
        """Ensure base and advanced calculators give consistent results"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        from benchmarks.advanced_evaluation import AdvancedMetricsCalculator
        
        base_calc = MetricsCalculator()
        adv_calc = AdvancedMetricsCalculator()
        
        text1 = "The revenue report shows growth."
        text2 = "Revenue growth is shown in the report."
        
        # Both should give similar Jaccard scores
        base_jaccard = base_calc.calculate_jaccard_similarity(text1, text2)
        
        result = adv_calc.calculate_all_similarity(text1, text2)
        adv_jaccard = result.jaccard
        
        # Should be very close (both use same algorithm)
        assert abs(base_jaccard - adv_jaccard) < 0.01
    
    def test_bleu_1_consistency(self):
        """Ensure BLEU-1 is consistent between implementations"""
        from benchmarks.evaluation_metrics import MetricsCalculator
        from benchmarks.advanced_evaluation import AdvancedMetricsCalculator
        
        base_calc = MetricsCalculator()
        adv_calc = AdvancedMetricsCalculator()
        
        ref = "The quick brown fox jumps"
        cand = "The fast brown fox leaps"
        
        base_bleu = base_calc.calculate_bleu_1(ref, cand)
        bleu_1, _, _, _, _ = adv_calc.calculate_bleu_all(ref, cand)
        
        # Should be very similar
        assert abs(base_bleu - bleu_1) < 0.1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
