"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS LLM ANALYTICS - BENCHMARKS PACKAGE
═══════════════════════════════════════════════════════════════════════════════

Research-grade benchmarking and evaluation framework.

Modules:
- benchmark_dataset.json: 160 domain-spanning queries
- benchmark_runner: Execute benchmarks with stratified sampling
- evaluation_metrics: Accuracy, quality, efficiency metrics
- baseline_comparisons: Compare against baseline systems

Version: 1.0.0
"""

from .evaluation_metrics import (
    MetricsCalculator,
    ResearchMetrics,
    AccuracyMetrics,
    EfficiencyMetrics,
    QualityMetrics,
    SystemMetrics,
    AggregateMetrics,
    evaluate
)

from .baseline_comparisons import (
    BaselineRunner,
    BaselineConfig,
    ComparisonResult,
    AblationStudy,
    run_full_evaluation
)

from .benchmark_runner import (
    BenchmarkRunner,
    QueryResult,
    BenchmarkReport
)

__all__ = [
    # Metrics
    'MetricsCalculator',
    'ResearchMetrics',
    'AccuracyMetrics',
    'EfficiencyMetrics',
    'QualityMetrics',
    'SystemMetrics',
    'AggregateMetrics',
    'evaluate',
    # Baselines
    'BaselineRunner',
    'BaselineConfig',
    'ComparisonResult',
    'AblationStudy',
    'run_full_evaluation',
    # Benchmark Runner
    'BenchmarkRunner',
    'QueryResult',
    'BenchmarkReport'
]
