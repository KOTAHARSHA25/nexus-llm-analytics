"""
Nexus Analytics Module
======================

This module contains research-grade tools for offline analysis, verification,
and system tuning. These components are designed for "deep dive" investigations
and are not typically used in the hot path of query execution.

Components:
-----------
- text_analytics: Advanced NLP metrics (BLEU, METEOR) and Semantic Chunking.
- validation: Statistical validation frameworks (K-Fold, Bootstrap).
- tuning: Hyperparameter optimization (Grid Search, Sensitivity Analysis).
- visuals: Research-quality visualization generators.
"""

from .text_analytics import (
    SemanticChunker,
    AdvancedMetricsCalculator,
    AdvancedSimilarityMetrics,
    StatisticalResult
)

from .validation import (
    KFoldCrossValidator,
    BootstrapValidator,
    LearningCurveAnalyzer,
    VarianceBiasAnalyzer,
    LeaveOneOutValidator,
    run_comprehensive_cv_analysis,
)

from .tuning import (
    HyperparameterSpace,
    GridSearchAnalyzer,
    SensitivityAnalyzer,
    InteractionAnalyzer,
    run_hyperparameter_analysis,
)

from .visuals import (
    ResearchVisualizer,
    ChartData,
    ASCIIChart,
)

__all__ = [
    # text_analytics
    "SemanticChunker",
    "AdvancedMetricsCalculator",
    "AdvancedSimilarityMetrics",
    "StatisticalResult",
    # validation
    "KFoldCrossValidator",
    "BootstrapValidator",
    "LearningCurveAnalyzer",
    "VarianceBiasAnalyzer",
    "LeaveOneOutValidator",
    "run_comprehensive_cv_analysis",
    # tuning
    "HyperparameterSpace",
    "GridSearchAnalyzer",
    "SensitivityAnalyzer",
    "InteractionAnalyzer",
    "run_hyperparameter_analysis",
    # visuals
    "ResearchVisualizer",
    "ChartData",
    "ASCIIChart",
]

