"""Visualization Module — Nexus LLM Analytics
=============================================

Deterministic, template-based chart generation using Plotly
and multi-library scaffold templates (matplotlib, seaborn,
plotly, altair, ggplot).

Submodules
----------
scaffold
    Library-specific code templates and best-practice instructions.
dynamic_charts
    Data-structure analyser and Plotly chart generator.

v2.0 Enterprise Additions
-------------------------
* Enterprise module docstring with submodule catalogue.
"""

from __future__ import annotations

from .scaffold import ChartScaffold, VisualizationGoal
from .dynamic_charts import ChartTypeAnalyzer, DynamicChartGenerator

__all__ = [
    'ChartScaffold',
    'VisualizationGoal',
    'ChartTypeAnalyzer',
    'DynamicChartGenerator'
]
