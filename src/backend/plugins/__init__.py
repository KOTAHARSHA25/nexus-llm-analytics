"""Nexus LLM Analytics — Plugin Agents Package
===============================================

Contains all specialist agent plugins that extend
:class:`~backend.core.plugin_system.BasePluginAgent`.

Agents are **auto-discovered** at runtime by
:func:`~backend.core.plugin_system.AgentRegistry.discover_agents`
which scans this directory for concrete ``BasePluginAgent`` subclasses.

Available Agents
----------------
``DataAnalystAgent``     General-purpose tabular / statistical analysis.
``FinancialAgent``       Domain-specific financial metrics and ratios.
``MLInsightsAgent``      Scikit-learn model training and evaluation.
``StatisticalAgent``     Hypothesis testing, distributions, correlations.
``TimeSeriesAgent``      Trend, seasonality, and forecasting analysis.
``RagAgent``             Retrieval-augmented generation over documents.
``ReporterAgent``        Structured report compilation.
``ReviewerAgent``        Secondary-model review and validation.
``VisualizerAgent``      Chart generation via Plotly / Matplotlib.
"""
