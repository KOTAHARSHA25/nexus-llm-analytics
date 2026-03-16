"""Nexus LLM Analytics — Services Package
=========================================

High-level service facades that orchestrate agents, caching, and
result interpretation for the API layer.

Submodules
----------
``analysis_service``
    :class:`AnalysisService` — singleton façade routing queries through
    the plugin registry, applying caching, Chain-of-Thought review,
    and result formatting.
"""
