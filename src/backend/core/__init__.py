"""
Backend Core Package — Nexus LLM Analytics v2.0
================================================

Enterprise-grade core layer providing the foundational services for the
Nexus multi-agent data-analysis platform.  Every public symbol is
re-exported here for convenient top-level access while preserving full
backward compatibility with v1.x import paths.

Subpackages
-----------
engine/
    Intelligent routing, model selection, self-correction, CoT parsing,
    fallback management, query orchestration, paper metrics, and user
    preferences.
security/
    RestrictedPython sandbox, security guards, audit logging, rate
    limiting, and threat detection.

Top-Level Modules
-----------------
llm_client           Synchronous & async Ollama communication with
                     circuit-breaker protection and adaptive timeouts.
config               Pydantic-based centralised configuration with
                     environment presets and runtime validation.
analysis_manager     Cancellable analysis lifecycle management.
chromadb_client      Vector storage with hybrid search & citation
                     tracking (Phase 3).
code_execution_history  Persistent code-execution replay system.
document_indexer     Semantic chunking and optimised RAG indexing.
dynamic_planner      Domain-agnostic analysis-plan generation.
enhanced_cache_integration  Multi-tier caching (L1/L2/L3) with smart
                     warming and tag-based invalidation.
enhanced_logging     JSON + coloured console log formatters.
enhanced_reports     Publication-ready PDF/Excel report generation.
error_handling       Centralised exception hierarchy with severity
                     levels, categories, and decorator support.
memory_optimizer     System RAM inspection and LLM readiness checks.
optimized_data_structures  High-performance trie, hash map, LRU cache.
optimizers           Unified memory/performance/startup optimisation.
phase1_integration   Coordinator aggregating Phase 1 subsystems.
plugin_system        Plug-and-play agent architecture with registry.
query_complexity_analyzer  Hierarchical decision-based complexity
                     scoring (V2, 95 % accuracy target).
query_parser         NLP query parser with intent classification.
rate_limiter         Token-bucket / sliding-window API protection.
semantic_mapper      Domain-agnostic column-concept mapping.
utils                Shared helpers (JSON formatter, audit trail).
websocket_manager    Real-time WebSocket progress broadcasting.

Backward Compatibility
----------------------
All v1.x import paths continue to work unchanged.  New v2.0 enterprise
classes are additive and opt-in.

.. versionadded:: 2.0
   Comprehensive re-exports and enterprise documentation.

Author
------
Nexus Analytics Research Team

Date
----
February 2026
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Lazy imports to avoid circular dependencies and speed up startup.
#
# Prefer importing specific modules directly:
#   from backend.core.llm_client import LLMClient
#   from backend.core.config import get_settings
#   from backend.core.engine import IntelligentRouter
#   from backend.core.security import SecurityGuards
# ---------------------------------------------------------------------------

__all__ = [
    # --- v1.x public API (preserved) ---
    'LLMClient',
    'AdvancedCache',

    # --- v2.0 additions ---
    # Subpackages (use ``from backend.core.engine import …``)
    'engine',
    'security',

    # Top-level singletons / accessors
    'get_settings',
    'analysis_manager',
    'get_execution_history',
    'get_semantic_mapper',
    'get_enhanced_cache_manager',
    'global_rate_limiter',
    'error_handler',
    'connection_manager',
    'progress_tracker',
]


def __getattr__(name: str):
    """Lazy accessor for heavyweight symbols — avoids circular imports."""
    # Map attribute names to their module paths for on-demand loading.
    _lazy_imports = {
        'LLMClient': 'backend.core.llm_client',
        'AdvancedCache': 'backend.core.enhanced_cache_integration',
        'get_settings': 'backend.core.config',
        'analysis_manager': 'backend.core.analysis_manager',
        'get_execution_history': 'backend.core.code_execution_history',
        'get_semantic_mapper': 'backend.core.semantic_mapper',
        'get_enhanced_cache_manager': 'backend.core.enhanced_cache_integration',
        'global_rate_limiter': 'backend.core.rate_limiter',
        'error_handler': 'backend.core.error_handling',
        'connection_manager': 'backend.core.websocket_manager',
        'progress_tracker': 'backend.core.websocket_manager',
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")