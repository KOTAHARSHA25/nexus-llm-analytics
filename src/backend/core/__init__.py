"""
Backend Core Package
====================
Core functionality for the Nexus LLM Analytics system.

Key Components:
- llm_client: LLM communication with Ollama
- intelligent_router: Query complexity-based model routing
- self_correction_engine: Generator→Critic→Feedback loop
- advanced_cache: High-performance caching
- plugin_system: Extensible agent plugins
"""

# Lazy imports to avoid circular dependencies and speed up startup
# Import specific modules as needed:
#   from backend.core.llm_client import LLMClient
#   from backend.core.advanced_cache import AdvancedCache

__all__ = [
    'LLMClient',
    'AdvancedCache',
]