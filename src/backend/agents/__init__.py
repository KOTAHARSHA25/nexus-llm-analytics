"""
Agents Module — Nexus LLM Analytics Platform
=============================================

Provides centralized model lifecycle management and component initialization
for the multi-agent analysis pipeline. This module is the single entry point
for acquiring LLM clients, vector stores, and orchestrator instances across
the entire backend.

Exports:
    ModelManager         — Singleton class managing lazy-loaded LLM and infra components.
    get_model_manager    — Thread-safe factory function for the ModelManager singleton.
    reset_model_manager  — Test-only helper to clear the singleton for test isolation.

Key Capabilities:
    - **Warmup**: ``manager.warmup()`` / ``await manager.async_warmup()`` sends a
      probe to Ollama, forcing model load into VRAM before the first real query.
    - **Model Inventory**: ``manager.list_available_models()`` returns all Ollama
      models with sizes, families, and quantization levels.
    - **Health Check**: ``manager.is_healthy()`` returns a simple bool for load
      balancer readiness probes.
    - **Hot-Reload**: ``manager.reload_models()`` swaps models at runtime
      without server restart after preference changes.
    - **Shutdown**: ``manager.shutdown()`` releases all resources cleanly.

Usage::

    from backend.agents import get_model_manager

    manager = get_model_manager()
    llm = manager.llm_client           # Lazy-initialized Ollama LLM client
    chroma = manager.chroma_client      # Lazy-initialized ChromaDB vector store
    orchestrator = manager.orchestrator  # Query routing & execution planning

    # Pre-flight warmup (async in FastAPI lifespan)
    result = await manager.async_warmup()
    assert result["success"]

    # Health check
    print(manager.is_healthy())  # True / False

    # Model inventory for UI dropdowns
    models = manager.list_available_models()
    print(models["count"], "models installed")

    # Hot-reload after user changes model preferences
    manager.reload_models()
"""

from .model_manager import ModelManager, get_model_manager, reset_model_manager

__all__ = [
    "ModelManager",
    "get_model_manager",
    "reset_model_manager",
]