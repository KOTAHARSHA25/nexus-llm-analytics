"""Models API — LLM Model Management & Configuration Endpoints
==============================================================

Exposes model discovery, configuration, testing, preference management,
and recommendation endpoints. Serves the frontend’s Model Settings page
and provides the backend’s model hot-swap capability.

Endpoints
---------
``GET  /available``
    Query Ollama for all installed models with metadata.
``GET  /recommendations``
    RAM-aware model configuration suggestions.
``POST /preferences``
    Persist user model choices to config and apply at runtime.
``GET  /preferences``
    Read current user model preferences.
``POST /test-model``
    Send a probe prompt to verify a specific model responds.
``POST /configure`` *(deprecated)*
    Legacy ``.env`` configuration method.
``POST /setup-complete``
    Mark first-time wizard as finished.
``GET  /test-results``
    Retrieve saved model test outcomes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

import requests

from backend.core.engine.model_selector import ModelSelector
from backend.core.engine.user_preferences import get_preferences_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# NOTE: /health endpoint removed (Dec 2025) - duplicates /api/health/
# Use GET /api/health/ for health checks instead


class ModelConfig(BaseModel):
    """User-submitted model configuration.

    Attributes:
        primary_model:              Preferred primary analysis LLM tag.
        review_model:               Preferred secondary review LLM tag.
        embedding_model:            Embedding model for vector search.
        auto_selection:             Let the system override choices based on RAM.
        allow_swap:                 Permit swap-backed model loading.
        enable_intelligent_routing: Route queries by complexity (experimental).
    """
    primary_model: str = Field(..., description="Primary LLM for analysis tasks", example="llama3.1:8b")
    review_model: str = Field(..., description="Secondary LLM for review/validation", example="phi3:mini")
    embedding_model: str = Field(..., description="Model for generating embeddings", example="nomic-embed-text:latest")
    auto_selection: bool = Field(True, description="Enable automatic model selection based on system resources")
    allow_swap: bool = Field(False, description="Allow using swap memory for larger models")
    enable_intelligent_routing: bool = Field(False, description="Enable intelligent routing based on query complexity (experimental)")


class ModelInfo(BaseModel):
    """Metadata for a single Ollama-installed model."""
    name: str = Field(..., description="Model name as registered in Ollama")
    size_gb: float = Field(..., description="Model size in gigabytes")
    description: str = Field(..., description="Human-readable model description")
    capabilities: List[str] = Field(..., description="List of model capabilities")
    compatible: bool = Field(..., description="Whether model is compatible with system")
    compatibility_message: str = Field(..., description="Details about compatibility")


class AvailableModelsResponse(BaseModel):
    """Response payload listing installed models with system context."""
    models: List[Dict[str, Any]] = Field(..., description="List of installed models")
    total_models: int = Field(..., description="Total number of models")
    system_memory: Dict[str, float] = Field(..., description="System memory information")
    ollama_running: bool = Field(..., description="Whether Ollama service is accessible")
    current_config: Dict[str, Any] = Field(..., description="Current model configuration")


@router.get(
    "/available",
    response_model=AvailableModelsResponse,
    summary="Get Available Models",
    description="Get all models installed on user's Ollama (not hardcoded)"
)
async def get_available_models() -> Dict[str, Any]:
    """Query Ollama for every installed model and return structured metadata.

    Returns:
        Dict with ``models`` list, ``total_models`` count, ``system_memory``
        snapshot, ``ollama_running`` bool, and ``current_config``.
    """
    try:
        # Get system info
        memory_info = ModelSelector.get_system_memory()
        
        # Fetch actual models from Ollama
        from backend.core.config import get_settings
        settings = get_settings()
        ollama_url = settings.ollama_base_url
        
        try:
            import asyncio
            response = await asyncio.to_thread(
                lambda: requests.get(f"{ollama_url}/api/tags", timeout=5)
            )
            ollama_models = response.json().get("models", [])
        except Exception as e:
            logger.warning("Could not fetch Ollama models: %s", e)
            ollama_models = []
        
        # Process actual models from user's Ollama
        models = []
        for model in ollama_models:
            model_name = model.get("name", "")
            size_bytes = model.get("size", 0)
            size_gb = size_bytes / (1024**3)  # Convert to GB
            
            # Determine model type based on name
            is_embedding = "embed" in model_name.lower()
            capabilities = []
            if is_embedding:
                capabilities = ["embeddings", "vector_search"]
            else:
                capabilities = ["text_generation", "analysis", "code_generation"]
            
            models.append({
                "name": model_name,
                "size_gb": round(size_gb, 2),
                "size_bytes": size_bytes,
                "modified": model.get("modified_at", ""),
                "digest": model.get("digest", "")[:12],  # Short digest
                "is_embedding": is_embedding,
                "capabilities": capabilities,
                "compatible": True,  # If it's installed, it's compatible
                "details": model.get("details", {})
            })
        
        # Get current preferences
        preferences_manager = get_preferences_manager()
        prefs = preferences_manager.load_preferences()
        
        return {
            "models": models,
            "total_models": len(models),
            "system_memory": memory_info,
            "ollama_running": len(ollama_models) > 0,
            "current_config": {
                "primary": prefs.primary_model,
                "review": prefs.review_model,
                "embedding": prefs.embedding_model,
                "auto_selection": prefs.auto_model_selection,
                "enable_intelligent_routing": prefs.enable_intelligent_routing
            }
        }
    except Exception as e:
        logger.error("Failed to get available models: %s", e, exc_info=True)
        return {"error": str(e), "models": [], "ollama_running": False}

# NOTE: /current endpoint removed (Dec 2025) - duplicates /available (current_config field)
# Use GET /api/models/available for current model config

@router.post("/configure", deprecated=True)
async def configure_models(config: ModelConfig) -> Dict[str, str]:
    """
    [DEPRECATED] Configure model preferences via .env format.
    
    Prefer using POST /api/models/preferences which saves directly to config.
    This endpoint is kept for users who prefer manual .env configuration.
    """
    try:
        # This would typically update environment variables or config file
        # For now, return what would be set
        return {
            "message": "Model configuration received. Restart server to apply changes.",
            "config": {
                "PREFERRED_PRIMARY_MODEL": config.primary_model,
                "PREFERRED_REVIEW_MODEL": config.review_model, 
                "EMBEDDING_MODEL": config.embedding_model,
                "AUTO_MODEL_SELECTION": str(config.auto_selection).lower(),
                "ALLOW_SWAP_USAGE": str(config.allow_swap).lower()
            },
            "note": "Add these to your .env file and restart the server"
        }
    except Exception as e:
        logger.error("Failed to configure models: %s", e, exc_info=True)
        return {"error": str(e)}

@router.get("/recommendations")
async def get_model_recommendations() -> Dict[str, Any]:
    """Return RAM-aware model configuration suggestions."""
    try:
        memory_info = ModelSelector.get_system_memory()
        available_ram = memory_info["available_gb"]
        total_ram = memory_info["total_gb"]
        
        recommendations = []
        
        if total_ram >= 16.0:
            recommendations.append({
                "config": "High Performance",
                "primary": "llama3.1:8b",
                "review": "llama3.1:8b", 
                "description": "Best quality analysis and code generation",
                "performance": "Excellent"
            })
        
        if total_ram >= 8.0:
            recommendations.append({
                "config": "Balanced",
                "primary": "llama3.1:8b",
                "review": "phi3:mini",
                "description": "Good quality with reasonable speed",
                "performance": "Good"
            })
        
        recommendations.append({
            "config": "Memory Efficient", 
            "primary": "phi3:mini",
            "review": "phi3:mini",
            "description": "Decent quality, works on most systems",
            "performance": "Fair" if available_ram >= 2.0 else "Slow"
        })
        
        return {
            "system_info": memory_info,
            "recommendations": recommendations,
            "current_availability": f"{available_ram:.1f}GB available of {total_ram:.1f}GB total"
        }
        
    except Exception as e:
        logger.error("Failed to generate recommendations: %s", e, exc_info=True)
        return {"error": str(e)}

@router.get("/preferences")
async def get_user_preferences() -> Dict[str, Any]:
    """Return the persisted user model preferences and first-time flag."""
    try:
        preferences_manager = get_preferences_manager()
        preferences = preferences_manager.load_preferences()
        
        return {
            "preferences": preferences.model_dump(),
            "is_first_time": preferences_manager.is_first_time_user(),
            "config_file_path": str(preferences_manager.preferences_file)
        }
    except Exception as e:
        logger.error("Failed to get user preferences: %s", e, exc_info=True)
        return {"error": str(e)}

@router.post("/preferences")
async def update_user_preferences(config: ModelConfig) -> Dict[str, str]:
    """Persist user model preferences and apply at runtime."""
    try:
        preferences_manager = get_preferences_manager()
        
        # Log the model change
        logger.info(
            "Model config changed: primary=%s, review=%s, embedding=%s, auto=%s, routing=%s",
            config.primary_model, config.review_model, config.embedding_model,
            config.auto_selection, config.enable_intelligent_routing,
        )
        
        success = preferences_manager.set_model_config(
            primary=config.primary_model,
            review=config.review_model,
            embedding=config.embedding_model,
            auto_selection=config.auto_selection
        )
        
        if config.allow_swap:
            preferences_manager.update_preferences(allow_swap_usage=config.allow_swap)
        
        # Update intelligent routing setting
        if hasattr(config, 'enable_intelligent_routing'):
            preferences_manager.update_preferences(enable_intelligent_routing=config.enable_intelligent_routing)
        
        if success:
            logger.info("Model configuration saved successfully")
            return {
                "message": "User preferences updated successfully",
                "status": "success",
                "note": "Changes will take effect on next analysis request"
            }
        else:
            logger.error("Failed to save model configuration", exc_info=True)
            return {"error": "Failed to save preferences", "status": "error"}
            
    except Exception as e:
        logger.error("Failed to update user preferences: %s", e, exc_info=True)
        return {"error": str(e), "status": "error"}

class ModelTestRequest(BaseModel):
    """Request payload for the ``/test-model`` endpoint."""
    model_name: str

def _is_embedding_model(model_name: str) -> bool:
    """Return ``True`` if the model tag indicates an embedding-only model."""
    embedding_indicators = ['embed', 'nomic-embed', 'bge', 'e5', 'gte', 'embedding']
    model_lower = model_name.lower()
    return any(indicator in model_lower for indicator in embedding_indicators)

def _test_embedding_model(model_name: str) -> dict:
    """Probe an embedding model via the ``/api/embeddings`` endpoint.

    Args:
        model_name: Ollama model tag to test.

    Returns:
        Dict with ``success``, ``response_time``, ``result``, and ``error`` keys.
    """
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model_name, "prompt": "test embedding"}
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        end_time = time.time()
        
        if "embedding" in data and len(data["embedding"]) > 0:
            return {
                "success": True,
                "response_time": end_time - start_time,
                "result": f"Embedding generated ({len(data['embedding'])} dimensions)",
                "error": None
            }
        else:
            return {
                "success": False,
                "response_time": end_time - start_time,
                "result": "",
                "error": "No embedding returned"
            }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "response_time": time.time() - start_time,
            "result": "",
            "error": f"HTTP Error: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "response_time": time.time() - start_time,
            "result": "",
            "error": str(e)
        }

@router.post("/test-model")
async def test_model(request: ModelTestRequest) -> Dict[str, Any]:
    """Send a probe prompt to verify a specific model responds."""
    try:
        from backend.agents.model_manager import get_model_manager
        
        start_time = time.time()
        
        # Determine model name (strip ollama/ prefix if present for direct API call)
        model_name = request.model_name.replace("ollama/", "")
        
        # Check if this is an embedding model - use different test method
        if _is_embedding_model(model_name):
            # Use embeddings API for embedding models
            test_result = _test_embedding_model(model_name)
            success = test_result["success"]
            response_time = test_result["response_time"]
            response_text = test_result["result"]
            error_msg = test_result["error"]
        else:
            # Use LLMClient directly to test text generation models
            test_query = "Hello, please respond with 'Model test successful'"
            
            initializer = get_model_manager()
            initializer.ensure_initialized()
            
            # Generate response using the specific model
            result = initializer.llm_client.generate(
                prompt=test_query,
                model=model_name,
                adaptive_timeout=False  # Fast test
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Parse result
            success = result.get("success", False)
            response_text = result.get("response", "")
            if not response_text:
                success = False
                error_msg = "Empty response from model"
            else:
                error_msg = result.get("error")
        
        # Save test result
        preferences_manager = get_preferences_manager()
        
        preferences_manager.save_model_test_result(
            model_name=request.model_name,
            success=success,
            response_time=response_time,
            error=error_msg
        )
        
        return {
            "model": request.model_name,
            "success": success,
            "response_time": response_time,
            "result": response_text,
            "error": error_msg,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error("Model test failed: %s", e, exc_info=True)
        preferences_manager = get_preferences_manager()
        preferences_manager.save_model_test_result(
            model_name=request.model_name,
            success=False,
            error=str(e)
        )
        return {
            "model": request.model_name,
            "success": False,
            "error": str(e),
            "status": "failed"
        }

@router.post("/setup-complete")
async def mark_setup_complete() -> Dict[str, str]:
    """Mark the first-time setup wizard as finished."""
    try:
        preferences_manager = get_preferences_manager()
        success = preferences_manager.mark_setup_complete()
        
        if success:
            return {"message": "Setup marked as complete", "status": "success"}
        else:
            return {"error": "Failed to update setup status", "status": "error"}
            
    except Exception as e:
        logger.error("Failed to mark setup complete: %s", e, exc_info=True)
        return {"error": str(e), "status": "error"}

@router.get("/test-results")
async def get_model_test_results() -> Dict[str, Any]:
    """Return persisted model test outcomes."""
    try:
        preferences_manager = get_preferences_manager()
        test_results = preferences_manager.get_model_test_results()
        
        return {
            "test_results": test_results,
            "status": "success"
        }
    except Exception as e:
        logger.error("Failed to get test results: %s", e, exc_info=True)
        return {"error": str(e), "status": "error"}