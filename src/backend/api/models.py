from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import os
import time
from backend.core.model_selector import ModelSelector
from backend.core.user_preferences import get_preferences_manager
from backend.services.analysis_service import get_analysis_service

# API endpoint for model management and selection

router = APIRouter()


@router.get("/health", summary="Health Check", description="Simple health check that doesn't initialize models")
async def health_check() -> Dict[str, str]:
    """Simple health check that doesn't initialize models"""
    return {"status": "healthy", "message": "Models API is running"}


class ModelConfig(BaseModel):
    """Configuration for model selection."""
    primary_model: str = Field(..., description="Primary LLM for analysis tasks", example="llama3.1:8b")
    review_model: str = Field(..., description="Secondary LLM for review/validation", example="phi3:mini")
    embedding_model: str = Field(..., description="Model for generating embeddings", example="nomic-embed-text:latest")
    auto_selection: bool = Field(True, description="Enable automatic model selection based on system resources")
    allow_swap: bool = Field(False, description="Allow using swap memory for larger models")
    enable_intelligent_routing: bool = Field(False, description="Enable intelligent routing based on query complexity (experimental)")


class ModelInfo(BaseModel):
    """Information about a single model."""
    name: str = Field(..., description="Model name as registered in Ollama")
    size_gb: float = Field(..., description="Model size in gigabytes")
    description: str = Field(..., description="Human-readable model description")
    capabilities: List[str] = Field(..., description="List of model capabilities")
    compatible: bool = Field(..., description="Whether model is compatible with system")
    compatibility_message: str = Field(..., description="Details about compatibility")


class AvailableModelsResponse(BaseModel):
    """Response containing available models and system information."""
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
    """Get all models installed on user's Ollama (not hardcoded)"""
    try:
        import requests
        
        # Get system info
        memory_info = ModelSelector.get_system_memory()
        
        # Fetch actual models from Ollama
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            ollama_models = response.json().get("models", [])
        except Exception as e:
            logging.warning(f"Could not fetch Ollama models: {e}")
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
        logging.error(f"Failed to get available models: {e}")
        return {"error": str(e), "models": [], "ollama_running": False}

@router.get("/current")
async def get_current_models() -> Dict[str, str]:
    """Get currently selected models"""
    try:
        primary, review, embedding = ModelSelector.select_optimal_models()
        return {
            "primary_model": primary,
            "review_model": review,
            "embedding_model": embedding,
            "selection_method": "auto" if os.getenv("AUTO_MODEL_SELECTION", "true").lower() == "true" else "manual"
        }
    except Exception as e:
        logging.error(f"Failed to get current models: {e}")
        return {"error": str(e)}

@router.post("/configure")
async def configure_models(config: ModelConfig) -> Dict[str, str]:
    """Configure model preferences (Note: Requires restart to take effect)"""
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
        logging.error(f"Failed to configure models: {e}")
        return {"error": str(e)}

@router.get("/recommendations")
async def get_model_recommendations() -> Dict[str, Any]:
    """Get intelligent model recommendations based on system specs"""
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
        logging.error(f"Failed to generate recommendations: {e}")
        return {"error": str(e)}

@router.get("/preferences")
async def get_user_preferences() -> Dict[str, Any]:
    """Get current user preferences"""
    try:
        preferences_manager = get_preferences_manager()
        preferences = preferences_manager.load_preferences()
        
        return {
            "preferences": preferences.model_dump(),
            "is_first_time": preferences_manager.is_first_time_user(),
            "config_file_path": str(preferences_manager.preferences_file)
        }
    except Exception as e:
        logging.error(f"Failed to get user preferences: {e}")
        return {"error": str(e)}

@router.post("/preferences")
async def update_user_preferences(config: ModelConfig) -> Dict[str, str]:
    """Update user preferences"""
    try:
        preferences_manager = get_preferences_manager()
        
        # Log the model change prominently to console
        print("\n" + "=" * 80)
        print("🔄 MODEL CONFIGURATION CHANGED:")
        print(f"   Primary Model:   {config.primary_model}")
        print(f"   Review Model:    {config.review_model}")
        print(f"   Embedding Model: {config.embedding_model}")
        print(f"   Auto Selection:  {config.auto_selection}")
        print(f"   Allow Swap:      {config.allow_swap}")
        print(f"   Intelligent Routing: {config.enable_intelligent_routing} (Experimental)")
        print("=" * 80 + "\n")
        
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
            print("✅ Model configuration saved successfully - Changes will take effect on next analysis\n")
            return {
                "message": "User preferences updated successfully",
                "status": "success",
                "note": "Changes will take effect on next analysis request"
            }
        else:
            print("❌ Failed to save model configuration\n")
            return {"error": "Failed to save preferences", "status": "error"}
            
    except Exception as e:
        logging.error(f"Failed to update user preferences: {e}")
        return {"error": str(e), "status": "error"}

class ModelTestRequest(BaseModel):
    model_name: str

@router.post("/test-model")
async def test_model(request: ModelTestRequest) -> Dict[str, Any]:
    """Test a specific model with a simple query"""
    try:
        import time
        from backend.agents.model_initializer import get_model_initializer
        
        start_time = time.time()
        
        # Simple test query
        test_query = "Hello, please respond with 'Model test successful'"
        
        # Determine model name (strip ollama/ prefix if present for direct API call)
        model_name = request.model_name.replace("ollama/", "")
        
        # Use LLMClient directly to test the SPECIFIC model requested
        # Bypassing AnalysisService which routes to agents using their default models
        initializer = get_model_initializer()
        initializer.ensure_initialized()
        
        # Generate response using the specific model
        result = initializer.llm_client.generate(
            prompt=test_query,
            model=model_name,
            adaptive_timeout=False # Fast test
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Parse result
        success = result.get("success", False)
        # Check if response contains the expected text or is just non-empty
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
        logging.error(f"Model test failed: {e}")
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
    """Mark first-time setup as complete"""
    try:
        preferences_manager = get_preferences_manager()
        success = preferences_manager.mark_setup_complete()
        
        if success:
            return {"message": "Setup marked as complete", "status": "success"}
        else:
            return {"error": "Failed to update setup status", "status": "error"}
            
    except Exception as e:
        logging.error(f"Failed to mark setup complete: {e}")
        return {"error": str(e), "status": "error"}

@router.get("/test-results")
async def get_model_test_results() -> Dict[str, Any]:
    """Get model test results"""
    try:
        preferences_manager = get_preferences_manager()
        test_results = preferences_manager.get_model_test_results()
        
        return {
            "test_results": test_results,
            "status": "success"
        }
    except Exception as e:
        logging.error(f"Failed to get test results: {e}")
        return {"error": str(e), "status": "error"}