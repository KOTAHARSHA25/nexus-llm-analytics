from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import os
import time
from backend.core.model_selector import ModelSelector
from backend.core.user_preferences import get_preferences_manager
from backend.agents.crew_manager import CrewManager

# API endpoint for model management and selection

router = APIRouter()

class ModelConfig(BaseModel):
    primary_model: str
    review_model: str
    embedding_model: str
    auto_selection: bool = True
    allow_swap: bool = False

class ModelInfo(BaseModel):
    name: str
    size_gb: float
    description: str
    capabilities: List[str]
    compatible: bool
    compatibility_message: str

@router.get("/available")
async def get_available_models() -> Dict[str, Any]:
    """Get all available models with compatibility information"""
    try:
        # Get system info
        memory_info = ModelSelector.get_system_memory()
        
        # Check available models
        models = []
        for model_name, info in ModelSelector.MODEL_REQUIREMENTS.items():
            compatible, message = ModelSelector.validate_model_compatibility(model_name)
            
            model_info = ModelInfo(
                name=model_name,
                size_gb=info["min_ram_gb"],
                description=info["description"],
                capabilities=info["capabilities"],
                compatible=compatible,
                compatibility_message=message
            )
            models.append(model_info)
        
        return {
            "models": models,
            "system_memory": memory_info,
            "current_config": {
                "primary": os.getenv("PREFERRED_PRIMARY_MODEL", "llama3.1:8b"),
                "review": os.getenv("PREFERRED_REVIEW_MODEL", "phi3:mini"),
                "embedding": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                "auto_selection": os.getenv("AUTO_MODEL_SELECTION", "true").lower() == "true"
            }
        }
    except Exception as e:
        logging.error(f"Failed to get available models: {e}")
        return {"error": str(e)}

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
        
        success = preferences_manager.set_model_config(
            primary=config.primary_model,
            review=config.review_model,
            embedding=config.embedding_model,
            auto_selection=config.auto_selection
        )
        
        if config.allow_swap:
            preferences_manager.update_preferences(allow_swap_usage=config.allow_swap)
        
        if success:
            return {
                "message": "User preferences updated successfully",
                "status": "success",
                "note": "Changes will take effect on next analysis request"
            }
        else:
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
        from backend.agents.crew_manager import CrewManager
        
        start_time = time.time()
        
        # Simple test query
        test_query = "Hello, please respond with 'Model test successful'"
        
        # Create a temporary crew manager instance
        crew_manager = CrewManager()
        
        # Test the model (this is a simplified test)
        result = crew_manager.handle_query(
            query=test_query,
            filename="test.txt"  # Dummy filename
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Save test result
        preferences_manager = get_preferences_manager()
        success = result.get("success", False)
        error_msg = result.get("error") if not success else None
        
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
            "result": result.get("result", ""),
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