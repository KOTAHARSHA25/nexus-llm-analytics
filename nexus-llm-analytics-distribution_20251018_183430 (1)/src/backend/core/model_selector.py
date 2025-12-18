# Smart Model Selection Based on System Resources
# Automatically chooses the best LLM model based on available RAM

import psutil
import logging
import os
import time
from typing import Tuple, Dict
from .user_preferences import get_preferences_manager

# Cache for expensive operations (5 minute TTL)
_system_memory_cache = {"data": None, "timestamp": 0, "ttl": 300}
_model_selection_cache = {"data": None, "timestamp": 0, "ttl": 300}
_compatibility_cache = {}

class ModelSelector:
    """
    Intelligent model selection based on system resources and model requirements.
    Ensures optimal performance while preventing out-of-memory errors.
    """
    
    # NO HARDCODED MODELS - Fetch dynamically from Ollama
    MODEL_REQUIREMENTS = {}  # Will be populated dynamically
    
    @staticmethod
    def _get_installed_models() -> Dict[str, Dict]:
        """Fetch all installed models from Ollama dynamically (NO HARDCODING)"""
        import requests
        
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            models_data = response.json().get("models", [])
            
            models_info = {}
            for model in models_data:
                model_name = model.get("name", "")
                size_bytes = model.get("size", 0)
                size_gb = size_bytes / (1024**3)
                
                # Estimate RAM requirements (roughly 1.2x model size for inference)
                min_ram = max(size_gb * 1.2, 0.5)  # At least 500MB
                recommended_ram = size_gb * 1.5
                
                is_embedding = "embed" in model_name.lower()
                
                models_info[model_name] = {
                    "min_ram_gb": min_ram,
                    "recommended_ram_gb": recommended_ram,
                    "size_gb": size_gb,
                    "description": f"{'Embedding model' if is_embedding else 'Text generation model'} ({size_gb:.1f}GB)",
                    "capabilities": ["embeddings"] if is_embedding else ["text_generation", "analysis"],
                    "is_embedding": is_embedding
                }
            
            logging.info(f"✅ Found {len(models_info)} installed models from Ollama")
            return models_info
            
        except Exception as e:
            # Use debug level to avoid log spam when Ollama is not running
            logging.debug(f"Could not fetch models from Ollama: {e}")
            return {}
    
    @staticmethod
    def get_system_memory() -> Dict[str, float]:
        """Get system memory information in GB (cached for performance)"""
        global _system_memory_cache
        
        current_time = time.time()
        
        # Return cached data if still valid
        if (_system_memory_cache["data"] is not None and 
            current_time - _system_memory_cache["timestamp"] < _system_memory_cache["ttl"]):
            return _system_memory_cache["data"]
        
        # Fetch fresh system memory data
        memory = psutil.virtual_memory()
        memory_data = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent
        }
        
        # Cache the result
        _system_memory_cache["data"] = memory_data
        _system_memory_cache["timestamp"] = current_time
        
        return memory_data
    
    @staticmethod
    def select_optimal_models() -> Tuple[str, str, str]:
        """
        Select optimal models based on available system memory and user preferences.
        DYNAMICALLY fetches installed models - NO HARDCODING!
        
        Returns:
            Tuple[primary_model, review_model, embedding_model]
        """
        memory_info = ModelSelector.get_system_memory()
        available_ram = memory_info["available_gb"]
        total_ram = memory_info["total_gb"]
        
        logging.info(f"System Memory: {total_ram:.1f}GB total, {available_ram:.1f}GB available")
        
        # Fetch installed models dynamically
        installed_models = ModelSelector._get_installed_models()
        if not installed_models:
            # Don't log as error - Ollama not running is expected during development
            logging.debug("No models found in Ollama (Ollama not running or no models installed)")
            raise RuntimeError("No models installed. Run: ollama pull <model-name>")
        
        # Update MODEL_REQUIREMENTS with actual installed models
        ModelSelector.MODEL_REQUIREMENTS = installed_models
        
        # Get user preferences
        preferences_manager = get_preferences_manager()
        user_prefs = preferences_manager.load_preferences()
        
        # If user has preferences set, use them
        if user_prefs.primary_model and user_prefs.review_model and user_prefs.embedding_model:
            preferred_primary = user_prefs.primary_model
            preferred_review = user_prefs.review_model
            preferred_embedding = user_prefs.embedding_model
            auto_selection = user_prefs.auto_model_selection
        else:
            # No preferences set - auto-select from installed models
            logging.info("No user preferences found - auto-selecting from installed models")
            auto_selection = True
            
            # Find first non-embedding model for primary
            non_embedding = [m for m, info in installed_models.items() if not info.get("is_embedding", False)]
            embedding_models = [m for m, info in installed_models.items() if info.get("is_embedding", False)]
            
            if not non_embedding:
                raise RuntimeError("No text generation models installed! Install: ollama pull llama3.1:8b")
            if not embedding_models:
                logging.warning("No embedding models found - RAG will not work properly")
                embedding_models = [non_embedding[0]]  # Fallback
            
            # Sort by size (larger = better quality)
            non_embedding.sort(key=lambda m: installed_models[m]["size_gb"], reverse=True)
            
            preferred_primary = non_embedding[0]
            preferred_review = non_embedding[-1] if len(non_embedding) > 1 else non_embedding[0]
            preferred_embedding = embedding_models[0]
        
        if not auto_selection:
            # Use exact user configuration
            primary = f"ollama/{preferred_primary}"
            review = f"ollama/{preferred_review}"
            embedding = f"ollama/{preferred_embedding}"
            
            logging.info(f"✅ Using user-selected models: Primary={preferred_primary}, Review={preferred_review}, Embedding={preferred_embedding}")
            return primary, review, embedding
        
        # Smart selection with memory validation
        allow_swap = user_prefs.allow_swap_usage if hasattr(user_prefs, 'allow_swap_usage') else False
        
        primary_model = ModelSelector._select_best_model(
            preferred_primary, "primary", available_ram, total_ram, allow_swap, installed_models
        )
        review_model = ModelSelector._select_best_model(
            preferred_review, "review", available_ram, total_ram, allow_swap, installed_models
        )
        embedding_model = f"ollama/{preferred_embedding}"
        
        return primary_model, review_model, embedding_model
    
    @staticmethod
    def _select_best_model(preferred: str, role: str, available_ram: float, total_ram: float, allow_swap: bool, installed_models: Dict) -> str:
        """Select the best model based on preferences and memory constraints - NO HARDCODING"""
        
        # Clean model name
        clean_preferred = preferred.replace("ollama/", "")
        
        # Check if preferred model is installed and can fit
        if clean_preferred in installed_models:
            required_ram = installed_models[clean_preferred]["min_ram_gb"]
            
            if available_ram >= required_ram:
                logging.info(f"✅ Using preferred {role} model: {clean_preferred} (needs {required_ram:.1f}GB, have {available_ram:.1f}GB)")
                return f"ollama/{clean_preferred}"
            elif allow_swap and total_ram >= required_ram:
                logging.warning(f"⚠️ Using {role} model with swap: {clean_preferred} (needs {required_ram:.1f}GB, have {available_ram:.1f}GB available)")
                logging.warning(f"💡 Performance will be slower due to swap usage")
                return f"ollama/{clean_preferred}"
            else:
                logging.warning(f"❌ Cannot use preferred {role} model: {clean_preferred} (needs {required_ram:.1f}GB, have {available_ram:.1f}GB)")
        
        # Dynamic fallback - find smallest non-embedding model that fits
        non_embedding = [(name, info) for name, info in installed_models.items() if not info.get("is_embedding", False)]
        
        if not non_embedding:
            raise RuntimeError("No text generation models available!")
        
        # Sort by RAM requirement (smallest first for fallback)
        non_embedding.sort(key=lambda x: x[1]["min_ram_gb"])
        
        # Try each model in order
        for model_name, model_info in non_embedding:
            required_ram = model_info["min_ram_gb"]
            
            if available_ram >= required_ram:
                logging.info(f"✅ Fallback to {role} model: {model_name} (needs {required_ram:.1f}GB, have {available_ram:.1f}GB)")
                return f"ollama/{model_name}"
            elif allow_swap and total_ram >= required_ram:
                logging.warning(f"⚠️ Fallback to {role} model with swap: {model_name}")
                return f"ollama/{model_name}"
        
        # If no models fit in memory, use the smallest one anyway (will be slow)
        smallest_model = non_embedding[0][0]
        logging.error(f"❌ No models fit in available RAM ({available_ram:.1f}GB)")
        logging.error(f"💡 Using smallest model anyway: {smallest_model} (will be slow)")
        return f"ollama/{smallest_model}"
    

    
    @staticmethod
    def validate_model_compatibility(model_name: str) -> Tuple[bool, str]:
        """
        Validate if a model can run on the current system (cached for performance).
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple[is_compatible, message]
        """
        global _compatibility_cache
        
        # Check cache first (compatibility rarely changes during runtime)
        if model_name in _compatibility_cache:
            return _compatibility_cache[model_name]
        
        # Clean model name (remove ollama/ prefix if present)
        clean_name = model_name.replace("ollama/", "")
        
        # Fetch installed models dynamically
        installed_models = ModelSelector._get_installed_models()
        
        if clean_name not in installed_models:
            result = (False, f"❌ Model '{clean_name}' not installed")
            _compatibility_cache[model_name] = result
            return result
        
        model_info = installed_models[clean_name]
        memory_info = ModelSelector.get_system_memory()
        
        required_ram = model_info["min_ram_gb"]
        available_ram = memory_info["available_gb"]
        
        # Add small buffer tolerance (50MB) for borderline cases
        buffer_gb = 0.05
        
        if available_ram >= (required_ram - buffer_gb):
            status = "✅" if available_ram >= required_ram else "⚠️"
            result = (True, f"{status} {clean_name} compatible (needs {required_ram:.1f}GB, have {available_ram:.1f}GB)")
        else:
            result = (False, f"❌ {clean_name} incompatible (needs {required_ram:.1f}GB, only {available_ram:.1f}GB available)")
        
        # Cache the result for future calls
        _compatibility_cache[model_name] = result
        return result
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict:
        """Get detailed information about a model"""
        clean_name = model_name.replace("ollama/", "")
        return ModelSelector.MODEL_REQUIREMENTS.get(clean_name, {
            "min_ram_gb": 0,
            "description": f"Unknown model: {clean_name}",
            "capabilities": ["unknown"]
        })
    
    @staticmethod
    def recommend_system_config() -> Dict[str, any]:
        """Provide system configuration recommendations"""
        memory_info = ModelSelector.get_system_memory()
        total_ram = memory_info["total_gb"]
        available_ram = memory_info["available_gb"]
        
        recommendations = {
            "current_config": {
                "total_ram_gb": total_ram,
                "available_ram_gb": available_ram,
                "optimal_models": ModelSelector.select_optimal_models()
            },
            "recommendations": []
        }
        
        if total_ram < 8:
            recommendations["recommendations"].append({
                "type": "hardware_upgrade",
                "priority": "high",
                "message": f"Consider upgrading to 16GB+ RAM for optimal performance with Llama 3.1 8B",
                "current": f"{total_ram:.1f}GB",
                "recommended": "16GB+"
            })
        
        if available_ram < total_ram * 0.5:
            recommendations["recommendations"].append({
                "type": "memory_optimization",
                "priority": "medium",
                "message": "Close unnecessary applications to free up memory",
                "current": f"{available_ram:.1f}GB available",
                "target": f"{total_ram * 0.7:.1f}GB available"
            })
        
        return recommendations

if __name__ == "__main__":
    # Test the model selector
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 Nexus LLM Analytics - Model Selector")
    print("=" * 50)
    
    # Show system info
    memory_info = ModelSelector.get_system_memory()
    print(f"💾 System Memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available")
    
    # Select optimal models
    primary, review, embedding = ModelSelector.select_optimal_models()
    print(f"🤖 Selected Models:")
    print(f"   Primary: {primary}")
    print(f"   Review: {review}")  
    print(f"   Embedding: {embedding}")
    
    # Validate compatibility
    print(f"\n🔍 Compatibility Check:")
    for model in [primary, review, embedding]:
        compatible, message = ModelSelector.validate_model_compatibility(model)
        print(f"   {message}")
    
    # Show recommendations
    recommendations = ModelSelector.recommend_system_config()
    if recommendations["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in recommendations["recommendations"]:
            print(f"   [{rec['priority'].upper()}] {rec['message']}")