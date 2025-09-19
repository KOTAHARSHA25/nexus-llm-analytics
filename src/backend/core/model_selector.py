# Smart Model Selection Based on System Resources
# Automatically chooses the best LLM model based on available RAM

import psutil
import logging
import os
from typing import Tuple, Dict
from .user_preferences import get_preferences_manager

class ModelSelector:
    """
    Intelligent model selection based on system resources and model requirements.
    Ensures optimal performance while preventing out-of-memory errors.
    """
    
    # Model memory requirements (in GB) - realistic estimates based on actual usage
    MODEL_REQUIREMENTS = {
        "llama3.1:8b": {
            "min_ram_gb": 6.0,      # Actual minimum to load model
            "recommended_ram_gb": 10.0,  # For smooth operation
            "description": "High-performance 8B parameter model",
            "capabilities": ["advanced_reasoning", "code_generation", "complex_analysis"]
        },
        "phi3:mini": {
            "min_ram_gb": 4.0,      # Updated based on actual Ollama requirements (3.8GB)
            "recommended_ram_gb": 5.0,   # Recommended for good performance
            "description": "Efficient 3.8B parameter model",
            "capabilities": ["basic_reasoning", "code_review", "simple_analysis"]
        },
        "tinyllama": {
            "min_ram_gb": 1.0,      # Very lightweight model
            "recommended_ram_gb": 1.5,   # Recommended for good performance
            "description": "Ultra-lightweight 1.1B parameter model",
            "capabilities": ["basic_reasoning", "simple_text_generation", "lightweight_analysis"]
        },
        "nomic-embed-text": {
            "min_ram_gb": 0.3,      # Embedding models are very light
            "recommended_ram_gb": 0.8,
            "description": "Embedding model for RAG",
            "capabilities": ["text_embeddings", "similarity_search"]
        }
    }
    
    @staticmethod
    def get_system_memory() -> Dict[str, float]:
        """Get system memory information in GB"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent
        }
    
    @staticmethod
    def select_optimal_models() -> Tuple[str, str, str]:
        """
        Select optimal models based on available system memory and user preferences.
        
        Returns:
            Tuple[primary_model, review_model, embedding_model]
        """
        memory_info = ModelSelector.get_system_memory()
        available_ram = memory_info["available_gb"]
        total_ram = memory_info["total_gb"]
        
        logging.info(f"System Memory: {total_ram:.1f}GB total, {available_ram:.1f}GB available")
        
        # Check if auto-selection is enabled
        auto_selection = os.getenv("AUTO_MODEL_SELECTION", "true").lower() == "true"
        allow_swap = os.getenv("ALLOW_SWAP_USAGE", "false").lower() == "true"
        memory_buffer = float(os.getenv("MEMORY_BUFFER_GB", "0.5"))
        
        # Get user preferences (prioritize user preferences over env vars)
        preferences_manager = get_preferences_manager()
        user_prefs = preferences_manager.load_preferences()
        
        # Use user preferences if available, fall back to env vars
        preferred_primary = user_prefs.primary_model if user_prefs.primary_model else os.getenv("PREFERRED_PRIMARY_MODEL", "llama3.1:8b")
        preferred_review = user_prefs.review_model if user_prefs.review_model else os.getenv("PREFERRED_REVIEW_MODEL", "phi3:mini")
        preferred_embedding = user_prefs.embedding_model if user_prefs.embedding_model else os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        
        # Override auto-selection with user preference
        if hasattr(user_prefs, 'auto_model_selection'):
            auto_selection = user_prefs.auto_model_selection
        
        if not auto_selection:
            # Use exact user configuration without modification
            primary = f"ollama/{preferred_primary}"
            review = f"ollama/{preferred_review}"
            embedding = f"ollama/{preferred_embedding}"
            
            logging.info("Auto model selection disabled - using exact user preferences")
            return primary, review, embedding
        
        # Smart model selection with user preferences and memory awareness
        logging.info(f"User preferences: Primary={preferred_primary}, Review={preferred_review}")
        
        # Try to use preferred models with memory validation
        primary_model = ModelSelector._select_best_model(
            preferred_primary, "primary", available_ram, total_ram, allow_swap
        )
        review_model = ModelSelector._select_best_model(
            preferred_review, "review", available_ram, total_ram, allow_swap
        )
        embedding_model = f"ollama/{preferred_embedding}"
        
        return primary_model, review_model, embedding_model
    
    @staticmethod
    def _select_best_model(preferred: str, role: str, available_ram: float, total_ram: float, allow_swap: bool) -> str:
        """Select the best model based on preferences and memory constraints"""
        
        # Clean model name
        clean_preferred = preferred.replace("ollama/", "")
        
        # Check if preferred model can fit
        if clean_preferred in ModelSelector.MODEL_REQUIREMENTS:
            required_ram = ModelSelector.MODEL_REQUIREMENTS[clean_preferred]["min_ram_gb"]
            
            if available_ram >= required_ram:
                logging.info(f"✅ Using preferred {role} model: {clean_preferred} (needs {required_ram}GB, have {available_ram:.1f}GB)")
                return f"ollama/{clean_preferred}"
            elif allow_swap and total_ram >= required_ram:
                logging.warning(f"⚠️ Using {role} model with swap: {clean_preferred} (needs {required_ram}GB, have {available_ram:.1f}GB available)")
                logging.warning(f"💡 Performance will be slower due to swap usage")
                return f"ollama/{clean_preferred}"
            else:
                logging.warning(f"❌ Cannot use preferred {role} model: {clean_preferred} (needs {required_ram}GB, have {available_ram:.1f}GB)")
        
        # Fallback logic based on available RAM
        if available_ram >= 1.0:
            fallback = "phi3:mini"
            logging.info(f"🔄 Fallback {role} model: {fallback}")
            return f"ollama/{fallback}"
        else:
            # Suggest installing a smaller model
            logging.error(f"🔴 Insufficient RAM for any model. Consider installing tinyllama or qwen2:0.5b")
            return f"ollama/phi3:mini"  # Still try, might work with swap
    
    @staticmethod
    def validate_model_compatibility(model_name: str) -> Tuple[bool, str]:
        """
        Validate if a model can run on the current system.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple[is_compatible, message]
        """
        # Clean model name (remove ollama/ prefix if present)
        clean_name = model_name.replace("ollama/", "")
        
        if clean_name not in ModelSelector.MODEL_REQUIREMENTS:
            return True, f"Unknown model '{clean_name}' - assuming compatible"
        
        model_info = ModelSelector.MODEL_REQUIREMENTS[clean_name]
        memory_info = ModelSelector.get_system_memory()
        
        required_ram = model_info["min_ram_gb"]
        available_ram = memory_info["available_gb"]
        
        # Add small buffer tolerance (50MB) for borderline cases
        buffer_gb = 0.05
        
        if available_ram >= (required_ram - buffer_gb):
            status = "✅" if available_ram >= required_ram else "⚠️"
            return True, f"{status} {clean_name} compatible (needs {required_ram}GB, have {available_ram:.1f}GB)"
        else:
            return False, f"❌ {clean_name} incompatible (needs {required_ram}GB, only {available_ram:.1f}GB available)"
    
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