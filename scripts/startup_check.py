# Nexus LLM Analytics - Smart Startup Script
# Automatically checks system resources and starts with optimal configuration

import os
import sys
import logging
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from core.model_selector import ModelSelector
from core.memory_optimizer import MemoryOptimizer

def check_system_readiness():
    """
    Check if system is ready for LLM operations and provide guidance.
    
    Returns:
        Tuple[is_ready, messages, recommendations]
    """
    print("🚀 Nexus LLM Analytics - System Readiness Check")
    print("=" * 55)
    
    # Get system information
    memory_info = ModelSelector.get_system_memory()
    optimization_plan = MemoryOptimizer.get_optimization_plan()
    
    print(f"💾 System Memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available")
    
    # Select optimal models
    primary_model, review_model, embedding_model = ModelSelector.select_optimal_models()
    print(f"🤖 Selected Models:")
    print(f"   Primary: {primary_model}")
    print(f"   Review: {review_model}")
    print(f"   Embedding: {embedding_model}")
    
    # Check compatibility
    print(f"\n🔍 Model Compatibility:")
    all_compatible = True
    for model_name, model_type in [(primary_model, "Primary"), (review_model, "Review"), (embedding_model, "Embedding")]:
        compatible, message = ModelSelector.validate_model_compatibility(model_name)
        print(f"   {message}")
        if not compatible:
            all_compatible = False
    
    # Provide recommendations if not all models are compatible
    if not all_compatible:
        print(f"\n💡 Memory Optimization Recommendations:")
        recommendations = optimization_plan["optimization_recommendations"]
        
        if recommendations:
            for rec in recommendations:
                print(f"   {rec}")
            
            estimated_available = optimization_plan["estimated_available_after_cleanup"]
            print(f"\n📈 Estimated Available After Cleanup: {estimated_available:.1f}GB")
            
            # Check if cleanup would help
            phi3_compatible_after = estimated_available >= 2.0
            llama_compatible_after = estimated_available >= 6.0
            
            if llama_compatible_after:
                print(f"   ✅ After cleanup: Could run Llama 3.1 8B (high performance)")
            elif phi3_compatible_after:
                print(f"   ✅ After cleanup: Could run Phi-3 Mini (good performance)")
            else:
                print(f"   ⚠️ Even after cleanup: Limited model options")
        else:
            print(f"   ✅ Memory usage is already optimized")
    
    # Final recommendations
    print(f"\n🎯 Recommendations:")
    
    if all_compatible:
        print(f"   ✅ System ready! All models can run with current memory.")
        return True, [], []
    elif memory_info['available_gb'] < 2.0:
        print(f"   🟡 Close some applications to free up memory before starting analysis")
        print(f"   🔧 Run: python -m backend.core.memory_optimizer for detailed guidance")
        return False, ["Low memory"], ["Close applications", "Run memory optimizer"]
    else:
        print(f"   ✅ System ready with lighter models (Phi-3 Mini)")
        return True, ["Using lighter models"], ["Consider memory optimization for better performance"]

def start_with_memory_check():
    """Start the system with automatic memory checking"""
    is_ready, messages, recommendations = check_system_readiness()
    
    if not is_ready:
        print(f"\n⚠️ System Not Ready")
        print(f"Please address the following before starting:")
        for rec in recommendations:
            print(f"   • {rec}")
        print(f"\nThen run this script again or start the backend manually.")
        return False
    
    print(f"\n🚀 System Ready! Starting Nexus LLM Analytics...")
    return True

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Check system readiness
    ready = start_with_memory_check()
    
    if ready:
        print(f"\n🔗 Next Steps:")
        print(f"   1. Start Ollama: ollama serve")
        print(f"   2. Load models: ollama pull phi3:mini && ollama pull nomic-embed-text")
        print(f"   3. Start backend: uvicorn backend.main:app --reload")
        print(f"   4. Start frontend: cd frontend && npm run dev")
        print(f"\n📖 For detailed memory optimization: python -m backend.core.memory_optimizer")
    else:
        print(f"\n🔧 Memory Optimization Help: python -m backend.core.memory_optimizer")