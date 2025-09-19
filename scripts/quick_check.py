# Quick Memory Check for Nexus LLM Analytics
# Simple script to check if you have enough RAM to run AI analysis

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from core.model_selector import ModelSelector
import psutil

def quick_memory_check():
    """Quick check and simple recommendations"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    
    print("🧠 Nexus LLM - Quick Memory Check")
    print("=" * 40)
    print(f"💾 Available RAM: {available_gb:.1f}GB of {total_gb:.1f}GB total")
    
    if available_gb >= 1.0:
        print("✅ Ready to run Phi-3 Mini model!")
        print("🚀 You can start the analysis now.")
        if available_gb < 1.3:
            print("⚠️ Running at minimum RAM - might be slower but should work!")
        return True
    else:
        print("🔴 Need more memory. Try closing:")
        print("   • Browser windows with many tabs")
        print("   • Other VS Code windows")
        print("   • Heavy applications")
        print(f"   Need: 1.0GB minimum, Have: {available_gb:.1f}GB")
        return False

if __name__ == "__main__":
    ready = quick_memory_check()
    
    if ready:
        print("\n🎯 Next steps:")
        print("1. ollama serve")
        print("2. ollama pull phi3:mini")
        print("3. ollama pull nomic-embed-text") 
        print("4. uvicorn backend.main:app --reload")
    else:
        print(f"\n🔄 Run this script again after freeing memory!")