"""Quick test to verify routing configuration"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.core.user_preferences import get_preferences_manager

# Load preferences
prefs_manager = get_preferences_manager()
prefs = prefs_manager.load_preferences()

print("═════════════════════════════════════════")
print("🔍 ROUTING CONFIGURATION CHECK")
print("═════════════════════════════════════════")
print(f"✅ Primary Model: {prefs.primary_model}")
print(f"✅ Review Model: {prefs.review_model}")
print(f"✅ Auto Selection: {prefs.auto_model_selection}")
print(f"🎯 Intelligent Routing: {prefs.enable_intelligent_routing}")
print("═════════════════════════════════════════")

if prefs.enable_intelligent_routing:
    print("✅ Routing is ENABLED - queries should show tier indicators")
else:
    print("❌ Routing is DISABLED - all queries use primary model")
    print("💡 To enable: Go to Settings → Toggle 'Intelligent Routing (Experimental)'")

print("\nConfig file location:", prefs_manager.config_file)
