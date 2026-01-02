"""
Advanced Test Suite for Fix 4: Smart Fallback Dynamic Models
Tests dynamic model discovery, fallback chain building, and edge cases
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

print("="*80)
print("🧪 ADVANCED TESTING FIX 4: SMART FALLBACK DYNAMIC MODELS")
print("="*80)

# Test 1: Check what models are actually installed
print("\n📝 Test 1: Verify Ollama is running and list installed models")
print("-"*80)

try:
    import requests
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    
    if response.status_code == 200:
        models_data = response.json().get("models", [])
        print(f"✅ Ollama is running")
        print(f"   Found {len(models_data)} models:")
        
        for model in models_data:
            name = model.get("name", "unknown")
            size_bytes = model.get("size", 0)
            size_gb = size_bytes / (1024**3)
            is_embedding = "embed" in name.lower() or "nomic" in name.lower()
            emoji = "🔤" if is_embedding else "🧠"
            print(f"   {emoji} {name:30s} - {size_gb:.2f} GB")
        
        # Filter text generation models
        text_models = [m.get("name") for m in models_data 
                      if "embed" not in m.get("name", "").lower() 
                      and "nomic" not in m.get("name", "").lower()]
        
        print(f"\n   Text generation models: {len(text_models)}")
        for tm in text_models:
            print(f"   ✓ {tm}")
            
        test1_pass = len(models_data) > 0
    else:
        print(f"❌ Ollama API returned status {response.status_code}")
        test1_pass = False
except Exception as e:
    print(f"❌ Cannot connect to Ollama: {e}")
    print("   Make sure Ollama is running: ollama serve")
    test1_pass = False

# Test 2: Test SmartFallbackManager initialization
print("\n📝 Test 2: SmartFallbackManager initialization with dynamic models")
print("-"*80)

try:
    from src.backend.core.smart_fallback import SmartFallbackManager
    
    manager = SmartFallbackManager()
    
    print(f"✅ SmartFallbackManager created")
    print(f"   Model chain strategies: {manager.model_chain.strategies}")
    print(f"   Number of fallback options: {len(manager.model_chain.strategies)}")
    
    # Verify it's not using hardcoded models
    hardcoded_models = ["llama3.1:8b", "phi3:mini", "tinyllama"]
    strategies = manager.model_chain.strategies
    
    # Check if all strategies are hardcoded (bad) or dynamic (good)
    all_hardcoded = all(s in hardcoded_models or s == "echo" for s in strategies)
    
    if not all_hardcoded and len(strategies) > 1:
        print(f"✅ PASS: Using dynamic model discovery (not hardcoded)")
        test2_pass = True
    elif test1_pass and len(text_models) > 0:
        # If we have models installed but manager is using hardcoded ones
        print(f"⚠️  WARNING: Models available but using hardcoded fallback")
        print(f"   Expected models: {text_models}")
        print(f"   Got strategies: {strategies}")
        test2_pass = False
    else:
        print(f"⚠️  Using default fallback (Ollama might not be available)")
        test2_pass = True  # This is acceptable if no models found
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test2_pass = False

# Test 3: Verify models are sorted by size (larger first)
print("\n📝 Test 3: Verify model ordering (larger models first)")
print("-"*80)

if test1_pass and test2_pass:
    try:
        # Get model sizes
        model_sizes = {}
        for model in models_data:
            name = model.get("name", "")
            if "embed" not in name.lower() and "nomic" not in name.lower():
                model_sizes[name] = model.get("size", 0)
        
        # Check if manager's chain follows size order
        manager_models = [m for m in manager.model_chain.strategies if m != "echo"]
        
        if len(manager_models) > 1:
            sizes = [model_sizes.get(m, 0) for m in manager_models]
            is_sorted = all(sizes[i] >= sizes[i+1] for i in range(len(sizes)-1))
            
            if is_sorted:
                print(f"✅ PASS: Models sorted by size (larger first)")
                for i, model in enumerate(manager_models):
                    size_gb = sizes[i] / (1024**3)
                    print(f"   {i+1}. {model:30s} - {size_gb:.2f} GB")
                test3_pass = True
            else:
                print(f"⚠️  Models not sorted by size")
                for i, model in enumerate(manager_models):
                    size_gb = sizes[i] / (1024**3)
                    print(f"   {i+1}. {model:30s} - {size_gb:.2f} GB")
                test3_pass = False
        else:
            print(f"⚠️  Only {len(manager_models)} model(s) in chain, cannot verify sorting")
            test3_pass = True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        test3_pass = False
else:
    print(f"⏭️  Skipped (previous tests failed)")
    test3_pass = False

# Test 4: Test fallback mechanism
print("\n📝 Test 4: Test fallback mechanism with model failure")
print("-"*80)

if test2_pass:
    try:
        from src.backend.core.smart_fallback import FallbackReason
        
        # Get first model
        first_model = manager.model_chain.strategies[0]
        print(f"   Starting with: {first_model}")
        
        # Simulate failure and get fallback
        fallback_model = manager.get_model_fallback(
            current_model=first_model,
            reason=FallbackReason.MODEL_UNAVAILABLE,
            error="Test failure"
        )
        
        print(f"   Fallback to: {fallback_model}")
        
        if fallback_model != first_model:
            print(f"✅ PASS: Fallback mechanism working")
            print(f"   Chain has {len(manager.model_chain.strategies) - 1} remaining options")
            test4_pass = True
        else:
            print(f"❌ FAIL: Fallback returned same model")
            test4_pass = False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        test4_pass = False
else:
    print(f"⏭️  Skipped (previous tests failed)")
    test4_pass = False

# Test 5: Edge case - No models installed (simulated)
print("\n📝 Test 5: Edge case - Behavior when no models detected")
print("-"*80)

try:
    # Create a manager that will fail to fetch models
    import unittest.mock as mock
    
    with mock.patch('src.backend.core.smart_fallback.SmartFallbackManager._get_installed_model_names', return_value=[]):
        fallback_manager = SmartFallbackManager()
        
        strategies = fallback_manager.model_chain.strategies
        print(f"   Strategies when no models: {strategies}")
        
        # Should have default fallback chain
        has_defaults = any(m in strategies for m in ["llama3.1:8b", "phi3:mini", "tinyllama"])
        has_echo = "echo" in strategies
        
        if has_defaults or (has_echo and len(strategies) > 0):
            print(f"✅ PASS: Falls back to default chain when no models detected")
            test5_pass = True
        else:
            print(f"❌ FAIL: No fallback chain when models unavailable")
            test5_pass = False
except Exception as e:
    print(f"❌ FAIL: {e}")
    test5_pass = False

# Test 6: Edge case - Ollama server down
print("\n📝 Test 6: Edge case - Ollama server unreachable")
print("-"*80)

try:
    import unittest.mock as mock
    
    def mock_get_error(*args, **kwargs):
        raise ConnectionError("Simulated connection error")
    
    with mock.patch('requests.get', side_effect=mock_get_error):
        error_manager = SmartFallbackManager()
        
        strategies = error_manager.model_chain.strategies
        print(f"   Strategies on connection error: {strategies}")
        
        if len(strategies) > 0 and "echo" in strategies:
            print(f"✅ PASS: Gracefully handles Ollama connection failure")
            print(f"   Using default fallback with {len(strategies)} options")
            test6_pass = True
        else:
            print(f"❌ FAIL: Crashes or no fallback when Ollama unreachable")
            test6_pass = False
except Exception as e:
    print(f"❌ FAIL: {e}")
    test6_pass = False

# Test 7: Verify 'echo' is always last resort
print("\n📝 Test 7: Verify 'echo' model as last resort fallback")
print("-"*80)

if test2_pass:
    try:
        strategies = manager.model_chain.strategies
        
        if "echo" in strategies:
            echo_index = strategies.index("echo")
            is_last = echo_index == len(strategies) - 1
            
            if is_last:
                print(f"✅ PASS: 'echo' is last resort (position {echo_index + 1}/{len(strategies)})")
                test7_pass = True
            else:
                print(f"⚠️  'echo' is not last (position {echo_index + 1}/{len(strategies)})")
                test7_pass = False
        else:
            print(f"⚠️  'echo' not in fallback chain")
            test7_pass = False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        test7_pass = False
else:
    print(f"⏭️  Skipped")
    test7_pass = False

# Test 8: Test exhaustion of fallback chain
print("\n📝 Test 8: Test complete fallback chain exhaustion")
print("-"*80)

if test2_pass:
    try:
        from src.backend.core.smart_fallback import FallbackReason
        
        exhaustion_manager = SmartFallbackManager()
        initial_count = len(exhaustion_manager.model_chain.strategies)
        
        print(f"   Initial chain length: {initial_count}")
        
        # Exhaust all fallbacks
        current = exhaustion_manager.model_chain.strategies[0]
        fallback_count = 0
        
        while exhaustion_manager.model_chain.has_fallback():
            next_model = exhaustion_manager.get_model_fallback(
                current_model=current,
                reason=FallbackReason.EXECUTION_ERROR,
                error="Test exhaustion"
            )
            fallback_count += 1
            print(f"   Fallback {fallback_count}: {current} → {next_model}")
            current = next_model
        
        print(f"\n   Chain exhausted after {fallback_count} fallbacks")
        
        # Try one more - should return last option
        final = exhaustion_manager.get_model_fallback(
            current_model=current,
            reason=FallbackReason.EXECUTION_ERROR,
            error="Final test"
        )
        
        print(f"   Final fallback attempt returns: {final}")
        
        if fallback_count > 0:
            print(f"✅ PASS: Fallback chain exhaustion handled gracefully")
            test8_pass = True
        else:
            print(f"❌ FAIL: No fallbacks occurred")
            test8_pass = False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        test8_pass = False
else:
    print(f"⏭️  Skipped")
    test8_pass = False

# Test 9: Embedding models should be filtered out
print("\n📝 Test 9: Verify embedding models are filtered from fallback chain")
print("-"*80)

if test1_pass and test2_pass:
    try:
        strategies = manager.model_chain.strategies
        
        # Check if any embedding models leaked into the chain
        embedding_keywords = ["embed", "nomic", "embedding"]
        has_embedding = any(
            any(keyword in str(s).lower() for keyword in embedding_keywords)
            for s in strategies if s != "echo"
        )
        
        if not has_embedding:
            print(f"✅ PASS: No embedding models in fallback chain")
            print(f"   Chain: {strategies}")
            test9_pass = True
        else:
            print(f"❌ FAIL: Embedding model found in chain")
            print(f"   Chain: {strategies}")
            test9_pass = False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        test9_pass = False
else:
    print(f"⏭️  Skipped")
    test9_pass = False

# Test 10: Test with different OLLAMA_BASE_URL
print("\n📝 Test 10: Test custom OLLAMA_BASE_URL support")
print("-"*80)

try:
    import unittest.mock as mock
    import os
    
    # Simulate custom URL
    custom_url = "http://custom-ollama:11434"
    
    with mock.patch.dict(os.environ, {'OLLAMA_BASE_URL': custom_url}):
        # Mock the request to custom URL
        with mock.patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": []}
            
            custom_manager = SmartFallbackManager()
            
            # Verify it tried to call the custom URL
            if mock_get.called:
                called_url = mock_get.call_args[0][0]
                if custom_url in called_url:
                    print(f"✅ PASS: Respects custom OLLAMA_BASE_URL")
                    print(f"   Called: {called_url}")
                    test10_pass = True
                else:
                    print(f"⚠️  Called wrong URL: {called_url}")
                    test10_pass = False
            else:
                print(f"⚠️  Request not made")
                test10_pass = False
except Exception as e:
    print(f"❌ FAIL: {e}")
    test10_pass = False

# Summary
print("\n" + "="*80)
print("📊 ADVANCED TEST SUMMARY - FIX 4")
print("="*80)

all_tests = [
    ("Ollama connection & model listing", test1_pass),
    ("Dynamic model discovery", test2_pass),
    ("Model size-based sorting", test3_pass),
    ("Fallback mechanism", test4_pass),
    ("No models edge case", test5_pass),
    ("Ollama unreachable edge case", test6_pass),
    ("Echo as last resort", test7_pass),
    ("Chain exhaustion handling", test8_pass),
    ("Embedding model filtering", test9_pass),
    ("Custom OLLAMA_BASE_URL", test10_pass),
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

for test_name, result in all_tests:
    icon = "✅" if result else "❌"
    print(f"   {icon} {test_name}")

print(f"\n{'✅' if passed == total else '⚠️'} Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

if passed == total:
    print("\n🎉 EXCELLENT! All Fix 4 tests passed!")
    print("   - Dynamic model discovery working")
    print("   - Fallback chain built from installed models")
    print("   - Edge cases handled gracefully")
    print("   - Embedding models filtered out")
    print("   - No hardcoded model dependencies")
elif passed >= total * 0.8:
    print(f"\n✅ GOOD! Most tests passed ({total - passed} issues)")
else:
    print(f"\n⚠️  WARNING: Multiple tests failed ({total - passed} failures)")
    print("   Review implementation")

print("="*80)
