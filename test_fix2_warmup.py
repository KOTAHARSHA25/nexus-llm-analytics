"""
Test Fix 2: Model Warmup - Verify first request is fast
"""
import requests
import time

print("="*80)
print("🧪 TESTING FIX 2: MODEL WARMUP")
print("="*80)

# Wait for server to be ready
print("\n⏳ Waiting for server to fully start...")
time.sleep(2)

# Test 1: Health check (should be instant)
print("\n📝 Test 1: Health check")
start = time.time()
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    elapsed = time.time() - start
    if response.status_code == 200:
        print(f"✅ PASS: Health check responded in {elapsed:.2f}s")
    else:
        print(f"❌ FAIL: Health check returned {response.status_code}")
except Exception as e:
    print(f"❌ FAIL: Health check error - {e}")

# Test 2: First LLM request (should be fast now - model is warmed up)
print("\n📝 Test 2: First LLM request - Get available models (uses LLM)")
print("   This should be FAST (~1-3s) instead of SLOW (~15-30s)")
start = time.time()
try:
    # This endpoint uses the LLM to get model info
    response = requests.get("http://localhost:8000/api/models", timeout=60)
    elapsed = time.time() - start
    if response.status_code == 200:
        if elapsed < 5:
            print(f"✅ PASS: First request was FAST ({elapsed:.2f}s) - Warmup working!")
            result = response.json()
            print(f"   Available models: {len(result.get('models', []))} models")
        elif elapsed < 10:
            print(f"⚠️  OKAY: First request took {elapsed:.2f}s - Acceptable but could be faster")
        else:
            print(f"❌ FAIL: First request was SLOW ({elapsed:.2f}s) - Warmup may not be working")
    else:
        print(f"❌ FAIL: Request returned {response.status_code}")
        print(f"   Error: {response.text[:200]}")
except Exception as e:
    print(f"❌ FAIL: Request error - {e}")

print("\n" + "="*80)
print("📊 FIX 2 TEST COMPLETE")
print("="*80)
print("\nExpected: First request should be <5 seconds")
print("Before Fix 2: First request was 15-30 seconds")
print("="*80)
