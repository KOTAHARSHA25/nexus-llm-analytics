"""
Task 1.3.2: Cache Validation Test
Tests that caching system is working correctly
"""

import requests
import time
import json

# Configuration
BASE_URL = "http://localhost:8000"
ANALYZE_URL = f"{BASE_URL}/analyze/"
HEALTH_URL = f"{BASE_URL}/health"

def test_cache_validation():
    """Test that cache is working by comparing response times"""
    
    print("\n" + "="*70)
    print("TASK 1.3.2: CACHE VALIDATION TEST")
    print("="*70)
    
    # Step 1: Check backend health
    print("\n[1/4] Checking backend health...")
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return
    
    # Step 2: Use simple.json (valid JSON, small file)
    filename = "simple.json"
    query = "What is the total sales amount?"
    
    print(f"\n[2/4] First query (cache MISS expected)...")
    print(f"   File: {filename}")
    print(f"   Query: {query}")
    
    payload = {
        "query": query,
        "filename": filename
    }
    
    # First request - should be cache MISS
    start_time = time.time()
    try:
        response1 = requests.post(ANALYZE_URL, json=payload, timeout=180)
        time1 = time.time() - start_time
        
        if response1.status_code == 200:
            result1 = response1.json()
            print(f"✅ First query completed")
            print(f"   Response time: {time1:.2f}s")
            print(f"   Answer: {result1.get('answer', 'N/A')[:100]}")
        else:
            print(f"❌ First query failed: {response1.status_code}")
            print(f"   Response: {response1.text[:200]}")
            return
    except Exception as e:
        print(f"❌ First query error: {e}")
        return
    
    # Step 3: Wait a moment
    print(f"\n[3/4] Waiting 2 seconds before second query...")
    time.sleep(2)
    
    # Step 4: Second request - should be cache HIT (faster)
    print(f"\n[4/4] Second query (cache HIT expected)...")
    print(f"   Same file: {filename}")
    print(f"   Same query: {query}")
    
    start_time = time.time()
    try:
        response2 = requests.post(ANALYZE_URL, json=payload, timeout=180)
        time2 = time.time() - start_time
        
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"✅ Second query completed")
            print(f"   Response time: {time2:.2f}s")
            print(f"   Answer: {result2.get('answer', 'N/A')[:100]}")
        else:
            print(f"❌ Second query failed: {response2.status_code}")
            return
    except Exception as e:
        print(f"❌ Second query error: {e}")
        return
    
    # Analysis
    print("\n" + "="*70)
    print("CACHE PERFORMANCE ANALYSIS")
    print("="*70)
    
    print(f"\n1st Query Time: {time1:.2f}s (cache MISS)")
    print(f"2nd Query Time: {time2:.2f}s (cache HIT expected)")
    
    speedup = ((time1 - time2) / time1) * 100 if time1 > 0 else 0
    
    if time2 < time1 * 0.5:  # Second query should be at least 50% faster
        print(f"\n✅ CACHE IS WORKING!")
        print(f"   Speedup: {speedup:.1f}% faster on cache hit")
        print(f"   Time saved: {time1 - time2:.2f}s")
        cache_working = True
    elif time2 < time1:
        print(f"\n⚠️  CACHE MAY BE WORKING (marginal improvement)")
        print(f"   Speedup: {speedup:.1f}% faster")
        print(f"   Expected: >50% speedup, got {speedup:.1f}%")
        cache_working = False
    else:
        print(f"\n❌ CACHE NOT WORKING")
        print(f"   Second query was SLOWER or same speed")
        cache_working = False
    
    # Check if answers are identical (cached results should match)
    if result1.get('answer') == result2.get('answer'):
        print(f"\n✅ Answers are identical (cache consistency verified)")
    else:
        print(f"\n⚠️  Answers differ (might indicate cache issue)")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if cache_working:
        print("\n✅ SUCCESS: Cache validation passed")
        print("   - Second query was significantly faster")
        print("   - Results are consistent")
        print("   - Task 1.3.2 validation: COMPLETE")
    else:
        print("\n⚠️  PARTIAL: Cache may need investigation")
        print("   - Cache exists in code but performance gain unclear")
        print("   - Possible reasons:")
        print("     * Cache may be working but LLM variability masks speedup")
        print("     * Different query paths may bypass cache")
        print("     * Cache key generation may need review")
    
    return cache_working

if __name__ == "__main__":
    test_cache_validation()
