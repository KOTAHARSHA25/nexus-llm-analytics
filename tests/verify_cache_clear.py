import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_cache_clearing():
    print("1. Checking initial cache status...")
    try:
        resp = requests.get(f"{BASE_URL}/api/health/cache-info")
        initial_stats = resp.json()
        print(f"   Initial L1 size: {initial_stats.get('l1_memory', {}).get('size', 0)}")
    except Exception as e:
        print(f"   Failed to get stats: {e}")

    print("\n2. Performing analysis to populate cache...")
    query = "What is 2+2?" 
    # specific query to ensure it's simple and likely not cached or easy to cache
    payload = {
        "query": query,
        "session_id": "test_cache_clear"
    }
    
    try:
        # We use sync analyze for simplicity in test
        requests.post(f"{BASE_URL}/api/analyze/", json=payload)
        # Run it twice to ensure it hits cache if working
        resp = requests.post(f"{BASE_URL}/api/analyze/", json=payload)
        data = resp.json()
        print(f"   Analysis result: {data.get('metrics', {}).get('cache_hit', False)}")
    except Exception as e:
        print(f"   Analysis failed: {e}")
        
    print("\n3. checking cache stats after analysis...")
    resp = requests.get(f"{BASE_URL}/api/health/cache-info")
    stats_after = resp.json()
    l1_size_after = stats_after.get('l1_memory', {}).get('size', 0)
    print(f"   L1 size after analysis: {l1_size_after}")

    print("\n4. Clearing cache via API...")
    resp = requests.post(f"{BASE_URL}/api/health/clear-cache")
    print(f"   Clear response: {resp.json()}")
    
    print("\n5. Verifying cache is empty...")
    resp = requests.get(f"{BASE_URL}/api/health/cache-info")
    final_stats = resp.json()
    l1_size_final = final_stats.get('l1_memory', {}).get('size', 0)
    print(f"   L1 size final: {l1_size_final}")
    
    if l1_size_final == 0:
        print("\nSUCCESS: Cache cleared successfully!")
    else:
        print(f"\nFAILURE: Cache not empty (size {l1_size_final})")
        sys.exit(1)

if __name__ == "__main__":
    test_cache_clearing()
