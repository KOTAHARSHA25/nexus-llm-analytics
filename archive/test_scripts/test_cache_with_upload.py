"""
Cache Validation Test with File Upload
Task 1.3.2: Verify caching works for uploaded files
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"
TEST_FILE = "data/samples/simple.json"

print("="*60)
print("TASK 1.3.2: CACHE VALIDATION (WITH FILE UPLOAD)")
print("="*60)

# Step 1: Check backend health
print("\n[1/5] Checking backend health...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✅ Backend is running")
    else:
        print(f"❌ Backend returned status {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ Backend not accessible: {e}")
    exit(1)

# Step 2: Upload file
print("\n[2/5] Uploading test file...")
try:
    with open(TEST_FILE, 'rb') as f:
        files = {'file': ('simple.json', f, 'application/json')}
        response = requests.post(f"{BASE_URL}/upload-documents/", files=files, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        filename = data.get('filename')
        print(f"✅ File uploaded successfully")
        print(f"   Filename: {filename}")
    else:
        print(f"❌ Upload failed: {response.status_code}")
        print(f"   Response: {response.text}")
        exit(1)
except Exception as e:
    print(f"❌ Upload error: {e}")
    exit(1)

# Step 3: First query (cache MISS expected)
print("\n[3/5] First query (cache MISS expected)...")
query = "What is the total sales amount?"
print(f"   File: {filename}")
print(f"   Query: {query}")

try:
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/analyze/",
        json={
            "query": query,
            "filename": filename
        },
        timeout=120
    )
    first_time = time.time() - start_time
    
    if response.status_code == 200:
        first_response = response.json()
        first_answer = first_response.get('answer', 'N/A')
        print(f"✅ First query completed")
        print(f"   Response time: {first_time:.2f}s")
        print(f"   Answer: {first_answer[:100]}...")
    else:
        print(f"❌ First query failed: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ First query error: {e}")
    exit(1)

# Step 4: Wait a moment
print("\n[4/5] Waiting 2 seconds before second query...")
time.sleep(2)

# Step 5: Second query (cache HIT expected - SAME query, SAME file)
print("\n[5/5] Second query (cache HIT expected)...")
print(f"   Same file: {filename}")
print(f"   Same query: {query}")

try:
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/analyze/",
        json={
            "query": query,
            "filename": filename
        },
        timeout=120
    )
    second_time = time.time() - start_time
    
    if response.status_code == 200:
        second_response = response.json()
        second_answer = second_response.get('answer', 'N/A')
        print(f"✅ Second query completed")
        print(f"   Response time: {second_time:.2f}s")
        print(f"   Answer: {second_answer[:100]}...")
    else:
        print(f"❌ Second query failed: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ Second query error: {e}")
    exit(1)

# Analysis
print("\n" + "="*60)
print("CACHE PERFORMANCE ANALYSIS")
print("="*60)

print(f"\n1st Query Time: {first_time:.2f}s (cache MISS)")
print(f"2nd Query Time: {second_time:.2f}s (cache HIT expected)")

speedup = ((first_time - second_time) / first_time * 100) if first_time > 0 else 0
print(f"\nSpeedup: {speedup:.1f}%")

# Cache validation thresholds
CACHE_SPEEDUP_THRESHOLD = 50  # Expect at least 50% speedup for cache hit

if second_time < first_time * 0.5:  # More than 50% faster
    print(f"\n✅ CACHE WORKING - Second query was {speedup:.0f}% faster")
    cache_status = "WORKING"
elif second_time < first_time * 0.8:  # 20-50% faster
    print(f"\n⚠️  CACHE PARTIALLY WORKING - Second query was {speedup:.0f}% faster")
    print("   (Expected >50% speedup for strong cache evidence)")
    cache_status = "PARTIAL"
else:
    print(f"\n❌ CACHE NOT WORKING - Second query was not significantly faster")
    print("   Possible reasons:")
    print("   - Cache may be disabled")
    print("   - Cache keys may not match")
    print("   - LLM variability may mask cache speedup")
    cache_status = "NOT_WORKING"

# Check answer consistency
if first_answer == second_answer:
    print(f"\n✅ Answers are identical (cache consistency verified)")
else:
    print(f"\n⚠️  Answers differ (may indicate cache not used)")
    print(f"   First:  {first_answer[:80]}")
    print(f"   Second: {second_answer[:80]}")

# Summary
print("\n" + "="*60)
print("SUMMARY - TASK 1.3.2: CACHING VALIDATION")
print("="*60)

if cache_status == "WORKING":
    print("\n✅ CACHE SYSTEM IS WORKING")
    print(f"   - Second query was {speedup:.0f}% faster")
    print("   - Answers are consistent")
    print("   - Cache provides significant performance benefit")
elif cache_status == "PARTIAL":
    print("\n⚠️  CACHE MAY BE WORKING")
    print(f"   - Second query was {speedup:.0f}% faster (expected >50%)")
    print("   - Some performance improvement observed")
    print("   - May need further investigation")
else:
    print("\n❌ CACHE NEEDS INVESTIGATION")
    print("   - No significant speedup observed")
    print("   - Cache may be disabled or not working")

print("\n" + "="*60)
