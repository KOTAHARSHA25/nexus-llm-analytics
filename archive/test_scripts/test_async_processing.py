"""
Task 1.3.3: Async Processing Validation Test
Verify that the backend can handle multiple concurrent requests without blocking
"""

import requests
import time
import threading
from queue import Queue

BASE_URL = "http://localhost:8000"
TEST_FILE = "data/samples/simple.json"

print("="*60)
print("TASK 1.3.3: ASYNC PROCESSING VALIDATION")
print("="*60)

# Step 1: Check backend health
print("\n[1/4] Checking backend health...")
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

# Step 2: Upload test file
print("\n[2/4] Uploading test file...")
try:
    with open(TEST_FILE, 'rb') as f:
        files = {'file': ('simple.json', f, 'application/json')}
        response = requests.post(f"{BASE_URL}/upload-documents/", files=files, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        filename = data.get('filename')
        print(f"✅ File uploaded: {filename}")
    else:
        print(f"❌ Upload failed: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ Upload error: {e}")
    exit(1)

# Step 3: Test concurrent requests (async verification)
print("\n[3/4] Testing concurrent request handling...")
print("   Sending 3 requests simultaneously...")

results_queue = Queue()

def send_query(query_id, query_text):
    """Send a query and record timing"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/analyze/",
            json={
                "query": query_text,
                "filename": filename
            },
            timeout=120
        )
        elapsed_time = time.time() - start_time
        
        results_queue.put({
            'query_id': query_id,
            'query': query_text,
            'time': elapsed_time,
            'status': response.status_code,
            'success': response.status_code == 200
        })
    except Exception as e:
        results_queue.put({
            'query_id': query_id,
            'query': query_text,
            'time': 0,
            'status': 0,
            'success': False,
            'error': str(e)
        })

# Define 3 different queries
queries = [
    ("Query 1", "What is the total sales amount?"),
    ("Query 2", "How many products are there?"),
    ("Query 3", "What is the average sales amount?")
]

# Start all requests at the same time
threads = []
start_time = time.time()

for query_id, query_text in queries:
    thread = threading.Thread(target=send_query, args=(query_id, query_text))
    threads.append(thread)
    thread.start()

# Wait for all to complete
for thread in threads:
    thread.join()

total_time = time.time() - start_time

# Collect results
results = []
while not results_queue.empty():
    results.append(results_queue.get())

results.sort(key=lambda x: x['query_id'])

# Step 4: Analyze async behavior
print("\n[4/4] Analyzing async behavior...")
print("-" * 60)

sum_of_individual_times = 0
for result in results:
    status_icon = "✅" if result['success'] else "❌"
    print(f"{status_icon} {result['query_id']}: {result['time']:.2f}s")
    if result['success']:
        sum_of_individual_times += result['time']

print("-" * 60)
print(f"Sum of individual times: {sum_of_individual_times:.2f}s")
print(f"Total wall-clock time:   {total_time:.2f}s")

# Analysis
print("\n" + "="*60)
print("ASYNC PROCESSING ANALYSIS")
print("="*60)

# If async works correctly:
# - Wall-clock time should be roughly equal to the LONGEST individual request
# - NOT the sum of all requests

# Calculate expected sequential time (if blocking)
max_individual_time = max([r['time'] for r in results if r['success']], default=0)

print(f"\nExpected if BLOCKING (sequential):  {sum_of_individual_times:.2f}s")
print(f"Expected if ASYNC (parallel):       ~{max_individual_time:.2f}s")
print(f"Actual wall-clock time:             {total_time:.2f}s")

# Calculate efficiency
if sum_of_individual_times > 0:
    parallelization_efficiency = (sum_of_individual_times / total_time)
    print(f"\nParallelization efficiency: {parallelization_efficiency:.2f}x")
    
    # If efficiency > 1.5x, async is working well
    if parallelization_efficiency > 1.5:
        print("\n✅ ASYNC PROCESSING IS WORKING")
        print(f"   System processed {len(queries)} requests in parallel")
        print(f"   {parallelization_efficiency:.1f}x faster than sequential processing")
        async_status = "WORKING"
    elif parallelization_efficiency > 1.1:
        print("\n⚠️  ASYNC PROCESSING PARTIALLY WORKING")
        print(f"   Some parallelization ({parallelization_efficiency:.1f}x speedup)")
        print("   May be limited by CPU or I/O bottlenecks")
        async_status = "PARTIAL"
    else:
        print("\n❌ ASYNC PROCESSING NOT WORKING")
        print("   Requests appear to be processed sequentially")
        print("   Possible blocking operations in the code")
        async_status = "NOT_WORKING"

# Summary
print("\n" + "="*60)
print("SUMMARY - TASK 1.3.3: ASYNC PROCESSING")
print("="*60)

successful_requests = sum(1 for r in results if r['success'])
print(f"\nRequests completed: {successful_requests}/{len(queries)}")

if async_status == "WORKING":
    print("\n✅ ASYNC PROCESSING VALIDATED")
    print("   - Concurrent requests handled efficiently")
    print("   - Non-blocking I/O operations confirmed")
    print(f"   - {parallelization_efficiency:.1f}x parallelization achieved")
elif async_status == "PARTIAL":
    print("\n⚠️  ASYNC PROCESSING NEEDS OPTIMIZATION")
    print("   - Some concurrency observed")
    print("   - Performance could be improved")
elif async_status == "NOT_WORKING":
    print("\n❌ ASYNC PROCESSING NEEDS FIXES")
    print("   - Requests processed sequentially")
    print("   - Blocking operations may exist")

print("\n" + "="*60)
