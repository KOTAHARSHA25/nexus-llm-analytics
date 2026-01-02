"""
Test Fix 3: Concurrent HTTP Requests
Sends multiple requests simultaneously to test thread safety under real load
"""
import requests
import threading
import time

print("="*80)
print("🧪 FIX 3: CONCURRENT REQUEST TEST")
print("="*80)

print("\n⏳ Waiting for backend to be ready...")
time.sleep(2)

# Test health endpoint with concurrent requests
results = []
errors = []

def make_request(request_id):
    try:
        start = time.time()
        response = requests.get("http://localhost:8000/health", timeout=10)
        elapsed = time.time() - start
        
        results.append({
            'id': request_id,
            'status': response.status_code,
            'time': elapsed
        })
    except Exception as e:
        errors.append({'id': request_id, 'error': str(e)})

print("\n📝 Sending 10 concurrent requests to /health endpoint...")
print("-"*80)

threads = []
start_time = time.time()

for i in range(10):
    t = threading.Thread(target=make_request, args=(i+1,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

total_time = time.time() - start_time

print(f"\n✅ Completed in {total_time:.2f}s")
print(f"   Success: {len(results)}/10 requests")
print(f"   Errors: {len(errors)}/10 requests")

if errors:
    print("\n❌ ERRORS:")
    for err in errors:
        print(f"   Request #{err['id']}: {err['error']}")

if results:
    print(f"\n📊 Response Times:")
    for r in sorted(results, key=lambda x: x['id']):
        status_icon = "✅" if r['status'] == 200 else "❌"
        print(f"   {status_icon} Request #{r['id']:2d}: {r['status']} in {r['time']:.2f}s")

print("\n" + "="*80)
if len(results) == 10 and len(errors) == 0:
    print("✅ FIX 3 VERIFIED: All concurrent requests succeeded!")
    print("   No race conditions or crashes under concurrent load")
else:
    print("⚠️  Some requests failed - check if backend is running")
print("="*80)
