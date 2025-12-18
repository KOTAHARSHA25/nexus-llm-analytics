import json
import glob
import os

# Find the most recent failure file
failure_files = glob.glob('tests/performance/stress_test_results/failures_*.json')
latest_file = max(failure_files, key=os.path.getctime)
print(f"Using: {latest_file}\n")

with open(latest_file) as f:
    failures = json.load(f)

# Find critical failures (complex queries routed to FAST)
critical = [f for f in failures if f['expected'] == 'full' and f['actual'] == 'fast']

print("\n=== CRITICAL SAFETY FAILURES ===\n")
for i, fail in enumerate(critical, 1):
    print(f"{i}. Query: {fail['query']}")
    print(f"   Complexity: {fail['complexity']:.3f}")
    print(f"   Semantic: {fail['semantic']:.3f} | Data: {fail['data']:.3f} | Operation: {fail['operation']:.3f}")
    print(f"   Reason: {fail['reason']}\n")

print(f"Total critical failures: {len(critical)}")

# Also check FAST failures (simple queries over-routed)
print("\n=== FAST TIER FAILURES (sample of simple queries over-routed) ===\n")
fast_fails = [f for f in failures if f['expected'] == 'fast'][:10]
for i, fail in enumerate(fast_fails, 1):
    print(f"{i}. Query: {fail['query'][:80]}")
    print(f"   Complexity: {fail['complexity']:.3f} | Routed to: {fail['actual']}")
    print(f"   Semantic: {fail['semantic']:.3f} | Data: {fail['data']:.3f} | Operation: {fail['operation']:.3f}\n")

print(f"Total FAST failures: {len(fast_fails)}")
