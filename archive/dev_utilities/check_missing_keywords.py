import json

# Load raw failures
with open('tests/performance/stress_test_results/failures_20251109_182842.json') as f:
    failures = json.load(f)

print("\n=== FULL TIER FAILURES (Complex queries under-routed) ===\n")
full_failures = [f for f in failures if f['expected'] == 'full']
for i, failure in enumerate(full_failures[:20], 1):
    print(f"{i}. {failure['query'][:100]}")
    print(f"   Complexity: {failure['complexity']:.3f} | Routed to: {failure['actual']} | Category: {failure['category']}\n")

print(f"\nTotal FULL failures: {len(full_failures)}")

print("\n\n=== BALANCED TIER FAILURES (Medium queries misrouted) ===\n")
balanced_failures = [f for f in failures if f['expected'] == 'balanced']
for i, failure in enumerate(balanced_failures[:10], 1):
    print(f"{i}. {failure['query'][:100]}")
    print(f"   Complexity: {failure['complexity']:.3f} | Routed to: {failure['actual']} | Category: {failure['category']}\n")

print(f"\nTotal BALANCED failures: {len(balanced_failures)}")
