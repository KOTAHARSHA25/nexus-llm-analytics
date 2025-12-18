"""
Analyze the 176 failures from 83% accuracy test to identify surgical improvements
"""
import json
from collections import defaultdict

# Load the test results
with open('tests/performance/stress_test_results/stress_test_report_20251109_174904.json') as f:
    data = json.load(f)

fa = data['failure_analysis']

print("=" * 80)
print("FAILURE ANALYSIS FOR 95% TARGET")
print("=" * 80)
print(f"\nTotal Failures: {fa['total_failures']} out of 1035 queries (17% failure rate)")
print(f"Target: Reduce to <52 failures (5% failure rate)")
print(f"Need to fix: {fa['total_failures'] - 52} failures")

print("\n" + "=" * 80)
print("FAILURES BY EXPECTED TIER:")
print("=" * 80)
for tier, count in sorted(fa['by_expected_tier'].items(), key=lambda x: -x[1]):
    print(f"  {tier:12s}: {count:3d} failures")

print("\n" + "=" * 80)
print("COMPLEXITY SCORE RANGE OF FAILURES:")
print("=" * 80)
cr = fa['complexity_range']
for key, value in cr.items():
    print(f"  {key:10s}: {value:.3f}")

print("\n" + "=" * 80)
print("TOP FAILURE CATEGORIES:")
print("=" * 80)
for cat, count in sorted(fa['by_category'].items(), key=lambda x: -x[1])[:15]:
    print(f"  {count:3d} - {cat}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("""
Key Insights:
1. 77 FAST failures: Simple queries being over-routed to BALANCED/FULL
   → Need to LOWER scoring for truly simple queries
   → Likely caused by length/structure scoring being too high

2. 56 BALANCED failures: Medium queries being under/over-routed  
   → Need more precise MEDIUM keyword detection
   → Better threshold calibration (0.25-0.45 range)

3. 42 FULL failures: Complex queries being under-routed
   → Some complex keywords still missing
   → Or complex keywords being outweighed by other factors

SURGICAL FIXES NEEDED:
1. Reduce SEMANTIC_WEIGHT (language complexity) - it's adding noise
2. Increase OPERATION_WEIGHT  - keywords are most reliable signal
3. Fine-tune thresholds: 0.25 → 0.28, 0.45 → 0.48 (widen safe zones)
4. Add "just/only" detection to FORCE simple tier
5. Reduce length-based scoring impact
""")
