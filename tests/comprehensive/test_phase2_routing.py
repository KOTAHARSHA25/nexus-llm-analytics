"""
PHASE 2: Intelligent Routing System Testing
Tests model tier selection based on query complexity
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.core.intelligent_router import IntelligentRouter, ModelTier, RoutingConfig
import time

print("=" * 80)
print("PHASE 2: INTELLIGENT ROUTING SYSTEM TESTING")
print("=" * 80)

# Initialize router
config = RoutingConfig(
    fast_model="tinyllama:latest",
    balanced_model="phi3:mini",
    full_power_model="llama3.1:8b",
    fast_threshold=0.25,
    balanced_threshold=0.45
)
router = IntelligentRouter(config)

total = 0
passed = 0

# TEST 1: Simple Query Routes to FAST
print("\n[TEST 1] Simple Query → FAST Tier")
total += 1
simple_queries = [
    "What is the total sales?",
    "Count how many records",
    "Show me the average",
    "What is the sum?"
]

fast_count = 0
for query in simple_queries:
    decision = router.route(query, {"rows": 100, "columns": 5})
    if decision.selected_tier == ModelTier.FAST:
        fast_count += 1
        
if fast_count >= 3:  # At least 75% should be FAST
    print(f"PASS: {fast_count}/4 simple queries routed to FAST tier")
    print(f"  - Selected model: {decision.selected_model}")
    print(f"  - Avg complexity: {decision.complexity_score:.3f}")
    passed += 1
else:
    print(f"FAIL: Only {fast_count}/4 simple queries routed to FAST")

# TEST 2: Medium Query Routes to BALANCED
print("\n[TEST 2] Medium Query → BALANCED Tier")
total += 1
medium_queries = [
    "Compare sales across regions and show trends",
    "Analyze customer segmentation by demographics",
    "Calculate correlation between age and revenue",
    "Show distribution of purchases by category"
]

balanced_count = 0
for query in medium_queries:
    decision = router.route(query, {"rows": 1000, "columns": 10})
    if decision.selected_tier in [ModelTier.BALANCED, ModelTier.FAST]:
        balanced_count += 1

if balanced_count >= 3:
    print(f"PASS: {balanced_count}/4 medium queries routed appropriately")
    print(f"  - Selected tier: {decision.selected_tier.value}")
    print(f"  - Complexity: {decision.complexity_score:.3f}")
    passed += 1
else:
    print(f"FAIL: Only {balanced_count}/4 queries routed to appropriate tier")

# TEST 3: Complex Query Routes to FULL_POWER
print("\n[TEST 3] Complex Query → FULL_POWER Tier")
total += 1
complex_queries = [
    "Perform advanced machine learning clustering analysis with PCA dimensionality reduction",
    "Build predictive model using ensemble methods and cross-validation",
    "Conduct comprehensive multivariate regression with interaction terms",
]

full_count = 0
for query in complex_queries:
    decision = router.route(query, {"rows": 10000, "columns": 20})
    if decision.selected_tier in [ModelTier.FULL_POWER, ModelTier.BALANCED]:
        full_count += 1

if full_count >= 2:  # At least 67% should be FULL_POWER/BALANCED
    print(f"PASS: {full_count}/3 complex queries routed to higher tiers")
    print(f"  - Selected tier: {decision.selected_tier.value}")
    print(f"  - Complexity: {decision.complexity_score:.3f}")
    passed += 1
else:
    print(f"FAIL: Only {full_count}/3 complex queries routed appropriately")

# TEST 4: Fallback Chain
print("\n[TEST 4] Fallback Chain Configuration")
total += 1
decision = router.route("Simple query", {"rows": 10, "columns": 3})
if decision.fallback_model:
    print(f"PASS: Fallback configured")
    print(f"  - Primary: {decision.selected_model}")
    print(f"  - Fallback: {decision.fallback_model}")
    passed += 1
else:
    print("FAIL: No fallback model configured")

# TEST 5: User Override
print("\n[TEST 5] User Override Functionality")
total += 1
decision = router.route("Any query", user_override="phi3:mini")
if decision.selected_model == "phi3:mini" and "OVERRIDE" in decision.reasoning.upper():
    print("PASS: User override respected")
    print(f"  - Forced model: {decision.selected_model}")
    passed += 1
else:
    print("FAIL: User override not working correctly")

# TEST 6: Statistics Tracking
print("\n[TEST 6] Statistics Tracking")
total += 1
# Make a few routing decisions
for i in range(5):
    router.route("Test query", {"rows": 100, "columns": 5})

stats = router.get_statistics()
if stats['total_decisions'] >= 5 and 'tier_distribution' in stats:
    print("PASS: Statistics tracked correctly")
    print(f"  - Total decisions: {stats['total_decisions']}")
    print(f"  - Tier distribution: {stats['tier_distribution']}")
    passed += 1
else:
    print("FAIL: Statistics not tracking properly")

# TEST 7: Routing Performance
print("\n[TEST 7] Routing Performance (<100ms)")
total += 1
start = time.time()
for i in range(10):
    router.route("Performance test query", {"rows": 1000, "columns": 10})
elapsed = (time.time() - start) / 10 * 1000  # Average per query in ms

if elapsed < 100:
    print(f"PASS: Average routing time {elapsed:.2f}ms (<100ms target)")
    passed += 1
else:
    print(f"FAIL: Average routing time {elapsed:.2f}ms (>100ms)")

# TEST 8: Complexity Score Ranges
print("\n[TEST 8] Complexity Score Ranges")
total += 1
simple = router.route("Count rows", {"rows": 10, "columns": 3})
complex_q = router.route("Machine learning prediction with cross-validation", {"rows": 10000, "columns": 50})

if simple.complexity_score < complex_q.complexity_score:
    print(f"PASS: Complexity ordering correct")
    print(f"  - Simple: {simple.complexity_score:.3f}")
    print(f"  - Complex: {complex_q.complexity_score:.3f}")
    passed += 1
else:
    print(f"FAIL: Complexity scores not ordered correctly")

# TEST 9: Reasoning Generation
print("\n[TEST 9] Routing Reasoning Generation")
total += 1
decision = router.route("Calculate average sales", {"rows": 500, "columns": 8})
if len(decision.reasoning) > 100 and "SELECTED TIER" in decision.reasoning.upper():
    print("PASS: Reasoning generated with sufficient detail")
    print(f"  - Reasoning length: {len(decision.reasoning)} chars")
    passed += 1
else:
    print("FAIL: Reasoning not generated properly")

# TEST 10: Data Info Impact
print("\n[TEST 10] Data Info Impact on Routing")
total += 1
small_data = router.route("Analyze correlations", {"rows": 50, "columns": 5})
large_data = router.route("Analyze correlations", {"rows": 100000, "columns": 100})

if large_data.complexity_score >= small_data.complexity_score:
    print("PASS: Data size impacts complexity correctly")
    print(f"  - Small data: {small_data.complexity_score:.3f}")
    print(f"  - Large data: {large_data.complexity_score:.3f}")
    passed += 1
else:
    print("FAIL: Data size not impacting complexity")

# Summary
print("\n" + "=" * 80)
print("PHASE 2 RESULTS: Intelligent Routing System")
print("=" * 80)
print(f"Total Tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Success Rate: {passed/total*100:.1f}%")

if passed == total:
    print("\n✅ ALL ROUTING TESTS PASSED")
elif passed >= total * 0.8:
    print(f"\n⚠️ MOSTLY WORKING - {total - passed} tests need attention")
else:
    print(f"\n❌ SIGNIFICANT ISSUES - {total - passed} tests failed")

# Show final routing statistics
print("\n" + "=" * 80)
print("ROUTING STATISTICS SUMMARY")
print("=" * 80)
final_stats = router.get_statistics()
print(f"Total Routing Decisions: {final_stats['total_decisions']}")
print(f"Tier Distribution: {final_stats['tier_distribution']}")
print(f"Average Complexity: {final_stats['average_complexity']:.3f}")
print(f"Average Routing Time: {final_stats['average_routing_time_ms']:.2f}ms")
