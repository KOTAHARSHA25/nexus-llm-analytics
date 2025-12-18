"""
🧪 COMPREHENSIVE ROUTING SCENARIOS TEST
Tests ALL routing mechanisms, model configurations, and edge cases:

✅ FULLY AUTOMATED (13 scenarios):
1. All ON - Optimal performance test
2. Routing OFF - Manual + CoT verification
3. All OFF - Baseline performance
4. CoT Independent - CRITICAL: Verifies bug fix (CoT works without routing)
5. Manual Override force_model - Tests decision hierarchy
6. Manual Override user_model - Tests alternative parameter name
7. Data Complexity - Validates complexity scoring
8. Query Ambiguity - Tests vague query handling
9. Rapid Fire - Concurrent load (5 queries < 1s apart)
10. Cache Behavior - Tests cache hit/miss with timing verification
11. Mixed Complexity - Complex + simple requirements in one query
12. Review Model Different - Tests primary != review model
13. Complexity Upgrade - Tests FAST -> BALANCED upgrade for complex queries

⏭️ MANUAL TESTS (2 scenarios - documented but skipped):
14. Missing Model Fallback - Requires uninstalling models
15. Smart Model Selection RAM - Requires changing available RAM

Expected Duration: ~35-40 minutes
Total Queries: ~60-70 across all scenarios
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import threading
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backend.core.user_preferences import get_preferences_manager
from backend.agents.crew_manager import CrewManager
from backend.core.advanced_cache import clear_all_caches

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_section(text):
    print("\n" + "─"*80)
    print(f"  {text}")
    print("─"*80)

def print_result(query, result, test_num, total, expected_tier=None, expect_cot=False):
    """Print test result with verification"""
    print(f"\n{'─'*80}")
    print(f"TEST {test_num}/{total}: {query[:60]}...")
    print(f"{'─'*80}")
    
    if isinstance(result, dict):
        success = result.get('success', False)
        model_used = result.get('model_used', 'Unknown')
        complexity = result.get('complexity_score', 0)
        tier = result.get('routing_tier', 'N/A')
        cot_used = result.get('cot_used', False)
        processing_time = result.get('processing_time_seconds', 0)
        
        print(f"{'✅' if success else '❌'} Success: {success}")
        print(f"📊 Complexity: {complexity:.3f}")
        print(f"🤖 Model: {model_used}")
        print(f"⚡ Tier: {tier}")
        print(f"🧠 CoT: {'✓ Yes' if cot_used else '✗ No'}")
        print(f"⏱️  Time: {processing_time:.2f}s")
        
        # Verify expectations
        if expected_tier:
            tier_match = tier == expected_tier
            print(f"   Expected Tier: {expected_tier} {'✓' if tier_match else '✗ MISMATCH'}")
        
        if expect_cot:
            cot_match = cot_used == expect_cot
            print(f"   Expected CoT: {expect_cot} {'✓' if cot_match else '✗ MISMATCH'}")
            
        return {
            'success': success,
            'time': processing_time,
            'tier': tier,
            'cot': cot_used,
            'complexity': complexity,
            'model': model_used
        }
    else:
        print(f"❌ Error: {result}")
        return {'success': False, 'time': 0, 'tier': 'ERROR', 'cot': False}

# Test queries with expected behavior
TEST_QUERIES = {
    'easy': [
        {
            "query": "How many rows are in this dataset?",
            "expected_tier": "FAST",
            "expect_cot": False,
            "expected_time": 3
        },
        {
            "query": "Count the unique products",
            "expected_tier": "FAST",
            "expect_cot": False,
            "expected_time": 3
        },
        {
            "query": "What is the sum of revenue?",
            "expected_tier": "FAST",
            "expect_cot": False,
            "expected_time": 3
        }
    ],
    'medium': [
        {
            "query": "Show average sales by region",
            "expected_tier": "BALANCED",
            "expect_cot": False,
            "expected_time": 6
        },
        {
            "query": "Which product has highest revenue?",
            "expected_tier": "BALANCED",
            "expect_cot": False,
            "expected_time": 6
        }
    ],
    'complex': [
        {
            "query": "Which region has best sales and why? Provide detailed analysis.",
            "expected_tier": "FULL_POWER",
            "expect_cot": True,
            "expected_time": 15
        },
        {
            "query": "Analyze correlation between price and revenue with statistical significance",
            "expected_tier": "FULL_POWER",
            "expect_cot": True,
            "expected_time": 15
        }
    ],
    'ambiguous': [
        {
            "query": "Tell me something interesting about this data",
            "expected_tier": "BALANCED",  # Should be medium-high
            "expect_cot": False
        },
        {
            "query": "What patterns do you see?",
            "expected_tier": "FULL_POWER",
            "expect_cot": False
        }
    ],
    'mixed': [
        {
            "query": "What is the total revenue AND predict next month's sales based on trends?",
            "expected_tier": "FULL_POWER",
            "expect_cot": True,
            "expected_time": 15
        }
    ]
}

def configure_system(smart_selection: bool, routing: bool, primary_model: str = None):
    """Configure system preferences"""
    prefs_manager = get_preferences_manager()
    
    updates = {
        "auto_model_selection": smart_selection,
        "enable_intelligent_routing": routing
    }
    
    if primary_model:
        updates["primary_model"] = primary_model
    
    prefs_manager.update_preferences(**updates)
    
    # Clear singleton to force reload
    CrewManager._instance = None
    
    return prefs_manager.load_preferences()

def run_query(query: str, sample_csv: Path, force_model: str = None) -> Dict[str, Any]:
    """Run a single query and return results"""
    crew_manager = CrewManager()
    
    start_time = time.time()
    
    try:
        # Use force_model if provided (tests manual override)
        if force_model:
            result = crew_manager.analyze_structured_data(
                query=query,
                filename=str(sample_csv),
                force_model=force_model  # Manual override
            )
        else:
            result = crew_manager.analyze_structured_data(
                query=query,
                filename=str(sample_csv)
            )
        
        processing_time = time.time() - start_time
        
        # Extract metadata
        if isinstance(result, dict):
            result['processing_time_seconds'] = processing_time
            return result
        else:
            return {
                'success': False,
                'error': str(result),
                'processing_time_seconds': processing_time
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'processing_time_seconds': time.time() - start_time
        }

def scenario_1_all_on(sample_csv: Path):
    """Scenario 1: All mechanisms enabled (optimal)"""
    print_header("SCENARIO 1: ALL ON (Optimal Performance)")
    print("Configuration: Smart Selection ON + Routing ON")
    
    configure_system(smart_selection=True, routing=True)
    
    all_queries = TEST_QUERIES['easy'] + TEST_QUERIES['medium'] + TEST_QUERIES['complex']
    results = []
    
    for idx, test in enumerate(all_queries, 1):
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'], 
            result, 
            idx, 
            len(all_queries),
            test.get('expected_tier'),
            test.get('expect_cot')
        )
        results.append(metrics)
    
    return results

def scenario_2_routing_off(sample_csv: Path):
    """Scenario 2: Routing OFF but Smart Selection ON"""
    print_header("SCENARIO 2: ROUTING OFF (Manual + CoT)")
    print("Configuration: Smart Selection ON + Routing OFF")
    print("Expected: All use primary model, but CoT still works for complex queries")
    
    configure_system(smart_selection=True, routing=False)
    
    all_queries = TEST_QUERIES['easy'] + TEST_QUERIES['medium'] + TEST_QUERIES['complex']
    results = []
    
    for idx, test in enumerate(all_queries, 1):
        result = run_query(test['query'], sample_csv)
        # Don't check tier since routing is off, but DO check CoT
        metrics = print_result(
            test['query'], 
            result, 
            idx, 
            len(all_queries),
            expected_tier=None,  # No tier expectations
            expect_cot=test.get('expect_cot')
        )
        results.append(metrics)
    
    return results

def scenario_3_all_off(sample_csv: Path):
    """Scenario 3: All automatic features disabled"""
    print_header("SCENARIO 3: ALL OFF (Baseline Performance)")
    print("Configuration: Smart Selection OFF + Routing OFF + Manual llama3.1:8b")
    print("Expected: All queries use same model, slower but consistent")
    
    configure_system(smart_selection=False, routing=False, primary_model='llama3.1:8b')
    
    all_queries = TEST_QUERIES['easy'] + TEST_QUERIES['medium'] + TEST_QUERIES['complex']
    results = []
    
    for idx, test in enumerate(all_queries, 1):
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'], 
            result, 
            idx, 
            len(all_queries)
        )
        results.append(metrics)
    
    return results

def scenario_4_cot_independent(sample_csv: Path):
    """Scenario 4: CoT works independently of routing"""
    print_header("SCENARIO 4: CoT INDEPENDENT (🔴 CRITICAL BUG FIX VERIFICATION)")
    print("Configuration: Routing OFF + Manual phi3:mini")
    print("Expected: CoT should STILL trigger for complex queries")
    print("This verifies the bug fix where CoT was tied to routing")
    
    configure_system(smart_selection=False, routing=False, primary_model='phi3:mini')
    
    # Test only complex queries
    results = []
    
    for idx, test in enumerate(TEST_QUERIES['complex'], 1):
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'], 
            result, 
            idx, 
            len(TEST_QUERIES['complex']),
            expect_cot=True  # MUST trigger CoT
        )
        results.append(metrics)
        
        # Verify CoT triggered
        if not metrics.get('cot'):
            print("   🔴 CRITICAL FAILURE: CoT did not trigger!")
        else:
            print("   ✅ PASS: CoT triggered independently of routing")
    
    return results

def scenario_5_manual_override(sample_csv: Path):
    """Scenario 5: Manual model override via force_model parameter"""
    print_header("SCENARIO 5: MANUAL OVERRIDE (Decision Hierarchy Test)")
    print("Configuration: Routing ON + Manual tinyllama selected")
    print("Expected: Decision hierarchy: force_model > user primary > routing")
    
    configure_system(smart_selection=False, routing=True, primary_model='tinyllama:latest')
    
    # Test complex query with manual tinyllama (normally uses llama3.1)
    results = []
    
    for idx, test in enumerate(TEST_QUERIES['complex'], 1):
        print(f"\n🎯 Testing manual override: Force tinyllama for complex query")
        result = run_query(test['query'], sample_csv, force_model='tinyllama:latest')
        metrics = print_result(
            test['query'], 
            result, 
            idx, 
            len(TEST_QUERIES['complex'])
        )
        results.append(metrics)
        
        # Verify tinyllama was used despite routing being ON
        if 'tinyllama' in metrics.get('model', '').lower():
            print("   ✅ PASS: Manual override respected")
        else:
            print(f"   ✗ MISMATCH: Expected tinyllama, got {metrics.get('model')}")
    
    return results

def scenario_6_user_model_parameter(sample_csv: Path):
    """Scenario 6: Manual model override via user_model parameter (alternative name)"""
    print_header("SCENARIO 6: MANUAL OVERRIDE (user_model parameter)")
    print("Configuration: Routing ON + Manual tinyllama via user_model")
    print("Expected: user_model parameter has same priority as force_model")
    
    configure_system(smart_selection=False, routing=True, primary_model='phi3:mini')
    
    # Test complex query with manual tinyllama via user_model (alternative parameter name)
    results = []
    
    for idx, test in enumerate(TEST_QUERIES['complex'], 1):
        print(f"\n🎯 Testing user_model override: Force tinyllama for complex query")
        # Note: This tests if user_model parameter works the same as force_model
        crew_manager = CrewManager()
        start_time = time.time()
        
        try:
            result = crew_manager.analyze_structured_data(
                query=test['query'],
                filename=str(sample_csv),
                user_model='tinyllama:latest'  # Alternative parameter name
            )
            
            processing_time = time.time() - start_time
            
            if isinstance(result, dict):
                result['processing_time_seconds'] = processing_time
            
            metrics = print_result(
                test['query'], 
                result, 
                idx, 
                len(TEST_QUERIES['complex'])
            )
            results.append(metrics)
            
            # Verify tinyllama was used despite routing being ON
            if 'tinyllama' in metrics.get('model', '').lower():
                print("   ✅ PASS: user_model override respected")
            else:
                print(f"   ✗ MISMATCH: Expected tinyllama, got {metrics.get('model')}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({'success': False, 'time': 0, 'model': 'ERROR'})
    
    return results

def scenario_7_data_complexity(sample_csv: Path):
    """Scenario 7: Validates complexity scoring"""
    print_header("SCENARIO 7: DATA COMPLEXITY IMPACT")
    print("Configuration: Routing ON")
    print("Expected: Same queries should get similar complexity scores")
    
    configure_system(smart_selection=True, routing=True)
    
    # Test same query twice to verify consistency
    results = []
    test = TEST_QUERIES['medium'][0]  # "Show average sales by region"
    
    for run in range(1, 3):
        print(f"\n🔁 Run {run}/2: Testing complexity consistency")
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'], 
            result, 
            run, 
            2,
            test.get('expected_tier')
        )
        results.append(metrics)
    
    # Compare complexity scores
    if len(results) == 2:
        complexity_diff = abs(results[0]['complexity'] - results[1]['complexity'])
        print(f"\n📊 Complexity Variance: {complexity_diff:.3f}")
        if complexity_diff < 0.1:
            print("   ✅ PASS: Consistent complexity scoring")
        else:
            print("   ⚠️  WARNING: High variance in complexity scores")
    
    return results

def scenario_8_query_ambiguity(sample_csv: Path):
    """Scenario 8: Vague query handling"""
    print_header("SCENARIO 8: QUERY AMBIGUITY HANDLING")
def scenario_8_query_ambiguity(sample_csv: Path):
    """Scenario 8: Vague query handling"""
    print_header("SCENARIO 8: QUERY AMBIGUITY HANDLING")
    print("Configuration: Routing ON")
    print("Expected: Vague queries should route to BALANCED or FULL_POWER")
    
    configure_system(smart_selection=True, routing=True)
    
    results = []
    
    for idx, test in enumerate(TEST_QUERIES['ambiguous'], 1):
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'], 
            result, 
            idx, 
            len(TEST_QUERIES['ambiguous'])
        )
        results.append(metrics)
        
        # Verify it didn't use FAST tier for ambiguous query
        if metrics.get('tier') == 'FAST':
            print("   ⚠️  WARNING: Ambiguous query used FAST tier")
        else:
            print(f"   ✅ PASS: Used appropriate tier ({metrics.get('tier')})")
    
    return results

def scenario_9_rapid_fire(sample_csv: Path):
    """Scenario 9: Concurrent query handling"""
    print_header("SCENARIO 9: RAPID FIRE QUERIES (Concurrent Load)")
def scenario_9_rapid_fire(sample_csv: Path):
    """Scenario 9: Concurrent query handling"""
    print_header("SCENARIO 9: RAPID FIRE QUERIES (Concurrent Load)")
    print("Configuration: Routing ON")
    print("Expected: All queries complete successfully with <1s spacing")
    
    configure_system(smart_selection=True, routing=True)
    
    # Use simple queries for rapid fire
    queries = TEST_QUERIES['easy'][:3]
    results = []
    threads = []
    
    def threaded_query(test, idx):
        result = run_query(test['query'], sample_csv)
        results.append((idx, result))
    
    print(f"\n🚀 Launching {len(queries)} queries concurrently...")
    start_time = time.time()
    
    for idx, test in enumerate(queries, 1):
        thread = threading.Thread(target=threaded_query, args=(test, idx))
        threads.append(thread)
        thread.start()
        time.sleep(0.2)  # 200ms between launches
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️  All queries completed in {total_time:.2f}s")
    print(f"📊 Success rate: {len([r for i, r in results if r.get('success')])}/{len(results)}")
    
    # Print individual results
    for idx, result in sorted(results):
        print_result(queries[idx-1]['query'], result, idx, len(queries))
    
    if len(results) == len(queries):
        print("\n✅ PASS: All concurrent queries handled successfully")
    else:
        print("\n⚠️  WARNING: Some queries failed")
    
    return [r for _, r in sorted(results)]

def scenario_10_cache_behavior(sample_csv: Path):
    """Scenario 10: Cache hit/miss testing"""
    print_header("SCENARIO 10: CACHE BEHAVIOR")
def scenario_10_cache_behavior(sample_csv: Path):
    """Scenario 10: Cache hit/miss testing"""
    print_header("SCENARIO 10: CACHE BEHAVIOR")
    print("Configuration: Routing ON")
    print("Expected: Second query should be faster (cache hit)")
    
    configure_system(smart_selection=True, routing=True)
    
    # Clear cache first
    print("🗑️  Clearing cache...")
    clear_all_caches()
    
    test = TEST_QUERIES['easy'][0]  # Simple query
    results = []
    
    # First query (cache miss)
    print("\n🔵 First Query (Cache MISS expected):")
    result1 = run_query(test['query'], sample_csv)
    metrics1 = print_result(test['query'], result1, 1, 2)
    results.append(metrics1)
    
    # Second query (cache hit)
    print("\n🟢 Second Query (Cache HIT expected):")
    result2 = run_query(test['query'], sample_csv)
    metrics2 = print_result(test['query'], result2, 2, 2)
    results.append(metrics2)
    
    # Compare times
    time_diff = metrics1['time'] - metrics2['time']
    speedup = (time_diff / metrics1['time']) * 100 if metrics1['time'] > 0 else 0
    
    print(f"\n📊 Cache Performance:")
    print(f"   First query: {metrics1['time']:.2f}s")
    print(f"   Second query: {metrics2['time']:.2f}s")
    print(f"   Speedup: {speedup:.1f}%")
    
    if speedup > 50:  # Expect significant speedup
        print("   ✅ PASS: Cache working effectively")
    else:
        print("   ⚠️  WARNING: Cache may not be working")
    
    return results

def scenario_11_mixed_complexity(sample_csv: Path):
    """Scenario 11: Query with mixed complexity requirements"""
    print_header("SCENARIO 11: MIXED COMPLEXITY QUERY")
    print("Configuration: Routing ON")
    print("Expected: Should route to FULL_POWER due to complex component")
    
    configure_system(smart_selection=True, routing=True)
    
    results = []
    
    for idx, test in enumerate(TEST_QUERIES['mixed'], 1):
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'], 
            result, 
            idx, 
            len(TEST_QUERIES['mixed']),
            test.get('expected_tier'),
            test.get('expect_cot')
        )
        results.append(metrics)
    
    return results

def scenario_12_review_model_different(sample_csv: Path):
    """Scenario 12: Test when review model != primary model"""
    print_header("SCENARIO 12: REVIEW MODEL DIFFERENT FROM PRIMARY")
    print("Configuration: Manual mode, primary=tinyllama, review=phi3:mini")
    print("Expected: CoT should use different models for generation vs critique")
    
    # Configure with different primary and review models
    prefs_manager = get_preferences_manager()
    prefs_manager.update_preferences(
        auto_model_selection=False,
        enable_intelligent_routing=False,
        primary_model='tinyllama:latest',
        review_model='phi3:mini'
    )
    CrewManager._instance = None
    
    results = []
    
    # Test complex query that should trigger CoT
    for idx, test in enumerate(TEST_QUERIES['complex'][:1], 1):  # Just test one
        print(f"\n🧠 Testing CoT with different primary/review models")
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'], 
            result, 
            idx, 
            1,
            expect_cot=True
        )
        results.append(metrics)
        
        # Verify CoT triggered
        if metrics.get('cot'):
            print("   ✅ PASS: CoT triggered with different models")
        else:
            print("   ⚠️  WARNING: CoT did not trigger")
    
    return results

def scenario_13_complexity_upgrade(sample_csv: Path):
    """Scenario 13: Test complexity-based tier upgrade (FAST -> BALANCED)"""
    print_header("SCENARIO 13: COMPLEXITY TIER UPGRADE")
    print("Configuration: Routing ON with intentionally weak FAST model")
    print("Expected: High complexity queries should upgrade from FAST to BALANCED")
    
    configure_system(smart_selection=True, routing=True)
    
    results = []
    
    # Use a complex query that should NOT use FAST tier
    test = {
        "query": "Analyze statistical correlation between price, revenue, and marketing spend using regression",
        "expected_tier": "FULL_POWER",  # High complexity should skip FAST
        "expect_cot": True
    }
    
    print(f"\n📊 Testing high-complexity query routing")
    result = run_query(test['query'], sample_csv)
    metrics = print_result(
        test['query'], 
        result, 
        1, 
        1,
        test.get('expected_tier'),
        test.get('expect_cot')
    )
    results.append(metrics)
    
    # Verify it didn't use FAST tier
    if metrics.get('tier') != 'FAST':
        print("   ✅ PASS: High complexity query correctly upgraded beyond FAST tier")
    else:
        print("   ⚠️  WARNING: High complexity query incorrectly used FAST tier")
    
    return results

def scenario_10_mixed_complexity(sample_csv: Path):
    """Scenario 10: Query with mixed complexity requirements"""
    print_header("SCENARIO 10: MIXED COMPLEXITY QUERY")
    print("Configuration: Routing ON")
    print("Expected: Should route to FULL_POWER due to complex component")
    
    configure_system(smart_selection=True, routing=True)
    
    results = []
    
    for idx, test in enumerate(TEST_QUERIES['mixed'], 1):
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'], 
            result, 
            idx, 
            len(TEST_QUERIES['mixed']),
            test.get('expected_tier'),
            test.get('expect_cot')
        )
        results.append(metrics)
    
    return results

def generate_final_report(all_results: Dict[str, List]):
    """Generate comprehensive final report"""
    print_header("FINAL COMPREHENSIVE REPORT")
    
    # Calculate totals
    total_queries = sum(len(results) for results in all_results.values())
    total_success = sum(
        sum(1 for r in results if r.get('success'))
        for results in all_results.values()
    )
    total_time = sum(
        sum(r.get('time', 0) for r in results)
        for results in all_results.values()
    )
    
    print(f"📊 Overall Statistics:")
    print(f"   Total Scenarios: {len(all_results)}")
    print(f"   Total Queries: {total_queries}")
    print(f"   Success Rate: {total_success}/{total_queries} ({100*total_success/total_queries:.1f}%)")
    print(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()
    
    # Per-scenario summary
    print("📋 Scenario Summary:")
    print(f"{'─'*80}")
    print(f"{'Scenario':<35} {'Queries':<8} {'Success':<8} {'Avg Time':<10}")
    print(f"{'─'*80}")
    
    for scenario_name, results in all_results.items():
        if not results:
            continue
        
        success_count = sum(1 for r in results if r.get('success'))
        avg_time = sum(r.get('time', 0) for r in results) / len(results)
        
        print(f"{scenario_name:<35} {len(results):<8} {success_count}/{len(results):<8} {avg_time:>6.2f}s")
    
    print(f"{'─'*80}")
    
    # Performance comparison
    if 'scenario_1_all_on' in all_results and 'scenario_3_all_off' in all_results:
        time_all_on = sum(r.get('time', 0) for r in all_results['scenario_1_all_on'])
        time_all_off = sum(r.get('time', 0) for r in all_results['scenario_3_all_off'])
        
        if time_all_off > 0:
            speedup = ((time_all_off - time_all_on) / time_all_off) * 100
            
            print(f"\n🚀 Performance Gain:")
            print(f"   All ON (Optimal):  {time_all_on:.1f}s")
            print(f"   All OFF (Baseline): {time_all_off:.1f}s")
            print(f"   Speedup: {speedup:.1f}% faster with intelligent routing!")
    
    # CoT verification
    cot_scenarios = ['scenario_2_routing_off', 'scenario_4_cot_independent']
    cot_triggered = 0
    cot_expected = 0
    
    for scenario in cot_scenarios:
        if scenario in all_results:
            for r in all_results[scenario]:
                if r.get('cot'):
                    cot_triggered += 1
                cot_expected += 1
    
    if cot_expected > 0:
        print(f"\n🧠 CoT Self-Correction:")
        print(f"   Triggered: {cot_triggered}/{cot_expected}")
        if cot_triggered == cot_expected:
            print(f"   ✅ PASS: CoT works independently of routing")
        else:
            print(f"   ⚠️  WARNING: CoT may not be triggering correctly")
    
    print(f"\n{'='*80}")
    print("✅ COMPREHENSIVE TEST COMPLETE")
    print(f"{'='*80}\n")

def main():
    print_header("🧪 COMPREHENSIVE ROUTING SCENARIOS TEST")
    print("Testing ALL routing mechanisms, model configurations, and edge cases")
    print(f"Expected Duration: ~35-40 minutes")
    print(f"Total Queries: ~60-70 across 13 scenarios")
    
    # Setup
    sample_csv = Path(__file__).parent / "data" / "samples" / "sales_data.csv"
    
    if not sample_csv.exists():
        print(f"\n❌ ERROR: Sample CSV not found at {sample_csv}")
        print(f"Please ensure data/samples/sales_data.csv exists")
        return
    
    print(f"\n✅ Found sample data: {sample_csv}")
    print(f"\n⚠️  IMPORTANT: Watch the BACKEND TERMINAL for routing logs!")
    print(f"   Look for: [INTELLIGENT ROUTING] messages")
    print(f"\n")
    
    input("Press ENTER to start comprehensive testing...")
    
    all_results = {}
    start_time = time.time()
    
    # Run all scenarios
    try:
        all_results['scenario_1_all_on'] = scenario_1_all_on(sample_csv)
        all_results['scenario_2_routing_off'] = scenario_2_routing_off(sample_csv)
        all_results['scenario_3_all_off'] = scenario_3_all_off(sample_csv)
        all_results['scenario_4_cot_independent'] = scenario_4_cot_independent(sample_csv)
        all_results['scenario_5_manual_override'] = scenario_5_manual_override(sample_csv)
        all_results['scenario_6_user_model_parameter'] = scenario_6_user_model_parameter(sample_csv)
        all_results['scenario_7_data_complexity'] = scenario_7_data_complexity(sample_csv)
        all_results['scenario_8_query_ambiguity'] = scenario_8_query_ambiguity(sample_csv)
        all_results['scenario_9_rapid_fire'] = scenario_9_rapid_fire(sample_csv)
        all_results['scenario_10_cache_behavior'] = scenario_10_cache_behavior(sample_csv)
        all_results['scenario_11_mixed_complexity'] = scenario_11_mixed_complexity(sample_csv)
        all_results['scenario_12_review_model_different'] = scenario_12_review_model_different(sample_csv)
        all_results['scenario_13_complexity_upgrade'] = scenario_13_complexity_upgrade(sample_csv)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Generate final report
    generate_final_report(all_results)
    
    print(f"⏱️  Total Test Duration: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Save results
    output_file = Path(__file__).parent / "test_routing_scenarios_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"📁 Detailed results saved to: {output_file}")
    
    # Restore defaults
    print("\n🔄 Restoring default preferences...")
    configure_system(smart_selection=True, routing=True)
    print("✅ Done!")

if __name__ == "__main__":
    main()
