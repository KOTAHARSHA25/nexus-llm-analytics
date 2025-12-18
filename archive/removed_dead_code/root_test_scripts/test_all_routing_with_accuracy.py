"""
🧪 COMPREHENSIVE ROUTING + ACCURACY TEST
Tests ALL routing mechanisms + model configurations + accuracy verification

✅ FULLY AUTOMATED (16 scenarios):
PART 1: ROUTING MECHANISMS (13 scenarios)
1. All ON - Optimal performance
2. Routing OFF - Manual + CoT
3. All OFF - Baseline
4. CoT Independent - Bug fix verification
5. Manual Override force_model
6. Manual Override user_model
7. Data Complexity - Consistency
8. Query Ambiguity
9. Rapid Fire - Concurrent
10. Cache Behavior
11. Mixed Complexity
12. Review Model Different
13. Complexity Upgrade

PART 2: MODEL CONFIGURATIONS + ACCURACY (3 scenarios)
14. LIGHTWEIGHT Model (tinyllama) - All queries with accuracy check
15. BALANCED Model (phi3:mini) - All queries with accuracy check
16. POWERFUL Model (llama3.1:8b) - All queries with accuracy check

Expected Duration: ~45-50 minutes
Total Queries: ~70-80 across all scenarios
"""

import sys
import os
import time
import json
import re
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

def print_result(query, result, test_num, total, expected_tier=None, expect_cot=False, expected_answer=None):
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
        answer = result.get('answer', '')
        
        print(f"{'✅' if success else '❌'} Success: {success}")
        print(f"📊 Complexity: {complexity:.3f}")
        print(f"🤖 Model: {model_used}")
        print(f"⚡ Tier: {tier}")
        print(f"🧠 CoT: {'✓ Yes' if cot_used else '✗ No'}")
        print(f"⏱️  Time: {processing_time:.2f}s")
        
        # Accuracy check
        accuracy_pass = None
        if expected_answer is not None:
            accuracy_pass = check_answer_accuracy(answer, expected_answer)
            print(f"🎯 Accuracy: {'✅ CORRECT' if accuracy_pass else '❌ WRONG'}")
            if expected_answer.get('details'):
                print(f"   Expected: {expected_answer['details']}")
        
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
            'model': model_used,
            'accurate': accuracy_pass
        }
    else:
        print(f"❌ Error: {result}")
        return {'success': False, 'time': 0, 'tier': 'ERROR', 'cot': False, 'accurate': False}

def check_answer_accuracy(answer: str, expected: Dict[str, Any]) -> bool:
    """Check if answer matches expected value"""
    if not answer:
        return False
    
    answer_lower = answer.lower()
    
    # Check based on type
    check_type = expected.get('type', 'exact')
    
    if check_type == 'number':
        # Extract number from answer
        numbers = re.findall(r'\d+\.?\d*', answer)
        if numbers:
            try:
                actual = float(numbers[0])
                expected_val = expected['value']
                tolerance = expected.get('tolerance', 0)
                return abs(actual - expected_val) <= tolerance
            except:
                return False
    
    elif check_type == 'contains':
        # Check if answer contains expected keyword
        keywords = expected.get('keywords', [])
        return any(keyword.lower() in answer_lower for keyword in keywords)
    
    elif check_type == 'range':
        # Check if number is in expected range
        numbers = re.findall(r'\d+\.?\d*', answer)
        if numbers:
            try:
                actual = float(numbers[0])
                min_val = expected.get('min', 0)
                max_val = expected.get('max', float('inf'))
                return min_val <= actual <= max_val
            except:
                return False
    
    return False

# Test queries with expected behavior AND accuracy verification
TEST_QUERIES_WITH_ACCURACY = {
    'easy': [
        {
            "query": "How many rows are in this dataset?",
            "expected_tier": "FAST",
            "expect_cot": False,
            "expected_time": 3,
            "expected_answer": {
                "type": "number",
                "value": 100,
                "tolerance": 0,
                "details": "100 rows"
            }
        },
        {
            "query": "Count the unique products",
            "expected_tier": "FAST",
            "expect_cot": False,
            "expected_time": 3,
            "expected_answer": {
                "type": "number",
                "value": 5,
                "tolerance": 0,
                "details": "5 products (Widget A-E)"
            }
        },
        {
            "query": "What is the sum of revenue?",
            "expected_tier": "FAST",
            "expect_cot": False,
            "expected_time": 3,
            "expected_answer": {
                "type": "range",
                "min": 2500000,
                "max": 2600000,
                "details": "~$2,563,044"
            }
        }
    ],
    'medium': [
        {
            "query": "Show average sales by region",
            "expected_tier": "BALANCED",
            "expect_cot": False,
            "expected_time": 6,
            "expected_answer": {
                "type": "contains",
                "keywords": ["region", "average", "North", "South", "East", "West"],
                "details": "Should mention 4 regions"
            }
        },
        {
            "query": "Which product has highest revenue?",
            "expected_tier": "BALANCED",
            "expect_cot": False,
            "expected_time": 6,
            "expected_answer": {
                "type": "contains",
                "keywords": ["Widget A", "Product_A", "Widget_A"],
                "details": "Product_A or Widget A"
            }
        }
    ],
    'complex': [
        {
            "query": "Which region has best sales and why? Provide detailed analysis.",
            "expected_tier": "FULL_POWER",
            "expect_cot": True,
            "expected_time": 15,
            "expected_answer": {
                "type": "contains",
                "keywords": ["North"],
                "details": "North region"
            }
        },
        {
            "query": "Analyze correlation between price and revenue with statistical significance",
            "expected_tier": "FULL_POWER",
            "expect_cot": True,
            "expected_time": 15,
            "expected_answer": {
                "type": "contains",
                "keywords": ["positive", "negative", "correlation"],
                "details": "Should mention correlation direction"
            }
        }
    ],
    'ambiguous': [
        {
            "query": "Tell me something interesting about this data",
            "expected_tier": "BALANCED"
        },
        {
            "query": "What patterns do you see?",
            "expected_tier": "FULL_POWER"
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

def configure_system(smart_selection: bool, routing: bool, primary_model: str = None, review_model: str = None):
    """Configure system preferences"""
    prefs_manager = get_preferences_manager()
    
    updates = {
        "auto_model_selection": smart_selection,
        "enable_intelligent_routing": routing
    }
    
    if primary_model:
        updates["primary_model"] = primary_model
    
    if review_model:
        updates["review_model"] = review_model
    
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
                force_model=force_model
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

# Import all scenario functions from test_all_routing_scenarios.py
# This avoids code duplication
def run_routing_scenarios(sample_csv: Path):
    """Run all 13 routing scenarios"""
    print_header("PART 1: ROUTING MECHANISMS (13 scenarios)")
    
    # Import the scenario functions
    import test_all_routing_scenarios as routing_test
    
    all_results = {}
    
    # Run all routing scenarios
    all_results['scenario_1_all_on'] = routing_test.scenario_1_all_on(sample_csv)
    all_results['scenario_2_routing_off'] = routing_test.scenario_2_routing_off(sample_csv)
    all_results['scenario_3_all_off'] = routing_test.scenario_3_all_off(sample_csv)
    all_results['scenario_4_cot_independent'] = routing_test.scenario_4_cot_independent(sample_csv)
    all_results['scenario_5_manual_override'] = routing_test.scenario_5_manual_override(sample_csv)
    all_results['scenario_6_user_model_parameter'] = routing_test.scenario_6_user_model_parameter(sample_csv)
    all_results['scenario_7_data_complexity'] = routing_test.scenario_7_data_complexity(sample_csv)
    all_results['scenario_8_query_ambiguity'] = routing_test.scenario_8_query_ambiguity(sample_csv)
    all_results['scenario_9_rapid_fire'] = routing_test.scenario_9_rapid_fire(sample_csv)
    all_results['scenario_10_cache_behavior'] = routing_test.scenario_10_cache_behavior(sample_csv)
    all_results['scenario_11_mixed_complexity'] = routing_test.scenario_11_mixed_complexity(sample_csv)
    all_results['scenario_12_review_model_different'] = routing_test.scenario_12_review_model_different(sample_csv)
    all_results['scenario_13_complexity_upgrade'] = routing_test.scenario_13_complexity_upgrade(sample_csv)
    
    return all_results

def scenario_14_lightweight_model_accuracy(sample_csv: Path):
    """Scenario 14: Test lightweight model (tinyllama) with accuracy verification"""
    print_header("SCENARIO 14: LIGHTWEIGHT MODEL + ACCURACY")
    print("Configuration: Manual tinyllama for all queries")
    print("Expected: Fast but may have lower accuracy on complex queries")
    
    configure_system(smart_selection=False, routing=False, primary_model='tinyllama:latest')
    
    all_queries = (TEST_QUERIES_WITH_ACCURACY['easy'] + 
                   TEST_QUERIES_WITH_ACCURACY['medium'] + 
                   TEST_QUERIES_WITH_ACCURACY['complex'])
    
    results = []
    
    for idx, test in enumerate(all_queries, 1):
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'],
            result,
            idx,
            len(all_queries),
            expected_answer=test.get('expected_answer')
        )
        results.append(metrics)
    
    # Accuracy summary
    accurate_count = sum(1 for r in results if r.get('accurate'))
    total_with_accuracy = sum(1 for r in results if r.get('accurate') is not None)
    
    print(f"\n📊 LIGHTWEIGHT MODEL ACCURACY:")
    print(f"   Correct Answers: {accurate_count}/{total_with_accuracy}")
    print(f"   Accuracy Rate: {100*accurate_count/total_with_accuracy:.1f}%" if total_with_accuracy > 0 else "   No accuracy checks")
    
    return results

def scenario_15_balanced_model_accuracy(sample_csv: Path):
    """Scenario 15: Test balanced model (phi3:mini) with accuracy verification"""
    print_header("SCENARIO 15: BALANCED MODEL + ACCURACY")
    print("Configuration: Manual phi3:mini for all queries")
    print("Expected: Moderate speed and good accuracy")
    
    configure_system(smart_selection=False, routing=False, primary_model='phi3:mini')
    
    all_queries = (TEST_QUERIES_WITH_ACCURACY['easy'] + 
                   TEST_QUERIES_WITH_ACCURACY['medium'] + 
                   TEST_QUERIES_WITH_ACCURACY['complex'])
    
    results = []
    
    for idx, test in enumerate(all_queries, 1):
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'],
            result,
            idx,
            len(all_queries),
            expected_answer=test.get('expected_answer')
        )
        results.append(metrics)
    
    # Accuracy summary
    accurate_count = sum(1 for r in results if r.get('accurate'))
    total_with_accuracy = sum(1 for r in results if r.get('accurate') is not None)
    
    print(f"\n📊 BALANCED MODEL ACCURACY:")
    print(f"   Correct Answers: {accurate_count}/{total_with_accuracy}")
    print(f"   Accuracy Rate: {100*accurate_count/total_with_accuracy:.1f}%" if total_with_accuracy > 0 else "   No accuracy checks")
    
    return results

def scenario_16_powerful_model_accuracy(sample_csv: Path):
    """Scenario 16: Test powerful model (llama3.1:8b) with accuracy verification"""
    print_header("SCENARIO 16: POWERFUL MODEL + ACCURACY")
    print("Configuration: Manual llama3.1:8b for all queries")
    print("Expected: Slower but highest accuracy")
    
    configure_system(smart_selection=False, routing=False, primary_model='llama3.1:8b')
    
    all_queries = (TEST_QUERIES_WITH_ACCURACY['easy'] + 
                   TEST_QUERIES_WITH_ACCURACY['medium'] + 
                   TEST_QUERIES_WITH_ACCURACY['complex'])
    
    results = []
    
    for idx, test in enumerate(all_queries, 1):
        result = run_query(test['query'], sample_csv)
        metrics = print_result(
            test['query'],
            result,
            idx,
            len(all_queries),
            expected_answer=test.get('expected_answer')
        )
        results.append(metrics)
    
    # Accuracy summary
    accurate_count = sum(1 for r in results if r.get('accurate'))
    total_with_accuracy = sum(1 for r in results if r.get('accurate') is not None)
    
    print(f"\n📊 POWERFUL MODEL ACCURACY:")
    print(f"   Correct Answers: {accurate_count}/{total_with_accuracy}")
    print(f"   Accuracy Rate: {100*accurate_count/total_with_accuracy:.1f}%" if total_with_accuracy > 0 else "   No accuracy checks")
    
    return results

def generate_comprehensive_report(all_results: Dict[str, List]):
    """Generate final comprehensive report with accuracy analysis"""
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
    
    # Accuracy totals
    total_accurate = sum(
        sum(1 for r in results if r.get('accurate') == True)
        for results in all_results.values()
    )
    total_accuracy_checks = sum(
        sum(1 for r in results if r.get('accurate') is not None)
        for results in all_results.values()
    )
    
    print(f"📊 Overall Statistics:")
    print(f"   Total Scenarios: {len(all_results)}")
    print(f"   Total Queries: {total_queries}")
    print(f"   Success Rate: {total_success}/{total_queries} ({100*total_success/total_queries:.1f}%)")
    print(f"   Accuracy Rate: {total_accurate}/{total_accuracy_checks} ({100*total_accurate/total_accuracy_checks:.1f}%)" if total_accuracy_checks > 0 else "   Accuracy: Not measured")
    print(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()
    
    # Per-scenario summary
    print("📋 Scenario Summary:")
    print(f"{'─'*80}")
    print(f"{'Scenario':<40} {'Queries':<8} {'Success':<10} {'Accuracy':<12} {'Avg Time'}")
    print(f"{'─'*80}")
    
    for scenario_name, results in all_results.items():
        if not results:
            continue
        
        success_count = sum(1 for r in results if r.get('success'))
        accurate_count = sum(1 for r in results if r.get('accurate') == True)
        accuracy_total = sum(1 for r in results if r.get('accurate') is not None)
        avg_time = sum(r.get('time', 0) for r in results) / len(results)
        
        accuracy_str = f"{accurate_count}/{accuracy_total}" if accuracy_total > 0 else "N/A"
        
        print(f"{scenario_name:<40} {len(results):<8} {success_count}/{len(results):<10} {accuracy_str:<12} {avg_time:>6.2f}s")
    
    print(f"{'─'*80}")
    
    # Model comparison
    print(f"\n📊 MODEL ACCURACY COMPARISON:")
    model_scenarios = {
        'scenario_14_lightweight_model_accuracy': 'Lightweight (tinyllama)',
        'scenario_15_balanced_model_accuracy': 'Balanced (phi3:mini)',
        'scenario_16_powerful_model_accuracy': 'Powerful (llama3.1:8b)'
    }
    
    for scenario_key, model_name in model_scenarios.items():
        if scenario_key in all_results:
            results = all_results[scenario_key]
            accurate = sum(1 for r in results if r.get('accurate') == True)
            total = sum(1 for r in results if r.get('accurate') is not None)
            avg_time = sum(r.get('time', 0) for r in results) / len(results)
            
            if total > 0:
                print(f"\n{model_name}:")
                print(f"   Accuracy: {accurate}/{total} ({100*accurate/total:.1f}%)")
                print(f"   Avg Time: {avg_time:.2f}s per query")
    
    print(f"\n{'='*80}")
    print("✅ COMPREHENSIVE TEST COMPLETE")
    print(f"{'='*80}\n")

def main():
    print_header("🧪 COMPREHENSIVE ROUTING + ACCURACY TEST")
    print("Testing ALL routing mechanisms + model configurations + accuracy")
    print(f"Expected Duration: ~45-50 minutes")
    print(f"Total Queries: ~70-80 across 16 scenarios")
    
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
        # PART 1: Routing scenarios (scenarios 1-13)
        routing_results = run_routing_scenarios(sample_csv)
        all_results.update(routing_results)
        
        # PART 2: Model + Accuracy scenarios (scenarios 14-16)
        print_header("PART 2: MODEL CONFIGURATIONS + ACCURACY (3 scenarios)")
        
        all_results['scenario_14_lightweight_model_accuracy'] = scenario_14_lightweight_model_accuracy(sample_csv)
        all_results['scenario_15_balanced_model_accuracy'] = scenario_15_balanced_model_accuracy(sample_csv)
        all_results['scenario_16_powerful_model_accuracy'] = scenario_16_powerful_model_accuracy(sample_csv)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Generate final report
    generate_comprehensive_report(all_results)
    
    print(f"⏱️  Total Test Duration: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Save results
    output_file = Path(__file__).parent / "test_routing_accuracy_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"📁 Detailed results saved to: {output_file}")
    
    # Restore defaults
    print("\n🔄 Restoring default preferences...")
    configure_system(smart_selection=True, routing=True)
    print("✅ Done!")

if __name__ == "__main__":
    main()
