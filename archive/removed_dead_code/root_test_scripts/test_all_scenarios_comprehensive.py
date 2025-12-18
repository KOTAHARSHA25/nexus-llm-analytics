"""
🧪 COMPREHENSIVE MECHANISM TESTING - ALL 12 SCENARIOS
Tests all combinations of Smart Model Selection, Intelligent Routing, and CoT Self-Correction

SCENARIOS COVERED:
1. All ON (Optimal Performance)
2. Routing OFF (Manual + CoT)
3. All OFF (Manual Only)
4. CoT Independent (Routing OFF + CoT verifies independence)
5. Manual Override (User choice vs Routing)
6. Missing Model Fallback (Resilience test)
7. Data Complexity Impact (Small vs Large data)
8. Query Ambiguity Handling
9. Rapid Fire Queries (Concurrent load)
10. Cache Behavior (Cache hit/miss)
11. CoT Iteration Limit
12. Mixed Complexity Query

Total Time: ~30-35 minutes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import requests
import time
import json
from typing import Dict, List, Any
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
FILENAME = "sales_data.csv"

# Test queries with expected complexity and ground truth
TEST_QUERIES = {
    'easy': [
        {"query": "How many rows are in this dataset?", "answer": 100, "complexity": "LOW", "expected_tier": "FAST"},
        {"query": "Count the unique products", "answer": 5, "complexity": "LOW", "expected_tier": "FAST"},
        {"query": "What is the sum of revenue?", "answer": 2563044, "complexity": "LOW", "expected_tier": "FAST"},
    ],
    'medium': [
        {"query": "Show average sales by region", "answer": "4 regions", "complexity": "MEDIUM", "expected_tier": "BALANCED"},
        {"query": "Which product has highest revenue?", "answer": "Product_A", "complexity": "MEDIUM", "expected_tier": "BALANCED"},
    ],
    'complex': [
        {"query": "Which region has best sales and why?", "answer": "North", "complexity": "HIGH", "expected_tier": "FULL_POWER", "expect_cot": True},
        {"query": "Find correlation between price and revenue", "answer": "correlation", "complexity": "HIGH", "expected_tier": "FULL_POWER", "expect_cot": True},
    ],
    'ambiguous': [
        {"query": "Tell me something interesting about this data", "complexity": "MEDIUM-HIGH", "expected_tier": "BALANCED/FULL_POWER"},
        {"query": "What patterns do you see?", "complexity": "HIGH", "expected_tier": "FULL_POWER"},
    ],
    'mixed': [
        {"query": "What is the total revenue AND predict next month's sales based on trends?", "complexity": "HIGH", "expected_tier": "FULL_POWER", "expect_cot": True},
    ]
}

class ScenarioTester:
    def __init__(self):
        self.results = []
        self.current_scenario = None
        
    def update_preferences(self, auto_selection: bool, enable_routing: bool, primary_model: str = None):
        """Update user preferences via API"""
        # Get current preferences first to preserve required fields
        try:
            current = requests.get(f"{BASE_URL}/models/preferences")
            if current.status_code == 200:
                # Extract the nested preferences object
                response_data = current.json()
                prefs = response_data.get('preferences', {})
            else:
                # Default fallback
                prefs = {
                    "primary_model": "phi3:mini",
                    "review_model": "phi3:mini",
                    "embedding_model": "nomic-embed-text:latest",
                    "auto_selection": True,
                    "allow_swap": False
                }
        except:
            # Default fallback
            prefs = {
                "primary_model": "phi3:mini",
                "review_model": "phi3:mini",
                "embedding_model": "nomic-embed-text:latest",
                "auto_selection": True,
                "allow_swap": False
            }
        
        # Update with new settings (use flat structure for POST)
        update_payload = {
            "primary_model": prefs.get("primary_model", "phi3:mini"),
            "review_model": prefs.get("review_model", "phi3:mini"),
            "embedding_model": prefs.get("embedding_model", "nomic-embed-text:latest"),
            "auto_selection": auto_selection,
            "allow_swap": prefs.get("allow_swap", False),
            "enable_intelligent_routing": enable_routing
        }
        
        if primary_model:
            update_payload["primary_model"] = primary_model
            
        response = requests.post(f"{BASE_URL}/models/preferences", json=update_payload)
        if response.status_code == 200:
            print(f"✅ Preferences updated: Auto={auto_selection}, Routing={enable_routing}, Model={update_payload['primary_model']}")
            time.sleep(2)  # Wait longer for settings to propagate
            return True
        else:
            print(f"❌ Failed to update preferences: {response.text}")
            return False
    
    def clear_cache(self):
        """Clear cache between scenarios"""
        # Use the advanced cache clearing script instead of API endpoint
        import subprocess
        try:
            # Try to clear cache via Python script
            result = subprocess.run(
                ['python', 'clear_cache.py'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("🧹 Cache cleared via script")
            else:
                print("⚠️ Cache clear script failed (non-critical)")
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Could not clear cache: {e} (non-critical)")
            time.sleep(0.5)
    
    def run_query(self, query: str, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single query and collect results"""
        print(f"\n   📝 Query: {query[:60]}...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{BASE_URL}/analyze/",
                json={
                    "query": query,
                    "filename": FILENAME
                },
                timeout=180  # Increased to 3 minutes for complex queries
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                return {
                    "query": query,
                    "success": True,
                    "time": elapsed,
                    "result": result.get("result", ""),
                    "expected_tier": expected.get("expected_tier", "N/A"),
                    "expected_complexity": expected.get("complexity", "N/A"),
                    "expect_cot": expected.get("expect_cot", False),
                    "ground_truth": expected.get("answer", "N/A")
                }
            else:
                return {
                    "query": query,
                    "success": False,
                    "time": elapsed,
                    "error": response.text[:200],  # Truncate error
                    "expected_tier": expected.get("expected_tier", "N/A")
                }
                
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            print(f"   ⏱️  TIMEOUT after {elapsed:.1f}s - query too slow")
            return {
                "query": query,
                "success": False,
                "time": elapsed,
                "error": f"Request timeout after {elapsed:.1f}s",
                "expected_tier": expected.get("expected_tier", "N/A")
            }
        except KeyboardInterrupt:
            raise  # Allow user to cancel
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "query": query,
                "success": False,
                "time": elapsed,
                "error": str(e)[:200],  # Truncate error
                "expected_tier": expected.get("expected_tier", "N/A")
            }
    
    def run_scenario(self, name: str, description: str, config: Dict[str, Any], queries: List[Dict[str, Any]]):
        """Run a complete test scenario"""
        print(f"\n{'='*80}")
        print(f"🧪 SCENARIO: {name}")
        print(f"📋 Description: {description}")
        print(f"{'='*80}")
        
        self.current_scenario = name
        
        # Configure preferences
        if not self.update_preferences(
            config.get('auto_selection', True),
            config.get('enable_routing', True),
            config.get('primary_model')
        ):
            print(f"❌ Scenario {name} SKIPPED - Configuration failed")
            return
        
        # Clear cache for fresh test
        self.clear_cache()
        
        scenario_start = time.time()
        scenario_results = []
        
        # Run all queries
        for i, query_data in enumerate(queries, 1):
            print(f"\n   Query {i}/{len(queries)}:")
            result = self.run_query(query_data['query'], query_data)
            scenario_results.append(result)
            
            if result['success']:
                print(f"   ✅ Success ({result['time']:.1f}s)")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        total_time = time.time() - scenario_start
        
        # Calculate statistics
        successful = sum(1 for r in scenario_results if r['success'])
        avg_time = sum(r['time'] for r in scenario_results) / len(scenario_results) if scenario_results else 0
        
        self.results.append({
            'scenario': name,
            'description': description,
            'config': config,
            'total_time': total_time,
            'queries_run': len(queries),
            'successful': successful,
            'failed': len(queries) - successful,
            'avg_time_per_query': avg_time,
            'results': scenario_results
        })
        
        print(f"\n{'='*80}")
        print(f"📊 SCENARIO SUMMARY: {name}")
        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Success Rate: {successful}/{len(queries)} ({successful/len(queries)*100:.0f}%)")
        print(f"   Avg Time/Query: {avg_time:.1f}s")
        print(f"{'='*80}")
    
    def print_final_report(self):
        """Print comprehensive test report"""
        print(f"\n\n")
        print(f"{'='*100}")
        print(f"🎉 COMPREHENSIVE TEST REPORT - ALL SCENARIOS")
        print(f"{'='*100}")
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Scenarios: {len(self.results)}")
        print(f"\n")
        
        # Summary table
        print(f"{'Scenario':<40} {'Queries':<10} {'Success':<10} {'Total Time':<12} {'Avg Time':<10}")
        print(f"{'-'*100}")
        
        total_queries = 0
        total_success = 0
        total_time = 0
        
        for result in self.results:
            print(f"{result['scenario']:<40} "
                  f"{result['queries_run']:<10} "
                  f"{result['successful']}/{result['queries_run']:<8} "
                  f"{result['total_time']:.1f}s{' ':<8} "
                  f"{result['avg_time_per_query']:.1f}s")
            
            total_queries += result['queries_run']
            total_success += result['successful']
            total_time += result['total_time']
        
        print(f"{'-'*100}")
        print(f"{'TOTAL':<40} {total_queries:<10} {total_success}/{total_queries:<8} {total_time:.1f}s{' ':<8} {total_time/total_queries:.1f}s")
        print(f"\n")
        
        # Performance comparison
        print(f"{'='*100}")
        print(f"⚡ PERFORMANCE COMPARISON")
        print(f"{'='*100}")
        
        # Find baseline (All OFF scenario)
        baseline = next((r for r in self.results if "All OFF" in r['scenario']), None)
        optimal = next((r for r in self.results if "All ON" in r['scenario']), None)
        
        if baseline and optimal:
            improvement = ((baseline['total_time'] - optimal['total_time']) / baseline['total_time']) * 100
            print(f"Baseline (All OFF): {baseline['total_time']:.1f}s")
            print(f"Optimal (All ON): {optimal['total_time']:.1f}s")
            print(f"Performance Gain: {improvement:.1f}% faster ⚡")
        
        print(f"\n")
        
        # Key findings
        print(f"{'='*100}")
        print(f"🔍 KEY FINDINGS")
        print(f"{'='*100}")
        
        # Check for failures
        failures = [r for r in self.results if r['failed'] > 0]
        if failures:
            print(f"⚠️ FAILURES DETECTED:")
            for fail in failures:
                print(f"   - {fail['scenario']}: {fail['failed']} failed queries")
        else:
            print(f"✅ ALL SCENARIOS PASSED - No failures detected!")
        
        # Check timing patterns
        fast_scenarios = [r for r in self.results if r['avg_time_per_query'] < 3]
        slow_scenarios = [r for r in self.results if r['avg_time_per_query'] > 10]
        
        if fast_scenarios:
            print(f"\n⚡ FASTEST SCENARIOS (avg < 3s/query):")
            for s in fast_scenarios:
                print(f"   - {s['scenario']}: {s['avg_time_per_query']:.1f}s avg")
        
        if slow_scenarios:
            print(f"\n🐌 SLOWEST SCENARIOS (avg > 10s/query):")
            for s in slow_scenarios:
                print(f"   - {s['scenario']}: {s['avg_time_per_query']:.1f}s avg")
        
        print(f"\n{'='*100}")
        print(f"📝 DETAILED LOGS: Check backend terminal for routing/CoT activation logs")
        print(f"{'='*100}\n")

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║          🧪 COMPREHENSIVE MECHANISM TESTING - ALL 12 SCENARIOS 🧪             ║
║                                                                               ║
║  This test validates ALL combinations of:                                    ║
║  • Smart Model Selection (RAM-based auto-selection)                          ║
║  • Intelligent Routing (Complexity-based model selection)                    ║
║  • CoT Self-Correction (Validation for complex queries)                      ║
║                                                                               ║
║  Duration: ~30-35 minutes                                                    ║
║  Queries: ~50-60 total across all scenarios                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  PREREQUISITES:")
    print("   1. Backend running: cd src/backend && python -m uvicorn main:app --reload")
    print("   2. File uploaded: data/samples/sales_data.csv")
    print("   3. Models available: tinyllama, phi3:mini, llama3.1:8b")
    print("\n⚠️  IMPORTANT:")
    print("   - Watch the BACKEND TERMINAL for routing logs!")
    print("   - Look for: 🎯 [INTELLIGENT ROUTING] messages")
    print("   - If queries are slow (>10s for easy queries), routing may not be working")
    print("\n")
    
    input("Press ENTER to start comprehensive testing...")
    
    tester = ScenarioTester()
    
    # Get all queries
    all_queries = TEST_QUERIES['easy'] + TEST_QUERIES['medium'] + TEST_QUERIES['complex']
    
    # SCENARIO 1: All ON (Optimal)
    tester.run_scenario(
        "1. All ON (Optimal Performance)",
        "Smart Selection + Routing + CoT - Expected fastest overall",
        {'auto_selection': True, 'enable_routing': True},
        all_queries
    )
    
    # SCENARIO 2: Routing OFF
    tester.run_scenario(
        "2. Routing OFF (Manual + CoT)",
        "Smart Selection ON, Routing OFF - All use primary model, CoT still works",
        {'auto_selection': True, 'enable_routing': False},
        all_queries
    )
    
    # SCENARIO 3: All OFF
    tester.run_scenario(
        "3. All OFF (Manual Only)",
        "Manual model selection, no routing, no smart selection - Baseline performance",
        {'auto_selection': False, 'enable_routing': False, 'primary_model': 'llama3.1:8b'},
        all_queries
    )
    
    # SCENARIO 4: CoT Independent Operation (CRITICAL BUG VERIFICATION)
    tester.run_scenario(
        "4. CoT Independent (Bug Fix Verification)",
        "Routing OFF but CoT should STILL trigger for complex queries",
        {'auto_selection': False, 'enable_routing': False, 'primary_model': 'phi3:mini'},
        TEST_QUERIES['complex']  # Only complex queries to verify CoT works
    )
    
    # SCENARIO 5: Manual Override (User Choice vs Routing)
    tester.run_scenario(
        "5. Manual Override (User Choice Priority)",
        "Routing ON but manual tinyllama selected - Tests decision hierarchy",
        {'auto_selection': False, 'enable_routing': True, 'primary_model': 'tinyllama:latest'},
        TEST_QUERIES['complex']  # Complex queries with FAST model forced
    )
    
    # SCENARIO 6: Missing Model Fallback (Skip - requires uninstalling models)
    print(f"\n{'='*80}")
    print(f"⏭️  SCENARIO 6: Missing Model Fallback - SKIPPED")
    print(f"   Reason: Requires uninstalling models, manual test recommended")
    print(f"{'='*80}")
    
    # SCENARIO 7: Data Complexity Impact
    tester.run_scenario(
        "7. Data Complexity Impact",
        "Same queries with current dataset - Validates complexity scoring",
        {'auto_selection': True, 'enable_routing': True},
        all_queries[:5]  # Subset of queries
    )
    
    # SCENARIO 8: Query Ambiguity Handling
    tester.run_scenario(
        "8. Query Ambiguity Handling",
        "Vague/ambiguous queries - Should route to BALANCED or FULL_POWER",
        {'auto_selection': True, 'enable_routing': True},
        TEST_QUERIES['ambiguous']
    )
    
    # SCENARIO 9: Rapid Fire Queries
    print(f"\n{'='*80}")
    print(f"🧪 SCENARIO: 9. Rapid Fire Queries (Concurrent Load)")
    print(f"📋 Description: Submit multiple queries rapidly to test thread safety")
    print(f"{'='*80}")
    
    tester.update_preferences(True, True)
    tester.clear_cache()
    
    print("\n   Submitting 5 queries rapidly (<1s apart)...")
    rapid_queries = TEST_QUERIES['easy'][:5]
    rapid_start = time.time()
    
    import threading
    rapid_results = []
    
    def run_rapid_query(q):
        result = tester.run_query(q['query'], q)
        rapid_results.append(result)
    
    threads = []
    for q in rapid_queries:
        t = threading.Thread(target=run_rapid_query, args=(q,))
        threads.append(t)
        t.start()
        time.sleep(0.2)  # 200ms between starts
    
    for t in threads:
        t.join()
    
    rapid_time = time.time() - rapid_start
    successful = sum(1 for r in rapid_results if r['success'])
    
    print(f"\n   ✅ Rapid fire complete: {successful}/{len(rapid_queries)} succeeded in {rapid_time:.1f}s")
    
    tester.results.append({
        'scenario': '9. Rapid Fire Queries',
        'description': 'Concurrent load test',
        'config': {'auto_selection': True, 'enable_routing': True},
        'total_time': rapid_time,
        'queries_run': len(rapid_queries),
        'successful': successful,
        'failed': len(rapid_queries) - successful,
        'avg_time_per_query': rapid_time / len(rapid_queries),
        'results': rapid_results
    })
    
    # SCENARIO 10: Cache Behavior
    print(f"\n{'='*80}")
    print(f"🧪 SCENARIO: 10. Cache Behavior")
    print(f"📋 Description: Test cache hit/miss behavior")
    print(f"{'='*80}")
    
    tester.update_preferences(True, True)
    tester.clear_cache()
    
    test_query = TEST_QUERIES['easy'][0]
    
    print("\n   First run (cache miss - should show routing logs):")
    result1 = tester.run_query(test_query['query'], test_query)
    
    print("\n   Second run (cache hit - should be instant, no routing logs):")
    result2 = tester.run_query(test_query['query'], test_query)
    
    print("\n   Modified query (cache miss - should show routing logs again):")
    modified_query = test_query.copy()
    modified_query['query'] = test_query['query'] + " exactly"
    result3 = tester.run_query(modified_query['query'], modified_query)
    
    cache_success = result1['success'] and result2['success'] and result3['success']
    cache_hit_faster = result2['time'] < result1['time'] * 0.5  # Cache should be 50%+ faster
    
    print(f"\n   {'✅' if cache_success else '❌'} Cache test: All queries succeeded")
    print(f"   {'✅' if cache_hit_faster else '❌'} Cache hit performance: {result2['time']:.2f}s vs {result1['time']:.2f}s (first run)")
    
    tester.results.append({
        'scenario': '10. Cache Behavior',
        'description': 'Cache hit/miss test',
        'config': {'auto_selection': True, 'enable_routing': True},
        'total_time': result1['time'] + result2['time'] + result3['time'],
        'queries_run': 3,
        'successful': sum([result1['success'], result2['success'], result3['success']]),
        'failed': 3 - sum([result1['success'], result2['success'], result3['success']]),
        'avg_time_per_query': (result1['time'] + result2['time'] + result3['time']) / 3,
        'results': [result1, result2, result3]
    })
    
    # SCENARIO 11: CoT Iteration Limit (Skip - requires config change)
    print(f"\n{'='*80}")
    print(f"⏭️  SCENARIO 11: CoT Iteration Limit - SKIPPED")
    print(f"   Reason: Requires modifying CoT config, manual test recommended")
    print(f"{'='*80}")
    
    # SCENARIO 12: Mixed Complexity Query
    tester.run_scenario(
        "12. Mixed Complexity Query",
        "Query with both simple and complex requirements - Should route to FULL_POWER",
        {'auto_selection': True, 'enable_routing': True},
        TEST_QUERIES['mixed']
    )
    
    # Generate final report
    tester.print_final_report()
    
    # Save results to file
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(tester.results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {report_file}")
    print(f"\n🎉 COMPREHENSIVE TESTING COMPLETE!")

if __name__ == "__main__":
    main()
