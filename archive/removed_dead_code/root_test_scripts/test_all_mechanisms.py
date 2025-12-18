"""
COMPREHENSIVE TEST: Smart Model Selection + Intelligent Routing + CoT Self-Correction
Tests all 3 mechanisms working together from EASY to COMPLEX queries
"""

import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import logging
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

backend_path = Path(__file__).parent / "src" / "backend"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def print_result(query, result, test_num, total):
    print(f"\n{'─'*80}")
    print(f"TEST {test_num}/{total}: {query}")
    print(f"{'─'*80}")
    
    if isinstance(result, dict):
        print(f"✅ Success: {result.get('success', False)}")
        print(f"📊 Complexity Score: {result.get('complexity_score', 'N/A'):.3f}" if 'complexity_score' in result else "📊 Complexity: N/A")
        print(f"🤖 Model Used: {result.get('model_used', 'Unknown')}")
        print(f"⚡ Routing Tier: {result.get('routing_tier', 'N/A')}")
        print(f"🧠 CoT Review: {'✓ Yes' if result.get('cot_used') else '✗ No'}")
        
        if result.get('cot_used'):
            cot_info = result.get('cot_metadata', {})
            print(f"   ├─ Initial Score: {cot_info.get('initial_confidence', 0):.2f}")
            print(f"   ├─ Final Score: {cot_info.get('final_confidence', 0):.2f}")
            print(f"   └─ Improvement: {cot_info.get('improvement_percentage', 0):.1f}%")
        
        print(f"⏱️  Processing Time: {result.get('processing_time_seconds', 0):.2f}s")
        
        # Show answer preview
        answer = result.get('answer', '')
        if answer:
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"\n📝 Answer Preview:\n{preview}")
    else:
        print(f"❌ Error: {result}")

def test_scenario(description, routing_enabled, smart_selection_enabled):
    """Test a specific configuration scenario"""
    print_section(f"SCENARIO: {description}")
    print(f"Configuration:")
    print(f"   |-- Smart Model Selection: {'ON' if smart_selection_enabled else 'OFF'}")
    print(f"   |-- Intelligent Routing: {'ON' if routing_enabled else 'OFF'}")
    print(f"   |-- CoT Self-Correction: ON (Auto at >=0.5 complexity)")
    
    # Update preferences
    from backend.core.user_preferences import get_preferences_manager
    prefs_manager = get_preferences_manager()
    prefs = prefs_manager.load_preferences()
    prefs.enable_intelligent_routing = routing_enabled
    prefs.auto_model_selection = smart_selection_enabled
    prefs_manager.save_preferences(prefs)
    
    print(f"\n✅ Configuration updated successfully")
    
    # Test queries from EASY to COMPLEX
    test_queries = [
        {
            "query": "How many total rows are in the dataset?",
            "expected_complexity": "LOW (0.0-0.3)",
            "expected_cot": False,
            "description": "Simple count - should be FAST"
        },
        {
            "query": "What is the sum of sales revenue?",
            "expected_complexity": "LOW (0.0-0.3)",
            "expected_cot": False,
            "description": "Basic aggregation - should be FAST"
        },
        {
            "query": "Show me the top 5 products by sales",
            "expected_complexity": "LOW-MEDIUM (0.2-0.4)",
            "expected_cot": False,
            "description": "Simple sorting - should be FAST/BALANCED"
        },
        {
            "query": "Compare sales performance between regions and identify trends",
            "expected_complexity": "MEDIUM (0.4-0.6)",
            "expected_cot": True,
            "description": "Comparison + analysis - should trigger CoT"
        },
        {
            "query": "Analyze correlation between price, marketing spend, and sales. Predict next quarter revenue using regression",
            "expected_complexity": "HIGH (0.7-1.0)",
            "expected_cot": True,
            "description": "ML/Statistics - should use POWERFUL model + CoT"
        },
        {
            "query": "Perform multivariate analysis to identify key factors driving customer churn. Use statistical significance testing",
            "expected_complexity": "VERY HIGH (0.8-1.0)",
            "expected_cot": True,
            "description": "Complex ML - must use POWERFUL model + CoT validation"
        }
    ]
    
    # Initialize crew manager
    from backend.agents.crew_manager import CrewManager
    crew_manager = CrewManager()
    
    results = []
    
    for idx, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        print(f"\n\n🧪 Test {idx}/{len(test_queries)}: {test_case['description']}")
        print(f"📝 Query: {query}")
        print(f"🎯 Expected: {test_case['expected_complexity']}, CoT: {test_case['expected_cot']}")
        
        try:
            # Use sample CSV data
            sample_csv = backend_path / "data" / "samples" / "sample_sales.csv"
            if not sample_csv.exists():
                # Create a simple sample if doesn't exist
                import pandas as pd
                sample_data = pd.DataFrame({
                    'product': ['A', 'B', 'C', 'D', 'E'] * 20,
                    'sales': [100, 200, 150, 300, 250] * 20,
                    'region': ['North', 'South', 'East', 'West', 'Central'] * 20,
                    'price': [10, 20, 15, 30, 25] * 20,
                    'marketing': [500, 600, 550, 700, 650] * 20
                })
                sample_csv.parent.mkdir(parents=True, exist_ok=True)
                sample_data.to_csv(sample_csv, index=False)
                print(f"✅ Created sample data at {sample_csv}")
            
            result = crew_manager.analyze_structured_data(
                query=query,
                filename=str(sample_csv)
            )
            
            # Extract metadata
            result_info = {
                'success': result.get('success', False),
                'query': query,
                'complexity_score': result.get('complexity_score', 0),
                'model_used': result.get('model_used', 'Unknown'),
                'routing_tier': result.get('routing_tier', 'N/A'),
                'cot_used': result.get('cot_review_performed', False),
                'cot_metadata': result.get('cot_metadata', {}),
                'processing_time_seconds': result.get('processing_time', 0),
                'answer': result.get('answer', ''),
                'expected_cot': test_case['expected_cot']
            }
            
            results.append(result_info)
            print_result(query, result_info, idx, len(test_queries))
            
            # Validation
            if test_case['expected_cot'] and not result_info['cot_used']:
                print(f"\n⚠️  WARNING: Expected CoT but it didn't trigger!")
            elif not test_case['expected_cot'] and result_info['cot_used']:
                print(f"\n⚠️  WARNING: CoT triggered unexpectedly for simple query!")
            else:
                print(f"\n✅ Behavior matches expectations")
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            result_info = {
                'success': False,
                'query': query,
                'error': str(e)
            }
            results.append(result_info)
    
    return results

def main():
    print_section("COMPREHENSIVE MECHANISM TEST")
    print("Testing: Smart Model Selection + Intelligent Routing + CoT Self-Correction")
    print("From EASY queries (counts) to COMPLEX queries (ML predictions)")
    
    all_results = {}
    
    # SCENARIO 1: All mechanisms ON
    print("\n\n")
    results_1 = test_scenario(
        "ALL MECHANISMS ENABLED",
        routing_enabled=True,
        smart_selection_enabled=True
    )
    all_results['all_on'] = results_1
    
    # SCENARIO 2: Routing OFF, Smart Selection ON
    print("\n\n")
    results_2 = test_scenario(
        "ROUTING OFF + SMART SELECTION ON",
        routing_enabled=False,
        smart_selection_enabled=True
    )
    all_results['routing_off'] = results_2
    
    # SCENARIO 3: All OFF (Manual mode)
    print("\n\n")
    results_3 = test_scenario(
        "MANUAL MODE (All Auto Features OFF)",
        routing_enabled=False,
        smart_selection_enabled=False
    )
    all_results['all_off'] = results_3
    
    # FINAL SUMMARY
    print_section("FINAL TEST SUMMARY")
    
    for scenario_name, results in all_results.items():
        print(f"\n{'─'*80}")
        print(f"Scenario: {scenario_name.upper()}")
        print(f"{'─'*80}")
        
        successful = sum(1 for r in results if r.get('success', False))
        cot_triggered = sum(1 for r in results if r.get('cot_used', False))
        avg_complexity = sum(r.get('complexity_score', 0) for r in results) / len(results) if results else 0
        
        print(f"✅ Successful: {successful}/{len(results)}")
        print(f"🧠 CoT Triggered: {cot_triggered}/{len(results)}")
        print(f"📊 Avg Complexity: {avg_complexity:.3f}")
        
        # Check if CoT behavior is correct
        cot_errors = 0
        for r in results:
            if r.get('expected_cot') and not r.get('cot_used'):
                cot_errors += 1
            elif not r.get('expected_cot') and r.get('cot_used'):
                cot_errors += 1
        
        if cot_errors == 0:
            print(f"✅ CoT behavior: PERFECT (triggered correctly for complex queries)")
        else:
            print(f"⚠️  CoT behavior: {cot_errors} mismatches detected")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")
    
    # Save results
    output_file = Path(__file__).parent / "test_mechanisms_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"📁 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
