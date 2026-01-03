"""
COMPREHENSIVE FIX 5 COMPARISON: Template vs Direct Approach
Tests with edge cases, ambiguous queries, complex scenarios
Multiple runs for statistical analysis
Focus: OUTPUT ACCURACY and RELIABILITY
"""
import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
import numpy as np
from backend.core.code_generator import CodeGenerator
from backend.core.llm_client import LLMClient

print("="*80)
print("🔬 COMPREHENSIVE FIX 5 COMPARISON TEST")
print("="*80)
print("Testing: Template-Based (Fix 5) vs Direct LLM Approach")
print("Focus: Accuracy, Edge Cases, Statistical Analysis")
print("="*80)

# Load real sample data
df = pd.read_csv('data/samples/sales_data.csv')
print(f"\n📊 Dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")

# Calculate ground truth for validation
ground_truth = {
    "max_sales": df['sales'].max(),
    "min_sales": df['sales'].min(),
    "avg_revenue": df['revenue'].mean(),
    "total_revenue": df['revenue'].sum(),
    "median_price": df['price'].median(),
    "product_with_max_sales": df.loc[df['sales'].idxmax(), 'product'],
    "top_3_products": df.nlargest(3, 'sales')['product'].tolist(),
    "products_above_5000_sales": len(df[df['sales'] > 5000]),
    "revenue_std": df['revenue'].std(),
    "north_region_count": len(df[df['region'] == 'North']),
}

print(f"\n✓ Ground truth calculated for {len(ground_truth)} metrics")

# Test scenarios - including edge cases NOT in prompts
test_scenarios = [
    # Basic queries (in prompt)
    {
        "query": "What is the maximum sales?",
        "expected": ground_truth["max_sales"],
        "validator": lambda x: abs(float(x) - ground_truth["max_sales"]) < 0.01,
        "category": "basic"
    },
    {
        "query": "What is the total revenue?",
        "expected": ground_truth["total_revenue"],
        "validator": lambda x: abs(float(x) - ground_truth["total_revenue"]) < 0.01,
        "category": "basic"
    },
    
    # Ambiguous queries (NOT in prompt)
    {
        "query": "Which one is highest?",  # Ambiguous - highest what?
        "expected": ground_truth["max_sales"],  # Should infer sales
        "validator": lambda x: ground_truth["product_with_max_sales"] in str(x) or abs(float(str(x).split()[-1]) - ground_truth["max_sales"]) < 0.01 if any(c.isdigit() for c in str(x)) else False,
        "category": "ambiguous"
    },
    {
        "query": "Show me the best products",  # Unclear metric
        "expected": ground_truth["top_3_products"],
        "validator": lambda x: any(p in str(x) for p in ground_truth["top_3_products"][:2]),
        "category": "ambiguous"
    },
    
    # Statistical queries (NOT in basic prompt)
    {
        "query": "What is the median price?",
        "expected": ground_truth["median_price"],
        "validator": lambda x: abs(float(x) - ground_truth["median_price"]) < 0.01,
        "category": "statistical"
    },
    {
        "query": "Calculate the standard deviation of revenue",
        "expected": ground_truth["revenue_std"],
        "validator": lambda x: abs(float(x) - ground_truth["revenue_std"]) < 1.0,
        "category": "statistical"
    },
    
    # Filtering queries (edge cases)
    {
        "query": "How many products have sales greater than 5000?",
        "expected": ground_truth["products_above_5000_sales"],
        "validator": lambda x: int(x) == ground_truth["products_above_5000_sales"],
        "category": "filtering"
    },
    {
        "query": "Count rows in North region",
        "expected": ground_truth["north_region_count"],
        "validator": lambda x: int(x) == ground_truth["north_region_count"],
        "category": "filtering"
    },
    
    # Comparison queries
    {
        "query": "What is the difference between max and min sales?",
        "expected": ground_truth["max_sales"] - ground_truth["min_sales"],
        "validator": lambda x: abs(float(x) - (ground_truth["max_sales"] - ground_truth["min_sales"])) < 0.01,
        "category": "comparison"
    },
    
    # Negation queries (tricky)
    {
        "query": "What is the minimum sales value?",
        "expected": ground_truth["min_sales"],
        "validator": lambda x: abs(float(x) - ground_truth["min_sales"]) < 0.01,
        "category": "negation"
    },
]

print(f"✓ Prepared {len(test_scenarios)} test scenarios")
print(f"  - Basic: {sum(1 for t in test_scenarios if t['category'] == 'basic')}")
print(f"  - Ambiguous: {sum(1 for t in test_scenarios if t['category'] == 'ambiguous')}")
print(f"  - Statistical: {sum(1 for t in test_scenarios if t['category'] == 'statistical')}")
print(f"  - Filtering: {sum(1 for t in test_scenarios if t['category'] == 'filtering')}")
print(f"  - Comparison: {sum(1 for t in test_scenarios if t['category'] == 'comparison')}")
print(f"  - Negation: {sum(1 for t in test_scenarios if t['category'] == 'negation')}")

# Initialize generator
llm_client = LLMClient()
generator = CodeGenerator(llm_client)

# Test configuration
TEST_MODEL = "phi3:mini"  # Small model where Fix 5 matters most
NUM_RUNS = 3  # Run each test 3 times for statistical reliability

print(f"\n🤖 Testing with: {TEST_MODEL}")
print(f"🔄 Runs per test: {NUM_RUNS}")

def extract_result_value(result_obj):
    """Extract actual value from execution result"""
    if result_obj is None:
        return None
    if isinstance(result_obj, dict) and 'result' in result_obj:
        result_obj = result_obj['result']
    if hasattr(result_obj, 'item'):  # numpy scalar
        return result_obj.item()
    if isinstance(result_obj, pd.DataFrame):
        return result_obj
    return result_obj

def test_template_approach(query, df, model, run_num):
    """Test with Fix 5 template-based approach (current implementation)"""
    start = time.time()
    try:
        generated = generator.generate_code(query, df, model)
        if not generated.is_valid:
            return {
                "success": False,
                "error": generated.error_message,
                "time": time.time() - start,
                "timeout": False
            }
        
        result = generator.execute_code(generated.code, df)
        elapsed = time.time() - start
        
        if result.success and result.result is not None:
            actual = extract_result_value(result.result)
            return {
                "success": True,
                "result": actual,
                "time": elapsed,
                "timeout": False,
                "code_length": len(generated.code)
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "time": elapsed,
                "timeout": False
            }
    except Exception as e:
        elapsed = time.time() - start
        is_timeout = elapsed > 300  # Consider 5+ minutes as timeout
        return {
            "success": False,
            "error": str(e),
            "time": elapsed,
            "timeout": is_timeout
        }

def test_direct_approach(query, df, model, run_num):
    """Test with direct LLM approach (minimal prompt)"""
    start = time.time()
    try:
        # Build minimal direct prompt
        columns_str = ", ".join(df.columns.tolist())
        direct_prompt = f"""You have a pandas DataFrame 'df' with columns: {columns_str}

Query: {query}

Generate Python code to answer this. Store result in variable 'result'.

Code:"""
        
        # Call LLM
        response = llm_client.generate(direct_prompt, model=model)
        
        if isinstance(response, dict):
            response_text = response.get('response', '')
            if response.get('error'):
                return {
                    "success": False,
                    "error": response.get('error'),
                    "time": time.time() - start,
                    "timeout": False
                }
        else:
            response_text = str(response)
        
        # Extract code
        code = generator._extract_code(response_text)
        if not code:
            return {
                "success": False,
                "error": "No code generated",
                "time": time.time() - start,
                "timeout": False
            }
        
        # Execute
        result = generator.execute_code(code, df)
        elapsed = time.time() - start
        
        if result.success and result.result is not None:
            actual = extract_result_value(result.result)
            return {
                "success": True,
                "result": actual,
                "time": elapsed,
                "timeout": False,
                "code_length": len(code)
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "time": elapsed,
                "timeout": False
            }
    except Exception as e:
        elapsed = time.time() - start
        is_timeout = elapsed > 300
        return {
            "success": False,
            "error": str(e),
            "time": elapsed,
            "timeout": is_timeout
        }

# Run comprehensive comparison
print("\n" + "="*80)
print("🔬 RUNNING COMPREHENSIVE TESTS")
print("="*80)

results = {
    "template": [],
    "direct": []
}

for scenario_idx, scenario in enumerate(test_scenarios, 1):
    print(f"\n{'='*80}")
    print(f"📝 Scenario {scenario_idx}/{len(test_scenarios)}: {scenario['category'].upper()}")
    print(f"Query: \"{scenario['query']}\"")
    print(f"Expected: {scenario['expected']}")
    print(f"{'='*80}")
    
    template_runs = []
    direct_runs = []
    
    for run in range(1, NUM_RUNS + 1):
        print(f"\n  Run {run}/{NUM_RUNS}:")
        
        # Test template approach
        print(f"    🔵 Template...", end=" ", flush=True)
        template_result = test_template_approach(scenario["query"], df, TEST_MODEL, run)
        
        if template_result["success"]:
            try:
                is_correct = scenario["validator"](template_result["result"])
                template_result["correct"] = is_correct
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                print(f"{status} ({template_result['time']:.1f}s)")
            except Exception as e:
                template_result["correct"] = False
                print(f"⚠️ VALIDATION ERROR ({template_result['time']:.1f}s)")
        else:
            template_result["correct"] = False
            if template_result["timeout"]:
                print(f"⏱️ TIMEOUT ({template_result['time']:.1f}s)")
            else:
                print(f"❌ FAILED ({template_result['time']:.1f}s)")
        
        template_runs.append(template_result)
        
        # Test direct approach
        print(f"    🟠 Direct...", end=" ", flush=True)
        direct_result = test_direct_approach(scenario["query"], df, TEST_MODEL, run)
        
        if direct_result["success"]:
            try:
                is_correct = scenario["validator"](direct_result["result"])
                direct_result["correct"] = is_correct
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                print(f"{status} ({direct_result['time']:.1f}s)")
            except Exception as e:
                direct_result["correct"] = False
                print(f"⚠️ VALIDATION ERROR ({direct_result['time']:.1f}s)")
        else:
            direct_result["correct"] = False
            if direct_result["timeout"]:
                print(f"⏱️ TIMEOUT ({direct_result['time']:.1f}s)")
            else:
                print(f"❌ FAILED ({direct_result['time']:.1f}s)")
        
        direct_runs.append(direct_result)
    
    # Aggregate results for this scenario
    template_correct = sum(1 for r in template_runs if r.get("correct", False))
    direct_correct = sum(1 for r in direct_runs if r.get("correct", False))
    
    print(f"\n  📊 Scenario Summary:")
    print(f"    Template: {template_correct}/{NUM_RUNS} correct ({template_correct/NUM_RUNS*100:.0f}%)")
    print(f"    Direct:   {direct_correct}/{NUM_RUNS} correct ({direct_correct/NUM_RUNS*100:.0f}%)")
    
    results["template"].append({
        "scenario": scenario_idx,
        "category": scenario["category"],
        "query": scenario["query"],
        "runs": template_runs,
        "correct_rate": template_correct / NUM_RUNS
    })
    
    results["direct"].append({
        "scenario": scenario_idx,
        "category": scenario["category"],
        "query": scenario["query"],
        "runs": direct_runs,
        "correct_rate": direct_correct / NUM_RUNS
    })

# Calculate overall statistics
print("\n" + "="*80)
print("📊 OVERALL STATISTICS")
print("="*80)

def calculate_stats(approach_results):
    total_tests = sum(len(s["runs"]) for s in approach_results)
    correct_count = sum(sum(1 for r in s["runs"] if r.get("correct", False)) for s in approach_results)
    timeout_count = sum(sum(1 for r in s["runs"] if r.get("timeout", False)) for s in approach_results)
    avg_time = np.mean([r["time"] for s in approach_results for r in s["runs"] if r["success"]])
    
    return {
        "total_tests": total_tests,
        "correct": correct_count,
        "accuracy": correct_count / total_tests * 100,
        "timeouts": timeout_count,
        "timeout_rate": timeout_count / total_tests * 100,
        "avg_time": avg_time
    }

template_stats = calculate_stats(results["template"])
direct_stats = calculate_stats(results["direct"])

print(f"\n🔵 TEMPLATE-BASED (Fix 5):")
print(f"  Accuracy:      {template_stats['correct']}/{template_stats['total_tests']} ({template_stats['accuracy']:.1f}%)")
print(f"  Timeouts:      {template_stats['timeouts']}/{template_stats['total_tests']} ({template_stats['timeout_rate']:.1f}%)")
print(f"  Avg Time:      {template_stats['avg_time']:.1f}s")

print(f"\n🟠 DIRECT LLM:")
print(f"  Accuracy:      {direct_stats['correct']}/{direct_stats['total_tests']} ({direct_stats['accuracy']:.1f}%)")
print(f"  Timeouts:      {direct_stats['timeouts']}/{direct_stats['total_tests']} ({direct_stats['timeout_rate']:.1f}%)")
print(f"  Avg Time:      {direct_stats['avg_time']:.1f}s")

# Category-wise breakdown
print(f"\n📈 ACCURACY BY CATEGORY:")
print(f"\n{'Category':<15} {'Template':<15} {'Direct':<15} {'Winner':<10}")
print("-" * 60)

categories = list(set(s["category"] for s in results["template"]))
for cat in categories:
    template_cat = [s for s in results["template"] if s["category"] == cat]
    direct_cat = [s for s in results["direct"] if s["category"] == cat]
    
    template_acc = np.mean([s["correct_rate"] for s in template_cat]) * 100
    direct_acc = np.mean([s["correct_rate"] for s in direct_cat]) * 100
    
    winner = "Template" if template_acc > direct_acc else "Direct" if direct_acc > template_acc else "Tie"
    
    print(f"{cat:<15} {template_acc:>6.1f}%        {direct_acc:>6.1f}%        {winner:<10}")

# Final recommendation
print("\n" + "="*80)
print("🎯 FINAL RECOMMENDATION")
print("="*80)

accuracy_diff = template_stats['accuracy'] - direct_stats['accuracy']
timeout_diff = direct_stats['timeout_rate'] - template_stats['timeout_rate']

if template_stats['accuracy'] >= 80 and direct_stats['accuracy'] < 70:
    verdict = "KEEP FIX 5 (Template-Based)"
    reason = f"Template approach is {accuracy_diff:.1f}% more accurate and {timeout_diff:.1f}% fewer timeouts"
elif direct_stats['accuracy'] >= 80 and template_stats['accuracy'] < 70:
    verdict = "SWITCH TO DIRECT"
    reason = f"Direct approach is {-accuracy_diff:.1f}% more accurate"
elif abs(accuracy_diff) < 5:
    verdict = "BOTH WORK WELL"
    reason = "Similar accuracy, choose based on other factors"
else:
    verdict = "KEEP FIX 5 (Template-Based)"
    reason = f"Template is more reliable ({accuracy_diff:+.1f}% accuracy, {timeout_diff:+.1f}% fewer timeouts)"

print(f"\n✅ VERDICT: {verdict}")
print(f"   Reason: {reason}")

print(f"\n💡 KEY INSIGHTS:")
if template_stats['accuracy'] > direct_stats['accuracy']:
    print(f"   • Template approach is more accurate ({template_stats['accuracy']:.1f}% vs {direct_stats['accuracy']:.1f}%)")
if template_stats['timeout_rate'] < direct_stats['timeout_rate']:
    print(f"   • Template has fewer timeouts ({template_stats['timeout_rate']:.1f}% vs {direct_stats['timeout_rate']:.1f}%)")
if template_stats['avg_time'] < direct_stats['avg_time']:
    print(f"   • Template is faster on average ({template_stats['avg_time']:.1f}s vs {direct_stats['avg_time']:.1f}s)")

print("\n" + "="*80)

# Save detailed results to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"fix5_comparison_results_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump({
        "test_config": {
            "model": TEST_MODEL,
            "num_runs": NUM_RUNS,
            "scenarios": len(test_scenarios),
            "timestamp": timestamp
        },
        "statistics": {
            "template": template_stats,
            "direct": direct_stats
        },
        "verdict": verdict,
        "reason": reason
    }, f, indent=2, default=str)

print(f"📄 Detailed results saved to: {results_file}")
print("="*80)
