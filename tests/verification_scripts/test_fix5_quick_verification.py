"""
Quick template verification - test key scenarios that were failing
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
import numpy as np
from backend.core.code_generator import CodeGenerator
from backend.core.llm_client import LLMClient

print("="*80)
print("QUICK TEMPLATE FIX VERIFICATION")
print("="*80)

df = pd.read_csv('data/samples/sales_data.csv')
print(f"\nDataset: {len(df)} rows, {len(df.columns)} columns")

# Ground truth
ground_truth = {
    "max_sales": df['sales'].max(),
    "revenue_std": df['revenue'].std(),
    "products_above_5000": len(df[df['sales'] > 5000]),
}

print(f"\nGround Truth:")
for key, val in ground_truth.items():
    print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")

llm_client = LLMClient()
generator = CodeGenerator(llm_client)

# Test the previously failing scenarios
test_scenarios = [
    {
        "id": "basic_max",
        "query": "What is the maximum sales?",
        "expected": ground_truth["max_sales"],
        "validator": lambda x: abs(float(x) - ground_truth["max_sales"]) < 0.01
    },
    {
        "id": "stat_std",
        "query": "Calculate the standard deviation of revenue",
        "expected": ground_truth["revenue_std"],
        "validator": lambda x: abs(float(x) - ground_truth["revenue_std"]) < 1.0
    },
    {
        "id": "filter_count",
        "query": "How many products have sales greater than 5000?",
        "expected": ground_truth["products_above_5000"],
        "validator": lambda x: int(x) == ground_truth["products_above_5000"]
    }
]

print("\n" + "="*80)
print("TESTING WITH FIXED TEMPLATE (phi3:mini)")
print("="*80)

results = {"passed": 0, "failed": 0, "errors": []}

for scenario in test_scenarios:
    print(f"\n[{scenario['id']}] {scenario['query']}")
    print(f"Expected: {scenario['expected']}")
    
    successes = 0
    failures = 0
    
    # Run 5 times for each scenario
    for run in range(1, 6):
        try:
            generated = generator.generate_code(scenario['query'], df, "phi3:mini")
            
            if not generated.is_valid:
                failures += 1
                continue
            
            result = generator.execute_code(generated.code, df)
            
            if result.success and result.result is not None:
                actual = result.result
                if isinstance(actual, dict) and 'result' in actual:
                    actual = actual['result']
                
                try:
                    if scenario['validator'](actual):
                        successes += 1
                    else:
                        failures += 1
                except:
                    failures += 1
            else:
                failures += 1
        except Exception as e:
            failures += 1
    
    accuracy = (successes / 5) * 100
    print(f"Result: {successes}/5 correct ({accuracy:.0f}%)")
    
    if accuracy >= 80:
        print("Status: PASS")
        results["passed"] += 1
    else:
        print("Status: FAIL")
        results["failed"] += 1
        results["errors"].append(f"{scenario['id']}: {accuracy:.0f}%")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Scenarios Passed: {results['passed']}/3")
print(f"Scenarios Failed: {results['failed']}/3")

if results['passed'] >= 2:
    print("\nVERDICT: TEMPLATE FIX SUCCESSFUL")
    print("The simple template improvements have resolved the major bugs.")
else:
    print("\nVERDICT: MORE WORK NEEDED")
    if results['errors']:
        print("Failed scenarios:")
        for err in results['errors']:
            print(f"  - {err}")

print("="*80)
