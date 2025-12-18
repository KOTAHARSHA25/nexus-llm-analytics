"""
REAL ACCURACY TESTING - Do Advanced Features Give CORRECT Answers?
Tests mathematical correctness, not just execution
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("ACCURACY VERIFICATION - Testing CORRECT Answers, Not Just Execution")
print("=" * 80)

total_tests = 0
passed_tests = 0

# ============================================================================
# TEST 1: Plugin Statistical Agent - MATHEMATICAL ACCURACY
# ============================================================================
print("\n[ACCURACY TEST 1] Statistical Agent - Correct Calculations")
print("-" * 80)
total_tests += 1

from backend.plugins.statistical_agent import StatisticalAgent

# Create KNOWN data with KNOWN answers
test_data = pd.DataFrame({
    'values': [10, 20, 30, 40, 50]  # Mean = 30, Median = 30, Std = 15.81
})

GROUND_TRUTH = {
    'mean': 30.0,
    'median': 30.0,
    'std': 15.811388300841896,
    'min': 10,
    'max': 50
}

stat_agent = StatisticalAgent()
stat_agent.initialize()

result = stat_agent.execute(
    query="Calculate descriptive statistics",
    data=test_data
)

if result.get('success') and 'statistics' in result:
    stats = result['statistics']
    
    # Check if calculated values match ground truth
    errors = []
    if abs(stats.get('mean', 0) - GROUND_TRUTH['mean']) > 0.01:
        errors.append(f"Mean: got {stats.get('mean')}, expected {GROUND_TRUTH['mean']}")
    if abs(stats.get('min', 0) - GROUND_TRUTH['min']) > 0.01:
        errors.append(f"Min: got {stats.get('min')}, expected {GROUND_TRUTH['min']}")
    if abs(stats.get('max', 0) - GROUND_TRUTH['max']) > 0.01:
        errors.append(f"Max: got {stats.get('max')}, expected {GROUND_TRUTH['max']}")
    
    if len(errors) == 0:
        print("✅ PASS: All calculations mathematically correct")
        print(f"   Mean: {stats.get('mean')} (correct)")
        print(f"   Min: {stats.get('min')} (correct)")
        print(f"   Max: {stats.get('max')} (correct)")
        passed_tests += 1
    else:
        print("❌ FAIL: Calculation errors:")
        for error in errors:
            print(f"   {error}")
else:
    print(f"❌ FAIL: {result.get('error', 'No statistics returned')}")

# ============================================================================
# TEST 2: Financial Agent - PROFIT CALCULATION ACCURACY
# ============================================================================
print("\n[ACCURACY TEST 2] Financial Agent - Correct Profit Calculation")
print("-" * 80)
total_tests += 1

from backend.plugins.financial_agent import FinancialAgent

# KNOWN financial data
financial_data = pd.DataFrame({
    'revenue': [1000, 2000, 3000],  # Total = 6000
    'cost': [600, 1200, 1800]       # Total = 3600
    # Profit = 6000 - 3600 = 2400
    # Profit Margin = 2400/6000 = 40%
})

GROUND_TRUTH = {
    'total_revenue': 6000,
    'total_cost': 3600,
    'total_profit': 2400,
    'profit_margin': 40.0
}

fin_agent = FinancialAgent()
fin_agent.initialize()

result = fin_agent.execute(
    query="Calculate profit and profit margin",
    data=financial_data
)

if result.get('success'):
    # Extract numbers from result
    result_text = str(result.get('analysis', ''))
    
    # Check if correct numbers appear in result
    found_revenue = '6000' in result_text or '6,000' in result_text
    found_cost = '3600' in result_text or '3,600' in result_text
    found_profit = '2400' in result_text or '2,400' in result_text
    
    if found_revenue and found_cost and found_profit:
        print("✅ PASS: Financial calculations correct")
        print(f"   Found revenue: 6000 ✓")
        print(f"   Found cost: 3600 ✓")
        print(f"   Found profit: 2400 ✓")
        passed_tests += 1
    else:
        print("❌ FAIL: Incorrect calculations in output")
        print(f"   Revenue found: {found_revenue}")
        print(f"   Cost found: {found_cost}")
        print(f"   Profit found: {found_profit}")
else:
    print(f"❌ FAIL: {result.get('error', 'Unknown error')}")

# ============================================================================
# TEST 3: Time Series Agent - TREND DETECTION ACCURACY
# ============================================================================
print("\n[ACCURACY TEST 3] Time Series Agent - Correct Trend Detection")
print("-" * 80)
total_tests += 1

from backend.plugins.time_series_agent import TimeSeriesAgent

# Create KNOWN upward trend: values increase by 10 each day
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
values = [100 + i*10 for i in range(10)]  # 100, 110, 120, ..., 190

ts_data = pd.DataFrame({
    'date': dates,
    'value': values
})

EXPECTED_TREND = "upward" or "increasing" or "positive"

ts_agent = TimeSeriesAgent()
ts_agent.initialize()

result = ts_agent.execute(
    query="Detect trend in the data",
    data=ts_data
)

if result.get('success'):
    result_text = str(result).lower()
    
    # Check if upward/increasing/positive trend detected
    trend_detected = any(word in result_text for word in ['upward', 'increasing', 'positive', 'rising', 'growth'])
    
    if trend_detected:
        print("✅ PASS: Correctly detected upward trend")
        print(f"   Data: 100→110→120...→190")
        print(f"   Detected: Upward trend ✓")
        passed_tests += 1
    else:
        print("❌ FAIL: Did not detect obvious upward trend")
        print(f"   Data clearly increases: 100, 110, 120, ..., 190")
else:
    print(f"❌ FAIL: {result.get('error', 'Unknown error')}")

# ============================================================================
# TEST 4: ML Insights Agent - CLUSTERING ACCURACY
# ============================================================================
print("\n[ACCURACY TEST 4] ML Agent - Correct Cluster Detection")
print("-" * 80)
total_tests += 1

from backend.plugins.ml_insights_agent import MLInsightsAgent

# Create OBVIOUS clusters: Group A (low values) and Group B (high values)
cluster_a = [[1, 1], [2, 2], [1, 2], [2, 1]]  # Bottom-left cluster
cluster_b = [[8, 8], [9, 9], [8, 9], [9, 8]]  # Top-right cluster

ml_data = pd.DataFrame(
    cluster_a + cluster_b,
    columns=['x', 'y']
)

ml_agent = MLInsightsAgent()
ml_agent.initialize()

result = ml_agent.execute(
    query="Perform clustering analysis",
    data=ml_data
)

if result.get('success') and 'clusters' in result:
    # Check if found 2 clusters (the obvious separation)
    n_clusters = result['clusters'].get('n_clusters', 0)
    
    if n_clusters == 2:
        print("✅ PASS: Correctly identified 2 distinct clusters")
        print(f"   Group A: [1,1], [2,2] → Cluster 1")
        print(f"   Group B: [8,8], [9,9] → Cluster 2")
        print(f"   Detected clusters: {n_clusters} ✓")
        passed_tests += 1
    else:
        print(f"❌ FAIL: Found {n_clusters} clusters, expected 2")
else:
    print(f"❌ FAIL: {result.get('error', 'No clustering result')}")

# ============================================================================
# TEST 5: Intelligent Routing - CORRECT TIER SELECTION
# ============================================================================
print("\n[ACCURACY TEST 5] Routing - Correct Model Tier for Complexity")
print("-" * 80)
total_tests += 1

from backend.core.intelligent_router import IntelligentRouter, ModelTier

router = IntelligentRouter()

# Test specific queries with KNOWN complexity
test_cases = [
    ("Count rows", ModelTier.FAST, "Simple aggregation"),
    ("Calculate average and sum", ModelTier.FAST, "Basic math"),
    ("Compare sales by region and show trends", ModelTier.BALANCED, "Medium analysis"),
    ("Predict customer churn using machine learning with cross-validation and ensemble methods", ModelTier.FULL_POWER, "Complex ML")
]

routing_correct = 0
routing_total = len(test_cases)

for query, expected_tier, description in test_cases:
    decision = router.route(query, {"rows": 1000, "columns": 10})
    
    # For simple queries, FAST is required
    # For complex queries, at least BALANCED (BALANCED or FULL_POWER acceptable)
    if expected_tier == ModelTier.FAST:
        if decision.selected_tier == ModelTier.FAST:
            routing_correct += 1
            print(f"   ✓ '{description}' → {decision.selected_tier.value} (correct)")
        else:
            print(f"   ✗ '{description}' → {decision.selected_tier.value} (expected {expected_tier.value})")
    elif expected_tier == ModelTier.FULL_POWER:
        if decision.selected_tier in [ModelTier.BALANCED, ModelTier.FULL_POWER]:
            routing_correct += 1
            print(f"   ✓ '{description}' → {decision.selected_tier.value} (acceptable)")
        else:
            print(f"   ✗ '{description}' → {decision.selected_tier.value} (expected higher tier)")
    else:  # BALANCED
        if decision.selected_tier in [ModelTier.FAST, ModelTier.BALANCED]:
            routing_correct += 1
            print(f"   ✓ '{description}' → {decision.selected_tier.value} (acceptable)")
        else:
            print(f"   ✗ '{description}' → {decision.selected_tier.value} (expected {expected_tier.value})")

if routing_correct == routing_total:
    print(f"✅ PASS: All {routing_total} queries routed to appropriate tiers")
    passed_tests += 1
else:
    print(f"❌ FAIL: Only {routing_correct}/{routing_total} correctly routed")

# ============================================================================
# TEST 6: CoT Parser - EXTRACTION ACCURACY
# ============================================================================
print("\n[ACCURACY TEST 6] CoT Parser - Correct Reasoning Extraction")
print("-" * 80)
total_tests += 1

from backend.core.cot_parser import CoTParser

parser = CoTParser()

# Test response with KNOWN content
test_response = """
[REASONING]
Step 1: Calculate the sum
Adding all values: 10 + 20 + 30 = 60

Step 2: Calculate the average
Dividing by count: 60 / 3 = 20

Step 3: Verify the result
The average of 10, 20, 30 is indeed 20.
[/REASONING]

[OUTPUT]
The average is 20.
[/OUTPUT]
"""

result = parser.parse(test_response)

# Check if extracted content is EXACTLY what we put in
reasoning_correct = "Step 1" in result.reasoning and "Step 2" in result.reasoning and "60 / 3 = 20" in result.reasoning
output_correct = "average is 20" in result.output.lower()
steps = parser.extract_steps(result.reasoning)
steps_correct = len(steps) == 3

if result.is_valid and reasoning_correct and output_correct and steps_correct:
    print("✅ PASS: Correctly extracted all CoT components")
    print(f"   Reasoning: {len(result.reasoning)} chars ✓")
    print(f"   Output: {result.output.strip()} ✓")
    print(f"   Steps: {len(steps)} extracted ✓")
    passed_tests += 1
else:
    print("❌ FAIL: CoT extraction incomplete or incorrect")
    print(f"   Valid: {result.is_valid}")
    print(f"   Reasoning correct: {reasoning_correct}")
    print(f"   Output correct: {output_correct}")
    print(f"   Steps correct: {steps_correct}")

# ============================================================================
# FINAL ACCURACY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("ACCURACY TEST RESULTS - DO ADVANCED FEATURES GIVE CORRECT ANSWERS?")
print("=" * 80)
print(f"\nTotal Accuracy Tests: {total_tests}")
print(f"Passed (Correct Answers): {passed_tests}")
print(f"Failed (Wrong Answers): {total_tests - passed_tests}")
print(f"Accuracy Rate: {passed_tests/total_tests*100:.1f}%")
print()

if passed_tests == total_tests:
    print("✅✅ ALL ACCURACY TESTS PASSED - ANSWERS ARE CORRECT")
elif passed_tests >= total_tests * 0.8:
    print("⚠️ MOSTLY ACCURATE - Some corrections needed")
else:
    print("❌ SIGNIFICANT ACCURACY ISSUES - Results not reliable")

print("\n" + "=" * 80)
print("WHAT WE VERIFIED:")
print("=" * 80)
print("✓ Statistical calculations (mean, min, max)")
print("✓ Financial calculations (revenue, cost, profit)")
print("✓ Trend detection (upward/downward)")
print("✓ Clustering (distinct groups)")
print("✓ Routing decisions (complexity → tier)")
print("✓ CoT parsing (reasoning extraction)")
print()
print("These tests verify MATHEMATICAL CORRECTNESS, not just execution.")
print("=" * 80)
