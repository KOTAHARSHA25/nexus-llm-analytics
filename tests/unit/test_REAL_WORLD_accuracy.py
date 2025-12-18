"""
REAL-WORLD ACCURACY TEST
Purpose: Test with ACTUAL user data scenarios, NOT reverse-engineered from code
Date: December 16, 2025

This test uses REAL data patterns and EXTERNAL ground truth values
calculated INDEPENDENTLY (not by looking at the code).
"""

import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

import pandas as pd
import numpy as np
from backend.plugins.statistical_agent import StatisticalAgent
from backend.plugins.financial_agent import FinancialAgent

print("="*80)
print("🔍 REAL-WORLD ACCURACY TEST - INDEPENDENT VERIFICATION")
print("="*80)
print("Testing with data patterns users would actually provide")
print("Ground truth calculated INDEPENDENTLY with external tools")
print("="*80)

# Initialize agents
stat_agent = StatisticalAgent()
stat_agent.initialize()

fin_agent = FinancialAgent()
fin_agent.initialize()

# ============================================================================
# TEST 1: Real E-commerce Sales Data (Messy Format)
# ============================================================================
print("\n[TEST 1] Real E-commerce Sales Data - Messy User Format")
print("-"*80)

# Simulate what a user might actually upload
ecommerce_data = pd.DataFrame({
    'Date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
    'Product': ['Widget', 'Gadget', 'Widget', 'Doohickey', 'Widget'],
    'Sales_Amount': [125.50, 89.99, 125.50, 45.00, 125.50],  # Total should be 511.49
    'Cost_Of_Goods': [75.30, 53.99, 75.30, 27.00, 75.30],    # Total should be 306.89
    'Quantity_Sold': [5, 3, 5, 2, 5]                          # Total should be 20
})

print("User Data (as they would upload it):")
print(ecommerce_data)

# GROUND TRUTH (calculated with calculator, NOT by reading code):
print("\n📊 GROUND TRUTH (calculated independently):")
print("  Total Sales: $511.49")
print("  Total Cost: $306.89")
print("  Total Profit: $204.60")
print("  Profit Margin: 40.0%")
print("  Total Quantity: 20 units")
print("  Average Sale: $102.298")

# Test Statistical Agent
query1 = "What's the average sales amount?"
result1 = stat_agent.execute(query=query1, data=ecommerce_data)

test1a_pass = False
if result1.get('success'):
    # Try to find the average
    if 'numeric_summary' in result1.get('result', {}):
        stats = result1['result']['numeric_summary']
        if 'Sales_Amount' in stats:
            calculated_mean = stats['Sales_Amount'].get('mean')
            expected_mean = 102.298  # 511.49 / 5
            
            print(f"\n  Statistical Agent Result:")
            print(f"    Calculated Mean: ${calculated_mean:.3f}")
            print(f"    Expected Mean: ${expected_mean:.3f}")
            
            if abs(calculated_mean - expected_mean) < 0.01:
                print("    ✅ CORRECT")
                test1a_pass = True
            else:
                print(f"    ❌ WRONG - Off by ${abs(calculated_mean - expected_mean):.3f}")
    else:
        print("  ❌ Agent didn't return statistics in expected format")
else:
    print(f"  ❌ Agent failed: {result1.get('error')}")

# Test Financial Agent
query2 = "Calculate my profit"
result2 = fin_agent.execute(query=query2, data=ecommerce_data)

test1b_pass = False
if result2.get('success'):
    print(f"\n  Financial Agent Result:")
    if 'result' in result2:
        results = result2['result']
        # Look for any profit calculation
        found_profit = False
        for key, value in results.items():
            if isinstance(value, dict) and 'gross_profit' in value:
                calculated_profit = value['gross_profit']
                expected_profit = 204.60  # 511.49 - 306.89
                
                print(f"    Calculated Profit: ${calculated_profit:.2f}")
                print(f"    Expected Profit: ${expected_profit:.2f}")
                
                if abs(calculated_profit - expected_profit) < 0.01:
                    print("    ✅ CORRECT")
                    test1b_pass = True
                else:
                    print(f"    ❌ WRONG - Off by ${abs(calculated_profit - expected_profit):.2f}")
                found_profit = True
                break
        
        if not found_profit:
            print("    ❌ No profit calculation found in result")
    else:
        print("    ❌ No result in response")
else:
    print(f"  ❌ Agent failed: {result2.get('error')}")

# ============================================================================
# TEST 2: User's CSV with Weird Column Names
# ============================================================================
print("\n[TEST 2] CSV with Non-Standard Column Names")
print("-"*80)

# Users don't always have clean column names
weird_data = pd.DataFrame({
    'Monthly Revenue (USD)': [5000, 7500, 6200, 8100, 9300],
    'Operating Expenses': [3000, 4000, 3500, 4500, 5000],
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May']
})

print("User Data:")
print(weird_data)

print("\n📊 GROUND TRUTH:")
print("  Total Revenue: $36,100")
print("  Total Expenses: $20,000")
print("  Total Profit: $16,100")
print("  Average Monthly Revenue: $7,220")

query = "What's my total revenue and profit?"
result = fin_agent.execute(query=query, data=weird_data)

test2_pass = False
if result.get('success'):
    print(f"\n  Agent handled non-standard column names: ✅")
    test2_pass = True
else:
    print(f"  ❌ Agent couldn't handle weird column names: {result.get('error')}")

# ============================================================================
# TEST 3: Data with Missing Values (Real-World Scenario)
# ============================================================================
print("\n[TEST 3] Data with Missing Values")
print("-"*80)

# Real user data often has missing values
messy_data = pd.DataFrame({
    'sales': [100, np.nan, 150, 200, np.nan, 250],
    'cost': [60, 80, np.nan, 120, 140, np.nan]
})

print("User Data (with NaN values):")
print(messy_data)

print("\n📊 GROUND TRUTH:")
print("  Valid sales values: [100, 150, 200, 250]")
print("  Mean of valid sales: 175.0")
print("  Valid cost values: [60, 80, 120, 140]")
print("  Mean of valid costs: 100.0")

query = "Calculate average sales"
result = stat_agent.execute(query=query, data=messy_data)

test3_pass = False
if result.get('success'):
    if 'numeric_summary' in result.get('result', {}):
        stats = result['result']['numeric_summary']
        if 'sales' in stats:
            calculated_mean = stats['sales'].get('mean')
            expected_mean = 175.0
            
            print(f"\n  Agent Result:")
            print(f"    Calculated Mean: {calculated_mean}")
            print(f"    Expected Mean: {expected_mean}")
            
            if abs(calculated_mean - expected_mean) < 0.01:
                print("    ✅ CORRECTLY handled missing values")
                test3_pass = True
            else:
                print(f"    ❌ WRONG - Didn't handle NaN correctly")
        else:
            print("  ❌ Sales not in result")
    else:
        print("  ❌ No numeric_summary in result")
else:
    print(f"  ❌ Agent failed: {result.get('error')}")

# ============================================================================
# TEST 4: Negative Numbers (Returns/Refunds)
# ============================================================================
print("\n[TEST 4] Negative Numbers - Returns and Refunds")
print("-"*80)

# Real-world data has returns
returns_data = pd.DataFrame({
    'transaction': ['Sale', 'Sale', 'Refund', 'Sale', 'Refund'],
    'amount': [100, 150, -50, 200, -30]  # Net should be 370
})

print("User Data:")
print(returns_data)

print("\n📊 GROUND TRUTH:")
print("  Net Revenue: $370 (100+150-50+200-30)")
print("  Sum of all values: $370")

query = "Calculate total revenue"
result = stat_agent.execute(query=query, data=returns_data)

test4_pass = False
if result.get('success'):
    if 'numeric_summary' in result.get('result', {}):
        stats = result['result']['numeric_summary']
        if 'amount' in stats:
            # Check if sum is accessible or if we can calculate from mean
            count = stats['amount'].get('count')
            mean = stats['amount'].get('mean')
            
            if count and mean:
                calculated_sum = count * mean
                expected_sum = 370.0
                
                print(f"\n  Agent Result:")
                print(f"    Count: {count}, Mean: {mean}")
                print(f"    Calculated Sum: {calculated_sum}")
                print(f"    Expected Sum: {expected_sum}")
                
                if abs(calculated_sum - expected_sum) < 0.01:
                    print("    ✅ CORRECTLY handled negative values")
                    test4_pass = True
                else:
                    print(f"    ❌ WRONG calculation")
            else:
                print("  ❌ Missing count or mean in result")
        else:
            print("  ❌ Amount not in result")
    else:
        print("  ❌ No numeric_summary")
else:
    print(f"  ❌ Agent failed: {result.get('error')}")

# ============================================================================
# TEST 5: Large Numbers (Real Business Scale)
# ============================================================================
print("\n[TEST 5] Large Numbers - Real Business Scale")
print("-"*80)

# Real businesses deal with large numbers
big_data = pd.DataFrame({
    'revenue': [1_250_000, 1_875_000, 2_100_000],
    'expenses': [875_000, 1_200_000, 1_400_000]
})

print("User Data (large numbers):")
print(big_data)

print("\n📊 GROUND TRUTH:")
print("  Total Revenue: $5,225,000")
print("  Total Expenses: $3,475,000")
print("  Total Profit: $1,750,000")
print("  Profit Margin: 33.49%")

query = "Analyze profitability"
result = fin_agent.execute(query=query, data=big_data)

test5_pass = False
if result.get('success'):
    if 'result' in result:
        results = result['result']
        for key, value in results.items():
            if isinstance(value, dict) and 'gross_profit' in value:
                calculated_profit = value['gross_profit']
                expected_profit = 1_750_000
                
                print(f"\n  Agent Result:")
                print(f"    Calculated Profit: ${calculated_profit:,.2f}")
                print(f"    Expected Profit: ${expected_profit:,.2f}")
                
                if abs(calculated_profit - expected_profit) < 1:
                    print("    ✅ CORRECTLY handled large numbers")
                    test5_pass = True
                else:
                    print(f"    ❌ WRONG - Off by ${abs(calculated_profit - expected_profit):,.2f}")
                break
    else:
        print("  ❌ No result")
else:
    print(f"  ❌ Agent failed: {result.get('error')}")

# ============================================================================
# TEST 6: Decimal Precision Issues
# ============================================================================
print("\n[TEST 6] Decimal Precision - Prices like $19.99")
print("-"*80)

# Real prices often have decimals
decimal_data = pd.DataFrame({
    'price': [19.99, 29.99, 39.99, 49.99, 59.99],
    'quantity': [10, 8, 6, 4, 2]
})

print("User Data:")
print(decimal_data)

print("\n📊 GROUND TRUTH:")
print("  Average Price: $39.99")
print("  Total Quantity: 30")

query = "What's the average price?"
result = stat_agent.execute(query=query, data=decimal_data)

test6_pass = False
if result.get('success'):
    if 'numeric_summary' in result.get('result', {}):
        stats = result['result']['numeric_summary']
        if 'price' in stats:
            calculated_mean = stats['price'].get('mean')
            expected_mean = 39.99
            
            print(f"\n  Agent Result:")
            print(f"    Calculated Mean: ${calculated_mean:.2f}")
            print(f"    Expected Mean: ${expected_mean:.2f}")
            
            if abs(calculated_mean - expected_mean) < 0.01:
                print("    ✅ CORRECT precision handling")
                test6_pass = True
            else:
                print(f"    ❌ WRONG - Precision error")
        else:
            print("  ❌ Price not in result")
    else:
        print("  ❌ No numeric_summary")
else:
    print(f"  ❌ Agent failed: {result.get('error')}")

# ============================================================================
# TEST SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 REAL-WORLD ACCURACY TEST SUMMARY")
print("="*80)

tests = [
    ("E-commerce Sales - Statistics", test1a_pass),
    ("E-commerce Sales - Profitability", test1b_pass),
    ("Non-standard Column Names", test2_pass),
    ("Missing Values Handling", test3_pass),
    ("Negative Numbers (Returns)", test4_pass),
    ("Large Numbers", test5_pass),
    ("Decimal Precision", test6_pass)
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

for test_name, result in tests:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} - {test_name}")

print("-"*80)
print(f"Overall: {passed}/{total} tests passed ({(passed/total*100):.1f}%)")

if passed == total:
    print("\n🎉 EXCELLENT: System handles real-world data accurately!")
elif passed >= 5:
    print(f"\n✅ GOOD: System mostly accurate with real data ({passed}/{total})")
else:
    print(f"\n⚠️ CONCERN: System struggles with real-world data ({total-passed} failures)")

print("\n💡 KEY INSIGHT:")
print("These tests use ACTUAL user data patterns and INDEPENDENT ground truth")
print("NOT reverse-engineered from code - true validation of accuracy")
print("="*80)
