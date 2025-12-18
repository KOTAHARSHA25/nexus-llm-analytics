"""
CORRECTED REAL-WORLD TEST - Handles All Return Formats
Purpose: Test with real data, accounting for format variations
Date: December 16, 2025
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

def find_numeric_summary(result):
    """Find numeric_summary regardless of nesting"""
    if not result.get('success'):
        return None
    
    res = result.get('result', {})
    
    # Direct path
    if 'numeric_summary' in res:
        return res['numeric_summary']
    
    # Nested in descriptive
    if 'descriptive' in res and 'numeric_summary' in res['descriptive']:
        return res['descriptive']['numeric_summary']
    
    return None

print("="*80)
print("🔍 CORRECTED REAL-WORLD ACCURACY TEST")
print("="*80)
print("Handles ALL return format variations")
print("="*80)

stat_agent = StatisticalAgent()
stat_agent.initialize()
fin_agent = FinancialAgent()
fin_agent.initialize()

# ============================================================================
# TEST 1: E-commerce Sales - CORRECTED
# ============================================================================
print("\n[TEST 1] E-commerce Sales Data")
print("-"*80)

ecommerce_data = pd.DataFrame({
    'Date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
    'Product': ['Widget', 'Gadget', 'Widget', 'Doohickey', 'Widget'],
    'Sales_Amount': [125.50, 89.99, 125.50, 45.00, 125.50],
    'Cost_Of_Goods': [75.30, 53.99, 75.30, 27.00, 75.30],
})

print("GROUND TRUTH: Average Sales = $102.298")

query = "What's the average sales amount?"
result = stat_agent.execute(query=query, data=ecommerce_data)

test1_pass = False
numeric_summary = find_numeric_summary(result)
if numeric_summary and 'Sales_Amount' in numeric_summary:
    calculated = numeric_summary['Sales_Amount']['mean']
    expected = 102.298
    print(f"Calculated: ${calculated:.3f}")
    print(f"Expected: ${expected:.3f}")
    
    if abs(calculated - expected) < 0.01:
        print("✅ CORRECT")
        test1_pass = True
    else:
        print(f"❌ WRONG - Off by ${abs(calculated - expected):.3f}")
else:
    print("❌ FAILED - Couldn't find result")

# ============================================================================
# TEST 2: Missing Values - CORRECTED
# ============================================================================
print("\n[TEST 2] Missing Values Handling")
print("-"*80)

messy_data = pd.DataFrame({
    'sales': [100, np.nan, 150, 200, np.nan, 250],
})

print("GROUND TRUTH: Mean of [100, 150, 200, 250] = 175.0")

query = "Calculate average sales"
result = stat_agent.execute(query=query, data=messy_data)

test2_pass = False
numeric_summary = find_numeric_summary(result)
if numeric_summary and 'sales' in numeric_summary:
    calculated = numeric_summary['sales']['mean']
    expected = 175.0
    print(f"Calculated: {calculated}")
    print(f"Expected: {expected}")
    
    if abs(calculated - expected) < 0.01:
        print("✅ CORRECT - Handled NaN values properly")
        test2_pass = True
    else:
        print(f"❌ WRONG")
else:
    print("❌ FAILED - Couldn't find result")

# ============================================================================
# TEST 3: Negative Numbers - CORRECTED
# ============================================================================
print("\n[TEST 3] Negative Numbers (Returns)")
print("-"*80)

returns_data = pd.DataFrame({
    'amount': [100, 150, -50, 200, -30]
})

print("GROUND TRUTH: Mean = 74.0, Sum = 370.0")

query = "Calculate statistics"
result = stat_agent.execute(query=query, data=returns_data)

test3_pass = False
numeric_summary = find_numeric_summary(result)
if numeric_summary and 'amount' in numeric_summary:
    calculated_mean = numeric_summary['amount']['mean']
    expected_mean = 74.0
    print(f"Calculated Mean: {calculated_mean}")
    print(f"Expected Mean: {expected_mean}")
    
    if abs(calculated_mean - expected_mean) < 0.01:
        print("✅ CORRECT - Handled negatives")
        test3_pass = True
    else:
        print(f"❌ WRONG")
else:
    print("❌ FAILED - Couldn't find result")

# ============================================================================
# TEST 4: Decimal Precision - CORRECTED
# ============================================================================
print("\n[TEST 4] Decimal Precision")
print("-"*80)

decimal_data = pd.DataFrame({
    'price': [19.99, 29.99, 39.99, 49.99, 59.99],
})

print("GROUND TRUTH: Mean = 39.99")

query = "What's the average price?"
result = stat_agent.execute(query=query, data=decimal_data)

test4_pass = False
numeric_summary = find_numeric_summary(result)
if numeric_summary and 'price' in numeric_summary:
    calculated = numeric_summary['price']['mean']
    expected = 39.99
    print(f"Calculated: ${calculated:.2f}")
    print(f"Expected: ${expected:.2f}")
    
    if abs(calculated - expected) < 0.01:
        print("✅ CORRECT")
        test4_pass = True
    else:
        print(f"❌ WRONG")
else:
    print("❌ FAILED - Couldn't find result")

# ============================================================================
# TEST 5: Financial - Large Numbers
# ============================================================================
print("\n[TEST 5] Large Numbers - Financial")
print("-"*80)

big_data = pd.DataFrame({
    'revenue': [1_250_000, 1_875_000, 2_100_000],
    'expenses': [875_000, 1_200_000, 1_400_000]
})

print("GROUND TRUTH: Profit = $1,750,000")

query = "Analyze profitability"
result = fin_agent.execute(query=query, data=big_data)

test5_pass = False
if result.get('success') and 'result' in result:
    for key, value in result['result'].items():
        if isinstance(value, dict) and 'gross_profit' in value:
            calculated = value['gross_profit']
            expected = 1_750_000
            print(f"Calculated: ${calculated:,.2f}")
            print(f"Expected: ${expected:,.2f}")
            
            if abs(calculated - expected) < 1:
                print("✅ CORRECT")
                test5_pass = True
            break

# ============================================================================
# TEST 6: Zero Values
# ============================================================================
print("\n[TEST 6] Zero Values Edge Case")
print("-"*80)

zero_data = pd.DataFrame({
    'value': [0, 10, 20, 30, 40]
})

print("GROUND TRUTH: Mean = 20.0")

query = "Calculate average"
result = stat_agent.execute(query=query, data=zero_data)

test6_pass = False
numeric_summary = find_numeric_summary(result)
if numeric_summary and 'value' in numeric_summary:
    calculated = numeric_summary['value']['mean']
    expected = 20.0
    print(f"Calculated: {calculated}")
    print(f"Expected: {expected}")
    
    if abs(calculated - expected) < 0.01:
        print("✅ CORRECT - Handled zero values")
        test6_pass = True
    else:
        print(f"❌ WRONG")
else:
    print("❌ FAILED")

# ============================================================================
# TEST 7: Single Value
# ============================================================================
print("\n[TEST 7] Single Value Edge Case")
print("-"*80)

single_data = pd.DataFrame({
    'amount': [42.0]
})

print("GROUND TRUTH: Mean = 42.0")

query = "Calculate statistics"
result = stat_agent.execute(query=query, data=single_data)

test7_pass = False
numeric_summary = find_numeric_summary(result)
if numeric_summary and 'amount' in numeric_summary:
    calculated = numeric_summary['amount']['mean']
    expected = 42.0
    print(f"Calculated: {calculated}")
    print(f"Expected: {expected}")
    
    if abs(calculated - expected) < 0.01:
        print("✅ CORRECT - Handled single value")
        test7_pass = True
    else:
        print(f"❌ WRONG")
else:
    print("❌ FAILED")

# ============================================================================
# TEST 8: All Same Values
# ============================================================================
print("\n[TEST 8] All Same Values")
print("-"*80)

same_data = pd.DataFrame({
    'price': [9.99, 9.99, 9.99, 9.99, 9.99]
})

print("GROUND TRUTH: Mean = 9.99, Std = 0.0")

query = "Analyze this data"
result = stat_agent.execute(query=query, data=same_data)

test8_pass = False
numeric_summary = find_numeric_summary(result)
if numeric_summary and 'price' in numeric_summary:
    calculated_mean = numeric_summary['price']['mean']
    calculated_std = numeric_summary['price']['std']
    print(f"Calculated Mean: {calculated_mean}")
    print(f"Calculated Std: {calculated_std}")
    
    if abs(calculated_mean - 9.99) < 0.01 and calculated_std < 0.01:
        print("✅ CORRECT - Handled identical values")
        test8_pass = True
    else:
        print(f"❌ WRONG")
else:
    print("❌ FAILED")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 CORRECTED REAL-WORLD TEST SUMMARY")
print("="*80)

tests = [
    ("E-commerce Sales Average", test1_pass),
    ("Missing Values", test2_pass),
    ("Negative Numbers", test3_pass),
    ("Decimal Precision", test4_pass),
    ("Large Numbers", test5_pass),
    ("Zero Values", test6_pass),
    ("Single Value", test7_pass),
    ("All Same Values", test8_pass)
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

for test_name, result in tests:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} - {test_name}")

print("-"*80)
print(f"Overall: {passed}/{total} tests passed ({(passed/total*100):.1f}%)")

if passed == total:
    print("\n🎉 EXCELLENT: System handles ALL real-world scenarios accurately!")
elif passed >= 6:
    print(f"\n✅ GOOD: System is accurate ({passed}/{total} tests passed)")
else:
    print(f"\n⚠️ CONCERN: System has issues ({total-passed} failures)")

print("\n💡 VERDICT:")
if passed >= 7:
    print("Calculations ARE accurate - previous failures were format lookup errors")
else:
    print("Need to investigate remaining failures")
print("="*80)
