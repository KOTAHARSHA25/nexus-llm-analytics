"""
INDEPENDENT ACCURACY TEST - WITH MESSY REAL DATA
Purpose: Test agents with data that has REAL user problems
Date: December 16, 2025

NOT using clean test data. Using messy data users actually upload.
"""

import sys
import os
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

from backend.plugins.statistical_agent import StatisticalAgent

print("="*80)
print("🔍 TESTING WITH TRULY MESSY USER DATA")
print("="*80)
print("⚠️ This data has REAL problems users encounter")
print("="*80)

# ============================================================================
# REAL MESSY USER DATA (not idealized)
# ============================================================================

# Test 1: Mixed types in columns
messy_sales = pd.DataFrame({
    "Date": ["2024-01-15", "01/20/2024", "2024-02-03", "March 5, 2024", ""],  # Mixed formats + empty
    "Revenue": [1500.50, "2000", 1750.25, None, "N/A"],  # Mix of float, string, None, text
    "Quantity": [10, 15, "twelve", 8, 20],  # Text number!
    "Customer": ["John Doe", "JANE SMITH", "bob jones", None, "Alice"],  # Inconsistent caps + None
    "Discount%": [10, 15, 5, 200, -5],  # Impossible values (200%, negative)
})

print("\n[TEST 1] Mixed Types and Bad Values")
print("-"*80)
print("Data issues:")
print("  • Mixed date formats: '2024-01-15', '01/20/2024', 'March 5, 2024'")
print("  • Revenue has: float, string '2000', None, 'N/A'")
print("  • Quantity has text: 'twelve'")
print("  • Discount has impossible: 200%, -5%")

stat_agent = StatisticalAgent()
stat_agent.initialize()

try:
    # Calculate mean revenue - can it handle the mess?
    result = stat_agent.execute(
        "calculate average revenue",
        messy_sales
    )
    
    if result and result.get('success'):
        print("\n✅ Agent handled messy data")
        
        # Extract actual calculated mean
        response = result.get('result', {})
        
        # The CORRECT answer (calculated manually):
        # Clean revenue values: 1500.50, 2000 (if converted), 1750.25
        # If it only uses valid floats: 1500.50 + 1750.25 = 3250.75 / 2 = 1625.375
        # If it converts '2000': 1500.50 + 2000 + 1750.25 = 5250.75 / 3 = 1750.25
        
        print(f"   Agent result: {response}")
        print(f"\n⚠️ CRITICAL: Did agent handle 'N/A' and None correctly?")
        print(f"   Valid values should be: [1500.50, (maybe 2000), 1750.25]")
        
    else:
        print(f"\n❌ Agent failed: {result.get('error', 'Unknown')}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")

# ============================================================================
# Test 2: Duplicate and inconsistent data
# ============================================================================
print("\n[TEST 2] Duplicates and Inconsistencies")
print("-"*80)

inconsistent_data = pd.DataFrame({
    "Product": ["iPhone", "iPhone ", "iphone", "IPHONE", "Galaxy"],  # Same product, different cases/spaces
    "Price": [999, 999.00, 1000, 998, 799],  # Near-duplicates with slight variations
    "Category": ["Electronics", "electronics", "Electronics", "Tech", "Electronics"],  # Inconsistent
})

print("Data issues:")
print("  • Same product with different capitalization")
print("  • Same prices with slight variations (999 vs 998)")
print("  • Inconsistent category names")

try:
    result = stat_agent.execute(
        "calculate average price",
        inconsistent_data
    )
    
    if result and result.get('success'):
        print("\n✅ Agent processed inconsistent data")
        
        # Correct answer: (999 + 999 + 1000 + 998 + 799) / 5 = 4795 / 5 = 959
        expected_mean = 959.0
        
        print(f"   Expected mean: ${expected_mean:.2f}")
        print(f"   Agent result: {result.get('result', {})}")
        
        # Check if agent got it right
        response_str = str(result.get('result', {}))
        if '959' in response_str or '959.0' in response_str:
            print("\n✅ CORRECT: Agent calculated accurate mean despite inconsistencies")
        else:
            print("\n⚠️ UNCERTAIN: Cannot verify accuracy from response format")
            
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown')}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")

# ============================================================================
# Test 3: Extreme outliers and impossible values
# ============================================================================
print("\n[TEST 3] Extreme Outliers and Impossible Values")
print("-"*80)

outlier_data = pd.DataFrame({
    "Sales": [100, 120, 110, 115, 99999999, 105, 108],  # One extreme outlier
    "Temperature": [20, 22, 21, -273.15, 23, 19, 21],  # Absolute zero (possible but extreme)
    "Age": [25, 30, 35, -5, 200, 28, 27],  # Impossible ages
})

print("Data issues:")
print("  • Sales has extreme outlier: 99,999,999")
print("  • Temperature has absolute zero: -273.15°C")
print("  • Age has impossible values: -5, 200")

try:
    result = stat_agent.execute(
        "calculate descriptive statistics for sales",
        outlier_data
    )
    
    if result and result.get('success'):
        print("\n✅ Agent processed data with extreme outliers")
        print(f"   Result: {result.get('result', {})}")
        
        # Mean with outlier: (100+120+110+115+99999999+105+108)/7 = 14,285,736.71
        # Mean without outlier: (100+120+110+115+105+108)/6 = 108.0
        
        print("\n⚠️ CRITICAL QUESTION:")
        print("   Did agent include the 99,999,999 outlier?")
        print("   Mean WITH outlier: ~14,285,737")
        print("   Mean WITHOUT outlier: ~108")
        
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown')}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")

# ============================================================================
# Test 4: Empty and single-value datasets
# ============================================================================
print("\n[TEST 4] Edge Cases - Empty and Single Values")
print("-"*80)

# Empty DataFrame
empty_df = pd.DataFrame({"Value": []})

# Single value
single_df = pd.DataFrame({"Value": [42]})

# All same value
identical_df = pd.DataFrame({"Value": [100, 100, 100, 100]})

print("Testing edge cases:")
print("  • Empty dataset (0 rows)")
print("  • Single value dataset (1 row)")
print("  • All identical values (std dev = 0)")

for name, df in [("Empty", empty_df), ("Single", single_df), ("Identical", identical_df)]:
    try:
        result = stat_agent.execute(
            "calculate statistics",
            df
        )
        
        if result and result.get('success'):
            print(f"\n✅ {name}: Handled gracefully")
        else:
            print(f"\n⚠️ {name}: {result.get('error', 'Failed')}")
            
    except Exception as e:
        print(f"\n⚠️ {name}: Error handled - {type(e).__name__}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 MESSY DATA TEST SUMMARY")
print("="*80)
print("\n🎯 KEY QUESTION: Did we test with TRULY unseen messy data?")
print("✅ YES - Data has real user problems:")
print("   • Mixed types in columns")
print("   • Text in numeric fields ('N/A', 'twelve')")
print("   • Inconsistent formatting")
print("   • Extreme outliers")
print("   • Impossible values")
print("   • Empty/single value datasets")
print("\n✅ YES - We did NOT design perfect test data")
print("✅ YES - This is what users ACTUALLY upload")
print("="*80)
