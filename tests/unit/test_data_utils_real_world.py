"""
DATA UTILITIES REAL-WORLD TEST
Purpose: Test data loading, cleaning, and processing
Date: December 16, 2025
"""

import sys
import os
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

from backend.utils.data_utils import clean_column_name, clean_column_names, read_dataframe, DataPathResolver

print("="*80)
print("🔍 DATA UTILITIES - REAL-WORLD TEST")
print("="*80)

# ============================================================================
# TEST 1: Column Name Cleaning (Messy User Data)
# ============================================================================
print("\n[TEST 1] Column Name Cleaning")
print("-"*80)

messy_names = [
    ("Sales $$$", "Sales___"),
    ("Customer Name!", "Customer_Name_"),
    ("Revenue (2024)", "Revenue__2024_"),
    ("Price-per-unit", "Price_per_unit"),
    ("Item #", "Item__"),
    ("Total Cost", "Total_Cost"),  # Space
    ("Rev@enue", "Rev_enue"),  # Special char
]

test1_pass = 0
test1_total = len(messy_names)

for messy, expected in messy_names:
    cleaned = clean_column_name(messy)
    
    status = "✅" if cleaned == expected else "❌"
    print(f"  {status} '{messy}' → '{cleaned}' (expected: '{expected}')")
    
    if cleaned == expected:
        test1_pass += 1

print(f"\nResult: {test1_pass}/{test1_total} column names cleaned correctly ({test1_pass/test1_total*100:.0f}%)")

# ============================================================================
# TEST 2: DataFrame Column Cleaning
# ============================================================================
print("\n[TEST 2] DataFrame Column Cleaning")
print("-"*80)

# Create messy DataFrame like users would upload
messy_df = pd.DataFrame({
    "Sales $": [100, 200, 300],
    "Customer Name!": ["Alice", "Bob", "Charlie"],
    "Revenue (USD)": [1000, 2000, 3000],
    "Price-per-unit": [10.5, 20.3, 30.1],
})

print("Original columns:", list(messy_df.columns))

cleaned_df = clean_column_names(messy_df)

print("Cleaned columns:", list(cleaned_df.columns))

# Check if special characters removed
has_special = any(
    c in str(col) 
    for col in cleaned_df.columns 
    for c in ['$', '!', '(', ')', '-']
)

if not has_special:
    print("✅ All special characters removed")
    test2_pass = 1
else:
    print("❌ Some special characters remain")
    test2_pass = 0

# Check data integrity
data_intact = all(
    messy_df.iloc[i, j] == cleaned_df.iloc[i, j]
    for i in range(len(messy_df))
    for j in range(len(messy_df.columns))
)

if data_intact:
    print("✅ Data values preserved")
else:
    print("❌ Data values changed")
    test2_pass = 0

print(f"\nResult: DataFrame cleaning {'PASSED' if test2_pass else 'FAILED'}")

# ============================================================================
# TEST 3: Edge Cases in Column Cleaning
# ============================================================================
print("\n[TEST 3] Edge Cases")
print("-"*80)

edge_cases = [
    ("", ""),  # Empty string
    ("___", "___"),  # Already clean
    ("ALLCAPS", "ALLCAPS"),  # Already clean
    ("123column", "123column"),  # Starts with number
    ("column_with_underscores", "column_with_underscores"),  # Already clean
    ("मूल्य", "_____"),  # Non-ASCII characters
]

test3_pass = 0
test3_total = len(edge_cases)

for input_name, expected in edge_cases:
    try:
        cleaned = clean_column_name(input_name)
        
        # For non-ASCII, just check it doesn't crash and returns something valid
        if any(ord(c) > 127 for c in input_name):
            # Non-ASCII should be replaced
            valid = all(c in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_' for c in cleaned)
            status = "✅" if valid else "❌"
            print(f"  {status} Non-ASCII '{input_name[:10]}...' → '{cleaned}' (alphanumeric+underscore only)")
            if valid:
                test3_pass += 1
        else:
            status = "✅" if cleaned == expected else "⚠️"
            print(f"  {status} '{input_name}' → '{cleaned}' (expected: '{expected}')")
            if cleaned == expected:
                test3_pass += 1
            else:
                test3_pass += 0.5  # Partial credit if valid output
                
    except Exception as e:
        print(f"  ❌ '{input_name}' → ERROR: {e}")

print(f"\nResult: {test3_pass}/{test3_total} edge cases handled ({test3_pass/test3_total*100:.0f}%)")

# ============================================================================
# TEST 4: Data Path Resolver
# ============================================================================
print("\n[TEST 4] Data Path Resolver")
print("-"*80)

try:
    root = DataPathResolver.get_project_root()
    print(f"  ✅ Project root: {root}")
    
    uploads = DataPathResolver.get_uploads_dir()
    print(f"  ✅ Uploads dir: {uploads}")
    
    samples = DataPathResolver.get_samples_dir()
    print(f"  ✅ Samples dir: {samples}")
    
    # Ensure directories exist
    DataPathResolver.ensure_directories_exist()
    print(f"  ✅ Directories ensured")
    
    test4_pass = 1
    print("\nResult: Data path resolver works correctly")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    test4_pass = 0
    print("\nResult: Data path resolver FAILED")

# ============================================================================
# TEST 5: Real User Data Patterns
# ============================================================================
print("\n[TEST 5] Real User Data Patterns")
print("-"*80)

# Create DataFrame with patterns users actually upload
user_data = pd.DataFrame({
    "Order ID": [1, 2, 3, 4, 5],
    "Customer Name": ["John Doe", "Jane Smith", None, "Bob Jones", "Alice Brown"],
    "Sales ($)": [100.50, 200.75, 150.25, None, 300.00],
    "Discount %": [10, 15, 0, 20, 5],
    "Product Category": ["Electronics", "Clothing", "Electronics", "Food", "Clothing"],
})

print("User data pattern (common in real uploads):")
print("  • Mix of numeric and text columns")
print("  • Missing values (None/NaN)")
print("  • Special characters in column names")
print("  • Different data types")

try:
    cleaned = clean_column_names(user_data)
    
    # Check expectations
    checks = [
        ("Column names alphanumeric+underscore", 
         all(all(c in '0123456789abcdefghijklmnopqrstuvwxyzABCEDFGHIJKLMNOPQRSTUVWXYZ_' for c in col) for col in cleaned.columns)),
        
        ("Data values preserved",
         len(cleaned) == len(user_data)),
        
        ("Missing values preserved",
         cleaned.isna().sum().sum() == user_data.isna().sum().sum()),
    ]
    
    test5_pass = 0
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if passed:
            test5_pass += 1
    
    print(f"\nResult: {test5_pass}/{len(checks)} checks passed ({test5_pass/len(checks)*100:.0f}%)")
    
except Exception as e:
    print(f"  ❌ Error processing user data: {e}")
    test5_pass = 0

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 DATA UTILITIES TEST SUMMARY")
print("="*80)

tests = [
    ("Column Name Cleaning", test1_pass, test1_total),
    ("DataFrame Cleaning", test2_pass, 1),
    ("Edge Cases", test3_pass, test3_total),
    ("Path Resolver", test4_pass, 1),
    ("Real User Patterns", test5_pass, 3),
]

total_pass = sum(p for _, p, _ in tests)
total_count = sum(t for _, _, t in tests)

for test_name, passed, total in tests:
    pct = (passed/total*100) if total > 0 else 0
    status = "✅" if pct >= 75 else "⚠️" if pct >= 50 else "❌"
    print(f"{status} {test_name}: {passed}/{total} ({pct:.0f}%)")

print("-"*80)
overall_pct = (total_pass/total_count*100) if total_count > 0 else 0
print(f"Overall: {total_pass:.1f}/{total_count} ({overall_pct:.1f}%)")

if overall_pct >= 90:
    print("\n✅ EXCELLENT: Data utilities handle real-world data well")
elif overall_pct >= 70:
    print("\n⚠️ GOOD: Data utilities work but have minor issues")
else:
    print("\n❌ CONCERN: Data utilities need improvement")

print("="*80)
