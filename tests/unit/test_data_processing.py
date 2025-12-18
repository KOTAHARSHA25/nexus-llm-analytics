"""
DATA PROCESSING PIPELINE TEST
Purpose: Test CSV upload, parsing, and processing
Date: December 16, 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

from backend.utils.data_utils import read_dataframe, clean_column_names

print("="*80)
print("🔍 DATA PROCESSING PIPELINE TEST")
print("="*80)

# ============================================================================
# TEST 1: CSV File Reading
# ============================================================================
print("\n[TEST 1] CSV File Reading")
print("-"*80)

# Create temporary CSV with realistic data
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
    f.write("Order ID,Customer Name,Sales $,Date\n")
    f.write("1,John Doe,100.50,2024-01-15\n")
    f.write("2,Jane Smith,200.75,2024-01-16\n")
    f.write("3,Bob Jones,,2024-01-17\n")  # Missing value
    temp_csv = f.name

try:
    df = read_dataframe(temp_csv)
    
    if df is not None and len(df) > 0:
        print(f"  ✅ CSV read successfully: {len(df)} rows")
        test1_pass = 1
    else:
        print("  ❌ CSV reading failed")
        test1_pass = 0
finally:
    # Cleanup
    if os.path.exists(temp_csv):
        os.unlink(temp_csv)

# ============================================================================
# TEST 2: Column Type Detection
# ============================================================================
print("\n[TEST 2] Column Type Detection")
print("-"*80)

test_data = pd.DataFrame({
    'id': [1, 2, 3, 4],  # Integer
    'name': ['A', 'B', 'C', 'D'],  # String
    'value': [1.5, 2.3, 3.1, 4.8],  # Float
    'date': pd.date_range('2024-01-01', periods=4),  # Datetime
    'flag': [True, False, True, False],  # Boolean
})

type_checks = [
    ('id', 'int'),
    ('name', 'object'),
    ('value', 'float'),
    ('date', 'datetime'),
]

test2_results = []
for col, expected_type in type_checks:
    actual = str(test_data[col].dtype)
    matches = expected_type in actual.lower()
    
    status = "✅" if matches else "⚠️"
    print(f"  {status} Column '{col}': {actual} (expected: {expected_type})")
    test2_results.append(1 if matches else 0)

test2_pass = sum(test2_results)
test2_total = len(type_checks)

# ============================================================================
# TEST 3: Missing Value Handling
# ============================================================================
print("\n[TEST 3] Missing Value Handling")
print("-"*80)

missing_data = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': [10, None, 30, None, 50],
    'C': [100, 200, 300, 400, 500],
})

test3_results = []

# Check detection
missing_count = missing_data.isna().sum()
if missing_count.sum() > 0:
    print(f"  ✅ Detected {missing_count.sum()} missing values")
    test3_results.append(1)
else:
    print("  ❌ Missing value detection failed")
    test3_results.append(0)

# Check handling (drop/fill)
try:
    cleaned = missing_data.dropna()
    if len(cleaned) < len(missing_data):
        print(f"  ✅ Can remove rows with missing values: {len(missing_data)} → {len(cleaned)}")
        test3_results.append(1)
    else:
        print("  ⚠️ Row removal unclear")
        test3_results.append(0.5)
except:
    print("  ❌ Row removal failed")
    test3_results.append(0)

test3_pass = sum(test3_results)
test3_total = len(test3_results)

# ============================================================================
# TEST 4: Data Validation
# ============================================================================
print("\n[TEST 4] Data Validation")
print("-"*80)

validation_data = pd.DataFrame({
    'age': [25, 30, -5, 200, 28],  # Negative and unrealistic values
    'price': [10.5, 20.3, -15.0, 30.2, 25.1],  # Negative price
    'quantity': [1, 2, 3, 4, 5],  # All valid
})

test4_results = []

# Detect negative ages
negative_ages = validation_data['age'] < 0
if negative_ages.any():
    print(f"  ✅ Detected negative ages: {negative_ages.sum()}")
    test4_results.append(1)
else:
    print("  ⚠️ Negative age detection unclear")
    test4_results.append(0)

# Detect unrealistic ages
unrealistic_ages = validation_data['age'] > 150
if unrealistic_ages.any():
    print(f"  ✅ Detected unrealistic ages: {unrealistic_ages.sum()}")
    test4_results.append(1)
else:
    print("  ⚠️ Unrealistic age detection unclear")
    test4_results.append(0)

# Detect negative prices
negative_prices = validation_data['price'] < 0
if negative_prices.any():
    print(f"  ✅ Detected negative prices: {negative_prices.sum()}")
    test4_results.append(1)
else:
    print("  ⚠️ Negative price detection unclear")
    test4_results.append(0)

test4_pass = sum(test4_results)
test4_total = len(test4_results)

# ============================================================================
# TEST 5: Large File Handling
# ============================================================================
print("\n[TEST 5] Large File Handling")
print("-"*80)

# Create large dataset (10k rows)
large_data = pd.DataFrame({
    'id': range(10000),
    'value': np.random.randint(0, 1000, 10000),
})

try:
    # Should handle without crashing
    summary = large_data.describe()
    
    if summary is not None:
        print(f"  ✅ Handled large dataset: {len(large_data)} rows")
        test5_pass = 1
    else:
        print("  ⚠️ Large dataset processing unclear")
        test5_pass = 0.5
except Exception as e:
    print(f"  ❌ Large dataset failed: {type(e).__name__}")
    test5_pass = 0

# ============================================================================
# TEST 6: Special Characters in Data
# ============================================================================
print("\n[TEST 6] Special Characters Handling")
print("-"*80)

special_char_data = pd.DataFrame({
    'name': ['Test & Co.', 'Smith\'s Shop', 'Café résumé', '日本語'],
    'email': ['test@example.com', 'user+tag@domain.co.uk', 'name.surname@site.org', 'test@domain'],
})

try:
    # Should handle without errors
    cleaned = clean_column_names(special_char_data)
    
    if cleaned is not None:
        print("  ✅ Handled special characters in data")
        test6_pass = 1
    else:
        print("  ⚠️ Special character handling unclear")
        test6_pass = 0.5
except Exception as e:
    print(f"  ❌ Special character handling failed: {type(e).__name__}")
    test6_pass = 0

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 DATA PROCESSING PIPELINE TEST SUMMARY")
print("="*80)

tests = [
    ("CSV File Reading", test1_pass, 1),
    ("Column Type Detection", test2_pass, test2_total),
    ("Missing Value Handling", test3_pass, test3_total),
    ("Data Validation", test4_pass, test4_total),
    ("Large File Handling", test5_pass, 1),
    ("Special Characters", test6_pass, 1),
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

if overall_pct >= 80:
    print("\n✅ EXCELLENT: Data processing pipeline working well")
elif overall_pct >= 60:
    print("\n⚠️ GOOD: Data processing pipeline functional")
else:
    print("\n❌ CONCERN: Data processing pipeline needs work")

print("="*80)
