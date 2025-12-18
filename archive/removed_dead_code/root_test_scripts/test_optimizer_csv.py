"""Debug: Check what data optimizer returns for sales_simple.csv"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.utils.data_optimizer import DataOptimizer

filepath = "data/samples/csv/sales_simple.csv"

print("="*60)
print("DIAGNOSTIC: Data Optimizer Output for sales_simple.csv")
print("="*60)
print(f"File: {filepath}")
print()

# Initialize optimizer
optimizer = DataOptimizer(max_rows=5, max_chars=3000)

# Get optimized data
result = optimizer.optimize_for_llm(filepath)

print("[SCHEMA]")
for col, info in result['schema'].items():
    print(f"  {col}: {info['type']} (unique: {info['unique_values']})")
print()

print("[STATS]")
print(f"  Total Rows: {result['stats']['total_rows']}")
print(f"  Total Columns: {result['stats']['total_columns']}")
print(f"  Columns: {result['stats']['columns']}")
print()

if 'numeric_summary' in result['stats']:
    print("[NUMERIC SUMMARY]")
    for col, stats in result['stats']['numeric_summary'].items():
        print(f"  {col}:")
        if 'sum' in stats:
            print(f"    Sum: {stats['sum']}")
        if 'mean' in stats:
            print(f"    Mean: {stats['mean']}")
        if 'count' in stats:
            print(f"    Count: {stats['count']}")
    print()

print("[SAMPLE DATA]")
for i, row in enumerate(result['sample'][:5], 1):
    print(f"  Row {i}: {row}")
print()

print("[PREVIEW - First 1000 chars]")
print(result['preview'][:1000])
print()
print("="*60)
