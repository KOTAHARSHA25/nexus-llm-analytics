"""Quick test for ML integration"""
import sys
import os
sys.path.insert(0, 'src')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
from backend.core.code_generator import CodeGenerator

# Load test data
df = pd.read_csv('data/samples/sales_data.csv')
print(f"[OK] Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")

# Initialize code generator
cg = CodeGenerator()
print("[OK] CodeGenerator initialized")

# Test 1: K-means clustering
print("\n=== Test 1: K-means Clustering ===")
query1 = "Perform K-means clustering with 3 clusters based on sales and revenue"
try:
    result1 = cg.generate_code(query1, df)
    print(f"Success: {result1.success}")
    if result1.success:
        print(f"Result type: {type(result1.result)}")
        print(f"Result:\n{result1.result}")
    else:
        print(f"Error: {result1.error}")
except Exception as e:
    print(f"Exception: {e}")

# Test 2: Correlation analysis
print("\n=== Test 2: Correlation Analysis ===")
query2 = "Find correlation between sales and revenue"
try:
    result2 = cg.generate_code(query2, df)
    print(f"Success: {result2.success}")
    if result2.success:
        print(f"Result: {result2.result}")
    else:
        print(f"Error: {result2.error}")
except Exception as e:
    print(f"Exception: {e}")

# Test 3: Random Forest feature importance
print("\n=== Test 3: Random Forest Classification ===")
query3 = "Build a random forest classifier to predict if revenue > 5000 using sales and price"
try:
    result3 = cg.generate_code(query3, df)
    print(f"Success: {result3.success}")
    if result3.success:
        print(f"Result type: {type(result3.result)}")
        print(f"Result:\n{result3.result}")
    else:
        print(f"Error: {result3.error}")
except Exception as e:
    print(f"Exception: {e}")

print("\n[OK] All tests completed")
