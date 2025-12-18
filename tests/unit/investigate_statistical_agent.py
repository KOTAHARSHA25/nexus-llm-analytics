"""
INVESTIGATE: What does Statistical Agent actually return?
"""

import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

import pandas as pd
import json
from backend.plugins.statistical_agent import StatisticalAgent

# Initialize agent
agent = StatisticalAgent()
agent.initialize()

# Test data
test_data = pd.DataFrame({
    'values': [10, 20, 30, 40, 50]
})

print("="*80)
print("🔍 INVESTIGATING: Statistical Agent Return Format")
print("="*80)

# Test 1: Descriptive Statistics
print("\n[TEST] Descriptive Statistics Query")
print("-"*80)
query = "Provide descriptive statistics for this dataset"
result = agent.execute(query=query, data=test_data)

print("\nFULL RESULT STRUCTURE:")
print(json.dumps(result, indent=2, default=str))

# Test 2: Different query
print("\n" + "="*80)
print("\n[TEST] Summary Statistics Query")
print("-"*80)
query2 = "Give me summary statistics"
result2 = agent.execute(query=query2, data=test_data)

print("\nFULL RESULT STRUCTURE:")
print(json.dumps(result2, indent=2, default=str))

# Test 3: Correlation
print("\n" + "="*80)
print("\n[TEST] Correlation Query")
print("-"*80)
corr_data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
})
query3 = "Perform correlation analysis"
result3 = agent.execute(query=query3, data=corr_data)

print("\nFULL RESULT STRUCTURE:")
print(json.dumps(result3, indent=2, default=str))
