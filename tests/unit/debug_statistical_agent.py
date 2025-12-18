"""
DEBUG: What is the Statistical Agent ACTUALLY returning?
"""

import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

import pandas as pd
import json
from backend.plugins.statistical_agent import StatisticalAgent

agent = StatisticalAgent()
agent.initialize()

# Test 1: E-commerce data
print("="*80)
print("TEST: E-commerce Sales Data")
print("="*80)

ecommerce_data = pd.DataFrame({
    'Sales_Amount': [125.50, 89.99, 125.50, 45.00, 125.50],
    'Cost_Of_Goods': [75.30, 53.99, 75.30, 27.00, 75.30],
})

query = "What's the average sales amount?"
result = agent.execute(query=query, data=ecommerce_data)

print("\nQuery:", query)
print("\nFULL RESULT STRUCTURE:")
print(json.dumps(result, indent=2, default=str))

# Test 2: Missing values
print("\n" + "="*80)
print("TEST: Data with Missing Values")
print("="*80)

import numpy as np
messy_data = pd.DataFrame({
    'sales': [100, np.nan, 150, 200, np.nan, 250],
})

query2 = "Calculate average sales"
result2 = agent.execute(query=query2, data=messy_data)

print("\nQuery:", query2)
print("\nFULL RESULT STRUCTURE:")
print(json.dumps(result2, indent=2, default=str))

# Test 3: Different query wording
print("\n" + "="*80)
print("TEST: Different Query Wording")
print("="*80)

query3 = "Describe this data"
result3 = agent.execute(query=query3, data=messy_data)

print("\nQuery:", query3)
print("\nFULL RESULT STRUCTURE:")
print(json.dumps(result3, indent=2, default=str))
