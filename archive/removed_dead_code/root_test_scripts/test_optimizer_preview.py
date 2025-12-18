"""
Test script to see what preview data the LLM receives for multi-file queries
"""
import sys
sys.path.append('src')

from backend.utils.data_optimizer import DataOptimizer

# Test with merged file
filepath = 'data/uploads/merged_customers_orders.csv'
optimizer = DataOptimizer(max_rows=5, max_chars=3000)
result = optimizer.optimize_for_llm(filepath, 'csv')

print("="*80)
print("FULL PREVIEW SENT TO LLM")
print("="*80)
print(result['preview'])
print("="*80)
print(f"Preview length: {len(result['preview'])} characters")
