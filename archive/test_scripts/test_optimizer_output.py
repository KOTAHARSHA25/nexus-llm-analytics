"""
Test what data the optimizer sends to the LLM
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
from backend.utils.data_optimizer import DataOptimizer

# Test simple.json
print("="*60)
print("TESTING DATA OPTIMIZER WITH simple.json")
print("="*60)

filepath = "data/samples/simple.json"

try:
    optimizer = DataOptimizer(max_rows=100)
    result = optimizer.optimize_for_llm(filepath)
    
    print("\n[1] Optimization successful!")
    print(f"    Original rows: {result.get('original_row_count', 'N/A')}")
    print(f"    Sampled rows: {result.get('sampled_row_count', 'N/A')}")
    print(f"    Columns: {result.get('column_count', 'N/A')}")
    
    print("\n[2] DATA PREVIEW (first 1500 chars):")
    print("-"*60)
    preview = result.get('preview', 'No preview')
    print(preview[:1500])
    print("-"*60)
    
    print("\n[3] SAMPLE DATA (what LLM sees):")
    print("-"*60)
    sample = result.get('sample_data', 'No sample')
    if isinstance(sample, str):
        print(sample[:1000])
    else:
        print(json.dumps(sample, indent=2)[:1000])
    print("-"*60)
    
    # Check if pre-calculated stats exist
    print("\n[4] CHECKING PRE-CALCULATED STATS:")
    if 'PRE-CALCULATED AGGREGATIONS' in preview:
        print("✅ Pre-calculated stats are in the preview")
        # Extract the stats section
        if 'AMOUNT:' in preview:
            start = preview.find('AMOUNT:')
            end = start + 500
            print("\nAMOUNT statistics:")
            print(preview[start:end])
    else:
        print("❌ Pre-calculated stats NOT FOUND in preview")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Expected correct answers
print("\nExpected Correct Answers:")
print("  Total sales: $940.49 (150 + 200 + 175.50 + 225 + 189.99)")
print("  Product count: 5")
print("  Average: $188.10")
print("\nIf pre-calculated stats show these values, CODE IS CORRECT")
print("If stats are wrong or missing, CODE HAS BUGS")
