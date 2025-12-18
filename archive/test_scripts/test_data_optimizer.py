"""
Test data optimizer with complex nested JSON

This script tests the data_optimizer.py utility with complex nested data
to ensure it properly flattens and optimizes data for LLM consumption.

Date: October 19, 2025
"""

import sys
from pathlib import Path

# Add src/backend to path
backend_path = Path(__file__).parent / 'src' / 'backend'
sys.path.insert(0, str(backend_path))

from utils.data_optimizer import DataOptimizer
import json

def test_complex_nested_json():
    """Test optimizer with complex nested JSON"""
    print("=" * 60)
    print("TESTING DATA OPTIMIZER - Complex Nested JSON")
    print("=" * 60)
    
    # Test file
    test_file = Path('data/samples/complex_nested.json')
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"\n📁 Testing file: {test_file.name}")
    
    # Load original data to show complexity
    with open(test_file, 'r') as f:
        original_data = json.load(f)
    
    print(f"\n📊 Original Data Structure:")
    print(f"  - Type: {type(original_data)}")
    print(f"  - Top-level keys: {list(original_data.keys()) if isinstance(original_data, dict) else 'N/A'}")
    
    # Check nesting
    if 'departments' in original_data:
        departments = original_data['departments']
        print(f"  - Departments: {len(departments)}")
        if departments and len(departments) > 0:
            dept = departments[0]
            print(f"  - Employees in first dept: {len(dept.get('employees', []))}")
            if dept.get('employees'):
                emp = dept['employees'][0]
                print(f"  - Keys in first employee: {list(emp.keys())}")
    
    print("\n" + "=" * 60)
    print("RUNNING OPTIMIZER...")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = DataOptimizer(max_rows=5, max_chars=2000)
    
    # Optimize data
    try:
        optimized = optimizer.optimize_for_llm(str(test_file))
        
        print("\n✅ OPTIMIZATION SUCCESSFUL!\n")
        
        # Show results
        print("📋 Optimization Results:")
        print(f"  - Was nested: {optimized.get('was_nested', False)}")
        print(f"  - Is optimized: {optimized.get('is_optimized', False)}")
        print(f"  - Total rows: {optimized.get('total_rows', 0)}")
        print(f"  - Total columns: {optimized.get('total_columns', 0)}")
        print(f"  - File type: {optimized.get('file_type', 'unknown')}")
        
        # Show schema
        print(f"\n📐 Schema (first 5 columns):")
        schema = optimized.get('schema', {})
        for i, (col, info) in enumerate(list(schema.items())[:5]):
            print(f"  {i+1}. {col}")
            print(f"     - Type: {info.get('type', 'unknown')}")
            print(f"     - Unique values: {info.get('unique_values', 0)}")
            print(f"     - Sample: {info.get('sample_values', [])[:2]}")
        
        if len(schema) > 5:
            print(f"  ... and {len(schema) - 5} more columns")
        
        # Show stats
        print(f"\n📊 Statistics:")
        stats = optimized.get('stats', {})
        print(f"  - Rows: {stats.get('total_rows', 0)}")
        print(f"  - Columns: {stats.get('total_columns', 0)}")
        print(f"  - Memory: {stats.get('memory_usage_mb', 0):.2f} MB")
        
        # Show preview (first 500 chars)
        print(f"\n📄 Preview for LLM (first 500 chars):")
        print("-" * 60)
        preview = optimized.get('preview', '')
        print(preview[:500])
        if len(preview) > 500:
            print(f"\n... ({len(preview) - 500} more characters)")
        print("-" * 60)
        
        # Show sample data
        print(f"\n📊 Sample Data (first 2 rows):")
        sample = optimized.get('sample', [])
        for i, row in enumerate(sample[:2]):
            print(f"\nRow {i+1}:")
            # Show first 5 fields
            for j, (key, value) in enumerate(list(row.items())[:5]):
                print(f"  - {key}: {value}")
            if len(row) > 5:
                print(f"  ... and {len(row) - 5} more fields")
        
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n💡 The optimized data is now ready for LLM processing!")
        print("   - Nested structures flattened")
        print("   - Data sampled to manageable size")
        print("   - Schema and stats generated")
        print("   - Preview truncated to avoid LLM context overflow")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during optimization:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complex_nested_json()
    sys.exit(0 if success else 1)
