"""
Test Fix 1: Sandbox Permissiveness for Data Operations
Tests that safe operations work while dangerous operations are blocked.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backend.core.sandbox import EnhancedSandbox

def test_sandbox_fix():
    """Test all scenarios for sandbox fix"""
    
    print("=" * 80)
    print("🧪 TESTING FIX 1: SANDBOX PERMISSIVENESS")
    print("=" * 80)
    
    # Create test data
    test_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    sandbox = EnhancedSandbox(max_memory_mb=512, max_cpu_seconds=30)
    
    # Test 1: Apply operation (previously blocked)
    print("\n📝 Test 1: df.apply() - SHOULD WORK")
    code1 = """
result = data['A'].apply(lambda x: x * 2)
"""
    try:
        output1 = sandbox.execute(code1, data=test_df)
        if 'error' in output1:
            print(f"❌ FAIL: apply() failed - {output1['error']}")
        else:
            print(f"✅ PASS: apply() works - result has {len(output1['result'])} vars")
    except Exception as e:
        print(f"❌ FAIL: apply() exception - {e}")
    
    # Test 2: Map operation (previously blocked)
    print("\n📝 Test 2: Series.map() - SHOULD WORK")
    code2 = """
result = data['C'].map({'x': 1, 'y': 2, 'z': 3})
"""
    try:
        output2 = sandbox.execute(code2, data=test_df)
        if 'error' in output2:
            print(f"❌ FAIL: map() failed - {output2['error']}")
        else:
            print(f"✅ PASS: map() works - result has {len(output2['result'])} vars")
    except Exception as e:
        print(f"❌ FAIL: map() exception - {e}")
    
    # Test 3: Transform operation (previously blocked)
    print("\n📝 Test 3: df.transform() - SHOULD WORK")
    code3 = """
result = data[['A', 'B']].transform(lambda x: x + 10)
"""
    try:
        output3 = sandbox.execute(code3, data=test_df)
        if 'error' in output3:
            print(f"❌ FAIL: transform() failed - {output3['error']}")
        else:
            print(f"✅ PASS: transform() works - result has {len(output3['result'])} vars")
    except Exception as e:
        print(f"❌ FAIL: transform() exception - {e}")
    
    # Test 4: Agg operation
    print("\n📝 Test 4: df.agg() - SHOULD WORK")
    code4 = """
result = data[['A', 'B']].agg(['mean', 'sum'])
"""
    try:
        output4 = sandbox.execute(code4, data=test_df)
        if 'error' in output4:
            print(f"❌ FAIL: agg() failed - {output4['error']}")
        else:
            print(f"✅ PASS: agg() works - result has {len(output4['result'])} vars")
    except Exception as e:
        print(f"❌ FAIL: agg() exception - {e}")
    
    # Test 5: Safe utilities (fillna, dropna, sort_values)
    print("\n📝 Test 5: Safe utilities (fillna, sort_values) - SHOULD WORK")
    code5 = """
df_sorted = data.sort_values('A', ascending=False)
result = df_sorted.head(3)
"""
    try:
        output5 = sandbox.execute(code5, data=test_df)
        if 'error' in output5:
            print(f"❌ FAIL: sort_values() failed - {output5['error']}")
        else:
            print(f"✅ PASS: sort_values() works - result has {len(output5['result'])} vars")
    except Exception as e:
        print(f"❌ FAIL: sort_values() exception - {e}")
    
    # Test 6: Melt operation
    print("\n📝 Test 6: pandas.melt() - SHOULD WORK")
    code6 = """
result = pandas.melt(data, id_vars=['C'], value_vars=['A', 'B'])
"""
    try:
        output6 = sandbox.execute(code6, data=test_df)
        if 'error' in output6:
            print(f"❌ FAIL: melt() failed - {output6['error']}")
        else:
            print(f"✅ PASS: melt() works - result has {len(output6['result'])} vars")
    except Exception as e:
        print(f"❌ FAIL: melt() exception - {e}")
    
    # Test 7: SECURITY - read_csv should be blocked
    print("\n📝 Test 7: pandas.read_csv() - SHOULD BE BLOCKED")
    code7 = """
result = pandas.read_csv('test.csv')
"""
    try:
        output7 = sandbox.execute(code7, data=test_df)
        if 'error' in output7:
            print(f"✅ PASS: read_csv() correctly blocked - {output7['error'][:80]}")
        else:
            print(f"❌ FAIL: read_csv() was NOT blocked - SECURITY ISSUE!")
    except Exception as e:
        print(f"✅ PASS: read_csv() correctly blocked - {str(e)[:80]}")
    
    # Test 8: SECURITY - to_csv should be blocked
    print("\n📝 Test 8: data.to_csv() - SHOULD BE BLOCKED")
    code8 = """
data.to_csv('output.csv')
result = 'done'
"""
    try:
        output8 = sandbox.execute(code8, data=test_df)
        if 'error' in output8:
            print(f"✅ PASS: to_csv() correctly blocked - {output8['error'][:80]}")
        else:
            print(f"❌ FAIL: to_csv() was NOT blocked - SECURITY ISSUE!")
    except Exception as e:
        print(f"✅ PASS: to_csv() correctly blocked - {str(e)[:80]}")
    
    # Test 9: SECURITY - to_excel should be blocked
    print("\n📝 Test 9: data.to_excel() - SHOULD BE BLOCKED")
    code9 = """
data.to_excel('output.xlsx')
result = 'done'
"""
    try:
        output9 = sandbox.execute(code9, data=test_df)
        if 'error' in output9:
            print(f"✅ PASS: to_excel() correctly blocked - {output9['error'][:80]}")
        else:
            print(f"❌ FAIL: to_excel() was NOT blocked - SECURITY ISSUE!")
    except Exception as e:
        print(f"✅ PASS: to_excel() correctly blocked - {str(e)[:80]}")
    
    # Test 10: Complex real-world scenario
    print("\n📝 Test 10: Complex real-world transformation - SHOULD WORK")
    code10 = """
# Calculate average by category
grouped = data.groupby('C')['A'].mean()
# Apply transformation
transformed = data['B'].apply(lambda x: x / 10)
# Combine results
result = pandas.DataFrame({'category': grouped.index, 'avg_A': grouped.values})
"""
    try:
        output10 = sandbox.execute(code10, data=test_df)
        if 'error' in output10:
            print(f"❌ FAIL: Complex transformation failed - {output10['error']}")
        else:
            print(f"✅ PASS: Complex transformation works - result has {len(output10['result'])} vars")
    except Exception as e:
        print(f"❌ FAIL: Complex transformation exception - {e}")
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print("Expected: Tests 1-6, 10 should PASS (safe operations work)")
    print("Expected: Tests 7-9 should PASS (dangerous operations blocked)")
    print("=" * 80)

if __name__ == "__main__":
    test_sandbox_fix()
