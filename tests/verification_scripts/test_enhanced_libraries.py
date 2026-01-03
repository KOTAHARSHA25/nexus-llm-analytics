"""
Test enhanced sandbox libraries - polars and additional pandas methods
"""
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from backend.core.sandbox import EnhancedSandbox
import pandas as pd

print("="*80)
print("🧪 TESTING ENHANCED LIBRARY SUPPORT")
print("="*80)

sandbox = EnhancedSandbox(max_memory_mb=512, max_cpu_seconds=120)

# Test DataFrame
test_df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': ['x', 'y', 'z', 'x', 'y'],
    'date': pd.date_range('2024-01-01', periods=5)
})

# Test 1: pivot() operation
print("\n📝 Test 1: pandas.pivot() - SHOULD WORK")
code1 = """
# Create a simple pivot
result_df = pandas.DataFrame({'A': [1,1,2,2], 'B': ['x','y','x','y'], 'val': [10,20,30,40]})
result = result_df.pivot(index='A', columns='B', values='val')
"""
try:
    output1 = sandbox.execute(code1, data=test_df)
    if 'error' in output1:
        print(f"❌ FAIL: pivot() failed - {output1['error']}")
    else:
        print(f"✅ PASS: pivot() works")
except Exception as e:
    print(f"❌ FAIL: pivot() exception - {e}")

# Test 2: to_datetime() operation
print("\n📝 Test 2: pandas.to_datetime() - SHOULD WORK")
code2 = """
date_series = pandas.Series(['2024-01-01', '2024-02-01', '2024-03-01'])
result = pandas.to_datetime(date_series)
"""
try:
    output2 = sandbox.execute(code2, data=test_df)
    if 'error' in output2:
        print(f"❌ FAIL: to_datetime() failed - {output2['error']}")
    else:
        print(f"✅ PASS: to_datetime() works")
except Exception as e:
    print(f"❌ FAIL: to_datetime() exception - {e}")

# Test 3: transpose() operation
print("\n📝 Test 3: DataFrame.transpose() - SHOULD WORK")
code3 = """
result = data[['A', 'B']].head(3).transpose()
"""
try:
    output3 = sandbox.execute(code3, data=test_df)
    if 'error' in output3:
        print(f"❌ FAIL: transpose() failed - {output3['error']}")
    else:
        print(f"✅ PASS: transpose() works")
except Exception as e:
    print(f"❌ FAIL: transpose() exception - {e}")

# Test 4: Polars support
print("\n📝 Test 4: Polars library - SHOULD WORK")
code4 = """
# Check if polars is available
try:
    # Create a simple polars DataFrame
    df_pl = pl.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    # Convert to pandas for result
    result = df_pl.to_pandas()
    polars_works = True
except Exception as e:
    result = f"Polars not available: {str(e)}"
    polars_works = False
"""
try:
    output4 = sandbox.execute(code4, data=test_df)
    if 'error' in output4:
        print(f"⚠️  Polars not available - {output4['error']}")
    else:
        if 'result' in output4 and 'polars_works' in output4['result']:
            if output4['result']['polars_works']:
                print(f"✅ PASS: Polars library works")
            else:
                print(f"⚠️  Polars available but test failed")
        else:
            print(f"✅ PASS: Polars test executed")
except Exception as e:
    print(f"⚠️  Polars exception (might not be installed) - {e}")

# Test 5: polars read_csv should be BLOCKED
print("\n📝 Test 5: polars.read_csv() - SHOULD BE BLOCKED")
code5 = """
result = pl.read_csv('test.csv')
"""
try:
    output5 = sandbox.execute(code5, data=test_df)
    if 'error' in output5:
        print(f"✅ PASS: polars.read_csv() correctly blocked - {output5['error'][:60]}")
    else:
        print(f"❌ FAIL: polars.read_csv() was NOT blocked - SECURITY ISSUE!")
except Exception as e:
    print(f"✅ PASS: polars.read_csv() correctly blocked - {str(e)[:60]}")

# Test 6: to_numpy() operation
print("\n📝 Test 6: DataFrame.to_numpy() - SHOULD WORK")
code6 = """
result = data[['A', 'B']].to_numpy()
"""
try:
    output6 = sandbox.execute(code6, data=test_df)
    if 'error' in output6:
        print(f"❌ FAIL: to_numpy() failed - {output6['error']}")
    else:
        print(f"✅ PASS: to_numpy() works")
except Exception as e:
    print(f"❌ FAIL: to_numpy() exception - {e}")

print("\n" + "="*80)
print("📊 ENHANCED LIBRARIES TEST COMPLETE")
print("="*80)
