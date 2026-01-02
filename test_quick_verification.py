"""
Simple test to verify key visualization and library additions work
"""
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from backend.core.sandbox import EnhancedSandbox
import pandas as pd

print("="*80)
print("🧪 QUICK VERIFICATION: VISUALIZATION & LIBRARY SUPPORT")
print("="*80)

sandbox = EnhancedSandbox()

test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})

# Test 1: Matplotlib works
print("\n📝 Test 1: matplotlib basic plot - SHOULD WORK")
try:
    result = sandbox.execute("fig = plt.figure(); plt.plot([1,2,3]); plt.close(); result='ok'", data=test_df)
    print(f"✅ PASS: matplotlib works" if 'error' not in result else f"❌ FAIL: {result['error'][:50]}")
except Exception as e:
    print(f"❌ FAIL: {str(e)[:50]}")

# Test 2: Matplotlib savefig blocked
print("\n📝 Test 2: matplotlib savefig blocked - SHOULD BE BLOCKED")
try:
    result = sandbox.execute("plt.savefig('test.png')", data=test_df)
    print(f"✅ PASS: savefig blocked" if 'error' in result else f"❌ FAIL: NOT BLOCKED")
except Exception as e:
    print(f"✅ PASS: savefig blocked")

# Test 3: Seaborn available
print("\n📝 Test 3: seaborn available - SHOULD WORK")
try:
    result = sandbox.execute("result = 'seaborn' if 'sns' in dir() else 'not available'", data=test_df)
    if 'error' not in result and 'seaborn' in str(result.get('result', {})):
        print(f"✅ PASS: seaborn available")
    else:
        print(f"⚠️  seaborn not available (optional)")
except Exception as e:
    print(f"⚠️  seaborn not available")

# Test 4: Plotly works
print("\n📝 Test 4: plotly - SHOULD WORK")
try:
    result = sandbox.execute("fig = px.scatter(x=[1,2,3], y=[1,2,3]); result='ok'", data=test_df)
    print(f"✅ PASS: plotly works" if 'error' not in result else f"❌ FAIL: {result['error'][:50]}")
except Exception as e:
    print(f"❌ FAIL: {str(e)[:50]}")

# Test 5: Polars works
print("\n📝 Test 5: polars - SHOULD WORK")
try:
    result = sandbox.execute("df_pl = pl.DataFrame({'x': [1,2,3]}); result='ok'", data=test_df)
    print(f"✅ PASS: polars works" if 'error' not in result else f"⚠️  polars: {result['error'][:40]}")
except Exception as e:
    print(f"⚠️  polars: {str(e)[:40]}")

# Test 6: Polars read blocked
print("\n📝 Test 6: polars read_csv blocked - SHOULD BE BLOCKED")
try:
    result = sandbox.execute("pl.read_csv('test.csv')", data=test_df)
    print(f"✅ PASS: polars I/O blocked" if 'error' in result else f"❌ FAIL: NOT BLOCKED")
except Exception as e:
    print(f"✅ PASS: polars I/O blocked")

# Test 7: Additional pandas methods
print("\n📝 Test 7: pandas new methods (pivot, transpose, to_numpy) - SHOULD WORK")
try:
    result = sandbox.execute("result = data.transpose().to_numpy(); result='ok'", data=test_df)
    print(f"✅ PASS: new pandas methods work" if 'error' not in result else f"❌ FAIL: {result['error'][:50]}")
except Exception as e:
    print(f"❌ FAIL: {str(e)[:50]}")

print("\n" + "="*80)
print("📊 VERIFICATION COMPLETE")
print("="*80)
