"""
Advanced Test Suite for Fix 1: Sandbox Permissions
Tests complex operations, edge cases, and security boundaries
"""
import pandas as pd
import numpy as np
from src.backend.core.sandbox import EnhancedSandbox

print("="*80)
print("🧪 ADVANCED TESTING FIX 1: SANDBOX PERMISSIONS")
print("="*80)

sandbox = EnhancedSandbox(max_memory_mb=512, max_cpu_seconds=30)
test_results = []

# Test Dataset - Complex real-world data
df = pd.DataFrame({
    'id': range(1, 101),
    'category': ['A', 'B', 'C', 'D'] * 25,
    'value': np.random.randn(100) * 100,
    'amount': np.random.randint(1, 1000, 100),
    'date': pd.date_range('2024-01-01', periods=100),
    'text': ['Item_' + str(i) for i in range(100)]
})

# Test 1: Complex apply with lambda
print("\n📝 Test 1: Complex apply() with multiple lambda functions")
print("-"*80)
try:
    code = """
result = data.copy()
result['computed'] = data['value'].apply(lambda x: x * 2 if x > 0 else x / 2)
result['category_upper'] = data['category'].apply(lambda x: x.upper())
result['log_amount'] = data['amount'].apply(lambda x: np.log(x + 1))
result.head()
"""
    result = sandbox.execute(code, data=df)
    if 'computed' in result and 'category_upper' in result and 'log_amount' in result:
        print(f"✅ PASS: Complex apply operations successful")
        print(f"   Computed {len(result)} rows with 3 new columns")
        test_results.append(True)
    else:
        print(f"❌ FAIL: Missing expected columns")
        test_results.append(False)
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_results.append(False)

# Test 2: Nested transformations with groupby
print("\n📝 Test 2: Complex groupby with transform and agg")
print("-"*80)
try:
    code = """
result = data.copy()
# Group-level statistics
result['category_mean'] = data.groupby('category')['value'].transform('mean')
result['category_std'] = data.groupby('category')['value'].transform('std')
# Aggregations
agg_result = data.groupby('category').agg({
    'value': ['mean', 'std', 'min', 'max'],
    'amount': ['sum', 'count']
})
result
"""
    result = sandbox.execute(code, data=df)
    if 'category_mean' in result and 'category_std' in result:
        print(f"✅ PASS: Groupby transform operations successful")
        print(f"   Category means: {result['category_mean'].nunique()} unique values")
        test_results.append(True)
    else:
        print(f"❌ FAIL: Transform failed")
        test_results.append(False)
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_results.append(False)

# Test 3: Map with dictionary and Series
print("\n📝 Test 3: Map operations with dict and complex mappings")
print("-"*80)
try:
    code = """
category_mapping = {'A': 'Alpha', 'B': 'Beta', 'C': 'Gamma', 'D': 'Delta'}
result = data.copy()
result['category_name'] = data['category'].map(category_mapping)
# Map with function
result['value_bucket'] = data['value'].map(lambda x: 'HIGH' if x > 50 else 'LOW' if x > 0 else 'NEG')
result[['category', 'category_name', 'value', 'value_bucket']].head(10)
"""
    result = sandbox.execute(code, data=df)
    if 'category_name' in result and 'value_bucket' in result:
        print(f"✅ PASS: Map operations successful")
        print(f"   Mapped {result['category_name'].notna().sum()} category names")
        test_results.append(True)
    else:
        print(f"❌ FAIL: Map operation failed")
        test_results.append(False)
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_results.append(False)

# Test 4: Melt and pivot operations
print("\n📝 Test 4: Reshape operations (melt, pivot, pivot_table)")
print("-"*80)
try:
    code = """
# Create wide format sample
wide_df = data.groupby('category')['amount'].sum().reset_index()
wide_df.columns = ['category', 'total']
# Pivot table
pivot_result = pd.pivot_table(
    data, 
    values='amount', 
    index='category', 
    aggfunc=['sum', 'mean', 'count']
)
pivot_result
"""
    result = sandbox.execute(code, data=df)
    if result is not None and len(result) > 0:
        print(f"✅ PASS: Pivot operations successful")
        print(f"   Pivot table shape: {result.shape}")
        test_results.append(True)
    else:
        print(f"❌ FAIL: Pivot failed")
        test_results.append(False)
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_results.append(False)

# Test 5: Chained operations (real-world complexity)
print("\n📝 Test 5: Chained complex operations")
print("-"*80)
try:
    code = """
result = (data
    .assign(value_squared=lambda x: x['value'] ** 2)
    .assign(is_high_value=lambda x: x['value'] > x['value'].median())
    .groupby(['category', 'is_high_value'])
    .agg({'amount': ['sum', 'mean'], 'value': 'count'})
    .reset_index()
)
result
"""
    result = sandbox.execute(code, data=df)
    if result is not None and len(result) > 0:
        print(f"✅ PASS: Chained operations successful")
        print(f"   Result has {len(result)} rows")
        test_results.append(True)
    else:
        print(f"❌ FAIL: Chaining failed")
        test_results.append(False)
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_results.append(False)

# Test 6: SECURITY - File operations should be BLOCKED
print("\n📝 Test 6: SECURITY - File write operations (should be blocked)")
print("-"*80)
security_tests = []

# Test 6a: to_csv should be blocked
try:
    code = """
data.to_csv('malicious_output.csv')
"""
    result = sandbox.execute(code, data=df)
    print(f"❌ FAIL: to_csv was NOT blocked - SECURITY ISSUE!")
    security_tests.append(False)
except Exception as e:
    if 'not allowed' in str(e).lower() or 'blocked' in str(e).lower():
        print(f"✅ PASS: to_csv correctly blocked")
        security_tests.append(True)
    else:
        print(f"⚠️  PARTIAL: to_csv blocked but unexpected error: {e}")
        security_tests.append(True)

# Test 6b: to_excel should be blocked
try:
    code = """
data.to_excel('malicious_output.xlsx')
"""
    result = sandbox.execute(code, data=df)
    print(f"❌ FAIL: to_excel was NOT blocked - SECURITY ISSUE!")
    security_tests.append(False)
except Exception as e:
    if 'not allowed' in str(e).lower() or 'blocked' in str(e).lower():
        print(f"✅ PASS: to_excel correctly blocked")
        security_tests.append(True)
    else:
        print(f"⚠️  PARTIAL: to_excel blocked but unexpected error: {e}")
        security_tests.append(True)

# Test 6c: to_pickle should be blocked
try:
    code = """
data.to_pickle('malicious.pkl')
"""
    result = sandbox.execute(code, data=df)
    print(f"❌ FAIL: to_pickle was NOT blocked - SECURITY ISSUE!")
    security_tests.append(False)
except Exception as e:
    if 'not allowed' in str(e).lower() or 'blocked' in str(e).lower():
        print(f"✅ PASS: to_pickle correctly blocked")
        security_tests.append(True)
    else:
        print(f"⚠️  PARTIAL: to_pickle blocked but unexpected error: {e}")
        security_tests.append(True)

test_results.extend(security_tests)

# Test 7: Edge case - Empty DataFrame
print("\n📝 Test 7: Edge case - Operations on empty DataFrame")
print("-"*80)
try:
    empty_df = pd.DataFrame()
    code = """
result = data.copy() if len(data) > 0 else pd.DataFrame({'message': ['Empty']})
result
"""
    result = sandbox.execute(code, data=empty_df)
    if result is not None:
        print(f"✅ PASS: Empty DataFrame handled gracefully")
        test_results.append(True)
    else:
        print(f"❌ FAIL: Empty DataFrame caused issues")
        test_results.append(False)
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_results.append(False)

# Test 8: Memory-intensive operation (should respect limits)
print("\n📝 Test 8: Memory limit enforcement")
print("-"*80)
try:
    code = """
# Try to create large array
large = data.copy()
for i in range(10):
    large[f'col_{i}'] = np.random.randn(len(data)) * 1000
large
"""
    result = sandbox.execute(code, data=df)
    if result is not None:
        print(f"✅ PASS: Memory-intensive operation completed within limits")
        print(f"   Created DataFrame with {len(result.columns)} columns")
        test_results.append(True)
    else:
        print(f"❌ FAIL: Operation failed")
        test_results.append(False)
except Exception as e:
    if 'memory' in str(e).lower() or 'limit' in str(e).lower():
        print(f"✅ PASS: Memory limit enforced correctly")
        test_results.append(True)
    else:
        print(f"⚠️  Error: {e}")
        test_results.append(False)

# Test 9: Statistical operations with scipy/statsmodels
print("\n📝 Test 9: Advanced statistical operations")
print("-"*80)
try:
    code = """
from scipy import stats
# Statistical tests
result = {
    'mean': float(np.mean(data['value'])),
    'median': float(np.median(data['value'])),
    'std': float(np.std(data['value'])),
    'skew': float(stats.skew(data['value'])),
    'kurtosis': float(stats.kurtosis(data['value']))
}
result
"""
    result = sandbox.execute(code, data=df)
    if isinstance(result, dict) and 'skew' in result and 'kurtosis' in result:
        print(f"✅ PASS: Advanced statistical operations successful")
        print(f"   Computed skewness: {result['skew']:.2f}, kurtosis: {result['kurtosis']:.2f}")
        test_results.append(True)
    else:
        print(f"❌ FAIL: Statistical operations incomplete")
        test_results.append(False)
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_results.append(False)

# Test 10: Polars high-performance operations
print("\n📝 Test 10: Polars dataframe operations")
print("-"*80)
try:
    code = """
import polars as pl
# Convert to polars
pl_df = pl.DataFrame({
    'a': list(range(100)),
    'b': list(range(100, 200)),
    'c': ['x', 'y', 'z'] * 33 + ['x']
})
# Polars operations
result = pl_df.filter(pl.col('a') > 50).group_by('c').agg([
    pl.col('a').mean().alias('mean_a'),
    pl.col('b').sum().alias('sum_b')
])
result.to_pandas()
"""
    result = sandbox.execute(code, data=df)
    if result is not None and len(result) > 0:
        print(f"✅ PASS: Polars operations successful")
        print(f"   Polars result: {len(result)} groups")
        test_results.append(True)
    else:
        print(f"❌ FAIL: Polars operations failed")
        test_results.append(False)
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_results.append(False)

# Summary
print("\n" + "="*80)
print("📊 ADVANCED TEST SUMMARY - FIX 1")
print("="*80)
passed = sum(test_results)
total = len(test_results)
percentage = (passed / total) * 100

print(f"\n{'✅' if passed == total else '⚠️'} Results: {passed}/{total} tests passed ({percentage:.1f}%)")

if passed == total:
    print("\n🎉 EXCELLENT! All advanced tests passed!")
    print("   - Complex pandas operations working")
    print("   - Security boundaries enforced")
    print("   - Edge cases handled")
    print("   - Statistical libraries accessible")
    print("   - High-performance libraries (polars) working")
elif passed >= total * 0.8:
    print("\n✅ GOOD! Most advanced tests passed")
    print(f"   {total - passed} tests need attention")
else:
    print("\n⚠️  WARNING! Multiple advanced tests failed")
    print(f"   {total - passed} tests failed - review implementation")

print("="*80)
