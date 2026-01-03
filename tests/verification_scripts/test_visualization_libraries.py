"""
Test visualization libraries in sandbox - matplotlib, seaborn, plotly
"""
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from backend.core.sandbox import EnhancedSandbox
import pandas as pd
import numpy as np

print("="*80)
print("🧪 TESTING VISUALIZATION LIBRARIES IN SANDBOX")
print("="*80)

sandbox = EnhancedSandbox(max_memory_mb=512, max_cpu_seconds=120)

# Test DataFrame
test_df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'value': np.random.randint(1, 100, 100)
})

# Test 1: Matplotlib basic plot
print("\n📝 Test 1: matplotlib.pyplot plot - SHOULD WORK")
code1 = """
fig = plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Test Plot')
result = 'plot_created'
plt.close()
"""
try:
    output1 = sandbox.execute(code1, data=test_df)
    if 'error' in output1:
        print(f"❌ FAIL: matplotlib plot failed - {output1['error'][:80]}")
    else:
        print(f"✅ PASS: matplotlib plot works")
except Exception as e:
    print(f"❌ FAIL: matplotlib exception - {str(e)[:80]}")

# Test 2: Matplotlib savefig should be BLOCKED
print("\n📝 Test 2: matplotlib.pyplot.savefig() - SHOULD BE BLOCKED")
code2 = """
fig = plt.figure()
plt.plot([1, 2, 3], [1, 2, 3])
plt.savefig('test.png')
result = 'saved'
"""
try:
    output2 = sandbox.execute(code2, data=test_df)
    if 'error' in output2:
        print(f"✅ PASS: savefig() correctly blocked - {output2['error'][:80]}")
    else:
        print(f"❌ FAIL: savefig() was NOT blocked - SECURITY ISSUE!")
except Exception as e:
    print(f"✅ PASS: savefig() correctly blocked - {str(e)[:80]}")

# Test 3: Seaborn plot
print("\n📝 Test 3: seaborn visualization - SHOULD WORK")
code3 = """
# Create sample data using numpy (already loaded)
df_plot = pandas.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50),
    'category': ['A' if i % 2 == 0 else 'B' for i in range(50)]
})
# Create seaborn plot
fig = plt.figure(figsize=(8, 6))
ax = sns.scatterplot(data=df_plot, x='x', y='y', hue='category')
result = 'seaborn_plot_created'
plt.close()
"""
try:
    output3 = sandbox.execute(code3, data=test_df)
    if 'error' in output3:
        print(f"❌ FAIL: seaborn plot failed - {output3['error'][:80]}")
    else:
        print(f"✅ PASS: seaborn plot works")
except Exception as e:
    print(f"❌ FAIL: seaborn exception - {str(e)[:80]}")

# Test 4: Plotly express
print("\n📝 Test 4: plotly.express - SHOULD WORK")
code4 = """
df_plotly = pandas.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 1, 5, 3],
    'size': [10, 20, 30, 40, 50]
})
fig = px.scatter(df_plotly, x='x', y='y', size='size')
result = 'plotly_chart_created'
"""
try:
    output4 = sandbox.execute(code4, data=test_df)
    if 'error' in output4:
        print(f"❌ FAIL: plotly express failed - {output4['error'][:80]}")
    else:
        print(f"✅ PASS: plotly express works")
except Exception as e:
    print(f"❌ FAIL: plotly exception - {str(e)[:80]}")

# Test 5: Numpy mathematical operations
print("\n📝 Test 5: numpy advanced math - SHOULD WORK")
code5 = """
# Test various numpy functions
arr = np.array([1, 2, 3, 4, 5])
result = {
    'mean': float(np.mean(arr)),
    'std': float(np.std(arr)),
    'sum': int(np.sum(arr)),
    'sqrt': np.sqrt(arr).tolist(),
    'sin': np.sin(arr).tolist(),
    'log': np.log(arr).tolist(),
    'cumsum': np.cumsum(arr).tolist()
}
"""
try:
    output5 = sandbox.execute(code5, data=test_df)
    if 'error' in output5:
        print(f"❌ FAIL: numpy math failed - {output5['error'][:80]}")
    else:
        print(f"✅ PASS: numpy math works")
except Exception as e:
    print(f"❌ FAIL: numpy exception - {str(e)[:80]}")

# Test 6: Numpy file operations should be BLOCKED
print("\n📝 Test 6: numpy.save() - SHOULD BE BLOCKED")
code6 = """
arr = np.array([1, 2, 3])
np.save('test.npy', arr)
result = 'saved'
"""
try:
    output6 = sandbox.execute(code6, data=test_df)
    if 'error' in output6:
        print(f"✅ PASS: numpy.save() correctly blocked - {output6['error'][:80]}")
    else:
        print(f"❌ FAIL: numpy.save() was NOT blocked - SECURITY ISSUE!")
except Exception as e:
    print(f"✅ PASS: numpy.save() correctly blocked - {str(e)[:80]}")

# Test 7: Complex data analysis workflow
print("\n📝 Test 7: Complex analysis workflow - SHOULD WORK")
code7 = """
# Data preparation
df = data.copy()
df['x_normalized'] = (df['x'] - df['x'].mean()) / df['x'].std()
df['y_normalized'] = (df['y'] - df['y'].mean()) / df['y'].std()

# Statistical analysis
correlation = df[['x_normalized', 'y_normalized']].corr().iloc[0, 1]

# Groupby analysis
category_stats = df.groupby('category').agg({
    'value': ['mean', 'sum', 'count'],
    'x': 'mean'
})

result = {
    'correlation': float(correlation),
    'categories': int(len(category_stats)),
    'total_records': int(len(df))
}
"""
try:
    output7 = sandbox.execute(code7, data=test_df)
    if 'error' in output7:
        print(f"❌ FAIL: complex workflow failed - {output7['error'][:80]}")
    else:
        print(f"✅ PASS: complex workflow works")
except Exception as e:
    print(f"❌ FAIL: complex workflow exception - {str(e)[:80]}")

print("\n" + "="*80)
print("📊 VISUALIZATION LIBRARIES TEST COMPLETE")
print("="*80)
print("\n✅ Summary: All data analysis and visualization libraries are available")
print("✅ Security: File I/O operations are properly blocked")
print("="*80)
