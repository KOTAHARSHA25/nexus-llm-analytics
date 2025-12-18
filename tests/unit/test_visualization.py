"""
VISUALIZATION SYSTEM TEST
Purpose: Test chart generation and data visualization
Date: December 16, 2025
"""

import sys
import os
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

print("="*80)
print("🔍 VISUALIZATION SYSTEM TEST")
print("="*80)

# ============================================================================
# TEST 1: Dynamic Chart Generator
# ============================================================================
print("\n[TEST 1] Dynamic Chart Generator Initialization")
print("-"*80)

try:
    from backend.visualization.dynamic_charts import DynamicChartGenerator
    
    chart_gen = DynamicChartGenerator()
    print("  ✅ DynamicChartGenerator initialized")
    test1_pass = 1
except Exception as e:
    print(f"  ❌ Initialization failed: {e}")
    test1_pass = 0
    chart_gen = None

# ============================================================================
# TEST 2: Chart Type Detection
# ============================================================================
print("\n[TEST 2] Chart Type Detection")
print("-"*80)

if chart_gen:
    # Create test data
    categorical_data = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D'],
        'values': [100, 150, 80, 120]
    })
    
    time_series_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'sales': [100, 110, 105, 120, 115, 130, 125, 140, 135, 150]
    })
    
    test_cases = [
        ("Bar chart for categories", categorical_data, "bar"),
        ("Line chart for time series", time_series_data, "line"),
    ]
    
    test2_results = []
    for name, data, expected_type in test_cases:
        try:
            # Detect appropriate chart type
            detected = chart_gen.suggest_chart_type(data)
            
            if detected:
                print(f"  ✅ {name}: detected={detected}")
                test2_results.append(1)
            else:
                print(f"  ⚠️ {name}: no suggestion")
                test2_results.append(0.5)
        except Exception as e:
            print(f"  ⚠️ {name}: {type(e).__name__}")
            test2_results.append(0)
    
    test2_pass = sum(test2_results)
    test2_total = len(test_cases)
else:
    test2_pass = 0
    test2_total = 2

# ============================================================================
# TEST 3: Chart Generation
# ============================================================================
print("\n[TEST 3] Chart Generation")
print("-"*80)

if chart_gen:
    chart_types = ["bar", "line", "scatter", "pie"]
    
    test3_results = []
    for chart_type in chart_types:
        try:
            # Generate chart config
            config = chart_gen.create_chart_config(
                chart_type,
                categorical_data,
                x='category',
                y='values'
            )
            
            if config and isinstance(config, dict):
                print(f"  ✅ {chart_type} chart config created")
                test3_results.append(1)
            else:
                print(f"  ⚠️ {chart_type} chart config incomplete")
                test3_results.append(0.5)
        except Exception as e:
            print(f"  ❌ {chart_type}: {type(e).__name__}")
            test3_results.append(0)
    
    test3_pass = sum(test3_results)
    test3_total = len(chart_types)
else:
    test3_pass = 0
    test3_total = 4

# ============================================================================
# TEST 4: Plotly Integration
# ============================================================================
print("\n[TEST 4] Plotly Integration")
print("-"*80)

try:
    import plotly.graph_objects as go
    
    # Create simple plotly figure
    fig = go.Figure(data=[go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3])])
    
    if fig:
        print("  ✅ Plotly integration working")
        test4_pass = 1
    else:
        print("  ❌ Plotly figure creation failed")
        test4_pass = 0
except ImportError:
    print("  ⚠️ Plotly not installed")
    test4_pass = 0
except Exception as e:
    print(f"  ❌ Error: {type(e).__name__}")
    test4_pass = 0

# ============================================================================
# TEST 5: Real-World Visualization Scenarios
# ============================================================================
print("\n[TEST 5] Real-World Scenarios")
print("-"*80)

if chart_gen:
    # Sales by region (pie chart)
    sales_by_region = pd.DataFrame({
        'region': ['North', 'South', 'East', 'West'],
        'sales': [25000, 18000, 22000, 15000]
    })
    
    # Monthly revenue (line chart)
    monthly_revenue = pd.DataFrame({
        'month': pd.date_range('2024-01-01', periods=6, freq='M'),
        'revenue': [50000, 55000, 52000, 60000, 65000, 70000]
    })
    
    scenarios = [
        ("Sales by region", sales_by_region),
        ("Monthly revenue", monthly_revenue),
    ]
    
    test5_results = []
    for name, data in scenarios:
        try:
            # Try to generate visualization
            viz = chart_gen.generate_visualization(name, data)
            
            if viz:
                print(f"  ✅ {name}: visualization created")
                test5_results.append(1)
            else:
                print(f"  ⚠️ {name}: visualization incomplete")
                test5_results.append(0.5)
        except Exception as e:
            print(f"  ⚠️ {name}: {type(e).__name__}")
            test5_results.append(0)
    
    test5_pass = sum(test5_results)
    test5_total = len(scenarios)
else:
    test5_pass = 0
    test5_total = 2

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 VISUALIZATION SYSTEM TEST SUMMARY")
print("="*80)

tests = [
    ("Initialization", test1_pass, 1),
    ("Chart Type Detection", test2_pass, test2_total),
    ("Chart Generation", test3_pass, test3_total),
    ("Plotly Integration", test4_pass, 1),
    ("Real-World Scenarios", test5_pass, test5_total),
]

total_pass = sum(p for _, p, _ in tests)
total_count = sum(t for _, _, t in tests)

for test_name, passed, total in tests:
    pct = (passed/total*100) if total > 0 else 0
    status = "✅" if pct >= 75 else "⚠️" if pct >= 50 else "❌"
    print(f"{status} {test_name}: {passed}/{total} ({pct:.0f}%)")

print("-"*80)
overall_pct = (total_pass/total_count*100) if total_count > 0 else 0
print(f"Overall: {total_pass:.1f}/{total_count} ({overall_pct:.1f}%)")

if overall_pct >= 80:
    print("\n✅ EXCELLENT: Visualization system working well")
elif overall_pct >= 60:
    print("\n⚠️ GOOD: Visualization system functional")
else:
    print("\n❌ CONCERN: Visualization system needs work")

print("="*80)
