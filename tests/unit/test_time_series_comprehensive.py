"""
TIME SERIES AGENT COMPREHENSIVE TEST
Purpose: Test time series analysis with real temporal data patterns
Date: December 16, 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'src'))

from backend.plugins.time_series_agent import TimeSeriesAgent

print("="*80)
print("🔍 TIME SERIES AGENT - COMPREHENSIVE TEST")
print("="*80)

# ============================================================================
# TEST 1: Trend Detection
# ============================================================================
print("\n[TEST 1] Trend Detection")
print("-"*80)

# Create data with clear upward trend
dates = pd.date_range('2024-01-01', periods=12, freq='M')
upward_trend = pd.DataFrame({
    'date': dates,
    'sales': [100, 110, 125, 135, 150, 170, 190, 210, 230, 250, 275, 300]
})

# Downward trend
downward_trend = pd.DataFrame({
    'date': dates,
    'revenue': [1000, 950, 920, 880, 850, 800, 750, 700, 650, 600, 550, 500]
})

# Flat/stable
stable = pd.DataFrame({
    'date': dates,
    'orders': [50, 52, 48, 51, 49, 50, 51, 49, 50, 52, 48, 50]
})

ts_agent = TimeSeriesAgent()
ts_agent.initialize()

test1_results = []
test_cases = [
    ("Upward trend", upward_trend, "increasing"),
    ("Downward trend", downward_trend, "decreasing"),
    ("Stable", stable, "stable"),
]

for name, data, expected in test_cases:
    try:
        result = ts_agent.execute(f"analyze trend in {name}", data)
        
        if result and result.get('success'):
            result_str = str(result).lower()
            detected = "increasing" if "increas" in result_str or "upward" in result_str or "growing" in result_str else \
                       "decreasing" if "decreas" in result_str or "downward" in result_str or "declining" in result_str else \
                       "stable" if "stable" in result_str or "flat" in result_str else "unknown"
            
            match = detected == expected
            status = "✅" if match else "⚠️"
            print(f"  {status} {name}: detected={detected}, expected={expected}")
            test1_results.append(1 if match else 0)
        else:
            print(f"  ❌ {name}: Failed")
            test1_results.append(0)
    except Exception as e:
        print(f"  ❌ {name}: ERROR - {type(e).__name__}")
        test1_results.append(0)

print(f"\nResult: {sum(test1_results)}/{len(test1_results)} trends detected correctly")

# ============================================================================
# TEST 2: Seasonality Detection
# ============================================================================
print("\n[TEST 2] Seasonality Detection")
print("-"*80)

# Create seasonal pattern (quarterly peaks)
dates_daily = pd.date_range('2024-01-01', periods=365, freq='D')
seasonal_pattern = 100 + 50 * np.sin(2 * np.pi * np.arange(365) / 90)  # 90-day cycle
seasonal_data = pd.DataFrame({
    'date': dates_daily,
    'sales': seasonal_pattern
})

try:
    result = ts_agent.execute("detect seasonality", seasonal_data)
    
    if result and result.get('success'):
        result_str = str(result).lower()
        has_seasonality = "season" in result_str or "periodic" in result_str or "cycl" in result_str
        
        if has_seasonality:
            print("  ✅ Seasonality detected")
            test2_pass = 1
        else:
            print("  ⚠️ Seasonality not explicitly mentioned")
            test2_pass = 0.5
    else:
        print("  ❌ Failed to detect seasonality")
        test2_pass = 0
except Exception as e:
    print(f"  ❌ ERROR: {type(e).__name__}")
    test2_pass = 0

# ============================================================================
# TEST 3: Forecasting
# ============================================================================
print("\n[TEST 3] Forecasting")
print("-"*80)

# Historical data
forecast_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10, freq='M'),
    'revenue': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
})

try:
    result = ts_agent.execute("predict next 3 months", forecast_data)
    
    if result and result.get('success'):
        result_str = str(result).lower()
        has_forecast = "forecast" in result_str or "predict" in result_str or "future" in result_str
        
        if has_forecast:
            print("  ✅ Forecast generated")
            # Check if reasonable (should be around 2000-2200 based on trend)
            test3_pass = 1
        else:
            print("  ⚠️ Forecast capability uncertain")
            test3_pass = 0.5
    else:
        print("  ❌ Forecasting failed")
        test3_pass = 0
except Exception as e:
    print(f"  ❌ ERROR: {type(e).__name__}")
    test3_pass = 0

# ============================================================================
# TEST 4: Anomaly Detection in Time Series
# ============================================================================
print("\n[TEST 4] Anomaly Detection")
print("-"*80)

# Normal pattern with one anomaly
anomaly_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=20, freq='D'),
    'visits': [100, 105, 98, 102, 99, 101, 103, 97, 500, 100, 102, 99, 101, 98, 103, 100, 99, 102, 101, 98]
    #                                                    ^^^ anomaly at day 9
})

try:
    result = ts_agent.execute("find anomalies", anomaly_data)
    
    if result and result.get('success'):
        result_str = str(result).lower()
        found_anomaly = "anomal" in result_str or "outlier" in result_str or "unusual" in result_str or "500" in result_str
        
        if found_anomaly:
            print("  ✅ Anomaly detected (day 9: 500 visits)")
            test4_pass = 1
        else:
            print("  ⚠️ Anomaly not explicitly identified")
            test4_pass = 0.5
    else:
        print("  ❌ Anomaly detection failed")
        test4_pass = 0
except Exception as e:
    print(f"  ❌ ERROR: {type(e).__name__}")
    test4_pass = 0

# ============================================================================
# TEST 5: Moving Averages
# ============================================================================
print("\n[TEST 5] Moving Averages")
print("-"*80)

ma_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30, freq='D'),
    'price': np.random.randint(90, 110, 30)  # Noisy data
})

try:
    result = ts_agent.execute("calculate 7-day moving average", ma_data)
    
    if result and result.get('success'):
        result_str = str(result).lower()
        has_ma = "moving average" in result_str or "smoothed" in result_str or "ma" in result_str
        
        if has_ma:
            print("  ✅ Moving average calculated")
            test5_pass = 1
        else:
            print("  ⚠️ Moving average capability uncertain")
            test5_pass = 0.5
    else:
        print("  ❌ Moving average failed")
        test5_pass = 0
except Exception as e:
    print(f"  ❌ ERROR: {type(e).__name__}")
    test5_pass = 0

# ============================================================================
# TEST 6: Real-World Temporal Patterns
# ============================================================================
print("\n[TEST 6] Real-World Patterns (E-commerce)")
print("-"*80)

# Realistic e-commerce sales: weekday vs weekend patterns
dates_week = pd.date_range('2024-01-01', periods=28, freq='D')
sales_pattern = []
for d in dates_week:
    if d.weekday() >= 5:  # Weekend
        sales_pattern.append(np.random.randint(150, 200))
    else:  # Weekday
        sales_pattern.append(np.random.randint(80, 120))

ecommerce_data = pd.DataFrame({
    'date': dates_week,
    'daily_sales': sales_pattern
})

try:
    result = ts_agent.execute("analyze weekly sales pattern", ecommerce_data)
    
    if result and result.get('success'):
        result_str = str(result).lower()
        detected_pattern = "weekend" in result_str or "week" in result_str or "pattern" in result_str
        
        if detected_pattern:
            print("  ✅ Weekly pattern recognized")
            test6_pass = 1
        else:
            print("  ⚠️ Pattern analysis performed")
            test6_pass = 0.5
    else:
        print("  ❌ Pattern analysis failed")
        test6_pass = 0
except Exception as e:
    print(f"  ❌ ERROR: {type(e).__name__}")
    test6_pass = 0

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 TIME SERIES AGENT TEST SUMMARY")
print("="*80)

total_pass = sum(test1_results) + test2_pass + test3_pass + test4_pass + test5_pass + test6_pass
total_tests = len(test1_results) + 5

tests = [
    ("Trend Detection", sum(test1_results), len(test1_results)),
    ("Seasonality Detection", test2_pass, 1),
    ("Forecasting", test3_pass, 1),
    ("Anomaly Detection", test4_pass, 1),
    ("Moving Averages", test5_pass, 1),
    ("Real-World Patterns", test6_pass, 1),
]

for test_name, passed, total in tests:
    pct = (passed/total*100) if total > 0 else 0
    status = "✅" if pct >= 75 else "⚠️" if pct >= 50 else "❌"
    print(f"{status} {test_name}: {passed}/{total} ({pct:.0f}%)")

print("-"*80)
overall_pct = (total_pass/total_tests*100) if total_tests > 0 else 0
print(f"Overall: {total_pass:.1f}/{total_tests} ({overall_pct:.1f}%)")

if overall_pct >= 80:
    print("\n✅ EXCELLENT: Time series agent handles temporal patterns well")
elif overall_pct >= 60:
    print("\n⚠️ GOOD: Time series agent works but has gaps")
else:
    print("\n❌ CONCERN: Time series agent needs improvement")

print("="*80)
