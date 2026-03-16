
import pandas as pd
import numpy as np
from scipy import stats
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)

def replicate_forecast_logic(filepath):
    print(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    
    # mimic _prepare_time_series_data
    datetime_col = 'order_date' # We know this is the col
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    data = data.sort_values(datetime_col)
    data.set_index(datetime_col, inplace=True)
    
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    ts_data = data[numeric_cols]
    
    print(f"Data sorted by date. Last 5 rows of total_amount:")
    print(ts_data['total_amount'].tail(5))
    
    results = {}
    
    for col in ts_data.columns:
        series = ts_data[col].dropna()
        
        # mimic _forecast_analysis logic
        col_results = {}
        
        # Moving Average
        if len(series) >= 3:
            window = min(3, len(series) // 3)
            # Rolling mean
            ma_series = series.rolling(window=window).mean()
            ma_forecast = ma_series.iloc[-1]
            col_results["moving_average"] = float(ma_forecast)
            
        # Linear Trend
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        next_point = slope * len(series) + intercept
        col_results["linear_trend"] = float(next_point)
        
        results[col] = col_results

    print("\n--- REPLICATION RESULTS ---")
    for col, res in results.items():
        print(f"{col}: MA={res.get('moving_average', 'N/A'):.2f}, Trend={res.get('linear_trend', 'N/A'):.2f}")

if __name__ == "__main__":
    replicate_forecast_logic("data/samples/comprehensive_ecommerce.csv")
