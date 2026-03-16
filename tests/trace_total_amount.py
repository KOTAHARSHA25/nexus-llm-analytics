
import pandas as pd
import numpy as np

def trace_total_amount():
    data = pd.read_csv("data/samples/comprehensive_ecommerce.csv")
    data['order_date'] = pd.to_datetime(data['order_date'])
    data = data.sort_values('order_date')
    
    last_3 = data['total_amount'].tail(3)
    print("\n--- Last 3 Values (Used for Moving Average) ---")
    print(last_3)
    
    avg = last_3.mean()
    print(f"\nCalculated Average: {avg:.2f}")
    
    # Linear Trend check
    series = data['total_amount'].reset_index(drop=True)
    x = np.arange(len(series))
    z = np.polyfit(x, series, 1)
    p = np.poly1d(z)
    trend_next = p(len(series))
    print(f"Calculated Linear Trend (Next Point): {trend_next:.2f}")

if __name__ == "__main__":
    trace_total_amount()
