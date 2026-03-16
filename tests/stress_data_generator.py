import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path("data/stress_test")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_complex_sales_data(rows=10000):
    """
    Generates a large, messy sales dataset with:
    - 10k rows
    - Null values in critical columns (to test robustness)
    - Outliers (to test average calculation skew)
    - Mixed date formats
    """
    np.random.seed(42)
    
    dates = [datetime(2025, 1, 1) + timedelta(days=x) for x in range(365)]
    categories = ['Electronics', 'Home', 'Clothing', 'Books', 'Toys']
    regions = ['North', 'South', 'East', 'West', None] # Null region
    
    data = {
        'order_id': range(1, rows + 1),
        'date': [np.random.choice(dates) for _ in range(rows)],
        'category': [np.random.choice(categories) for _ in range(rows)],
        'region': [np.random.choice(regions) for _ in range(rows)],
        'sales_amount': np.random.exponential(scale=200, size=rows), # Skewed distribution
        'quantity': np.random.randint(1, 20, size=rows),
        'discount': np.random.choice([0, 0.05, 0.1, 0.2, None], size=rows) # Mixed nulls
    }
    
    df = pd.DataFrame(data)
    
    # Introduce outliers
    df.loc[0, 'sales_amount'] = 500000.00  # Massive outlier
    
    # Calculated expected Truths
    total_revenue = df['sales_amount'].sum()
    avg_sale = df['sales_amount'].mean()
    top_cat = df.groupby('category')['sales_amount'].sum().idxmax()
    
    # Save
    csv_path = DATA_DIR / "complex_sales_10k.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {csv_path} with {rows} rows.")
    
    return {
        "dataset": "complex_sales_10k.csv",
        "total_revenue": float(total_revenue),
        "avg_sale": float(avg_sale),
        "top_category": top_cat,
        "notes": "Includes 1 outlier (500k) and null regions"
    }

def generate_ground_truth(truths):
    with open(DATA_DIR / "stress_ground_truth.json", "w") as f:
        json.dump(truths, f, indent=4)
    print("Saved ground truth.")

if __name__ == "__main__":
    t1 = generate_complex_sales_data()
    generate_ground_truth([t1])
