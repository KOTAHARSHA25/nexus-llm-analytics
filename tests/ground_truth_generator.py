import pandas as pd
import json
import os
from pathlib import Path

# Use absolute path relative to project root (2 levels up from tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "samples"

def calculate_ground_truth():
    results = {}
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return

    # --- E-Commerce Analysis ---
    ecommerce_path = DATA_DIR / "comprehensive_ecommerce.csv"
    if ecommerce_path.exists():
        print(f"✅ Found: {ecommerce_path}")
        df = pd.read_csv(ecommerce_path)
        # 1. Total Revenue
        if 'total_amount' in df.columns.str.lower():
            # Standardize col name finding
            col = [c for c in df.columns if c.lower() == 'total_amount' or c.lower() == 'total amount'][0]
            total_rev = df[col].sum()
            results['ecommerce_total_revenue'] = float(total_rev)
            
        # 2. Top Category by Sales
        cat_col = [c for c in df.columns if 'category' in c.lower()][0]
        amt_col = [c for c in df.columns if 'amount' in c.lower()][0]
        
        if cat_col and amt_col:
            top_cat = df.groupby(cat_col)[amt_col].sum().idxmax()
            results['ecommerce_top_category'] = top_cat
            
        # 3. Average Order Value
        if amt_col:
            avg_order = df[amt_col].mean()
            results['ecommerce_avg_order_value'] = float(avg_order)
    else:
        print(f"❌ Not Found: {ecommerce_path}")
            
    # --- University Analysis ---
    uni_path = DATA_DIR / "university_academic_data.csv"
    if uni_path.exists():
        print(f"✅ Found: {uni_path}")
        df = pd.read_csv(uni_path)
        # 4. Average GPA
        gpa_col = [c for c in df.columns if 'gpa' in c.lower()][0]
        if gpa_col:
            avg_gpa = df[gpa_col].mean()
            results['university_avg_gpa'] = float(avg_gpa)
            
        # 5. Top Major Count
        major_col = [c for c in df.columns if 'major' in c.lower()][0]
        if major_col:
            top_major = df[major_col].mode()[0]
            results['university_top_major'] = top_major
    else:
        print(f"❌ Not Found: {uni_path}")

    print("\n--- GROUND TRUTH ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    calculate_ground_truth()
