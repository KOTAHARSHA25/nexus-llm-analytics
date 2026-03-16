
"""
Test Data Generator Script
Generates diverse datasets for comprehensive backend testing.
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
TEST_DATA_DIR = SCRIPT_DIR.parent / "data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_ecommerce_data():
    print("Generating ecommerce_comprehensive.csv...")
    rows = 500
    
    categories = ['Electronics', 'Home & Garden', 'Clothing', 'Books', 'Toys']
    products = {
        'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Smart Watch', 'Tablet'],
        'Home & Garden': ['Plant Pot', 'Garden Hose', 'Chair', 'Table', 'Lamp'],
        'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Hat'],
        'Books': ['Novel', 'Textbook', 'Biography', 'Cookbook', 'Comic'],
        'Toys': ['Lego Set', 'Doll', 'Action Figure', 'Puzzle', 'Board Game']
    }
    
    data = []
    start_date = datetime(2025, 1, 1)
    
    for i in range(rows):
        cat = random.choice(categories)
        prod = random.choice(products[cat])
        qty = random.randint(1, 10)
        price = round(random.uniform(10.0, 1000.0), 2)
        
        data.append({
            'transaction_id': f'TXN-{10000+i}',
            'customer_id': f'CUST-{random.randint(100, 200)}',
            'date': (start_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
            'category': cat,
            'product_name': prod,
            'quantity': qty,
            'unit_price': price,
            'total_amount': round(qty * price, 2),
            'payment_method': random.choice(['Credit Card', 'PayPal', 'Debit Card']),
            'status': random.choice(['Completed', 'Pending', 'Returned', 'Cancelled'])
        })
        
    df = pd.DataFrame(data)
    df.to_csv(TEST_DATA_DIR / "ecommerce_comprehensive.csv", index=False)

def generate_healthcare_data():
    print("Generating healthcare_patients.csv...")
    rows = 200
    
    diagnoses = ['Hypertension', 'Diabetes', 'Asthma', 'Arthritis', 'Healthy', 'Flu']
    
    data = []
    for i in range(rows):
        admission_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 300))
        discharge_date = admission_date + timedelta(days=random.randint(1, 14))
        
        data.append({
            'patient_id': f'P-{1000+i}',
            'age': random.randint(1, 90),
            'gender': random.choice(['M', 'F']),
            'admission_date': admission_date.strftime('%Y-%m-%d'),
            'discharge_date': discharge_date.strftime('%Y-%m-%d'),
            'diagnosis': random.choice(diagnoses),
            'bill_amount': round(random.uniform(500.0, 50000.0), 2),
            'insurance_provider': random.choice(['BlueCross', 'Aetna', 'UnitedHealth', 'Medicare', None]),
            'readmission_flag': random.choice([True, False, False, False]) # Mostly False
        })
        
    df = pd.DataFrame(data)
    df.to_csv(TEST_DATA_DIR / "healthcare_patients.csv", index=False)

def generate_finance_data():
    print("Generating finance_stock_data.csv...")
    # Time series data
    dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    data = []
    for ticker in tickers:
        price = random.uniform(100, 200)
        for date in dates:
            change = random.uniform(-0.02, 0.02)
            price = price * (1 + change)
            high = price * (1 + random.uniform(0, 0.01))
            low = price * (1 - random.uniform(0, 0.01))
            volume = random.randint(100000, 10000000)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'open': round(price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': volume
            })
            
    df = pd.DataFrame(data)
    df.to_csv(TEST_DATA_DIR / "finance_stock_data.csv", index=False)

def generate_iot_data():
    print("Generating manufacturing_iot.json...")
    # Nested structure
    rows = 100
    
    data = []
    for i in range(rows):
        data.append({
            'sensor_id': f'SENS-{random.randint(1, 10)}',
            'timestamp': (datetime.now() - timedelta(minutes=i*10)).isoformat(),
            'location': {
                'factory': 'Plant-A',
                'zone': random.choice(['Zone-1', 'Zone-2', 'Zone-3'])
            },
            'readings': {
                'temperature': round(random.uniform(20.0, 80.0), 1),
                'pressure': round(random.uniform(100.0, 200.0), 1),
                'vibration': round(random.uniform(0.0, 5.0), 2)
            },
            'status': 'normal' if random.random() > 0.05 else 'error'
        })
        
    with open(TEST_DATA_DIR / "manufacturing_iot.json", 'w') as f:
        json.dump(data, f, indent=2)

def generate_edge_cases():
    print("Generating quality_issues.csv...")
    # Mixed types, nulls, infinities
    data = []
    for i in range(50):
        data.append({
            'id': i,
            'mixed_col': random.choice([1, 'two', 3.0, None, '5']),
            'null_heavy': random.choice(['value', None, None, None]),
            'dirty_text': random.choice([' Clean ', '  Dirty  ', 'Mixed Case', 'Special@#Chars']),
            'numbers_with_text': f'{random.randint(10,99)} USD'
        })
    df = pd.DataFrame(data)
    df.to_csv(TEST_DATA_DIR / "quality_issues.csv", index=False)
    
    print("Generating malformed_structure.json...")
    # Just a valid JSON but complex structure that might trip up simple parsers
    complex_data = {
        "metadata": {"version": 1.0, "source": "test"},
        "records": [
            {"id": 1, "values": [1, 2, 3]},
            {"id": 2, "values": {"a": 1, "b": 2}}, # Inconsistent type
            {"id": 3, "values": None}
        ]
    }
    with open(TEST_DATA_DIR / "malformed_structure.json", 'w') as f:
        json.dump(complex_data, f, indent=2)

if __name__ == "__main__":
    generate_ecommerce_data()
    generate_healthcare_data()
    generate_finance_data()
    generate_iot_data()
    generate_edge_cases()
    print(f"Done! Data generated in {TEST_DATA_DIR}")
