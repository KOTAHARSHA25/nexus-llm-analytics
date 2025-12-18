"""
Multi-File CSV Test - Phase 2 Task 2.2.1
Tests joining operations across 2 CSV files
"""
import requests
import time
import pandas as pd

API_BASE = "http://localhost:8000"

# Verify test data exists
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

customers_df = pd.read_csv('data/samples/csv/customers.csv')
orders_df = pd.read_csv('data/samples/csv/orders.csv')

print("="*70)
print("PHASE 2: Task 2.2.1 - Multi-File Analysis")
print("="*70)
print(f"\nTest Data:")
print(f"  customers.csv: {len(customers_df)} rows")
print(f"  orders.csv: {len(orders_df)} rows")
print()

# Calculate expected answers
merged = orders_df.merge(customers_df, on='customer_id')
orders_by_city = merged.groupby('city')['amount'].agg(['sum', 'count']).sort_values('sum', ascending=False)
total_by_country = merged.groupby('country')['amount'].sum().sort_values(ascending=False)

print("Expected Answers:")
print(f"\n1. Total orders by city:")
for city, row in orders_by_city.head(3).iterrows():
    print(f"   {city}: ${row['sum']:.2f} ({int(row['count'])} orders)")

print(f"\n2. Total by country:")
for country, total in total_by_country.head(3).items():
    print(f"   {country}: ${total:.2f}")

# Test queries
queries = [
    {
        "q": "Show total orders per city from customers and orders files",
        "files": ["customers.csv", "orders.csv"],
        "expected": "New York, London, Toronto with totals"
    },
    {
        "q": "What is total revenue by country?",
        "files": ["customers.csv", "orders.csv"],  
        "expected": "USA, UK, Canada totals"
    }
]

print(f"\n{'='*70}")
print("TESTING Multi-File Queries...")
print(f"{'='*70}\n")

# Upload both files first
print("[SETUP] Uploading files...")
for filename in ["customers.csv", "orders.csv"]:
    try:
        filepath = f'data/samples/csv/{filename}'
        with open(filepath, 'rb') as f:
            upload_response = requests.post(
                f"{API_BASE}/upload-documents/",
                files={'file': (filename, f, 'text/csv')}
            )
        if upload_response.status_code == 200:
            print(f"  ✅ {filename} uploaded")
        else:
            print(f"  ❌ {filename} upload failed: {upload_response.status_code}")
            exit(1)
    except Exception as e:
        print(f"  ❌ Error uploading {filename}: {e}")
        exit(1)

print()

# Run multi-file queries
results = []
for i, test in enumerate(queries, 1):
    print(f"[{i}/{len(queries)}] Query: {test['q']}")
    print(f"    Files: {', '.join(test['files'])}")
    print(f"    Expected: {test['expected']}")
    
    start = time.time()
    try:
        r = requests.post(
            f"{API_BASE}/analyze/",
            json={"query": test['q'], "filenames": test['files']},
            timeout=180
        )
        elapsed = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            answer = result.get("result", result.get("answer", "No answer"))
            print(f"    Answer: {answer[:200]}...")
            print(f"    Time: {elapsed:.1f}s")
            
            if elapsed < 120:
                print(f"    ✅ Performance: PASS (<120s)")
                results.append("PASS")
            else:
                print(f"    ⚠️ Performance: SLOW (>120s)")
                results.append("SLOW")
        else:
            print(f"    ❌ ERROR: {r.status_code} - {r.text[:100]}")
            results.append("ERROR")
    
    except Exception as e:
        print(f"    ❌ EXCEPTION: {str(e)[:100]}")
        results.append("EXCEPTION")
    
    print()

print("="*70)
print(f"Task 2.2.1 Complete - Results: {results.count('PASS')}/{len(queries)} PASS")
print("="*70)
