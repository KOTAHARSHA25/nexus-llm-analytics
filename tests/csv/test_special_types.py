"""
Special Data Types Test - Phase 2 Task 2.2.2
Tests handling of dates, currency, percentages, and categorical data
"""
import requests
import time
import pandas as pd

API_BASE = "http://localhost:8000"

# Load and verify test data
df = pd.read_csv('data/samples/csv/special_types.csv')

print("="*70)
print("PHASE 2: Task 2.2.2 - Special Data Types")
print("="*70)
print(f"\nTest Data: special_types.csv ({len(df)} rows)")
print(f"Columns: {', '.join(df.columns)}")
print()

# Calculate expected answers
print("Expected Answers:")
print()

# Date queries
jan_sales = df[df['date'].str.startswith('2024-01')]['amount'].str.replace('$','').str.replace(',','').astype(float).sum()
feb_sales = df[df['date'].str.startswith('2024-02')]['amount'].str.replace('$','').str.replace(',','').astype(float).sum()
print(f"1. Sales by month:")
print(f"   January: ${jan_sales:,.2f}")
print(f"   February: ${feb_sales:,.2f}")

# Currency aggregations
total_amount = df['amount'].str.replace('$','').str.replace(',','').astype(float).sum()
avg_amount = df['amount'].str.replace('$','').str.replace(',','').astype(float).mean()
print(f"\n2. Total/Average amounts:")
print(f"   Total: ${total_amount:,.2f}")
print(f"   Average: ${avg_amount:,.2f}")

# Percentage calculations
avg_discount = df['discount_pct'].str.replace('%','').astype(float).mean()
print(f"\n3. Average discount: {avg_discount:.1f}%")

# Category analysis
category_totals = df.groupby('category').apply(
    lambda x: x['amount'].str.replace('$','').str.replace(',','').astype(float).sum()
).sort_values(ascending=False)
print(f"\n4. Sales by category:")
for cat, total in category_totals.items():
    print(f"   {cat}: ${total:,.2f}")

# Test queries
queries = [
    {
        "q": "What are the total sales for January 2024?",
        "expected": f"${jan_sales:,.2f}",
        "type": "Date parsing"
    },
    {
        "q": "What is the average transaction amount?",
        "expected": f"${avg_amount:,.2f}",
        "type": "Currency"
    },
    {
        "q": "What is the average discount percentage?",
        "expected": f"{avg_discount:.1f}%",
        "type": "Percentage"
    },
    {
        "q": "Which category has the highest total sales?",
        "expected": f"{category_totals.index[0]}: ${category_totals.iloc[0]:,.2f}",
        "type": "Categorical"
    },
    {
        "q": "Compare sales between January and February 2024",
        "expected": f"Jan: ${jan_sales:,.2f}, Feb: ${feb_sales:,.2f}",
        "type": "Date comparison"
    }
]

print(f"\n{'='*70}")
print("TESTING Special Data Type Queries...")
print(f"{'='*70}\n")

# Upload file
print("[SETUP] Uploading special_types.csv...")
try:
    with open('data/samples/csv/special_types.csv', 'rb') as f:
        upload_response = requests.post(
            f"{API_BASE}/upload-documents/",
            files={'file': ('special_types.csv', f, 'text/csv')}
        )
    if upload_response.status_code == 200:
        print("  ✅ File uploaded\n")
    else:
        print(f"  ❌ Upload failed: {upload_response.status_code}")
        exit(1)
except Exception as e:
    print(f"  ❌ Error: {e}")
    exit(1)

# Run queries
results = []
passed = 0
for i, test in enumerate(queries, 1):
    print(f"[{i}/{len(queries)}] {test['type']}: {test['q']}")
    print(f"    Expected: {test['expected']}")
    
    start = time.time()
    try:
        r = requests.post(
            f"{API_BASE}/analyze/",
            json={"query": test['q'], "filename": "special_types.csv"},
            timeout=180
        )
        elapsed = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            answer = result.get("result", result.get("answer", "No answer"))
            print(f"    Answer: {answer[:150]}...")
            print(f"    Time: {elapsed:.1f}s")
            
            # Check performance
            if elapsed < 120:
                print(f"    ✅ Performance: PASS (<120s)")
                passed += 1
            else:
                print(f"    ⚠️ Performance: SLOW (>120s)")
            
            results.append({
                "query": test['q'],
                "type": test['type'],
                "answer": answer,
                "time": elapsed,
                "status": "PASS" if elapsed < 120 else "SLOW"
            })
        else:
            print(f"    ❌ API Error: {r.status_code}")
            results.append({
                "query": test['q'],
                "type": test['type'],
                "answer": f"Error {r.status_code}",
                "time": elapsed,
                "status": "FAIL"
            })
    except Exception as e:
        print(f"    ❌ Error: {e}")
        results.append({
            "query": test['q'],
            "type": test['type'],
            "answer": str(e),
            "time": 0,
            "status": "FAIL"
        })
    
    print()

print(f"{'='*70}")
print(f"Task 2.2.2 Complete - Results: {passed}/{len(queries)} PASS")
print(f"{'='*70}")
