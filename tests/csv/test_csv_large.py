"""Large CSV test for Phase 2 Task 2.1.3 - Transactions"""
import requests
import time

API_BASE = "http://localhost:8000"
filename = "transactions_large.csv"
filepath = "data/samples/csv/transactions_large.csv"

queries = [
    {"q": "What is the total transaction volume?", "expected": "$1,272,076.58"},
    {"q": "What is the average transaction amount?", "expected": "~$254.42"}
]

print("="*60)
print("PHASE 2: Task 2.1.3 - Large CSV Testing (5,000 rows)")
print("="*60)
print(f"File: {filename}")
print()

# STEP 1: Upload the file first
print("[SETUP] Uploading CSV file...")
try:
    with open(filepath, 'rb') as f:
        upload_response = requests.post(
            f"{API_BASE}/upload-documents/",
            files={'file': (filename, f, 'text/csv')}
        )
    
    if upload_response.status_code == 200:
        upload_result = upload_response.json()
        print(f"✅ Upload successful: {upload_result.get('message')}")
        print(f"   Columns: {upload_result.get('columns')}")
        print()
    else:
        print(f"❌ Upload failed: {upload_response.status_code}")
        print(f"   {upload_response.text[:200]}")
        exit(1)
except Exception as e:
    print(f"❌ Upload error: {e}")
    exit(1)

# STEP 2: Run queries
print("[TESTING] Running queries...")
print()

results = []
for i, test in enumerate(queries, 1):
    print(f"[{i}/2] Query: {test['q']}")
    print(f"    Expected: {test['expected']}")
    
    start = time.time()
    try:
        r = requests.post(
            f"{API_BASE}/analyze/",
            json={"query": test['q'], "filename": filename},
            timeout=180
        )
        elapsed = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            answer = result.get("result", result.get("answer", "No answer"))
            print(f"    Answer: {answer[:300]}...")
            print(f"    Time: {elapsed:.1f}s")
            
            if elapsed < 120:
                print(f"    ✅ Performance: PASS (<120s)")
                results.append("PASS")
            else:
                print(f"    ⚠️ Performance: SLOW (>{120}s)")
                results.append("SLOW")
        else:
            print(f"    ❌ ERROR: {r.status_code} - {r.text[:100]}")
            results.append("ERROR")
    
    except Exception as e:
        print(f"    ❌ EXCEPTION: {str(e)[:100]}")
        results.append("EXCEPTION")
    
    print()

print("="*60)
print(f"Task 2.1.3 Complete - Results: {results.count('PASS')}/2 PASS")
print("="*60)
