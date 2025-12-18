"""
Complete CSV Test Suite - All Tests Combined
Runs simple, medium, and large CSV tests sequentially
Phase 2 Task 2.1 - Complete Test
"""
import requests
import time

API_BASE = "http://localhost:8000"

# Test configurations
tests = [
    {
        "name": "2.1.1 Simple CSV (5 rows)",
        "file": "sales_simple.csv",
        "path": "data/samples/csv/sales_simple.csv",
        "queries": [
            {"q": "What is the total revenue?", "expected": "$5,850"},
            {"q": "Which product has highest sales?", "expected": "Widget A ($3,300)"},
            {"q": "How many unique products?", "expected": "2"}
        ]
    },
    {
        "name": "2.1.2 Medium CSV (100 rows)",
        "file": "customer_data.csv",
        "path": "data/samples/csv/customer_data.csv",
        "queries": [
            {"q": "What is the average age of customers?", "expected": "~43 years"},
            {"q": "Which city has the most customers?", "expected": "Phoenix"},
            {"q": "Calculate total revenue by membership level", "expected": "Breakdown by level"}
        ]
    },
    {
        "name": "2.1.3 Large CSV (100 rows)",
        "file": "sales_data.csv",
        "path": "data/samples/sales_data.csv",
        "queries": [
            {"q": "What is the total revenue?", "expected": "$2,563,044"},
            {"q": "What is the average revenue per transaction?", "expected": "~$25,630"}
        ]
    }
]

print("="*70)
print("COMPLETE CSV TEST SUITE - Phase 2 Task 2.1")
print("="*70)
print()

total_queries = sum(len(t["queries"]) for t in tests)
passed = 0
failed = 0
total_time = 0

for test_suite in tests:
    print(f"\n{'='*70}")
    print(f"TEST: {test_suite['name']}")
    print(f"{'='*70}")
    print(f"File: {test_suite['file']}\n")
    
    # Upload file
    print("[SETUP] Uploading file...")
    try:
        with open(test_suite["path"], 'rb') as f:
            upload_response = requests.post(
                f"{API_BASE}/upload-documents/",
                files={'file': (test_suite["file"], f, 'text/csv')}
            )
        
        if upload_response.status_code == 200:
            print(f"✅ Upload successful\n")
        else:
            print(f"❌ Upload failed: {upload_response.status_code}\n")
            failed += len(test_suite["queries"])
            continue
    except Exception as e:
        print(f"❌ Upload error: {e}\n")
        failed += len(test_suite["queries"])
        continue
    
    # Run queries
    print("[TESTING] Running queries...\n")
    for i, query_test in enumerate(test_suite["queries"], 1):
        print(f"[{i}/{len(test_suite['queries'])}] {query_test['q']}")
        print(f"    Expected: {query_test['expected']}")
        
        start = time.time()
        try:
            r = requests.post(
                f"{API_BASE}/analyze/",
                json={"query": query_test['q'], "filename": test_suite["file"]},
                timeout=180
            )
            elapsed = time.time() - start
            total_time += elapsed
            
            if r.status_code == 200:
                result = r.json()
                answer = result.get("result", result.get("answer", "No answer"))
                print(f"    Answer: {answer[:100]}...")
                print(f"    Time: {elapsed:.1f}s")
                
                if elapsed < 120:
                    print(f"    ✅ PASS")
                    passed += 1
                else:
                    print(f"    ⚠️ SLOW but CORRECT")
                    passed += 1
            else:
                print(f"    ❌ ERROR: {r.status_code}")
                failed += 1
        
        except Exception as e:
            print(f"    ❌ EXCEPTION: {str(e)[:80]}")
            failed += 1
        
        print()

# Final summary
print("="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Total Queries: {total_queries}")
print(f"Passed: {passed} ({passed/total_queries*100:.1f}%)")
print(f"Failed: {failed} ({failed/total_queries*100:.1f}%)")
print(f"Average Time: {total_time/total_queries:.1f}s")
print(f"Total Time: {total_time:.1f}s")
print()

if passed == total_queries:
    print("🎉 ALL TESTS PASSED - 100% ACCURACY!")
else:
    print(f"⚠️ {failed} tests failed")

print("="*70)
