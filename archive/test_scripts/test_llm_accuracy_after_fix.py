"""
Test LLM accuracy with FIXED optimizer
After fixing the data optimizer bug, test if both models now give correct answers
"""
import requests
import json
import time

API_BASE = "http://localhost:8000"

# Upload simple.json
print("=" * 70)
print("TESTING LLM ACCURACY WITH FIXED OPTIMIZER")
print("=" * 70)
print()

# Upload the file
print("[1] Uploading simple.json...")
with open("data/samples/simple.json", "rb") as f:
    files = {"file": ("simple.json", f, "application/json")}
    upload_resp = requests.post(f"{API_BASE}/upload-documents/", files=files)

if upload_resp.status_code != 200:
    print(f"❌ Upload failed: {upload_resp.text}")
    exit(1)

upload_data = upload_resp.json()
filename = upload_data["filename"]
print(f"✅ File uploaded: {filename}")
print()

# Test queries with CORRECT expected answers
queries = [
    {
        "query": "What is the total sales amount?",
        "expected": "$940.49",
        "tolerance": 0.01
    },
    {
        "query": "How many products are there?",
        "expected": "5",
        "exact": True
    },
    {
        "query": "What is the average sale amount?",
        "expected": "$188.10",
        "tolerance": 0.01
    }
]

results = []

for i, test in enumerate(queries, 1):
    print(f"[{i}] Query: {test['query']}")
    print(f"    Expected: {test['expected']}")
    
    start = time.time()
    resp = requests.post(
        f"{API_BASE}/analyze/",
        json={
            "query": test["query"],
            "filename": filename
        }
    )
    elapsed = time.time() - start
    
    if resp.status_code != 200:
        print(f"    ❌ API Error: {resp.text}")
        results.append({
            "query": test["query"],
            "expected": test["expected"],
            "actual": "ERROR",
            "correct": False,
            "time": elapsed
        })
        continue
    
    data = resp.json()
    answer = data.get("result", data.get("answer", "No answer"))  # Check 'result' first, then 'answer'
    
    print(f"    Actual: {answer[:100]}...")  # Show first 100 chars
    print(f"    Time: {elapsed:.2f}s")
    
    # Check correctness
    correct = False
    if test.get("exact"):
        correct = test["expected"] in answer
    else:
        # Check if answer contains the expected value
        correct = test["expected"] in answer
    
    status = "✅ CORRECT" if correct else "❌ WRONG"
    print(f"    {status}")
    print()
    
    results.append({
        "query": test["query"],
        "expected": test["expected"],
        "actual": answer,
        "correct": correct,
        "time": elapsed
    })

# Summary
print("=" * 70)
print("ACCURACY SUMMARY")
print("=" * 70)
correct_count = sum(1 for r in results if r["correct"])
total = len(results)
accuracy = (correct_count / total * 100) if total > 0 else 0

print(f"Correct: {correct_count}/{total} ({accuracy:.1f}%)")
print(f"Model: phi3:mini")
print(f"Average response time: {sum(r['time'] for r in results) / total:.2f}s")
print()

if accuracy == 100:
    print("🎉 SUCCESS! All answers are correct after fixing the optimizer!")
    print("   The problem was CODE BUG in data_optimizer.py, NOT the model!")
else:
    print("⚠️  Some answers still wrong. Further investigation needed.")
    print()
    for r in results:
        if not r["correct"]:
            print(f"   ❌ {r['query']}")
            print(f"      Expected: {r['expected']}")
            print(f"      Got: {r['actual']}")
