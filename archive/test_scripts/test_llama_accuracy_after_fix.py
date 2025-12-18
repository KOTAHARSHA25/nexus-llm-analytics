"""
Test llama3.1:8b accuracy with FIXED optimizer
Check numerical accuracy, not just string matching
"""
import requests
import json
import time
import re

API_BASE = "http://localhost:8000"

def extract_number(text):
    """Extract first number from text (handles $123.45 or 123.45)"""
    match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    return None

# Upload simple.json
print("=" * 70)
print("TESTING LLAMA3.1:8B ACCURACY WITH FIXED OPTIMIZER")
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

# Test queries with numerical validation
queries = [
    {
        "query": "What is the total sales amount?",
        "expected_number": 940.49,
        "tolerance": 0.01
    },
    {
        "query": "How many products are there?",
        "expected_number": 5,
        "tolerance": 0
    },
    {
        "query": "What is the average sale amount?",
        "expected_number": 188.10,
        "tolerance": 0.01
    }
]

results = []

for i, test in enumerate(queries, 1):
    print(f"[{i}] Query: {test['query']}")
    print(f"    Expected: {test['expected_number']}")
    
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
            "expected": test["expected_number"],
            "actual": None,
            "correct": False,
            "time": elapsed
        })
        continue
    
    data = resp.json()
    answer = data.get("result", data.get("answer", "No answer"))
    
    # Extract number from answer
    actual_number = extract_number(answer)
    
    print(f"    Actual answer: {answer[:80]}...")
    print(f"    Extracted number: {actual_number}")
    print(f"    Time: {elapsed:.2f}s")
    
    # Check numerical correctness
    correct = False
    if actual_number is not None:
        diff = abs(actual_number - test['expected_number'])
        correct = diff <= test['tolerance']
    
    status = "✅ CORRECT" if correct else "❌ WRONG"
    print(f"    {status}")
    print()
    
    results.append({
        "query": test["query"],
        "expected": test["expected_number"],
        "actual": actual_number,
        "correct": correct,
        "time": elapsed,
        "answer": answer
    })

# Summary
print("=" * 70)
print("LLAMA3.1:8B ACCURACY SUMMARY")
print("=" * 70)
correct_count = sum(1 for r in results if r["correct"])
total = len(results)
accuracy = (correct_count / total * 100) if total > 0 else 0

print(f"Numerical Accuracy: {correct_count}/{total} ({accuracy:.1f}%)")
print(f"Model: llama3.1:8b")
print(f"Average response time: {sum(r['time'] for r in results) / total:.2f}s")
print()

if accuracy == 100:
    print("🎉 SUCCESS! llama3.1:8b is 100% numerically accurate!")
else:
    print("⚠️  Some numerical errors detected:")
    print()
    for r in results:
        if not r["correct"]:
            print(f"   ❌ {r['query']}")
            print(f"      Expected: {r['expected']}")
            print(f"      Got: {r['actual']}")
            print(f"      Answer: {r['answer'][:100]}...")
