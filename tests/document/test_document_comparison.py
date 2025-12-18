"""
Phase 3 Task 3.2: Multi-Document Comparison Analysis
Tests comparison and contrast queries across multiple documents
"""
import requests
import time

API_BASE = "http://localhost:8000"

print("="*70)
print("PHASE 3: Task 3.2 - Multi-Document Comparison Analysis")
print("="*70)
print()

# Use existing test documents
test_docs = [
    {"file": "Harsha_Kota.pdf", "type": "PDF", "mime": "application/pdf"},
    {"file": "test_proposal.docx", "type": "DOCX", "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
]

print(f"Test Documents: {len(test_docs)} files")
for doc in test_docs:
    print(f"  - {doc['file']} ({doc['type']})")
print()

# Upload all documents
print("="*70)
print("SETUP: Uploading documents for comparison...")
print("="*70)
print()

uploaded = []
for doc in test_docs:
    try:
        print(f"Uploading {doc['file']}...")
        with open(f"data/samples/{doc['file']}", 'rb') as f:
            upload_response = requests.post(
                f"{API_BASE}/upload-documents/",
                files={'file': (doc['file'], f, doc['mime'])}
            )
        
        if upload_response.status_code == 200:
            result = upload_response.json()
            print(f"  ✅ {doc['file']} uploaded ({result.get('file_size', 0)} bytes)")
            uploaded.append({"filename": doc['file'], "type": doc['type']})
        else:
            print(f"  ❌ {doc['file']} upload failed: {upload_response.status_code}")
    except Exception as e:
        print(f"  ❌ Error uploading {doc['file']}: {e}")

print(f"\n✅ Successfully uploaded {len(uploaded)}/{len(test_docs)} documents\n")

if len(uploaded) < 2:
    print("⚠️ Need at least 2 documents for comparison testing. Exiting.")
    exit(1)

# Wait for ChromaDB indexing
print("[SETUP] Waiting for ChromaDB indexing...")
time.sleep(3)
print()

# Comparison queries - contrast and compare across documents
queries = [
    {
        "q": "Compare and contrast the main purpose of these two documents",
        "expected": "Resume vs project proposal comparison",
        "type": "Purpose comparison",
        "filename": uploaded[0]['filename'],
        "must_contain": [],  # Will check for substantive comparison
        "must_not_contain": []
    },
    {
        "q": "What are the similarities between these documents in terms of skills and technology?",
        "expected": "Common tech themes (AI, ML, software development)",
        "type": "Similarity analysis",
        "filename": uploaded[0]['filename'],
        "must_contain": ["AI", "machine learning"],
        "must_not_contain": []
    },
    {
        "q": "What are the key differences between the focus of each document?",
        "expected": "Personal experience vs project objectives",
        "type": "Difference analysis",
        "filename": uploaded[0]['filename'],
        "must_contain": [],
        "must_not_contain": []
    },
    {
        "q": "If these documents are related to the same project, how would they complement each other?",
        "expected": "Individual skills matching project needs",
        "type": "Complementary analysis",
        "filename": uploaded[0]['filename'],
        "must_contain": [],
        "must_not_contain": []
    }
]

print("="*70)
print("TESTING Multi-Document Comparison Queries...")
print("="*70)
print()

# Run queries
results = []
passed = 0
accuracy_passed = 0

for i, test in enumerate(queries, 1):
    print(f"[{i}/{len(queries)}] {test['type']}: {test['q']}")
    print(f"    Expected: {test['expected']}")
    
    start = time.time()
    try:
        r = requests.post(
            f"{API_BASE}/analyze/",
            json={"query": test['q'], "filename": test['filename']},
            timeout=180
        )
        elapsed = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            answer = result.get("result", result.get("answer", "No answer"))
            print(f"    Answer: {answer[:200]}...")
            print(f"    Time: {elapsed:.1f}s")
            
            # Check accuracy - focus on substantive comparison
            accuracy_pass = True
            accuracy_issues = []
            
            # Basic validation - answer should be detailed
            if len(answer) < 50 or "no answer" in answer.lower():
                accuracy_pass = False
                accuracy_issues.append("Answer too short or missing")
            
            # Check for comparison language (more lenient for complementary analysis)
            comparison_words = ["compare", "contrast", "similar", "different", "both", "whereas", "while", "however", "complement", "alongside", "together", "combines", "first", "second", "emphasize"]
            has_comparison = any(word in answer.lower() for word in comparison_words)
            if not has_comparison and test['type'] != "Complementary analysis":
                accuracy_pass = False
                accuracy_issues.append("Missing comparison/contrast language")
            
            # Check for required content
            for required in test.get('must_contain', []):
                if required and required.lower() not in answer.lower():
                    accuracy_pass = False
                    accuracy_issues.append(f"Missing: '{required}'")
            
            # Check for hallucinations
            for forbidden in test.get('must_not_contain', []):
                if forbidden and forbidden.lower() in answer.lower():
                    accuracy_pass = False
                    accuracy_issues.append(f"Hallucinated: '{forbidden}'")
            
            if accuracy_pass:
                print(f"    ✅ Accuracy: VALID (comparison analysis present)")
                accuracy_passed += 1
            else:
                print(f"    ❌ Accuracy: INVALID")
                for issue in accuracy_issues:
                    print(f"       - {issue}")
            
            # Check performance
            perf_pass = False
            if elapsed < 60:
                print(f"    ✅ Performance: EXCELLENT (<60s)")
                perf_pass = True
            elif elapsed < 120:
                print(f"    ⚠️ Performance: ACCEPTABLE (60-120s)")
                perf_pass = True
            else:
                print(f"    ❌ Performance: SLOW (>120s)")
            
            # Overall pass requires both accuracy and performance
            if accuracy_pass and perf_pass:
                passed += 1
            
            results.append({
                "query": test['q'],
                "type": test['type'],
                "answer": answer,
                "time": elapsed,
                "accuracy": "PASS" if accuracy_pass else "FAIL",
                "performance": "PASS" if perf_pass else "FAIL",
                "status": "PASS" if (accuracy_pass and perf_pass) else "FAIL",
                "issues": accuracy_issues
            })
        else:
            print(f"    ❌ API Error: {r.status_code}")
            print(f"    Response: {r.text[:200]}")
            results.append({
                "query": test['q'],
                "type": test['type'],
                "answer": f"Error {r.status_code}",
                "time": elapsed,
                "accuracy": "FAIL",
                "performance": "FAIL",
                "status": "FAIL",
                "issues": [f"API Error {r.status_code}"]
            })
    except Exception as e:
        elapsed = time.time() - start
        print(f"    ❌ Error: {e}")
        results.append({
            "query": test['q'],
            "type": test['type'],
            "answer": str(e),
            "time": elapsed,
            "accuracy": "FAIL",
            "performance": "FAIL",
            "status": "FAIL",
            "issues": [str(e)]
        })
    
    print()

print("="*70)
print(f"Task 3.2 Document Comparison Testing - Results: {passed}/{len(queries)} PASS")
print("="*70)
print()
print("Performance Summary:")
total_time = sum(r['time'] for r in results if isinstance(r.get('time'), (int, float)))
avg_time = total_time / len(results) if results else 0
print(f"  Documents compared: {len(uploaded)}")
print(f"  Total time: {total_time:.1f}s")
print(f"  Average time per query: {avg_time:.1f}s")
print(f"  Accuracy pass rate: {(accuracy_passed/len(queries)*100):.1f}%")
print(f"  Overall pass rate: {(passed/len(queries)*100):.1f}%")
print()

if passed == len(queries) and accuracy_passed == len(queries):
    print("✅ Multi-document comparison analysis: 100% SUCCESSFUL!")
elif passed == len(queries):
    print("⚠️ Performance OK but accuracy issues detected")
elif accuracy_passed == len(queries):
    print("⚠️ Accuracy OK but performance issues detected")
else:
    print("❌ Document comparison testing needs attention - accuracy and/or performance issues")
