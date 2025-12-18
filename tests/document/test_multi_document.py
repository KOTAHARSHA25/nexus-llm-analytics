"""
Phase 3 Task 3.1.2: Multi-Document Analysis Testing
Tests querying across multiple PDF documents with ChromaDB
"""
import requests
import time

API_BASE = "http://localhost:8000"

print("="*70)
print("PHASE 3: Task 3.1.2 - Multi-Document Analysis")
print("="*70)
print()

# Test with 2 resume PDFs (we have 2 in samples folder)
test_docs = [
    "Harsha_Kota.pdf",
    "HARSHA Kota Resume.pdf"  # Likely a different version or similar name
]

print(f"Test Documents: {len(test_docs)} PDFs")
for doc in test_docs:
    print(f"  - {doc}")
print()

# Upload all documents
print("="*70)
print("SETUP: Uploading documents...")
print("="*70)
print()

uploaded = []
for doc in test_docs:
    try:
        print(f"Uploading {doc}...")
        with open(f'data/samples/{doc}', 'rb') as f:
            upload_response = requests.post(
                f"{API_BASE}/upload-documents/",
                files={'file': (doc, f, 'application/pdf')}
            )
        
        if upload_response.status_code == 200:
            result = upload_response.json()
            print(f"  ✅ {doc} uploaded ({result.get('file_size', 0)} bytes)")
            uploaded.append(doc)
        else:
            print(f"  ❌ {doc} upload failed: {upload_response.status_code}")
    except Exception as e:
        print(f"  ❌ Error uploading {doc}: {e}")

print(f"\n✅ Successfully uploaded {len(uploaded)}/{len(test_docs)} documents\n")

if len(uploaded) < 2:
    print("⚠️ Need at least 2 documents for multi-document testing. Exiting.")
    exit(1)

# Multi-document queries with accuracy validation
# Note: ChromaDB searches ALL indexed documents regardless of filename
# We pass first document name to trigger RAG mode
queries = [
    {
        "q": "What are the common skills across all documents?",
        "expected": "Skills mentioned in multiple resumes",
        "type": "Cross-document analysis",
        "filename": uploaded[0],  # Pass any filename to trigger RAG search
        "must_contain": ["Python", "Java"],  # Must mention these skills
        "must_not_contain": ["PyTorch", "C++"]  # Should not hallucinate these
    },
    {
        "q": "What is the name of the person in these resumes?",
        "expected": "HARSHA KOTA or Harsha Kota",
        "type": "Name extraction",
        "filename": uploaded[0],
        "must_contain": ["Harsha", "Kota"],
        "must_not_contain": ["Kotahara"]  # Common LLM error
    },
    {
        "q": "What degree is this person pursuing and at which institution?",
        "expected": "B.Tech at MLR Institute of Technology",
        "type": "Education extraction",
        "filename": uploaded[0],
        "must_contain": ["B.Tech", "MLR Institute"],
        "must_not_contain": []
    }
]

print("="*70)
print("TESTING Multi-Document Queries...")
print("="*70)
print()

# For multi-document queries, we query without specifying filename
# ChromaDB should return relevant chunks from ALL indexed documents
results = []
passed = 0
accuracy_passed = 0

for i, test in enumerate(queries, 1):
    print(f"[{i}/{len(queries)}] {test['type']}: {test['q']}")
    print(f"    Expected: {test['expected']}")
    
    start = time.time()
    try:
        # Pass a filename to trigger RAG mode - ChromaDB will search ALL documents
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
            
            # Check accuracy first
            accuracy_pass = True
            accuracy_issues = []
            
            # Check for required content
            for required in test.get('must_contain', []):
                if required.lower() not in answer.lower():
                    accuracy_pass = False
                    accuracy_issues.append(f"Missing: '{required}'")
            
            # Check for hallucinations
            for forbidden in test.get('must_not_contain', []):
                if forbidden.lower() in answer.lower():
                    accuracy_pass = False
                    accuracy_issues.append(f"Hallucinated: '{forbidden}'")
            
            if accuracy_pass:
                print(f"    ✅ Accuracy: CORRECT")
                accuracy_passed += 1
            else:
                print(f"    ❌ Accuracy: INCORRECT")
                for issue in accuracy_issues:
                    print(f"       - {issue}")
            
            # Check performance (target: <60s)
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
print(f"Task 3.1.2 Multi-Document Testing - Results: {passed}/{len(queries)} PASS")
print("="*70)
print()
print("Performance Summary:")
total_time = sum(r['time'] for r in results if isinstance(r.get('time'), (int, float)))
avg_time = total_time / len(results) if results else 0
print(f"  Documents indexed: {len(uploaded)}")
print(f"  Total time: {total_time:.1f}s")
print(f"  Average time per query: {avg_time:.1f}s")
print(f"  Accuracy pass rate: {(accuracy_passed/len(queries)*100):.1f}%")
print(f"  Overall pass rate: {(passed/len(queries)*100):.1f}%")
print()

if passed == len(queries) and accuracy_passed == len(queries):
    print("✅ Multi-document RAG retrieval: 100% ACCURATE!")
elif passed == len(queries):
    print("⚠️ Performance OK but accuracy issues detected")
elif accuracy_passed == len(queries):
    print("⚠️ Accuracy OK but performance issues detected")
else:
    print("❌ Multi-document testing needs attention - accuracy and/or performance issues")
