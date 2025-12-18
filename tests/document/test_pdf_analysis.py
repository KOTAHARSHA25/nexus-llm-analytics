"""
Phase 3 Task 3.1.1: PDF Document Analysis Testing
Tests single PDF document upload, indexing, and querying via RAG
"""
import requests
import time

API_BASE = "http://localhost:8000"

print("="*70)
print("PHASE 3: Task 3.1.1 - PDF Document Analysis")
print("="*70)
print()

# Test document
test_pdf = "Harsha_Kota.pdf"
print(f"Test Document: {test_pdf}")
print()

# Test queries for resume
queries = [
    {
        "q": "What is the person's name in this resume?",
        "expected": "Harsha Kota",
        "type": "Simple extraction",
        "must_contain": ["Harsha", "Kota"],
        "must_not_contain": ["Kotahara"]
    },
    {
        "q": "What are the key skills mentioned?",
        "expected": "Programming languages, frameworks, technical skills",
        "type": "List extraction",
        "must_contain": ["Python", "Java"],
        "must_not_contain": ["PyTorch", "C++"]  # Common hallucinations
    },
    {
        "q": "What is the educational background?",
        "expected": "Degree, university, graduation year",
        "type": "Section extraction",
        "must_contain": ["B.Tech", "MLR Institute"],
        "must_not_contain": []
    },
    {
        "q": "Summarize this resume in 2-3 sentences",
        "expected": "Brief professional summary",
        "type": "Summarization",
        "must_contain": ["Computer Vision", "Deep Learning"],
        "must_not_contain": []
    }
]

print("="*70)
print("TESTING PDF Document Queries...")
print("="*70)
print()

# Upload PDF
print(f"[SETUP] Uploading {test_pdf}...")
try:
    with open(f'data/samples/{test_pdf}', 'rb') as f:
        upload_response = requests.post(
            f"{API_BASE}/upload-documents/",
            files={'file': (test_pdf, f, 'application/pdf')}
        )
    
    if upload_response.status_code == 200:
        result = upload_response.json()
        print(f"  ✅ PDF uploaded successfully")
        print(f"  File size: {result.get('file_size', 'unknown')} bytes")
        if result.get('extracted_text_path'):
            print(f"  Text extracted to: {result.get('extracted_text_path')}")
        print()
    else:
        print(f"  ❌ Upload failed: {upload_response.status_code}")
        print(f"  Response: {upload_response.text}")
        exit(1)
except Exception as e:
    print(f"  ❌ Error: {e}")
    exit(1)

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
            json={"query": test['q'], "filename": test_pdf},
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
            
            # Check performance (RAG target: <60s for simple queries)
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
print(f"Task 3.1.1 PDF Testing - Results: {passed}/{len(queries)} PASS")
print("="*70)
print()
print("Performance Summary:")
total_time = sum(r['time'] for r in results if isinstance(r.get('time'), (int, float)))
avg_time = total_time / len(results) if results else 0
print(f"  Total time: {total_time:.1f}s")
print(f"  Average time per query: {avg_time:.1f}s")
print(f"  Accuracy pass rate: {(accuracy_passed/len(queries)*100):.1f}%")
print(f"  Overall pass rate: {(passed/len(queries)*100):.1f}%")
print()

if passed == len(queries) and accuracy_passed == len(queries):
    print("✅ PDF RAG analysis: 100% ACCURATE!")
elif passed == len(queries):
    print("⚠️ Performance OK but accuracy issues detected")
elif accuracy_passed == len(queries):
    print("⚠️ Accuracy OK but performance issues detected")
else:
    print("❌ PDF testing needs attention - accuracy and/or performance issues")
