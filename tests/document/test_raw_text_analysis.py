"""
Phase 3 Task 3.1.4: Raw Text Input Analysis Testing
Tests direct text input (user typing in website) with ChromaDB RAG
"""
import requests
import time

API_BASE = "http://localhost:8000"

print("="*70)
print("PHASE 3: Task 3.1.4 - Raw Text Input Analysis")
print("="*70)
print()

# Sample raw text input - simulating user typing directly in website
raw_text_content = """
Project Meeting Notes - October 26, 2025

Attendees: Sarah Chen (Project Manager), Mike Johnson (Lead Developer), 
Lisa Wang (Data Scientist), Alex Kumar (UX Designer)

Key Discussion Points:
1. Database Migration
   - Completed migration to PostgreSQL
   - Performance improved by 40%
   - Zero downtime achieved

2. Machine Learning Model
   - Accuracy: 94.5% on test dataset
   - Training time reduced to 3 hours
   - Model size: 450MB
   - Ready for production deployment

3. User Interface Updates
   - New dashboard design approved
   - Mobile responsiveness implemented
   - Dark mode support added
   - Accessibility score: 98/100

4. Next Sprint Goals
   - Implement real-time notifications
   - Add export functionality (PDF, Excel)
   - Optimize API response times
   - Complete documentation

Budget Status: $45,000 spent of $60,000 allocated
Timeline: On track for December 15, 2025 launch
"""

print("Test Input: Raw text (meeting notes)")
print(f"Text length: {len(raw_text_content)} characters")
print()

# Test queries for the raw text
queries = [
    {
        "q": "Who attended the meeting?",
        "expected": "Attendee names",
        "type": "Name extraction",
        "must_contain": ["Sarah Chen", "Mike Johnson"],
        "must_not_contain": ["John Doe", "Jane Smith"]
    },
    {
        "q": "What is the machine learning model accuracy?",
        "expected": "94.5%",
        "type": "Numeric extraction",
        "must_contain": ["94.5"],
        "must_not_contain": ["90", "95", "100"]
    },
    {
        "q": "What are the next sprint goals?",
        "expected": "Sprint tasks",
        "type": "List extraction",
        "must_contain": ["notifications", "export"],
        "must_not_contain": []
    },
    {
        "q": "What is the current budget status?",
        "expected": "Budget information",
        "type": "Financial extraction",
        "must_contain": ["45,000", "60,000"],
        "must_not_contain": ["100,000"]
    }
]

print("="*70)
print("TESTING Raw Text Input...")
print("="*70)
print()

# Upload raw text
print(f"[SETUP] Submitting raw text input...")
try:
    upload_response = requests.post(
        f"{API_BASE}/upload-documents/raw-text",
        json={
            "text": raw_text_content,
            "title": "Project Meeting Notes Oct 2025",
            "description": "Meeting notes from project status discussion"
        }
    )
    
    if upload_response.status_code == 200:
        result = upload_response.json()
        filename = result.get('filename')
        print(f"  ✅ Raw text uploaded successfully")
        print(f"  Filename: {filename}")
        print(f"  Text size: {result.get('text_size', 'unknown')} characters")
        print()
    else:
        print(f"  ❌ Upload failed: {upload_response.status_code}")
        print(f"  Response: {upload_response.text}")
        exit(1)
except Exception as e:
    print(f"  ❌ Error: {e}")
    exit(1)

# Wait for ChromaDB indexing
print("[SETUP] Waiting for ChromaDB indexing...")
time.sleep(2)
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
            json={"query": test['q'], "filename": filename},
            timeout=180
        )
        elapsed = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            answer = result.get("result", result.get("answer", "No answer"))
            print(f"    Answer: {answer[:200]}...")
            print(f"    Time: {elapsed:.1f}s")
            
            # Check accuracy
            accuracy_pass = True
            accuracy_issues = []
            
            # Basic validation
            if len(answer) < 10 or "no answer" in answer.lower():
                accuracy_pass = False
                accuracy_issues.append("Answer too short or missing")
            
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
                print(f"    ✅ Accuracy: CORRECT")
                accuracy_passed += 1
            else:
                print(f"    ❌ Accuracy: INCORRECT")
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
print(f"Task 3.1.4 Raw Text Testing - Results: {passed}/{len(queries)} PASS")
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
    print("✅ Raw text RAG analysis: 100% SUCCESSFUL!")
elif passed == len(queries):
    print("⚠️ Performance OK but accuracy issues detected")
elif accuracy_passed == len(queries):
    print("⚠️ Accuracy OK but performance issues detected")
else:
    print("❌ Raw text testing needs attention - accuracy and/or performance issues")
