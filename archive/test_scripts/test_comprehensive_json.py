"""
COMPREHENSIVE JSON TEST - Complete validation before Task 1.4
Tests all JSON scenarios: simple nested, complex nested, flat structures
"""
import requests
import json
import time
import re
from pathlib import Path

API_BASE = "http://localhost:8000"

def extract_number(text):
    """Extract first number from text (handles $123.45 or 123.45 or word numbers)"""
    # Word to number mapping
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Try to find digit-based numbers first
    match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    
    # Try word numbers (case insensitive)
    text_lower = text.lower()
    for word, num in word_to_num.items():
        if word in text_lower:
            return float(num)
    
    return None

def upload_file(filepath):
    """Upload a file to the backend"""
    with open(filepath, 'rb') as f:
        files = {'file': (Path(filepath).name, f, 'application/json')}
        response = requests.post(f"{API_BASE}/upload-documents/", files=files)
        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.text}")
        return response.json()['filename']

def query_analysis(filename, query):
    """Send a query to analyze the uploaded file"""
    response = requests.post(
        f"{API_BASE}/analyze/",
        json={"query": query, "filename": filename}
    )
    if response.status_code != 200:
        raise Exception(f"Query failed: {response.text}")
    data = response.json()
    return data.get("result", data.get("answer", "No answer"))

print("=" * 80)
print("COMPREHENSIVE JSON TEST SUITE - ALL SCENARIOS")
print("=" * 80)
print()

# Test configurations
test_suites = [
    {
        "name": "SIMPLE NESTED JSON (simple.json)",
        "file": "data/samples/simple.json",
        "tests": [
            {
                "query": "What is the total sales amount?",
                "expected": 940.49,
                "tolerance": 0.01,
                "description": "Sum aggregation"
            },
            {
                "query": "How many products are there?",
                "expected": 5,
                "tolerance": 0,
                "description": "Count aggregation"
            },
            {
                "query": "What is the average sale amount?",
                "expected": 188.10,
                "tolerance": 0.01,
                "description": "Average aggregation"
            },
            {
                "query": "What is the highest sale amount?",
                "expected": 225.00,
                "tolerance": 0.01,
                "description": "Max aggregation"
            },
            {
                "query": "What is the lowest sale amount?",
                "expected": 150.00,
                "tolerance": 0.01,
                "description": "Min aggregation"
            }
        ]
    },
    {
        "name": "FLAT JSON (1.json)",
        "file": "data/samples/1.json",
        "tests": [
            {
                "query": "How many records are there?",
                "expected": 1,
                "tolerance": 0,
                "description": "Count in flat structure"
            },
            {
                "query": "What is the rollNumber?",
                "expected": None,
                "tolerance": None,
                "description": "Text field retrieval",
                "validate_type": "text_contains",
                "expected_text": "22r21a6695"
            }
        ]
    },
    {
        "name": "ANALYZE JSON (analyze.json)",
        "file": "data/samples/analyze.json",
        "tests": [
            {
                "query": "What is the total value?",
                "expected": 3,  # 1 + 2 = 3
                "tolerance": 0.01,
                "description": "Sum aggregation"
            },
            {
                "query": "How many records are there?",
                "expected": 2,
                "tolerance": 0,
                "description": "Record count"
            },
            {
                "query": "What is the average value?",
                "expected": 1.5,  # (1 + 2) / 2
                "tolerance": 0.01,
                "description": "Average calculation"
            }
        ]
    }
]

# Track overall results
all_results = []
suite_summaries = []

for suite in test_suites:
    print("=" * 80)
    print(f"TEST SUITE: {suite['name']}")
    print("=" * 80)
    print()
    
    try:
        # Upload file
        print(f"📁 Uploading {suite['file']}...")
        filename = upload_file(suite['file'])
        print(f"✅ Uploaded as: {filename}")
        print()
        
        suite_results = []
        
        # Run all tests for this file
        for i, test in enumerate(suite['tests'], 1):
            print(f"[{i}/{len(suite['tests'])}] {test['description']}")
            print(f"    Query: {test['query']}")
            
            start = time.time()
            answer = query_analysis(filename, test['query'])
            elapsed = time.time() - start
            
            actual_number = extract_number(answer)
            
            print(f"    Answer: {answer[:100]}...")
            print(f"    Extracted: {actual_number}")
            print(f"    Time: {elapsed:.2f}s")
            
            # Validate result
            correct = False
            error_msg = None
            
            if test.get('validate_type') == 'text_contains':
                # Check if answer contains expected text
                expected_text = test.get('expected_text', '')
                correct = expected_text in answer
                if not correct:
                    error_msg = f"Expected text '{expected_text}' not found in answer"
            elif test.get('validate_type') == 'number':
                correct = actual_number is not None
                if not correct:
                    error_msg = "Expected a number, got none"
            elif test.get('validate_type') == 'positive_number':
                correct = actual_number is not None and actual_number > 0
                if not correct:
                    error_msg = f"Expected positive number, got {actual_number}"
            elif test['expected'] is not None:
                if actual_number is None:
                    correct = False
                    error_msg = "No number found in answer"
                else:
                    diff = abs(actual_number - test['expected'])
                    correct = diff <= test['tolerance']
                    if not correct:
                        error_msg = f"Expected {test['expected']}, got {actual_number} (diff: {diff})"
            
            status = "✅ PASS" if correct else "❌ FAIL"
            print(f"    {status}")
            if error_msg:
                print(f"    Error: {error_msg}")
            print()
            
            suite_results.append({
                "test": test['description'],
                "query": test['query'],
                "expected": test['expected'],
                "actual": actual_number,
                "answer": answer,
                "correct": correct,
                "time": elapsed,
                "error": error_msg
            })
            
            all_results.append(suite_results[-1])
        
        # Suite summary
        passed = sum(1 for r in suite_results if r['correct'])
        total = len(suite_results)
        avg_time = sum(r['time'] for r in suite_results) / total
        
        print("-" * 80)
        print(f"SUITE SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print(f"Average time: {avg_time:.2f}s")
        print()
        
        suite_summaries.append({
            "name": suite['name'],
            "passed": passed,
            "total": total,
            "avg_time": avg_time,
            "success": passed == total
        })
        
    except Exception as e:
        print(f"❌ SUITE FAILED: {e}")
        print()
        suite_summaries.append({
            "name": suite['name'],
            "passed": 0,
            "total": len(suite['tests']),
            "avg_time": 0,
            "success": False,
            "error": str(e)
        })

# Overall summary
print("=" * 80)
print("FINAL COMPREHENSIVE TEST SUMMARY")
print("=" * 80)
print()

total_passed = sum(1 for r in all_results if r['correct'])
total_tests = len(all_results)
overall_success = total_passed == total_tests

print(f"Overall: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
print()

print("Suite Breakdown:")
for suite in suite_summaries:
    status = "✅" if suite['success'] else "❌"
    print(f"{status} {suite['name']}: {suite['passed']}/{suite['total']} passed")
    if 'error' in suite:
        print(f"   Error: {suite['error']}")
print()

if overall_success:
    print("🎉 SUCCESS! ALL JSON TESTS PASSED!")
    print("✅ System is ready to proceed to Task 1.4: Frontend Manual Testing")
else:
    print("⚠️ FAILURES DETECTED - Review failed tests:")
    print()
    for r in all_results:
        if not r['correct']:
            print(f"❌ {r['test']}")
            print(f"   Query: {r['query']}")
            print(f"   Expected: {r['expected']}")
            print(f"   Got: {r['actual']}")
            print(f"   Error: {r['error']}")
            print()
    print("🔴 DO NOT PROCEED TO TASK 1.4 UNTIL ALL TESTS PASS")

print()
print("=" * 80)
