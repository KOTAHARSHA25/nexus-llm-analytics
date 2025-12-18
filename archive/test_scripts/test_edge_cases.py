"""
EDGE CASE VALIDATION TEST - Quick Test of All Edge Cases
Tests the robustness improvements with real edge case files
"""
import requests
import json
import time

API_BASE = "http://localhost:8000"

test_cases = [
    {
        "file": "data/samples/edge_cases/null_values.json",
        "query": "How many records have null values?",
        "description": "Null handling test"
    },
    {
        "file": "data/samples/edge_cases/special_keys.json",
        "query": "How many users are there?",
        "description": "Special characters in keys"
    },
    {
        "file": "data/samples/edge_cases/unicode_data.json",
        "query": "What is the total amount?",
        "description": "Unicode/international characters"
    },
    {
        "file": "data/samples/edge_cases/boolean_fields.json",
        "query": "How many active users?",
        "description": "Boolean fields"
    },
    {
        "file": "data/samples/edge_cases/date_formats.json",
        "query": "What is the total amount?",
        "description": "Date parsing"
    },
    {
        "file": "data/samples/edge_cases/deep_nested.json",
        "query": "What is the total budget?",
        "description": "Deep nesting (5 levels)"
    },
    {
        "file": "data/samples/edge_cases/nested_arrays.json",
        "query": "What data is in the matrix?",
        "description": "Arrays within arrays"
    },
    {
        "file": "data/samples/edge_cases/mixed_types.json",
        "query": "How many records?",
        "description": "Mixed data types"
    },
    {
        "file": "data/samples/edge_cases/large_nested_array.json",
        "query": "How many products?",
        "description": "Large nested array (150 items)"
    }
]

# Test empty files separately (expect errors)
error_cases = [
    {
        "file": "data/samples/edge_cases/empty_array.json",
        "description": "Empty array - should error gracefully"
    },
    {
        "file": "data/samples/edge_cases/empty_object.json",
        "description": "Empty object - should error gracefully"
    }
]

print("="*80)
print("EDGE CASE VALIDATION TEST")
print("="*80)
print()

results = []
errors = []

# Test regular cases
for i, test in enumerate(test_cases, 1):
    print(f"[{i}/{len(test_cases)}] Testing: {test['description']}")
    print(f"    File: {test['file']}")
    
    try:
        # Upload file
        with open(test['file'], 'rb') as f:
            files = {'file': (test['file'].split('/')[-1], f, 'application/json')}
            upload_resp = requests.post(f"{API_BASE}/upload-documents/", files=files, timeout=30)
        
        if upload_resp.status_code != 200:
            print(f"    ❌ UPLOAD FAILED: {upload_resp.text[:100]}")
            errors.append({"test": test['description'], "stage": "upload", "error": upload_resp.text})
            print()
            continue
        
        filename = upload_resp.json()['filename']
        
        # Query
        start = time.time()
        resp = requests.post(
            f"{API_BASE}/analyze/",
            json={"query": test['query'], "filename": filename},
            timeout=120
        )
        elapsed = time.time() - start
        
        if resp.status_code != 200:
            print(f"    ❌ QUERY FAILED: {resp.text[:100]}")
            errors.append({"test": test['description'], "stage": "query", "error": resp.text})
        else:
            data = resp.json()
            answer = data.get("result", data.get("answer", "No answer"))[:100]
            print(f"    ✅ SUCCESS: {answer}...")
            print(f"    Time: {elapsed:.2f}s")
            results.append({"test": test['description'], "success": True, "time": elapsed})
        
    except Exception as e:
        print(f"    ❌ ERROR: {str(e)[:100]}")
        errors.append({"test": test['description'], "stage": "exception", "error": str(e)})
    
    print()

# Test error cases
print("="*80)
print("TESTING ERROR HANDLING (Empty Files)")
print("="*80)
print()

for i, test in enumerate(error_cases, 1):
    print(f"[{i}/{len(error_cases)}] Testing: {test['description']}")
    print(f"    File: {test['file']}")
    
    try:
        with open(test['file'], 'rb') as f:
            files = {'file': (test['file'].split('/')[-1], f, 'application/json')}
            upload_resp = requests.post(f"{API_BASE}/upload-documents/", files=files, timeout=30)
        
        if upload_resp.status_code == 200:
            print(f"    ⚠️  Upload succeeded (expected error)")
        else:
            print(f"    ✅ GRACEFUL ERROR: {upload_resp.json().get('detail', 'Unknown')[:100]}")
    
    except Exception as e:
        print(f"    ❌ CRASH: {str(e)[:100]}")
        errors.append({"test": test['description'], "stage": "error_handling", "error": str(e)})
    
    print()

# Summary
print("="*80)
print("EDGE CASE TEST SUMMARY")
print("="*80)
passed = len([r for r in results if r['success']])
total = len(test_cases)
print(f"\nRegular Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
print(f"Errors encountered: {len(errors)}")

if errors:
    print("\n❌ FAILURES:")
    for err in errors:
        print(f"  - {err['test']} ({err['stage']}): {err['error'][:80]}...")
else:
    print("\n✅ All edge cases handled successfully!")

print("\n" + "="*80)
