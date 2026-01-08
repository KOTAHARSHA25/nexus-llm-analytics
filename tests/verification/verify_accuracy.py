
import requests
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

BASE_URL = "http://localhost:8000/analyze"
DATA_DIR = os.path.join(os.getcwd(), "src", "backend", "tests", "data")

def run_test(name, filename, query, expected_keywords, context=None):
    print(f"\n--- Running Test: {name} ---")
    
    # 1. Read file path
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
        
    print(f"Testing with file: {filename}")
    print(f"Query: {query}")
    
    # 2. Construct Payload
    payload = {
        "query": query,
        "filename": file_path, # Sending absolute path for local test
        "force_refresh": True # Ensure we test logic, not cache
    }
    if context:
        payload.update(context)
        
    # 3. Send Request
    try:
        response = requests.post(BASE_URL, json=payload, timeout=60) # High timeout for LLM
        
        if response.status_code != 200:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return False
            
        result = response.json()
        result_text = str(result.get("result", "")).lower()
        print(f"Result: {result_text[:200]}...") # Print first 200 chars
        
        # 4. Verify Accuracy
        success = True
        missing = []
        for kw in expected_keywords:
            if kw.lower() not in result_text:
                success = False
                missing.append(kw)
        
        if success:
            print("✅ Accuracy Check Passed")
            return True
        else:
            print(f"❌ Accuracy Check Failed. Missing keywords: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return False

if __name__ == "__main__":
    print(f"Starting Accuracy Verification against http://localhost:8000")
    print(f"Data Directory: {DATA_DIR}")
    
    tests = [
        {
            "name": "Ecommerce - Total Sales",
            "filename": "ecommerce_comprehensive.csv", 
            "query": "What are the total sales?",
            "expected": ["total", "sales"] # LLM output varies, but should mention these
        },
        {
            "name": "Finance - Trend Analysis",
            "filename": "finance_stock_data.csv",
            "query": "Describe the trend of the open price.",
            "expected": ["trend", "open"] 
        },
        {
            "name": "Healthcare - Patient Count",
            "filename": "healthcare_patients.csv",
            "query": "How many patients are there?",
            "expected": ["count", "patient"]
        }
    ]
    
    results = []
    for t in tests:
        passed = run_test(t["name"], t["filename"], t["query"], t["expected"])
        results.append(passed)
        
    print("\n\n=== Final Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    if all(results):
        print("✅ SYSTEM ACCURACY VERIFIED")
        sys.exit(0)
    else:
        print("❌ SYSTEM ACCURACY FAILED")
        sys.exit(1)
