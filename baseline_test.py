"""
Baseline Test Script for Accuracy Comparison
Runs identical queries against OLD and NEW versions to establish ground truth.
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
RESULTS_FILE = "baseline_results.json"

# Test queries categorized by expected complexity
TEST_QUERIES = {
    "simple": [
        {"query": "What is the total revenue?", "file": "sales_data.csv"},
        {"query": "How many rows are in this dataset?", "file": "sales_data.csv"},
        {"query": "What is the average sales?", "file": "sales_data.csv"},
        {"query": "Show the top 5 products by revenue", "file": "sales_data.csv"},
        {"query": "What is the maximum marketing_spend?", "file": "sales_data.csv"},
    ],
    "medium": [
        {"query": "Compare sales by region", "file": "sales_data.csv"},
        {"query": "What is the correlation between sales and revenue?", "file": "sales_data.csv"},
        {"query": "Group by product and show total sales for each", "file": "sales_data.csv"},
        {"query": "Which region has the highest average revenue?", "file": "sales_data.csv"},
        {"query": "Show the distribution of stress levels", "file": "StressLevelDataset.csv"},
    ],
    "complex": [
        {"query": "Analyze the relationship between anxiety_level and stress_level", "file": "StressLevelDataset.csv"},
        {"query": "What factors most strongly predict high stress levels?", "file": "StressLevelDataset.csv"},
        {"query": "Identify any outliers in the sales data", "file": "sales_data.csv"},
        {"query": "Calculate year-over-year growth pattern by region", "file": "sales_data.csv"},
        {"query": "Perform statistical analysis on marketing_spend effectiveness", "file": "sales_data.csv"},
    ]
}

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health/", timeout=5)
        return response.status_code == 200
    except:
        return False

def run_query(query: str, filename: str, timeout: int = 300) -> dict:
    """Send a query to the backend and capture response"""
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze/",
            json={"query": query, "filename": filename},
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "status_code": response.status_code,
                "result": data.get("result", ""),
                "error": data.get("error"),
                "elapsed_seconds": round(elapsed, 2),
                "raw_response": data
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "result": "",
                "error": response.text,
                "elapsed_seconds": round(elapsed, 2)
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "result": "",
            "error": f"Timeout after {timeout}s",
            "elapsed_seconds": timeout
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "result": "",
            "error": str(e),
            "elapsed_seconds": time.time() - start_time
        }

def run_baseline_tests(version_name: str) -> dict:
    """Run all test queries and collect results"""
    print(f"\n{'='*60}")
    print(f"BASELINE TEST: {version_name}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    # Check backend health first
    print("Checking backend health...")
    if not check_backend_health():
        print("ERROR: Backend is not responding at", BACKEND_URL)
        print("Make sure the backend is running and try again.")
        return None
    print("Backend is healthy!\n")
    
    results = {
        "version": version_name,
        "timestamp": datetime.now().isoformat(),
        "backend_url": BACKEND_URL,
        "categories": {}
    }
    
    total_queries = sum(len(queries) for queries in TEST_QUERIES.values())
    current = 0
    
    for category, queries in TEST_QUERIES.items():
        print(f"\n--- {category.upper()} QUERIES ---")
        results["categories"][category] = []
        
        for test in queries:
            current += 1
            query = test["query"]
            filename = test["file"]
            
            print(f"\n[{current}/{total_queries}] Query: {query[:50]}...")
            print(f"    File: {filename}")
            
            result = run_query(query, filename)
            
            test_result = {
                "query": query,
                "filename": filename,
                **result
            }
            results["categories"][category].append(test_result)
            
            if result["success"]:
                # Truncate result for display
                result_preview = result["result"][:100] + "..." if len(result["result"]) > 100 else result["result"]
                print(f"    ✓ Success ({result['elapsed_seconds']}s)")
                print(f"    Result: {result_preview}")
            else:
                print(f"    ✗ Failed ({result['elapsed_seconds']}s)")
                print(f"    Error: {result['error'][:100]}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_success = 0
    total_failed = 0
    total_time = 0
    
    for category, cat_results in results["categories"].items():
        success = sum(1 for r in cat_results if r["success"])
        failed = len(cat_results) - success
        avg_time = sum(r["elapsed_seconds"] for r in cat_results) / len(cat_results)
        
        total_success += success
        total_failed += failed
        total_time += sum(r["elapsed_seconds"] for r in cat_results)
        
        print(f"{category.upper()}: {success}/{len(cat_results)} passed, avg {avg_time:.1f}s")
    
    print(f"\nTOTAL: {total_success}/{total_success + total_failed} passed")
    print(f"Total time: {total_time:.1f}s")
    
    results["summary"] = {
        "total_queries": total_success + total_failed,
        "passed": total_success,
        "failed": total_failed,
        "total_time_seconds": round(total_time, 2)
    }
    
    return results

def save_results(results: dict, filename: str):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")

def main():
    import sys
    
    version_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"baseline_{version_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print(f"Running baseline tests for version: {version_name}")
    
    results = run_baseline_tests(version_name)
    
    if results:
        save_results(results, output_file)
        print(f"\n✓ Baseline test complete for {version_name}")
        return 0
    else:
        print(f"\n✗ Baseline test failed - backend not available")
        return 1

if __name__ == "__main__":
    exit(main())
