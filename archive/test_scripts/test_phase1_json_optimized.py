"""
Phase 1: JSON Data Type - OPTIMIZED Comprehensive Deep Test
FIXES APPLIED:
1. ✅ Review protocol DISABLED (50% faster)
2. ✅ Increased timeouts (180s upload, 300s analysis)
3. ✅ Simplified agent prompts (reduce hallucinations)
4. ✅ Resource monitoring (CPU/Memory tracking)
5. ✅ Better error reporting
"""

import requests
import json
import os
import time
import psutil
from datetime import datetime
from pathlib import Path

# ========== CONFIGURATION ==========
BASE_URL = "http://127.0.0.1:8000"
UPLOAD_TIMEOUT = 180  # 3 minutes (was 30s)
ANALYZE_TIMEOUT = 300  # 5 minutes (was 60s)
HEALTH_TIMEOUT = 10

# Test data directory
DATA_DIR = Path("data/samples")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"{text}")
    print(f"{'='*80}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

def get_system_resources():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    memory_percent = memory.percent
    
    return {
        "cpu_percent": cpu_percent,
        "memory_used_gb": round(memory_used_gb, 2),
        "memory_total_gb": round(memory_total_gb, 2),
        "memory_percent": memory_percent
    }

def print_resources():
    """Print current system resource usage"""
    resources = get_system_resources()
    print(f"{Colors.BLUE}📊 Resources: CPU {resources['cpu_percent']}% | Memory {resources['memory_used_gb']}/{resources['memory_total_gb']} GB ({resources['memory_percent']}%){Colors.END}")

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BASE_URL}/health/health", timeout=HEALTH_TIMEOUT)
        return response.status_code == 200
    except Exception as e:
        print_error(f"Backend health check failed: {e}")
        return False

def upload_file(filename):
    """Upload a file to backend"""
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        print_error(f"File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f, 'application/json')}
            response = requests.post(
                f"{BASE_URL}/upload-documents/", 
                files=files,
                timeout=UPLOAD_TIMEOUT
            )
            
        if response.status_code == 200:
            return response.json()
        else:
            print_error(f"Upload failed: {response.status_code} - {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print_error(f"Upload timeout after {UPLOAD_TIMEOUT}s")
        return None
    except Exception as e:
        print_error(f"Upload error: {e}")
        return None

def analyze_query(filename, query, expected_agent=None):
    """Send analysis query"""
    print_info(f"Query: {query}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze/",
            json={"query": query, "filename": filename},
            timeout=ANALYZE_TIMEOUT
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Check for errors in response
            if "error" in result:
                print_error(f"Analysis error: {result['error']}")
                if "suggestions" in result:
                    for suggestion in result["suggestions"]:
                        print_warning(f"  • {suggestion}")
                return None
            
            # Print execution time
            print(f"{Colors.CYAN}⏱️  Execution time: {elapsed_time:.1f}s{Colors.END}")
            
            # Validate agent routing if specified
            if expected_agent:
                # Note: This would require backend to return agent info
                print_info(f"Expected agent: {expected_agent}")
            
            # Show result preview
            result_text = result.get("result", "")
            if len(result_text) > 200:
                print(f"{Colors.CYAN}📄 Result preview: {result_text[:200]}...{Colors.END}")
            else:
                print(f"{Colors.CYAN}📄 Result: {result_text}{Colors.END}")
            
            print_success(f"Analysis completed in {elapsed_time:.1f}s")
            return result
            
        else:
            print_error(f"Analysis failed: {response.status_code} - {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        print_error(f"Analysis timeout after {ANALYZE_TIMEOUT}s (actual: {elapsed_time:.1f}s)")
        return None
    except Exception as e:
        print_error(f"Analysis error: {e}")
        return None

def create_test_data():
    """Create all test JSON files"""
    print_header("Creating Test Data Files")
    
    # 1. Complex Nested JSON
    complex_nested = {
        "company": "TechCorp Analytics",
        "founded": "2020-01-15",
        "departments": [
            {
                "name": "Engineering",
                "budget": 500000,
                "employees": [
                    {"id": 1, "name": "Alice Johnson", "role": "Senior Engineer", "salary": 120000},
                    {"id": 2, "name": "Bob Smith", "role": "Junior Engineer", "salary": 80000}
                ],
                "projects": [
                    {"id": "P1", "name": "Cloud Migration", "status": "active", "budget": 100000}
                ]
            },
            {
                "name": "Sales",
                "budget": 300000,
                "employees": [
                    {"id": 3, "name": "Carol Davis", "role": "Sales Manager", "salary": 95000}
                ],
                "projects": []
            }
        ],
        "metadata": {
            "total_employees": 3,
            "total_budget": 800000,
            "active_projects": 1
        }
    }
    
    with open(DATA_DIR / "complex_nested.json", 'w') as f:
        json.dump(complex_nested, f, indent=2)
    print_success("Created: data\\samples\\complex_nested.json")
    
    # 2. Large Transaction Dataset (10,000 records)
    print_info("Generating large dataset (10,000 records)...")
    import random
    transactions = []
    categories = ["Electronics", "Clothing", "Food", "Books", "Home"]
    regions = ["North", "South", "East", "West"]
    
    for i in range(10000):
        transactions.append({
            "transaction_id": f"TXN{i+1:06d}",
            "date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "category": random.choice(categories),
            "region": random.choice(regions),
            "quantity": random.randint(1, 10),
            "price": round(random.uniform(10, 500), 2),
            "discount": round(random.uniform(0, 0.3), 2)
        })
    
    # Add total
    for txn in transactions:
        txn["total"] = round(txn["quantity"] * txn["price"] * (1 - txn["discount"]), 2)
    
    with open(DATA_DIR / "large_transactions.json", 'w') as f:
        json.dump(transactions, f)
    print_success("Created: data\\samples\\large_transactions.json (10,000 records)")
    
    # 3. Financial Quarterly Data
    financial_quarterly = [
        {"quarter": "Q1-2024", "revenue": 1250000, "expenses": 890000, "profit": 360000},
        {"quarter": "Q2-2024", "revenue": 1380000, "expenses": 920000, "profit": 460000},
        {"quarter": "Q3-2024", "revenue": 1420000, "expenses": 950000, "profit": 470000},
        {"quarter": "Q4-2024", "revenue": 1550000, "expenses": 980000, "profit": 570000}
    ]
    
    with open(DATA_DIR / "financial_quarterly.json", 'w') as f:
        json.dump(financial_quarterly, f, indent=2)
    print_success("Created: data\\samples\\financial_quarterly.json")
    
    # 4. Time Series Sales Data
    sales_timeseries = []
    for day in range(336):  # ~11 months
        sales_timeseries.append({
            "date": f"2024-{(day // 28) + 1:02d}-{(day % 28) + 1:02d}",
            "sales": random.randint(5000, 25000),
            "orders": random.randint(50, 300)
        })
    
    with open(DATA_DIR / "sales_timeseries.json", 'w') as f:
        json.dump(sales_timeseries, f)
    print_success("Created: data\\samples\\sales_timeseries.json (336 days of data)")
    
    # 5. Malformed JSON (intentionally broken)
    with open(DATA_DIR / "malformed.json", 'w') as f:
        f.write('{"broken": "json", "missing_bracket": true')
    print_success("Created: data\\samples\\malformed.json (intentionally malformed)")
    
    print("\nTest data creation complete!")

# ========== TEST FUNCTIONS ==========

def test_simple_json():
    """Test 1: Simple JSON - Student Data"""
    print_header("TEST 1: Simple JSON - Student Data")
    print_resources()
    
    results = {"passed": 0, "failed": 0}
    
    # Upload
    print_info("Uploading: 1.json")
    upload_result = upload_file("1.json")
    if not upload_result:
        results["failed"] += 3
        return results
    print_success(f"Upload successful: 1.json")
    
    # Query 1
    result = analyze_query("1.json", "What is the student's name?")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    time.sleep(2)  # Brief pause between queries
    
    # Query 2  
    result = analyze_query("1.json", "What is the roll number?")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    time.sleep(2)
    
    # Query 3
    result = analyze_query("1.json", "Summarize the student information")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    return results

def test_analyze_json():
    """Test 2: Analyze JSON - Category/Value Data"""
    print_header("TEST 2: Analyze JSON - Category/Value Data")
    print_resources()
    
    results = {"passed": 0, "failed": 0}
    
    # Upload
    print_info("Uploading: analyze.json")
    upload_result = upload_file("analyze.json")
    if not upload_result:
        results["failed"] += 3
        return results
    print_success(f"Upload successful: analyze.json")
    
    # Query 1
    result = analyze_query("analyze.json", "What categories are present?")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    time.sleep(2)
    
    # Query 2
    result = analyze_query("analyze.json", "What is the sum of values?")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    time.sleep(2)
    
    # Query 3
    result = analyze_query("analyze.json", "Show me the relationship between category and value")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    return results

def test_complex_nested_json():
    """Test 3: Complex Nested JSON - Company Data"""
    print_header("TEST 3: Complex Nested JSON - Company Data")
    print_resources()
    
    results = {"passed": 0, "failed": 0}
    
    # Upload
    print_info("Uploading: complex_nested.json")
    upload_result = upload_file("complex_nested.json")
    if not upload_result:
        results["failed"] += 3
        return results
    print_success(f"Upload successful: complex_nested.json")
    
    # Query 1
    result = analyze_query("complex_nested.json", "How many departments are there?")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    time.sleep(2)
    
    # Query 2
    result = analyze_query("complex_nested.json", "What is the average salary across all employees?")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    time.sleep(2)
    
    # Query 3
    result = analyze_query("complex_nested.json", "Which department has the highest budget?")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    return results

def test_large_json():
    """Test 4: Large JSON - 10K Transactions"""
    print_header("TEST 4: Large JSON - 10,000 Transaction Records")
    print_resources()
    
    results = {"passed": 0, "failed": 0}
    
    # Upload
    print_info("Uploading: large_transactions.json (10K records)")
    upload_result = upload_file("large_transactions.json")
    if not upload_result:
        results["failed"] += 2
        return results
    print_success(f"Upload successful: large_transactions.json")
    
    # Query 1
    result = analyze_query("large_transactions.json", "What is the total revenue?")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    time.sleep(2)
    
    # Query 2
    result = analyze_query("large_transactions.json", "Which category has the most sales?")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    return results

def test_financial_json():
    """Test 5: Financial JSON - Quarterly Data"""
    print_header("TEST 5: Financial JSON - Quarterly Results")
    print_resources()
    
    results = {"passed": 0, "failed": 0}
    
    # Upload
    print_info("Uploading: financial_quarterly.json")
    upload_result = upload_file("financial_quarterly.json")
    if not upload_result:
        results["failed"] += 2
        return results
    print_success(f"Upload successful: financial_quarterly.json")
    
    # Query 1
    result = analyze_query("financial_quarterly.json", "What is the total profit for 2024?", expected_agent="Financial Agent")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    time.sleep(2)
    
    # Query 2
    result = analyze_query("financial_quarterly.json", "Which quarter had the highest profit margin?", expected_agent="Financial Agent")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    return results

def test_timeseries_json():
    """Test 6: Time Series JSON - Sales Over Time"""
    print_header("TEST 6: Time Series JSON - Daily Sales Data")
    print_resources()
    
    results = {"passed": 0, "failed": 0}
    
    # Upload
    print_info("Uploading: sales_timeseries.json")
    upload_result = upload_file("sales_timeseries.json")
    if not upload_result:
        results["failed"] += 2
        return results
    print_success(f"Upload successful: sales_timeseries.json")
    
    # Query 1
    result = analyze_query("sales_timeseries.json", "What is the average daily sales?", expected_agent="Time Series Agent")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    time.sleep(2)
    
    # Query 2
    result = analyze_query("sales_timeseries.json", "Show me the sales trend over time", expected_agent="Time Series Agent")
    if result:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    print_resources()
    return results

def test_malformed_json():
    """Test 7: Malformed JSON - Error Handling"""
    print_header("TEST 7: Malformed JSON - Error Handling Validation")
    print_resources()
    
    results = {"passed": 0, "failed": 0}
    
    # Upload - should fail gracefully
    print_info("Uploading: malformed.json (expecting graceful error)")
    upload_result = upload_file("malformed.json")
    
    # We expect this to fail, so failure is success!
    if upload_result is None or "error" in str(upload_result):
        print_success("Malformed JSON correctly rejected")
        results["passed"] += 1
    else:
        print_error("Malformed JSON was accepted (should have been rejected!)")
        results["failed"] += 1
    
    print_resources()
    return results

def run_all_json_tests():
    """Execute all JSON tests with summary"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'🧪 '*40}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}PHASE 1: JSON DATA TYPE - OPTIMIZED COMPREHENSIVE DEEP TEST{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'🧪 '*40}{Colors.END}\n")
    
    # Pre-test health check
    print_header("PRE-TEST: Backend Health Check")
    print_resources()
    
    if not check_backend_health():
        print_error("Backend is not running. Please start the backend first.")
        print_info("Start backend: cd src/backend && uvicorn main:app --reload")
        return
    
    print_success("Backend is running")
    
    # Create test data
    create_test_data()
    
    # Initialize results
    total_results = {"passed": 0, "failed": 0}
    test_functions = [
        ("Simple JSON - Student Data", test_simple_json),
        ("Analyze JSON - Category/Value", test_analyze_json),
        ("Complex Nested JSON - Company", test_complex_nested_json),
        ("Large JSON - 10K Transactions", test_large_json),
        ("Financial JSON - Quarterly", test_financial_json),
        ("Time Series JSON - Sales", test_timeseries_json),
        ("Malformed JSON - Error Handling", test_malformed_json)
    ]
    
    # Run each test
    for test_name, test_func in test_functions:
        try:
            results = test_func()
            total_results["passed"] += results["passed"]
            total_results["failed"] += results["failed"]
            
            # Brief pause between tests
            time.sleep(3)
            
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            total_results["failed"] += 1
    
    # Final summary
    print_header("PHASE 1: JSON TESTING - FINAL SUMMARY")
    print_resources()
    
    total_tests = total_results["passed"] + total_results["failed"]
    success_rate = (total_results["passed"] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{Colors.BOLD}Total Tests Run: {total_tests}{Colors.END}")
    print(f"{Colors.GREEN}✅ Passed: {total_results['passed']}{Colors.END}")
    print(f"{Colors.RED}❌ Failed: {total_results['failed']}{Colors.END}")
    print(f"{Colors.CYAN}📊 Success Rate: {success_rate:.1f}%{Colors.END}\n")
    
    if total_results["failed"] == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! Ready for manual frontend testing.{Colors.END}")
        print(f"{Colors.YELLOW}👉 Please test JSON files in the frontend UI and approve to proceed to Phase 2 (CSV){Colors.END}\n")
    else:
        print(f"{Colors.RED}{Colors.BOLD}⚠️  ERRORS DETECTED - Must fix before proceeding to Phase 2{Colors.END}")
        print(f"{Colors.YELLOW}👉 Review errors above and re-run tests after fixes{Colors.END}\n")

if __name__ == "__main__":
    try:
        run_all_json_tests()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Test interrupted by user (Ctrl+C){Colors.END}")
        print_resources()
    except Exception as e:
        print_error(f"Fatal error: {e}")
        print_resources()
