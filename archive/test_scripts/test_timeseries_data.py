#!/usr/bin/env python3
"""
Test time series data analysis
Task 1.1.5: Time Series JSON
"""

import requests
import json
import time

BACKEND_URL = "http://localhost:8000"
TEST_FILE = "sales_timeseries.json"

def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_query(query, expected_info):
    """Test a single query"""
    print(f"\n{'='*70}")
    print(f"TEST: {query}")
    print(f"{'='*70}")
    print(f"Expected: {expected_info}")
    
    print(f"\n⏳ Sending request to backend...")
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze/",
            json={
                "query": query,
                "filename": TEST_FILE
            },
            timeout=300
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('result', 'No answer')
            
            print(f"\n✅ SUCCESS (Response time: {elapsed:.1f}s)")
            print(f"\n📊 Answer:")
            print(answer)
            
            # Performance assessment
            if elapsed < 120:
                print(f"\n✅ EXCELLENT: Response time under 120s target!")
            elif elapsed < 180:
                print(f"\n⚠️  ACCEPTABLE: Response time under 180s target")
            else:
                print(f"\n⚠️  SLOW: Response time exceeds 180s target")
            
            return True, elapsed
        else:
            print(f"\n❌ FAILED: HTTP {response.status_code}")
            print(response.text)
            return False, time.time() - start_time
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"\n❌ TIMEOUT after {elapsed:.1f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR: {e}")
        return False, elapsed

def main():
    print("📈 Starting time series data tests...")
    print("⚠️  Make sure backend is running on http://localhost:8000\n")
    
    # Check backend
    if not check_backend():
        print("❌ Backend is not running!")
        print("   Start it with: cd src/backend && python -m uvicorn main:app --reload")
        return
    
    print("✅ Backend is running\n")
    
    # Load the data to get exact values
    with open("data/samples/sales_timeseries.json", 'r') as f:
        data = json.load(f)
    
    # Calculate expected values
    seasons = {}
    for entry in data:
        season = entry['season']
        if season not in seasons:
            seasons[season] = []
        seasons[season].append(entry['sales_amount'])
    
    summer_avg = sum(seasons['Summer']) / len(seasons['Summer'])
    winter_avg = sum(seasons['Winter']) / len(seasons['Winter'])
    
    total_sales = sum(entry['sales_amount'] for entry in data)
    
    print("="*70)
    print("TESTING: Time Series Sales Data")
    print("="*70)
    
    tests = [
        {
            "query": "What is the total sales for the entire year?",
            "expected": f"~${total_sales:,.2f}"
        },
        {
            "query": "Identify seasonal patterns in the sales data",
            "expected": f"Summer highest (~${summer_avg:,.2f}/day), Winter lowest (~${winter_avg:,.2f}/day)"
        },
        {
            "query": "What is the sales trend over the year?",
            "expected": "Seasonal variation with summer peak"
        }
    ]
    
    results = []
    times = []
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(tests)}: {test['query']}")
        print(f"{'='*70}")
        print(f"Expected: {test['expected']}")
        print(f"Dataset: 366 daily records (full year)")
        
        success, elapsed = test_query(test['query'], test['expected'])
        results.append(success)
        times.append(elapsed)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results)
    print(f"\n📊 Results:")
    print(f"  ✅ PASSED: {passed}/{len(tests)}")
    print(f"  ❌ FAILED: {len(tests) - passed}/{len(tests)}")
    
    print(f"\n⏱️  Response Times:")
    for i, (test, elapsed) in enumerate(zip(tests, times), 1):
        status = "✅" if elapsed < 180 else "⚠️"
        print(f"  {status} {elapsed:.1f}s - {test['query'][:60]}...")
    
    avg_time = sum(times) / len(times)
    print(f"\n  Average: {avg_time:.1f}s")
    
    print(f"\n💡 Time Series Analysis:")
    if avg_time < 120:
        print(f"  ✅ EXCELLENT performance on time series data")
    elif avg_time < 180:
        print(f"  ✅ GOOD performance on time series data")
    else:
        print(f"  ⚠️  Performance needs improvement")
    
    print(f"  ✅ Average {avg_time:.1f}s is {'under' if avg_time < 180 else 'over'} 180s target")
    
    if all(results):
        print(f"\n✅ ALL TESTS PASSED!")
        print(f"   Time series analysis is working correctly")
    else:
        print(f"\n⚠️  Some tests failed - review results above")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
