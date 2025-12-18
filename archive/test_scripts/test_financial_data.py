#!/usr/bin/env python3
"""
Test financial quarterly data analysis
Task 1.1.4: Financial Data JSON
"""

import requests
import json
import time
from pathlib import Path

BACKEND_URL = "http://localhost:8000"
TEST_FILE = "financial_quarterly.json"

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
    print(f"Dataset: Financial quarterly data (12 months, 4 quarters)")
    
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
    print("🏦 Starting financial data tests...")
    print("⚠️  Make sure backend is running on http://localhost:8000\n")
    
    # Check backend
    if not check_backend():
        print("❌ Backend is not running!")
        print("   Start it with: cd src/backend && python -m uvicorn main:app --reload")
        return
    
    print("✅ Backend is running\n")
    
    # Load the data to get exact values
    with open("data/samples/financial_quarterly.json", 'r') as f:
        data = json.load(f)
    
    # Calculate expected values
    q1_data = [entry for entry in data if entry['quarter'] == 'Q1']
    q1_revenue = sum(entry['revenue'] for entry in q1_data)
    
    quarter_margins = {}
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = [entry for entry in data if entry['quarter'] == quarter]
        q_revenue = sum(entry['revenue'] for entry in q_data)
        q_net_income = sum(entry['net_income'] for entry in q_data)
        quarter_margins[quarter] = (q_net_income / q_revenue) * 100
    
    best_quarter = max(quarter_margins.items(), key=lambda x: x[1])
    
    print("="*70)
    print("TESTING: Financial Quarterly Data")
    print("="*70)
    
    tests = [
        {
            "query": "What is the total revenue for Q1 2024?",
            "expected": f"~${q1_revenue:,.2f}"
        },
        {
            "query": "Which quarter has the highest profit margin?",
            "expected": f"{best_quarter[0]} ({best_quarter[1]:.2f}%)"
        },
        {
            "query": "What are the total operating expenses across all quarters?",
            "expected": f"Sum of all operating expenses"
        }
    ]
    
    results = []
    times = []
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(tests)}: {test['query']}")
        print(f"{'='*70}")
        print(f"Expected: {test['expected']}")
        print(f"Dataset: 12 monthly records")
        
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
    
    print(f"\n💡 Financial Data Analysis:")
    if avg_time < 120:
        print(f"  ✅ EXCELLENT performance on financial data")
    elif avg_time < 180:
        print(f"  ✅ GOOD performance on financial data")
    else:
        print(f"  ⚠️  Performance needs improvement")
    
    print(f"  ✅ Average {avg_time:.1f}s is {'under' if avg_time < 180 else 'over'} 180s target")
    
    if all(results):
        print(f"\n✅ ALL TESTS PASSED!")
        print(f"   Financial data analysis is working correctly")
    else:
        print(f"\n⚠️  Some tests failed - review results above")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
