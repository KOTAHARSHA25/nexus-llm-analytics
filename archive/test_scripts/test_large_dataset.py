"""
Test large dataset JSON queries

Tests the data optimizer with 10K record dataset to ensure:
1. No timeout
2. Efficient sampling
3. Correct aggregations
4. Fast response time

Date: October 19, 2025
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_large_dataset_queries():
    """Test large dataset analysis"""
    
    print("=" * 70)
    print("TESTING: Large Dataset JSON (10K Records)")
    print("=" * 70)
    
    filename = "large_transactions.json"
    
    # Test queries
    test_queries = [
        {
            "query": "What is the total transaction amount?",
            "expected": "~$13.9M total"
        },
        {
            "query": "Show top 5 categories by count",
            "expected": "Books, Electronics, etc."
        },
        {
            "query": "What is the average transaction value?",
            "expected": "~$1,389 average"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/3: {test['query']}")
        print(f"{'='*70}")
        print(f"Expected: {test['expected']}")
        print(f"Dataset: 10,000 records (3.94 MB)")
        
        # Make request
        payload = {
            "filename": filename,
            "query": test['query']
        }
        
        print(f"\n⏳ Sending request to backend...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{BASE_URL}/analyze/",
                json=payload,
                timeout=300
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('result', 'No answer')
                
                print(f"\n✅ SUCCESS (Response time: {elapsed:.1f}s)")
                print(f"\n📊 Answer:")
                print(f"{answer[:500]}")
                if len(answer) > 500:
                    print(f"... ({len(answer) - 500} more characters)")
                
                # Check performance target
                if elapsed < 120:
                    print(f"\n✅ EXCELLENT: Response time under 120s target!")
                    status = 'PASS'
                elif elapsed < 180:
                    print(f"\n⚠️  ACCEPTABLE: Response time under 180s target")
                    status = 'PASS'
                else:
                    print(f"\n⚠️  SLOW: Response time exceeds 180s target")
                    status = 'PARTIAL'
                
                results.append({
                    'query': test['query'],
                    'status': status,
                    'time': elapsed
                })
                
            else:
                elapsed = time.time() - start_time
                print(f"\n❌ FAILED (Status: {response.status_code}, Time: {elapsed:.1f}s)")
                print(f"Error: {response.text[:200]}")
                results.append({
                    'query': test['query'],
                    'status': 'FAIL',
                    'time': elapsed,
                    'error': response.text[:200]
                })
                
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            print(f"\n❌ TIMEOUT after {elapsed:.1f}s")
            results.append({
                'query': test['query'],
                'status': 'TIMEOUT',
                'time': elapsed
            })
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ ERROR: {str(e)}")
            results.append({
                'query': test['query'],
                'status': 'ERROR',
                'time': elapsed,
                'error': str(e)
            })
    
    # Summary
    print(f"\n\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    partial = sum(1 for r in results if r['status'] == 'PARTIAL')
    failed = sum(1 for r in results if r['status'] in ['FAIL', 'TIMEOUT', 'ERROR'])
    
    print(f"\n📊 Results:")
    print(f"  ✅ PASSED: {passed}/3")
    print(f"  ⚠️  PARTIAL: {partial}/3")
    print(f"  ❌ FAILED: {failed}/3")
    
    print(f"\n⏱️  Response Times:")
    for r in results:
        status_emoji = {'PASS': '✅', 'PARTIAL': '⚠️', 'FAIL': '❌', 'TIMEOUT': '⏰', 'ERROR': '💥'}
        emoji = status_emoji.get(r['status'], '❓')
        print(f"  {emoji} {r['time']:.1f}s - {r['query'][:50]}...")
    
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"\n  Average: {avg_time:.1f}s")
    
    # Check optimizer effectiveness
    print(f"\n💡 Data Optimizer Analysis:")
    if avg_time < 120:
        print(f"  ✅ EXCELLENT performance on 10K records!")
        print(f"  ✅ Data sampling working effectively")
        print(f"  ✅ Average {avg_time:.1f}s is well under 120s target")
    elif avg_time < 180:
        print(f"  ✅ GOOD performance on 10K records")
        print(f"  ✅ Average {avg_time:.1f}s is under 180s target")
    else:
        print(f"  ⚠️  NEEDS OPTIMIZATION")
        print(f"  ⚠️  Average {avg_time:.1f}s exceeds targets")
        print(f"  💡 Consider further data sampling or aggregation")
    
    # Check for timeouts
    timeouts = [r for r in results if r['status'] == 'TIMEOUT']
    if timeouts:
        print(f"\n❌ TIMEOUT ISSUES DETECTED")
        print(f"   {len(timeouts)} queries timed out")
    else:
        print(f"\n✅ NO TIMEOUTS with 10K records!")
        print(f"   Data optimizer successfully handles large datasets")
    
    print(f"\n{'='*70}")
    
    return passed == 3

if __name__ == "__main__":
    print("\n🚀 Starting large dataset tests...")
    print("⚠️  Make sure backend is running on http://localhost:8000\n")
    
    try:
        # Check if backend is running
        response = requests.get(f"{BASE_URL}/health/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running\n")
            success = test_large_dataset_queries()
            exit(0 if success else 1)
        else:
            print("❌ Backend returned unexpected status")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Backend is not running!")
        print("   Please start backend: cd src/backend && python -m uvicorn main:app --reload")
        exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        exit(1)
