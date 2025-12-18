"""
Test complex nested JSON queries with the optimized data handler

This tests if the timeout issue is fixed after implementing data_optimizer.py

Date: October 19, 2025
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_complex_nested_queries():
    """Test complex nested JSON analysis"""
    
    print("=" * 70)
    print("TESTING: Complex Nested JSON with Data Optimizer")
    print("=" * 70)
    
    filename = "complex_nested.json"
    
    # Test queries
    test_queries = [
        {
            "query": "How many departments are there?",
            "expected": "2 departments"
        },
        {
            "query": "What is the average salary across all employees?",
            "expected": "average salary calculation"
        },
        {
            "query": "List all unique job titles",
            "expected": "Senior Engineer, Junior Engineer, Sales Manager"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/3: {test['query']}")
        print(f"{'='*70}")
        print(f"Expected: {test['expected']}")
        
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
                timeout=300  # 5 minute timeout
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', 'No answer')
                
                print(f"\n✅ SUCCESS (Response time: {elapsed:.1f}s)")
                print(f"\n📊 Answer:")
                print(f"{answer[:500]}")
                if len(answer) > 500:
                    print(f"... ({len(answer) - 500} more characters)")
                
                # Check if it's a direct answer (not code)
                if '```' in answer or 'df[' in answer or 'pandas' in answer.lower():
                    print(f"\n⚠️  WARNING: Response contains code, not direct answer!")
                    results.append({
                        'query': test['query'],
                        'status': 'PARTIAL',
                        'time': elapsed,
                        'issue': 'Contains code instead of direct answer'
                    })
                else:
                    print(f"\n✅ Direct answer provided (no code)")
                    results.append({
                        'query': test['query'],
                        'status': 'PASS',
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
    
    # Check if optimization helped
    if avg_time < 180:
        print(f"\n✅ OPTIMIZATION SUCCESSFUL!")
        print(f"   Average response time ({avg_time:.1f}s) is under 180s target")
    else:
        print(f"\n⚠️  NEEDS MORE OPTIMIZATION")
        print(f"   Average response time ({avg_time:.1f}s) exceeds 180s target")
    
    # Check for timeouts
    timeouts = [r for r in results if r['status'] == 'TIMEOUT']
    if timeouts:
        print(f"\n❌ TIMEOUT ISSUE NOT FIXED")
        print(f"   {len(timeouts)} queries still timing out")
    else:
        print(f"\n✅ NO TIMEOUTS!")
        print(f"   Data optimizer successfully prevented timeouts")
    
    print(f"\n{'='*70}")
    
    return passed == 3

if __name__ == "__main__":
    print("\n🚀 Starting tests...")
    print("⚠️  Make sure backend is running on http://localhost:8000\n")
    
    try:
        # Check if backend is running
        response = requests.get(f"{BASE_URL}/health/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running\n")
            success = test_complex_nested_queries()
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
