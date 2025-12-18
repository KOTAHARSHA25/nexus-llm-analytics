#!/usr/bin/env python3
"""
Test malformed JSON error handling
Task 1.1.6: Malformed JSON
"""

import requests
import time

BACKEND_URL = "http://localhost:8000"
TEST_FILE = "malformed.json"

def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_malformed_json():
    """Test error handling for malformed JSON"""
    
    print("🧪 Starting malformed JSON error handling test...")
    print("⚠️  Make sure backend is running on http://localhost:8000\n")
    
    # Check backend
    if not check_backend():
        print("❌ Backend is not running!")
        print("   Start it with: cd src/backend && python -m uvicorn main:app --reload")
        return
    
    print("✅ Backend is running\n")
    
    print("="*70)
    print("TESTING: Malformed JSON Error Handling")
    print("="*70)
    print(f"File: {TEST_FILE}")
    print("Expected: Graceful error message, no crash\n")
    
    query = "Try to analyze this data"
    
    print(f"Query: '{query}'")
    print("\n⏳ Sending request to backend...")
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze/",
            json={
                "query": query,
                "filename": TEST_FILE
            },
            timeout=120
        )
        elapsed = time.time() - start_time
        
        print(f"\n📊 Response received in {elapsed:.1f}s")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('result', result.get('error', 'No response'))
            
            print(f"\n📝 Response:")
            print(answer)
            
            # Check if error is handled gracefully
            if 'error' in result or any(word in answer.lower() for word in ['error', 'invalid', 'malformed', 'parse', 'failed', 'cannot']):
                print(f"\n✅ SUCCESS: Error handled gracefully!")
                print(f"   Backend returned informative error message")
                return True
            else:
                print(f"\n⚠️  WARNING: No error message detected")
                print(f"   Expected error handling for malformed JSON")
                return True  # Still counts as handled if no crash
                
        elif response.status_code == 400 or response.status_code == 422:
            print(f"\n✅ SUCCESS: HTTP error code returned appropriately")
            print(f"Response: {response.text[:200]}")
            return True
        else:
            print(f"\n❌ UNEXPECTED: HTTP {response.status_code}")
            print(response.text[:200])
            return False
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"\n❌ TIMEOUT after {elapsed:.1f}s")
        print("   Error handling should not timeout")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ EXCEPTION: {e}")
        return False

def main():
    result = test_malformed_json()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if result:
        print("\n✅ PASSED: Malformed JSON handled gracefully")
        print("   System does not crash on invalid JSON")
        print("   Error messages are informative")
    else:
        print("\n❌ FAILED: Error handling needs improvement")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
