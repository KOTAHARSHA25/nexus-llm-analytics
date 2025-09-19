#!/usr/bin/env python3
"""
Frontend-Backend Integration Test
"""
import requests
import time
import sys

def test_backend_api():
    """Test if backend API is accessible"""
    print("🧪 Testing Backend API...")
    
    try:
        # Test the main API documentation
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Backend API docs accessible")
        else:
            print(f"⚠️ Backend API docs returned: {response.status_code}")
            
        # Test a specific API endpoint if available
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend root endpoint responding")
        else:
            print(f"⚠️ Backend root endpoint: {response.status_code}")
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend API at http://127.0.0.1:8000")
        return False
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False

def test_frontend():
    """Test if frontend is accessible"""
    print("\n🧪 Testing Frontend...")
    
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend accessible at localhost:3000")
            
            # Check if it's actually a Next.js app
            if "next" in response.text.lower() or "react" in response.text.lower():
                print("✅ Frontend appears to be a Next.js/React app")
            else:
                print("⚠️ Frontend doesn't appear to be Next.js/React")
                
            return True
        else:
            print(f"⚠️ Frontend returned status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to frontend at http://localhost:3000")
        print("💡 Make sure 'npm run dev' is running in the frontend directory")
        return False
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def test_cors_configuration():
    """Test if CORS is configured for frontend-backend communication"""
    print("\n🧪 Testing CORS Configuration...")
    
    try:
        # Test OPTIONS request (preflight)
        response = requests.options("http://127.0.0.1:8000/", 
                                  headers={
                                      "Origin": "http://localhost:3000",
                                      "Access-Control-Request-Method": "GET"
                                  }, 
                                  timeout=5)
        
        if response.status_code in [200, 204]:
            print("✅ CORS preflight request accepted")
            
            # Check CORS headers
            cors_headers = response.headers.get("Access-Control-Allow-Origin")
            if cors_headers:
                print(f"✅ CORS headers present: {cors_headers}")
            else:
                print("⚠️ CORS headers not found")
                
            return True
        else:
            print(f"⚠️ CORS preflight returned: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ CORS test failed: {e}")
        return False

def main():
    """Run comprehensive integration tests"""
    print("🚀 NEXUS LLM ANALYTICS - INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Backend API", test_backend_api),
        ("Frontend", test_frontend), 
        ("CORS Configuration", test_cors_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 INTEGRATION TEST RESULTS")
    print("=" * 40)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Frontend and Backend are ready for full integration")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        print("🛠️ Some components need attention before full integration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)