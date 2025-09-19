#!/usr/bin/env python3
"""
End-to-End Upload Test
"""
import requests
import tempfile
import os
import json
import time

def create_test_csv():
    """Create a test CSV file for upload"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("name,age,department,salary\n")
        f.write("Alice,28,Engineering,75000\n") 
        f.write("Bob,32,Marketing,65000\n")
        f.write("Charlie,26,Design,60000\n")
        f.write("Diana,30,Engineering,80000\n")
        return f.name

def test_upload_endpoint():
    """Test the upload endpoint with a real file"""
    print("🧪 Testing Upload Endpoint...")
    
    # Create test file
    csv_file = create_test_csv()
    
    try:
        # Test upload
        with open(csv_file, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            response = requests.post('http://127.0.0.1:8000/upload-documents/', files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"📊 Response: {result}")
            return True
        else:
            print(f"❌ Upload failed with status: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend - make sure it's running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(csv_file):
            os.unlink(csv_file)

def test_basic_endpoints():
    """Test basic API endpoints"""
    print("\n🧪 Testing Basic API Endpoints...")
    
    endpoints = [
        ("Root", "http://127.0.0.1:8000/"),
        ("Docs", "http://127.0.0.1:8000/docs"),
        ("OpenAPI", "http://127.0.0.1:8000/openapi.json"),
    ]
    
    results = []
    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} endpoint working")
                results.append(True)
            else:
                print(f"⚠️ {name} endpoint returned: {response.status_code}")
                results.append(False)
        except requests.exceptions.ConnectionError:
            print(f"❌ {name} endpoint not accessible")
            results.append(False)
        except Exception as e:
            print(f"❌ {name} endpoint failed: {e}")
            results.append(False)
    
    return all(results)

def main():
    """Run end-to-end tests"""
    print("🚀 END-TO-END FUNCTIONALITY TEST")
    print("=" * 50)
    print("⚠️ Make sure both servers are running:")
    print("   Backend: uvicorn main:app --reload (port 8000)")
    print("   Frontend: npm run dev (port 3000)")
    print()
    
    # Wait a moment for user to start servers
    input("Press Enter when both servers are running...")
    
    tests = [
        ("Basic API Endpoints", test_basic_endpoints),
        ("Upload Functionality", test_upload_endpoint),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 END-TO-END TEST RESULTS")
    print("=" * 40)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL END-TO-END TESTS PASSED!")
        print("✅ The application is working end-to-end!")
        print("\n🚀 NEXUS LLM ANALYTICS IS READY!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        print("🛠️ Some functionality needs attention")
        return False

if __name__ == "__main__":
    success = main()