"""
Test script to verify text input functionality
"""
import requests
import json

API_BASE = "http://localhost:8000"

def test_text_input():
    """Test analyzing text input directly"""
    print("="*70)
    print("TESTING TEXT INPUT FUNCTIONALITY")
    print("="*70)
    
    # Sample text data (simulating what user types in the textarea)
    text_data = """my name is harsha and i study at mlrit in the course of aiml"""
    
    print(f"\n📝 Text Input: {text_data}")
    print(f"\n🔍 Query: What information can you extract from this text?")
    
    # Make the request just like the frontend does
    response = requests.post(
        f"{API_BASE}/analyze",
        json={
            "query": "What information can you extract from this text?",
            "text_data": text_data
        },
        timeout=60
    )
    
    print(f"\n📊 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ SUCCESS!")
        print(f"\nExtracted Information:")
        print(f"   Status: {result.get('status', 'N/A')}")
        print(f"   Result: {result.get('result', 'N/A')}")
        print(f"   Execution Time: {result.get('execution_time', 0):.2f}s")
        
        if result.get('code'):
            print(f"\n📝 Generated Code:")
            print(f"   {result.get('code')[:200]}...")
        
        return True
    else:
        print(f"\n❌ FAILED!")
        print(f"   Error: {response.text[:300]}")
        return False


def test_csv_text_input():
    """Test analyzing CSV data pasted as text"""
    print("\n" + "="*70)
    print("TESTING CSV TEXT INPUT")
    print("="*70)
    
    # Sample CSV data pasted directly
    csv_text = """Name,Age,Score
Alice,25,95
Bob,30,88
Charlie,22,92
David,28,85"""
    
    print(f"\n📝 CSV Data Pasted:")
    print(csv_text)
    print(f"\n🔍 Query: What is the average score?")
    
    response = requests.post(
        f"{API_BASE}/analyze",
        json={
            "query": "What is the average score?",
            "text_data": csv_text
        },
        timeout=60
    )
    
    print(f"\n📊 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ SUCCESS!")
        print(f"\nAnalysis Result:")
        print(f"   Answer: {result.get('result', 'N/A')}")
        print(f"   Execution Time: {result.get('execution_time', 0):.2f}s")
        return True
    else:
        print(f"\n❌ FAILED!")
        print(f"   Error: {response.text[:300]}")
        return False


if __name__ == "__main__":
    results = []
    
    # Test 1: Plain text input
    results.append(("Text Input", test_text_input()))
    
    # Test 2: CSV text input
    results.append(("CSV Text Input", test_csv_text_input()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.0f}%)")
    
    if passed_count == total_count:
        print("\n🎉 All text input tests passed!")
        print("   Text input functionality is working correctly!")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
