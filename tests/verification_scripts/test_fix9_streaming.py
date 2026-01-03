"""
Test Fix 9: Streaming Responses (SSE)

Verifies that the streaming endpoint:
1. Returns proper SSE format
2. Sends progressive updates
3. Handles errors gracefully
4. Completes successfully
"""

import sys
import requests
import json
import time
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
STREAM_ENDPOINT = f"{BASE_URL}/api/analyze/stream"


def test_streaming_simple_query():
    """Test streaming with a simple query"""
    print("\n🌊 Testing Simple Streaming Query...")
    
    # Prepare request
    payload = {
        "query": "Count the rows",
        "filename": "sales_data.csv"
    }
    
    try:
        # Make streaming request
        response = requests.post(
            STREAM_ENDPOINT,
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"  ❌ HTTP {response.status_code}: {response.text}")
            return 0, 1
        
        # Parse SSE events
        events = []
        steps_seen = set()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_json = line_str[6:]  # Remove 'data: ' prefix
                    try:
                        event = json.loads(data_json)
                        events.append(event)
                        step = event.get('step')
                        steps_seen.add(step)
                        print(f"    📡 {step}: {event.get('message', 'N/A')}")
                    except json.JSONDecodeError as e:
                        print(f"    ⚠️ Failed to parse JSON: {data_json[:100]}")
        
        # Verify we got all expected steps
        expected_steps = {'init', 'validation', 'loading', 'analyzing', 'formatting', 'complete'}
        missing_steps = expected_steps - steps_seen
        
        if not missing_steps:
            print(f"  ✅ All {len(expected_steps)} steps received")
            print(f"  ✅ Total events: {len(events)}")
            return 1, 1
        else:
            print(f"  ⚠️ Missing steps: {missing_steps}")
            print(f"  ⚠️ Received steps: {steps_seen}")
            return 0, 1
    
    except requests.exceptions.ConnectionError:
        print("  ⚠️ Connection refused - backend server not running?")
        print("     Start server with: python -m uvicorn backend.main:app")
        return 0, 1
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 1


def test_streaming_error_handling():
    """Test streaming with invalid request"""
    print("\n❌ Testing Error Handling...")
    
    # Test 1: Missing query field (caught by FastAPI validation)
    payload1 = {
        "filename": "test.csv"
        # Missing 'query' field
    }
    
    # Test 2: Empty query (caught by streaming logic)
    payload2 = {
        "query": "",
        "filename": "test.csv"
    }
    
    test_passed = False
    
    try:
        # Test FastAPI validation error
        response1 = requests.post(
            STREAM_ENDPOINT,
            json=payload1,
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=10
        )
        
        # FastAPI validation should return 422
        if response1.status_code == 422:
            print(f"    ✅ FastAPI validation error (422) for missing query")
            test_passed = True
        
        # Test empty query error
        response2 = requests.post(
            STREAM_ENDPOINT,
            json=payload2,
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=10
        )
        
        # Parse events from empty query test
        error_seen = False
        for line in response2.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        event = json.loads(line_str[6:])
                        if event.get('step') == 'error':
                            error_seen = True
                            print(f"    ✅ Error event received: {event.get('message')}")
                    except:
                        pass
        
        if error_seen or test_passed:
            print("  ✅ Error handling works correctly")
            return 1, 1
        else:
            print("  ⚠️ No error event received for empty query")
            return 0, 1
    
    except requests.exceptions.ConnectionError:
        print("  ⚠️ Connection refused - backend not running")
        return 0, 1
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 1


def test_streaming_progress_values():
    """Test that progress values increase monotonically"""
    print("\n📈 Testing Progress Values...")
    
    payload = {
        "query": "Show first 5 rows",
        "filename": "sales_data.csv"
    }
    
    try:
        response = requests.post(
            STREAM_ENDPOINT,
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"  ⚠️ HTTP {response.status_code}")
            return 0, 1
        
        progress_values = []
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        event = json.loads(line_str[6:])
                        if 'progress' in event:
                            progress_values.append(event['progress'])
                    except:
                        pass
        
        # Check monotonic increase
        is_monotonic = all(progress_values[i] <= progress_values[i+1] 
                          for i in range(len(progress_values)-1))
        
        if is_monotonic and len(progress_values) > 0:
            print(f"  ✅ Progress values increase: {progress_values}")
            print(f"  ✅ Final progress: {progress_values[-1]}%")
            return 1, 1
        else:
            print(f"  ❌ Progress not monotonic: {progress_values}")
            return 0, 1
    
    except requests.exceptions.ConnectionError:
        print("  ⚠️ Backend not running")
        return 0, 1
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 1


def test_sse_format():
    """Test that response follows SSE format specification"""
    print("\n📋 Testing SSE Format Compliance...")
    
    payload = {
        "query": "Count rows",
        "filename": "test.csv"
    }
    
    try:
        response = requests.post(
            STREAM_ENDPOINT,
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=10
        )
        
        # Check headers
        content_type = response.headers.get('Content-Type', '')
        cache_control = response.headers.get('Cache-Control', '')
        
        checks = []
        
        # Check Content-Type
        if 'text/event-stream' in content_type:
            print("  ✅ Content-Type: text/event-stream")
            checks.append(True)
        else:
            print(f"  ❌ Wrong Content-Type: {content_type}")
            checks.append(False)
        
        # Check Cache-Control
        if 'no-cache' in cache_control:
            print("  ✅ Cache-Control: no-cache")
            checks.append(True)
        else:
            print(f"  ⚠️ Cache-Control: {cache_control}")
            checks.append(False)
        
        # Check data format
        valid_format = True
        for line in response.iter_lines(decode_unicode=True):
            if line and not line.startswith('data: ') and line.strip() != '':
                print(f"  ❌ Invalid SSE format: {line[:50]}")
                valid_format = False
                break
        
        if valid_format:
            print("  ✅ All lines follow 'data: ...' format")
            checks.append(True)
        else:
            checks.append(False)
        
        passed = sum(checks)
        return passed, len(checks)
    
    except requests.exceptions.ConnectionError:
        print("  ⚠️ Backend not running")
        return 0, 3
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 3


def main():
    """Run all streaming tests"""
    print("=" * 60)
    print("FIX 9: STREAMING RESPONSES (SSE) TEST")
    print("=" * 60)
    print("\nℹ️  Note: Backend server must be running at http://localhost:8000")
    print("   Start with: python -m uvicorn backend.main:app")
    print()
    
    results = []
    
    # Run all tests
    results.append(test_streaming_simple_query())
    results.append(test_streaming_error_handling())
    results.append(test_streaming_progress_values())
    results.append(test_sse_format())
    
    # Calculate totals
    total_passed = sum(r[0] for r in results)
    total_tests = sum(r[1] for r in results)
    percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed}/{total_tests} tests passed ({percentage:.1f}%)")
    print("=" * 60)
    
    if percentage >= 90:
        print("\n🎯 Fix 9 VERIFIED - Streaming responses working!")
        print("   ✅ SSE format compliance")
        print("   ✅ Progressive updates")
        print("   ✅ Error handling")
        print("   ✅ Progress tracking")
    elif percentage >= 70:
        print("\n⚠️ Fix 9 PARTIAL - Most features working")
    elif percentage == 0 and total_tests > 0:
        print("\n⚠️ Backend server not running")
        print("   Start server: python -m uvicorn backend.main:app --reload")
    else:
        print("\n❌ Fix 9 NEEDS WORK")
    
    return percentage >= 90


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
