"""
PHASE 4: Visualization System Testing
Tests chart generation, suggestions, and different visualization types
"""
import sys
import requests
import json
from pathlib import Path

print("=" * 80)
print("PHASE 4: VISUALIZATION SYSTEM TESTING")
print("=" * 80)

API_BASE = "http://localhost:8000"
total = 0
passed = 0

# First, upload a test file
print("\n[SETUP] Uploading test data file...")
test_file = "data/samples/sales_data.csv"
if not Path(test_file).exists():
    print(f"SKIP: Test file {test_file} not found")
    sys.exit(0)

try:
    with open(test_file, 'rb') as f:
        upload_resp = requests.post(
            f"{API_BASE}/upload",
            files={'file': ('sales_data.csv', f, 'text/csv')},
            timeout=10
        )
    
    if upload_resp.status_code == 200:
        upload_data = upload_resp.json()
        filename = 'sales_data.csv'
        conversation_id = upload_data.get('conversation_id')
        print(f"✓ File uploaded: {filename}")
        print(f"  - Conversation ID: {conversation_id}")
    else:
        print(f"✗ Upload failed: {upload_resp.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Upload error: {e}")
    sys.exit(1)

# TEST 1: Chart Suggestions Endpoint
print("\n[TEST 1] Chart Suggestions Generation")
total += 1
try:
    resp = requests.post(
        f"{API_BASE}/visualize/suggestions",
        json={"filename": filename},
        timeout=10
    )
    
    if resp.status_code == 200:
        data = resp.json()
        if 'suggestions' in data and len(data['suggestions']) > 0:
            print(f"PASS: Generated {len(data['suggestions'])} chart suggestions")
            print(f"  - Recommended: {data.get('recommended', 'N/A')}")
            passed += 1
        else:
            print("FAIL: No suggestions returned")
    else:
        print(f"FAIL: Status {resp.status_code}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 2: Bar Chart Generation
print("\n[TEST 2] Bar Chart Generation")
total += 1
try:
    resp = requests.post(
        f"{API_BASE}/visualize/goal-based",
        json={
            "filename": filename,
            "goal": "Show revenue by product as a bar chart",
            "library": "plotly"
        },
        timeout=15
    )
    
    if resp.status_code == 200:
        data = resp.json()
        if data.get('success') and 'visualization' in data:
            print("PASS: Bar chart generated")
            print(f"  - Chart type: {data.get('selected_chart', {}).get('type', 'N/A')}")
            passed += 1
        else:
            print(f"FAIL: {data.get('error', 'Unknown error')}")
    else:
        print(f"FAIL: Status {resp.status_code}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 3: Line Chart Generation
print("\n[TEST 3] Line Chart Generation")
total += 1
try:
    resp = requests.post(
        f"{API_BASE}/visualize/goal-based",
        json={
            "filename": filename,
            "goal": "Show trend over time as a line chart",
            "library": "plotly"
        },
        timeout=15
    )
    
    if resp.status_code == 200:
        data = resp.json()
        if data.get('success'):
            print("PASS: Line chart generated")
            passed += 1
        else:
            print(f"FAIL: {data.get('error', 'Unknown error')}")
    else:
        print(f"FAIL: Status {resp.status_code}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 4: Pie Chart Generation
print("\n[TEST 4] Pie Chart Generation")
total += 1
try:
    resp = requests.post(
        f"{API_BASE}/visualize/goal-based",
        json={
            "filename": filename,
            "goal": "Show distribution as a pie chart",
            "library": "plotly"
        },
        timeout=15
    )
    
    if resp.status_code == 200:
        data = resp.json()
        if data.get('success'):
            print("PASS: Pie chart generated")
            passed += 1
        else:
            print(f"FAIL: {data.get('error', 'Unknown error')}")
    else:
        print(f"FAIL: Status {resp.status_code}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 5: Chart Type Detection
print("\n[TEST 5] Automatic Chart Type Selection")
total += 1
try:
    resp = requests.post(
        f"{API_BASE}/visualize/goal-based",
        json={
            "filename": filename,
            "goal": "Visualize the data appropriately",
            "library": "plotly"
        },
        timeout=15
    )
    
    if resp.status_code == 200:
        data = resp.json()
        if data.get('success') and data.get('selected_chart'):
            print("PASS: Automatic chart type selected")
            print(f"  - Selected: {data['selected_chart'].get('type', 'N/A')}")
            passed += 1
        else:
            print("FAIL: No chart type selected")
    else:
        print(f"FAIL: Status {resp.status_code}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 6: Multiple Library Support
print("\n[TEST 6] Multiple Library Support (Matplotlib)")
total += 1
try:
    resp = requests.post(
        f"{API_BASE}/visualize/goal-based",
        json={
            "filename": filename,
            "goal": "Create a simple bar chart",
            "library": "matplotlib"
        },
        timeout=15
    )
    
    if resp.status_code == 200:
        data = resp.json()
        if data.get('success'):
            print("PASS: Matplotlib library works")
            passed += 1
        else:
            print(f"FAIL: {data.get('error', 'Unknown error')}")
    else:
        print(f"FAIL: Status {resp.status_code}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 7: Visualization JSON Format
print("\n[TEST 7] Visualization JSON Format Validation")
total += 1
try:
    resp = requests.post(
        f"{API_BASE}/visualize/goal-based",
        json={
            "filename": filename,
            "goal": "Create any chart",
            "library": "plotly"
        },
        timeout=15
    )
    
    if resp.status_code == 200:
        data = resp.json()
        if data.get('success') and 'visualization' in data:
            viz = data['visualization']
            # Check if it has figure_json (plotly) or image_data (matplotlib)
            if 'figure_json' in viz or 'image_data' in viz:
                print("PASS: Visualization JSON format correct")
                passed += 1
            else:
                print("FAIL: Missing visualization data")
        else:
            print("FAIL: No visualization in response")
    else:
        print(f"FAIL: Status {resp.status_code}")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# TEST 8: Chart Performance (<5s)
print("\n[TEST 8] Chart Generation Performance (<5s)")
total += 1
import time
try:
    start = time.time()
    resp = requests.post(
        f"{API_BASE}/visualize/goal-based",
        json={
            "filename": filename,
            "goal": "Create a bar chart",
            "library": "plotly"
        },
        timeout=15
    )
    elapsed = time.time() - start
    
    if resp.status_code == 200 and elapsed < 5:
        print(f"PASS: Chart generated in {elapsed:.2f}s (<5s target)")
        passed += 1
    else:
        print(f"FAIL: Took {elapsed:.2f}s or failed")
except Exception as e:
    print(f"FAIL: Exception - {str(e)[:100]}")

# Summary
print("\n" + "=" * 80)
print("PHASE 4 RESULTS: Visualization System")
print("=" * 80)
print(f"Total Tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Success Rate: {passed/total*100:.1f}%")

if passed == total:
    print("\n✅ ALL VISUALIZATION TESTS PASSED")
elif passed >= total * 0.75:
    print(f"\n⚠️ MOSTLY WORKING - {total - passed} tests need attention")
else:
    print(f"\n❌ SIGNIFICANT ISSUES - {total - passed} tests failed")
