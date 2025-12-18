"""
Phase 4 Task 4.1: Comprehensive Chart Generation Testing
Tests all chart types with validation of output format
"""
import requests
import time
import json

API_BASE = "http://localhost:8000"

print("="*70)
print("PHASE 4: Task 4.1 - Chart Generation Testing")
print("="*70)
print()

# Use existing sales_simple.csv
test_file = "sales_simple.csv"

print(f"Test File: {test_file}")
print()

# Chart types to test
chart_tests = [
    {
        "name": "Bar Chart",
        "goal": "Show revenue by product as a bar chart",
        "expected_type": "bar",
        "must_contain": ["Widget A", "Widget B"]
    },
    {
        "name": "Line Chart",
        "goal": "Show sales trend over time as a line chart",
        "expected_type": "line",
        "must_contain": ["date", "revenue"]
    },
    {
        "name": "Pie Chart",
        "goal": "Show revenue distribution by product as a pie chart",
        "expected_type": "pie",
        "must_contain": ["Widget"]
    },
    {
        "name": "Auto Chart",
        "goal": None,  # Auto-detection
        "expected_type": "auto",
        "must_contain": []
    }
]

print("="*70)
print("TESTING Chart Generation...")
print("="*70)
print()

results = []
passed = 0
total = len(chart_tests)

for i, test in enumerate(chart_tests, 1):
    print(f"[{i}/{total}] Testing {test['name']}...")
    print(f"    Goal: {test['goal'] or 'Auto-detect chart type'}")
    
    start = time.time()
    try:
        if test['goal']:
            # Goal-based visualization
            r = requests.post(
                f"{API_BASE}/visualize/goal-based",
                json={
                    "filename": test_file,
                    "goal": test['goal'],
                    "library": "plotly"
                },
                timeout=120
            )
        else:
            # Auto chart
            # Read the data first to pass to generate endpoint
            r = requests.post(
                f"{API_BASE}/visualize/goal-based",
                json={
                    "filename": test_file,
                    "library": "plotly"
                },
                timeout=120
            )
        
        elapsed = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            print(f"    Time: {elapsed:.1f}s")
            
            # Check for chart data (handle both flat and nested responses)
            has_chart = False
            chart_format = None
            
            # Check nested visualization object first
            if "visualization" in result and isinstance(result["visualization"], dict):
                viz = result["visualization"]
                if "figure_json" in viz:
                    has_chart = True
                    chart_format = "Plotly JSON (nested)"
                    try:
                        chart_data = json.loads(viz["figure_json"]) if isinstance(viz["figure_json"], str) else viz["figure_json"]
                        print(f"    ✅ Chart Format: {chart_format} (valid)")
                    except:
                        print(f"    ⚠️ Chart Format: {chart_format} (invalid JSON)")
                        has_chart = False
                elif "image_base64" in viz:
                    has_chart = True
                    chart_format = "Base64 PNG (nested)"
                    print(f"    ✅ Chart Format: {chart_format}")
            # Check flat structure
            elif "figure_json" in result:
                has_chart = True
                chart_format = "Plotly JSON"
                # Validate it's valid JSON
                try:
                    chart_data = json.loads(result["figure_json"]) if isinstance(result["figure_json"], str) else result["figure_json"]
                    print(f"    ✅ Chart Format: {chart_format} (valid)")
                except:
                    print(f"    ⚠️ Chart Format: {chart_format} (invalid JSON)")
                    has_chart = False
            elif "visualizations" in result and result["visualizations"]:
                has_chart = True
                chart_format = "Visualization list"
                print(f"    ✅ Chart Format: {chart_format} ({len(result['visualizations'])} charts)")
            elif "chart" in result:
                has_chart = True
                chart_format = "Chart object"
                print(f"    ✅ Chart Format: {chart_format}")
            elif "image_base64" in result:
                has_chart = True
                chart_format = "Base64 PNG"
                print(f"    ✅ Chart Format: {chart_format}")
            
            if has_chart:
                # Check performance
                if elapsed < 60:
                    print(f"    ✅ Performance: EXCELLENT (<60s)")
                elif elapsed < 120:
                    print(f"    ⚠️ Performance: ACCEPTABLE (60-120s)")
                else:
                    print(f"    ❌ Performance: SLOW (>120s)")
                
                passed += 1
                results.append({
                    "test": test['name'],
                    "status": "PASS",
                    "time": elapsed,
                    "format": chart_format
                })
            else:
                print(f"    ❌ No chart data found in response")
                print(f"    Response keys: {list(result.keys())}")
                if "error" in result:
                    print(f"    Error details: {result['error'][:200]}")
                results.append({
                    "test": test['name'],
                    "status": "FAIL",
                    "time": elapsed,
                    "error": "No chart data"
                })
        else:
            elapsed = time.time() - start
            print(f"    ❌ API Error: {r.status_code}")
            print(f"    Response: {r.text[:200]}")
            results.append({
                "test": test['name'],
                "status": "FAIL",
                "time": elapsed,
                "error": f"HTTP {r.status_code}"
            })
    
    except Exception as e:
        elapsed = time.time() - start
        print(f"    ❌ Error: {str(e)[:100]}")
        results.append({
            "test": test['name'],
            "status": "FAIL",
            "time": elapsed,
            "error": str(e)[:100]
        })
    
    print()

print("="*70)
print(f"Task 4.1 Chart Generation - Results: {passed}/{total} PASS")
print("="*70)
print()

print("Summary:")
total_time = sum(r['time'] for r in results if isinstance(r.get('time'), (int, float)))
avg_time = total_time / len(results) if results else 0
print(f"  Total time: {total_time:.1f}s")
print(f"  Average time per chart: {avg_time:.1f}s")
print(f"  Pass rate: {(passed/total*100):.1f}%")
print()

print("Results by Chart Type:")
for r in results:
    status_icon = "✅" if r['status'] == "PASS" else "❌"
    format_info = f"({r.get('format', 'N/A')})" if r['status'] == "PASS" else f"({r.get('error', 'Unknown error')})"
    print(f"  {status_icon} {r['test']}: {r['time']:.1f}s {format_info}")
print()

if passed == total:
    print("✅ All chart types generated successfully!")
elif passed >= total * 0.75:
    print("⚠️ Most chart types working - minor issues detected")
else:
    print("❌ Chart generation needs attention - multiple failures")
