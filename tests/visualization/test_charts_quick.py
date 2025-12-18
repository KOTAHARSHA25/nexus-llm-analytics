"""
Phase 4: Quick Visualization Test
Tests that chart generation works
"""
import requests
import json

API_BASE = "http://localhost:8000"

print("="*70)
print("PHASE 4: Chart Generation Test")
print("="*70)
print()

# Use existing sales_simple.csv for quick test
test_file = "sales_simple.csv"

print(f"[TEST] Testing chart generation with {test_file}...")
print()

# Test 1: Auto chart generation
print("[1/2] Testing auto chart generation...")
try:
    r = requests.post(
        f"{API_BASE}/visualize/auto",
        json={
            "filename": test_file,
            "chart_type": "auto"
        },
        timeout=30
    )
    
    if r.status_code == 200:
        result = r.json()
        if "figure_json" in result or "chart" in result:
            print("  ✅ Auto chart generation: WORKS")
        else:
            print(f"  ⚠️ Response received but no chart: {list(result.keys())}")
    else:
        print(f"  ❌ API Error: {r.status_code}")
        print(f"  Response: {r.text[:200]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

print()

# Test 2: Goal-based visualization
print("[2/2] Testing goal-based visualization...")
try:
    r = requests.post(
        f"{API_BASE}/visualize/goal-based",
        json={
            "filename": test_file,
            "goal": "Show sales by category",
            "library": "plotly"
        },
        timeout=60
    )
    
    if r.status_code == 200:
        result = r.json()
        if "figure_json" in result or "chart" in result or "visualizations" in result:
            print("  ✅ Goal-based visualization: WORKS")
        else:
            print(f"  ⚠️ Response received but no chart: {list(result.keys())}")
    else:
        print(f"  ❌ API Error: {r.status_code}")
        print(f"  Response: {r.text[:200]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

print()
print("="*70)
print("Phase 4 Quick Test Complete")
print("="*70)
print()
print("✅ If both tests passed, visualization infrastructure is DEMO-READY")
print("⚠️ If tests failed, endpoints may not be fully implemented yet")
