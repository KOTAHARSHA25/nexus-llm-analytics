"""
Debug script to test visualization generation and see actual response
"""
import requests
import json

BACKEND_URL = "http://localhost:8000"

def test_visualization_flow():
    print("="*70)
    print("DEBUGGING VISUALIZATION FLOW")
    print("="*70)
    
    filename = "test_sales_monthly.csv"
    
    # Step 1: Get suggestions
    print("\n1️⃣  Testing /visualize/suggestions endpoint")
    print(f"   Filename: {filename}")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/visualize/suggestions",
            json={"filename": filename},
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS")
            print(f"   Response keys: {list(data.keys())}")
            
            if 'suggestions' in data:
                print(f"   Suggestions count: {len(data['suggestions'])}")
                if data['suggestions']:
                    print(f"   First suggestion type: {data['suggestions'][0].get('type')}")
        else:
            print(f"   ❌ FAILED")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Step 2: Generate chart
    print("\n2️⃣  Testing /visualize/goal-based endpoint")
    print(f"   Filename: {filename}")
    print(f"   Library: plotly")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/visualize/goal-based",
            json={
                "filename": filename,
                "library": "plotly"
            },
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS")
            print(f"   Response keys: {list(data.keys())}")
            
            # Check for visualization data
            if 'visualization' in data:
                viz = data['visualization']
                print(f"   ✅ Has 'visualization' key")
                print(f"   Visualization keys: {list(viz.keys())}")
                
                if 'figure_json' in viz:
                    fig_json = viz['figure_json']
                    if isinstance(fig_json, str):
                        print(f"   ✅ figure_json is string (length: {len(fig_json)} chars)")
                        # Try to parse it
                        try:
                            parsed = json.loads(fig_json)
                            print(f"   ✅ figure_json is valid JSON")
                            print(f"   Parsed keys: {list(parsed.keys())}")
                            if 'data' in parsed:
                                print(f"   ✅ Has 'data' array (length: {len(parsed['data'])})")
                            if 'layout' in parsed:
                                print(f"   ✅ Has 'layout' object")
                        except:
                            print(f"   ❌ figure_json is NOT valid JSON")
                    elif isinstance(fig_json, dict):
                        print(f"   ✅ figure_json is dict")
                        print(f"   Keys: {list(fig_json.keys())}")
                else:
                    print(f"   ❌ Missing 'figure_json' in visualization")
            else:
                print(f"   ❌ Missing 'visualization' key in response")
                
            # Check success flag
            if 'success' in data:
                print(f"   Success flag: {data['success']}")
                
            # Check selected_chart
            if 'selected_chart' in data:
                chart = data['selected_chart']
                print(f"   Selected chart type: {chart.get('type')}")
            
            # Print full response structure (formatted)
            print(f"\n📊 FULL RESPONSE STRUCTURE:")
            print(json.dumps({k: type(v).__name__ for k, v in data.items()}, indent=2))
            
        else:
            print(f"   ❌ FAILED")
            print(f"   Response: {response.text[:500]}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Step 3: Test with explicit goal
    print("\n3️⃣  Testing /visualize/goal-based with explicit goal")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/visualize/goal-based",
            json={
                "filename": filename,
                "goal": "Show revenue trends over time",
                "library": "plotly"
            },
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS")
            has_viz = 'visualization' in data
            has_success = data.get('success', False)
            print(f"   Has visualization: {has_viz}")
            print(f"   Success flag: {has_success}")
            
            if has_viz:
                has_figure = 'figure_json' in data['visualization']
                print(f"   Has figure_json: {has_figure}")
        else:
            print(f"   ❌ FAILED")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

if __name__ == "__main__":
    test_visualization_flow()
