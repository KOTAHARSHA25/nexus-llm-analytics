
import requests
import json
import sys
import time
import sqlite3
from pathlib import Path
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000/api"
DATA_FILE = Path("data/samples/comprehensive_ecommerce.csv")
DB_PATH = Path("reports/nexus_history.db") # The backend creates it here relative to run dir? 
# Actually backend code says: settings.get_reports_path().parent / DB_FILENAME
# If reports path is 'reports/', parent is root? No, usually 'data/reports'.
# Let's check backend config later, but for now we can check the API for history.

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_step(step, msg):
    print(f"\n[Step {step}] {msg}")

def run_simulation():
    print_header("NEXUS LLM ANALYTICS - END-TO-END USER SIMULATION")
    print(f"Target: {API_URL}")
    print(f"Time:   {datetime.now().isoformat()}")

    # 1. Health Check
    print_step(1, "Checking Backend Health...")
    try:
        # Use root health check for reliability
        r = requests.get("http://localhost:8000/health")
        if r.status_code == 200:
            print("✅ Backend is HEALTHY")
            print(f"   Status: {r.json()}")
        else:
            print(f"❌ Backend Unhealthy: {r.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        sys.exit(1)

    # 2. Upload File
    print_step(2, f"Uploading Data File: {DATA_FILE.name}")
    try:
        with open(DATA_FILE, 'rb') as f:
            files = {'file': (DATA_FILE.name, f, 'text/csv')}
            # Upload endpoint is at /api/upload (mounted at /api/upload with / route)
            r = requests.post(f"{API_URL}/upload/", files=files) 
            if r.status_code != 200:
                 # Try without trailing slash
                 r = requests.post(f"{API_URL}/upload", files=files)

            if r.status_code == 200:
                print("✅ Upload Successful")
                print(f"   Response: {r.json()}")
            else:
                print(f"❌ Upload Failed: {r.status_code} - {r.text}")
                sys.exit(1)
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        sys.exit(1)

    # 3. Simulate Query (Streaming)
    query = "What is the total revenue by Region? Plot this data."
    print_step(3, f"Sending Analyst Query: '{query}'")
    
    session_id = f"sim_user_{int(time.time())}"
    payload = {
        "query": query,
        "session_id": session_id,
        "filename": DATA_FILE.name,
        "preferred_plugin": "Auto-Select Agent"
    }

    print("   ⬇️  STREAMING RESPONSE STARTED  ⬇️")
    print("-" * 60)
    
    full_result_text = ""
    final_json = None
    
    try:
        # Stream endpoint is /api/analyze/stream
        with requests.post(f"{API_URL}/analyze/stream", json=payload, stream=True) as r:
            if r.status_code != 200:
                print(f"❌ Stream Request Failed: {r.status_code} - {r.text}")
                sys.exit(1)
                
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        try:
                            data = json.loads(data_str)
                            
                            # Handle different event types
                            if data.get("step") == "token":
                                token = data.get("token", "")
                                print(token, end="", flush=True)
                                full_result_text += token
                            elif data.get("step") == "error":
                                print(f"\n❌ STREAM ERROR: {data.get('error')}")
                            elif data.get("step") == "complete":
                                print("\n\n✅ STREAM COMPLETE")
                                final_json = data.get("result")
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
         print(f"\n❌ Stream Connection Error: {e}")
         sys.exit(1)

    print("-" * 60)

    # 4. Validation: Plotly
    print_step(4, "Verifying Interactive Dashboard (Phase 14 Feature)...")
    if final_json and final_json.get("metadata", {}).get("visualization"):
        viz = final_json["metadata"]["visualization"]
        # Plotly JSON usually has 'data' and 'layout'
        if "data" in viz and "layout" in viz:
            print("✅ Plotly Visualization Found!")
            print(f"   Chart Title: {viz.get('layout', {}).get('title', {}).get('text', 'Unknown')}")
            print(f"   Data Points: {len(viz.get('data', [])[0].get('x', []))} items")
        else:
             print("⚠️ Visualization found but format unknown.")
             print(str(viz)[:200])
    else:
        print("❌ No Visualization generated!")
        # Check generated code to see why
        print(f"Generated Code:\n{final_json.get('code')}")

    # 5. Validation: SQLite History
    print_step(5, "Verifying SQLite Persistence (Phase 14 Feature)...")
    # We can check via API to be safe (since DB path might vary)
    try:
        r = requests.get(f"{API_URL}/history")
        if r.status_code == 200:
            history = r.json().get("history", [])
            # Look for our query
            found = False
            for item in history:
                if item["query"] == query:
                    found = True
                    print("✅ Query found in persistent history database")
                    print(f"   Timestamp: {item.get('timestamp')}")
                    break
            if not found:
                print("❌ Query NOT found in history!")
                print(f"   Last 3 queries: {[h['query'] for h in history[:3]]}")
        else:
            print(f"❌ Failed to fetch history: {r.status_code}")
    except Exception as e:
        print(f"❌ History Check Failed: {e}")

    print_header("SIMULATION COMPLETE")

if __name__ == "__main__":
    run_simulation()
