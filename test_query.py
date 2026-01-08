import requests
import json
import time

url = "http://localhost:8000/api/analyze/"
payload = {
    "query": "Forecast the total daily sales for the next 3 days.",
    "filename": "MultiAgent_Demo_Data.csv"
}

print(f"Sending query to {url}...")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    start_time = time.time()
    response = requests.post(url, json=payload, timeout=120)  # Long timeout for LLM
    print(f"Time taken: {time.time() - start_time:.2f}s")
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("\nResponse:")
        print(json.dumps(data, indent=2))
        
        # Validation checks
        result_text = data.get("result", "")
        agent = data.get("agent", "Unknown")
        print(f"\nAgent used: {agent}")
        if "forecast" in result_text.lower() or "sales" in result_text.lower():
            print("✅ Verification Passed: Response contains forecast/sales data.")
        else:
            print("⚠️ Verification Warning: Response might not contain forecast.")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Exception: {e}")
