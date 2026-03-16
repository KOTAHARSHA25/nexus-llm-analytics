
import requests
import json

url = "http://localhost:8000/analyze"
payload = {
    "query": "test query",
    "text_data": "sample data",
    "force_refresh": True,
    "review_level": "optional"
}

try:
    print(f"Sending request to {url} with payload: {payload}")
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:", response.json().get("status"))
        print("Success! API accepted new parameters.")
    else:
        print("Error:", response.text)
except Exception as e:
    print(f"Connection failed: {e}")
