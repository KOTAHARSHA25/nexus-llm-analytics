
import requests
import json

url = "http://localhost:8000/api/analyze/"
payload = {
    "query": "Calculate the average sales and forecast next month based on trends.",
    "filename": "sales_data.csv"
}

try:
    res = requests.post(url, json=payload)
    print(f"Status: {res.status_code}")
    print("Response:")
    print(json.dumps(res.json(), indent=2))
except Exception as e:
    print(e)
