
import requests
try:
    res = requests.post("http://localhost:8000/api/analyze/", json={"query": "test"})
    print(f"Status: {res.status_code}")
    print(f"Body: {res.text}")
except Exception as e:
    print(e)
