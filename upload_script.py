import requests
import os

file_path = "data/samples/MultiAgent_Demo_Data.csv"
url = "http://localhost:8000/api/upload/"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, 'rb') as f:
    files = {'file': (os.path.basename(file_path), f, 'text/csv')}
    try:
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
