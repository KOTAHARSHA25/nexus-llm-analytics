
import requests
import os

URL = "http://localhost:8000/api/upload/"
FILE_PATH = "src/backend/tests/data/ecommerce_comprehensive.csv"

def test_upload():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print(f"Uploading {FILE_PATH} to {URL}...")
    try:
        with open(FILE_PATH, 'rb') as f:
            files = {'file': (os.path.basename(FILE_PATH), f, 'text/csv')}
            response = requests.post(URL, files=files)
            
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Upload Successful!")
        else:
            print("❌ Upload Failed!")
            
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_upload()
