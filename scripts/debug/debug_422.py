
import requests

URL = "http://localhost:8000/api/upload/"

def test_missing_file():
    print(f"Sending request without file to {URL}...")
    try:
        # Send empty post to trigger 422
        response = requests.post(URL, data={})
            
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_missing_file()
