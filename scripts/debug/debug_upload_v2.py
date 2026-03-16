
import requests
import json

URL = "http://localhost:8000/api/upload/"

def test_upload():
    print(f"Uploading dummy.csv to {URL}...")
    
    content = b"col1,col2\nval1,val2"
    files = {'file': ('dummy.csv', content, 'text/csv')}
    
    try:
        response = requests.post(URL, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Raw Content: {response.content}")
        try:
            print(f"JSON: {response.json()}")
        except:
            print("Response is not JSON")
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    test_upload()
