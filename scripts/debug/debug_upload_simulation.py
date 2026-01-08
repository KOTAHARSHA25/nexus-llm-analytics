
import requests

URL = "http://localhost:8000/api/upload/"

def test_upload():
    print(f"Uploading dummy.csv to {URL}...")
    
    # Create dummy content
    content = b"col1,col2\nval1,val2"
    files = {'file': ('dummy.csv', content, 'text/csv')}
    
    try:
        response = requests.post(URL, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Response Body: {response.text}")
        
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_upload()
