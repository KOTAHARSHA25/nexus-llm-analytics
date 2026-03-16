
import requests
import time
import json

def test_ollama():
    print("Testing Ollama API...")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi3:mini",
        "prompt": "Hello, are you working?",
        "stream": False
    }
    
    try:
        start = time.time()
        response = requests.post(url, json=payload, timeout=30)
        end = time.time()
        
        if response.status_code == 200:
            print(f"✅ Success! Response time: {end - start:.2f}s")
            print(f"Response: {response.json().get('response')}")
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_ollama()
