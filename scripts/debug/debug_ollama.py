
import requests
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from backend.core.config import get_settings

try:
    settings = get_settings()
    url = settings.ollama_base_url
    print(f"Target URL: {url}")
    
    print(f"Querying {url}/api/tags ...")
    response = requests.get(f"{url}/api/tags", timeout=5)
    print(f"Status Code: {response.status_code}")
    
    data = response.json()
    print("Response Keys:", data.keys())
    
    models = data.get("models", [])
    print(f"Models Found: {len(models)}")
    
    for m in models:
        print(f" - Name: {m.get('name')}, Size: {m.get('size')}")

except Exception as e:
    print(f"ERROR: {e}")
