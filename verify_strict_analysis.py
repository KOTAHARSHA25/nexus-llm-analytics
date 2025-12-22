
import requests
import json

def test_analysis():
    url = "http://localhost:8000/api/analyze/"
    
    # Payload matching the frontend request format
    payload = {
        "query": "what is the most listened track",
        "filename": "spotify_data_clean.csv",
        "file_type": "csv"
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Analysis Success!")
            print("-" * 50)
            print(result.get("result", "No result field"))
            print("-" * 50)
            
            # Validation
            output = result.get("result", "")
            if "Data Quality Status" in output:
                print("✅ Found 'Data Quality Status' in output.")
            else:
                print("❌ 'Data Quality Status' missing from output.")
                
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_analysis()
