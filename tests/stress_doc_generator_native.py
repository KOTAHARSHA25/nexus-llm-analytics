import json
from pathlib import Path

DATA_DIR = Path("data/stress_test_docs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_massive_text():
    """Generates a 50KB text file with a specific hidden key."""
    content = "This is a standard clause in the contract. " * 2000
    # The Needle
    content += "\n\nCRITICAL CLAUSE: The release code for the project is 77-ALPHA-OMEGA.\n\n"
    content += "End of document filler. " * 1000
    
    with open(DATA_DIR / "massive_contract.txt", "w") as f:
        f.write(content)
    print("Generated massive_contract.txt")

def generate_complex_json():
    """Generates a deeply nested JSON that mimics a document structure."""
    data = {
        "document": {
            "meta": {"author": "Director Johnson", "version": 1.0},
            "sections": [
                {"title": "Introduction", "content": "..."}
            ]
        }
    }
    # Add depth
    current = data["document"]["sections"][0]
    for i in range(50):
        current["sub_section"] = {"level": i, "text": f"Deep content {i}"}
        current = current["sub_section"]
    
    current["FINAL_SECRET"] = "The treasure is in the depth."
    
    with open(DATA_DIR / "deep_nested_doc.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Generated deep_nested_doc.json")

if __name__ == "__main__":
    generate_massive_text()
    generate_complex_json()
