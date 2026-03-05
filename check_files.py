import os

base_dir = r"c:\Users\mitta\OneDrive\Documents\Major Project\Phase_2\nexus-llm-analytics-dist(Main)\nexus-llm-analytics-dist"

print(f"Listing: {base_dir}")
try:
    files = os.listdir(base_dir)
    for f in files:
        if "original" in f.lower():
            print(f"Found: '{f}' (Request: original paper.docx)")
            print(f"Bytes: {f.encode('utf-8')}")
            full_path = os.path.join(base_dir, f)
            print(f"Exists? {os.path.exists(full_path)}")
except Exception as e:
    print(f"Error listing: {e}")
