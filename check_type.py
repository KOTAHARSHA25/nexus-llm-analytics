import os

base_dir = r"c:\Users\mitta\OneDrive\Documents\Major Project\Phase_2\nexus-llm-analytics-dist(Main)\nexus-llm-analytics-dist"
file_path = os.path.join(base_dir, "original paper.docx")

if os.path.exists(file_path):
    size = os.path.getsize(file_path)
    print(f"Size: {size}")
    if size > 0:
        with open(file_path, "rb") as f:
            head = f.read(4)
            print(f"Header: {head}")
            if head == b'PK\x03\x04':
                print("Header matches Zip/Docx format.")
            else:
                print("Header DOES NOT match Zip/Docx format.")
    else:
        print("File is empty.")
else:
    print("File not found (again).")
