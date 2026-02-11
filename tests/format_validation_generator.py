import csv
import json
import random
import os
import pandas as pd
from pathlib import Path

# Setup
DATA_DIR = Path("data/format_validation")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_csv_variants():
    base_name = "test_csv"
    
    # 1. Clean
    pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}).to_csv(DATA_DIR / f"{base_name}_clean.csv", index=False)
    
    # 2. Missing Headers (Just data)
    with open(DATA_DIR / f"{base_name}_no_header.csv", 'w') as f:
        f.write("1,x\n2,y")
        
    # 3. Mixed Types
    with open(DATA_DIR / f"{base_name}_mixed.csv", 'w') as f:
        f.write("col1,col2\n1,text\ntext,2\n3.5,True")
        
    # 4. Large Numbers
    pd.DataFrame({'val': [10**20, -10**20]}).to_csv(DATA_DIR / f"{base_name}_large_nums.csv", index=False)
    
    # 5. Empty
    open(DATA_DIR / f"{base_name}_empty.csv", 'w').close()
    
    # 6. Corrupted (Broken CSV structure)
    with open(DATA_DIR / f"{base_name}_corrupt.csv", 'w') as f:
        f.write("col1,col2\n1,2,3\n4") # Row length mismatch
        
    # 7. Multilingual
    pd.DataFrame({'text': ['こんにちは', 'مرحبا', 'Hello']}).to_csv(DATA_DIR / f"{base_name}_unicode.csv", index=False, encoding='utf-8-sig')
    
    # 8. Special Chars
    pd.DataFrame({'schars': ['@#$%', '^&*()', '|}{']}).to_csv(DATA_DIR / f"{base_name}_special.csv", index=False)
    
    # 9. Duplicate Rows
    pd.DataFrame({'id': [1, 1, 1], 'val': ['a', 'a', 'a']}).to_csv(DATA_DIR / f"{base_name}_dupes.csv", index=False)
    
    # 10. Incorrect Schema (Logic for testing mostly, file itself is valid CSV)
    pd.DataFrame({'age': ['twelve', '20']}).to_csv(DATA_DIR / f"{base_name}_bad_schema.csv", index=False)

def generate_json_variants():
    base_name = "test_json"
    
    # 1. Clean
    with open(DATA_DIR / f"{base_name}_clean.json", 'w') as f:
        json.dump([{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}], f)
        
    # 6. Corrupted
    with open(DATA_DIR / f"{base_name}_corrupt.json", 'w') as f:
        f.write("{'a': 1, ") # Incomplete
        
    # 7. Multilingual
    with open(DATA_DIR / f"{base_name}_unicode.json", 'w', encoding='utf-8') as f:
        json.dump([{'msg': 'こんにちは'}], f, ensure_ascii=False)
        
    # Other variants are similar to CSV logic structure-wise or handled by the generic generator
    
def generate_txt_variants():
    base_name = "test_txt"
    with open(DATA_DIR / f"{base_name}_clean.txt", 'w') as f:
        f.write("Simple text content.\nLine 2.")
    with open(DATA_DIR / f"{base_name}_unicode.txt", 'w', encoding='utf-8') as f:
        f.write("Hello world.\n你好\nLine 3.")
        
if __name__ == "__main__":
    print("Generating CSVs...")
    generate_csv_variants()
    print("Generating JSONs...")
    generate_json_variants()
    print("Generating TXTs...")
    generate_txt_variants()
    print(f"Done. Files in {DATA_DIR.absolute()}")
