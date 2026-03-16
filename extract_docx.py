import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def extract_text(docx_path, output_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
        
        tree = ET.fromstring(xml_content)
        
        paragraphs = []
        for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
            texts = [node.text for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(paragraphs))
            
        return f"Successfully extracted text to {output_path}"
    except Exception as e:
        return f"Error reading {docx_path}: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_docx.py <path_to_docx> <output_txt_path>")
        sys.exit(1)
    
    docx_path = sys.argv[1]
    output_path = sys.argv[2]
    print(extract_text(docx_path, output_path))
