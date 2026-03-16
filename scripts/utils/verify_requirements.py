import os
import ast
import sys
import pkg_resources
from typing import Set, Dict, List

# Standard library modules (simplified list for 3.8+)
STD_LIB = {
    'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'contextlib', 'copy', 'csv', 'datetime',
    'decimal', 'email', 'enum', 'functools', 'glob', 'hashlib', 'html', 'http', 'importlib', 'inspect',
    'io', 'json', 'logging', 'math', 'mimetypes', 'multiprocessing', 'os', 'pathlib', 'pickle', 'platform',
    'pprint', 'random', 're', 'shutil', 'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string',
    'subprocess', 'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest', 'urllib',
    'uuid', 'warnings', 'weakref', 'xml', 'zipfile', 'zoneinfo', 'typing_extensions'
}

# Mapping common import names to package names (e.g. yaml -> PyYAML)
IMPORT_MAP = {
    'yaml': 'PyYAML',
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'bs4': 'beautifulsoup4',
    'dotenv': 'python-dotenv',
    'fitz': 'pymupdf',
    'chromadb': 'chromadb',
    'plotly': 'plotly',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'statsmodels': 'statsmodels',
    'seaborn': 'seaborn',
    'matplotlib': 'matplotlib',
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'pydantic': 'pydantic',
    'requests': 'requests',
    'aiofiles': 'aiofiles',
    'werkzeug': 'werkzeug',
    'langchain_ollama': 'langchain-ollama',
    'langchain_community': 'langchain-community',
    'PyPDF2': 'PyPDF2',
    'pdfplumber': 'pdfplumber',
    'openpyxl': 'openpyxl',
    'docx': 'python-docx',
    'pptx': 'python-pptx',
    'striprtf': 'striprtf',
    'reportlab': 'reportlab',
    'jinja2': 'jinja2',
    'RestrictedPython': 'RestrictedPython',
    'psutil': 'psutil',
    'structlog': 'structlog',
    'colorlog': 'colorlog',
    'pkg_resources': 'setuptools',
    'magic': 'python-magic',
    'chardet': 'chardet',
    'html2text': 'html2text',
    'sentence_transformers': 'sentence-transformers',
    'multipart': 'python-multipart'
}

def get_imported_modules(root_dir: str) -> Set[str]:
    imports = set()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=path)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module.split('.')[0])
                except Exception as e:
                    print(f"Warning: Could not parse {path}: {e}")
    return imports

def get_requirements(req_file: str) -> Set[str]:
    reqs = set()
    if not os.path.exists(req_file):
        return reqs
        
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle version specifiers (e.g., pandas>=2.1.0)
                pkg = line.split('>=')[0].split('==')[0].split('[')[0].strip().lower()
                reqs.add(pkg)
    return reqs

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    src_dir = os.path.join(base_dir, 'src')
    req_file = os.path.join(base_dir, 'requirements.txt')
    
    print(f"Scanning imports in {src_dir}...")
    imported = get_imported_modules(src_dir)
    
    print(f"Reading {req_file}...")
    listed_reqs = get_requirements(req_file)
    
    # Analyze
    missing = set()
    unknown = set()
    
    for mod in imported:
        if mod in STD_LIB:
            continue
        if mod in {'backend', 'src', 'tests', 'scripts'}: # Internal modules
            continue
            
        # Map import name to package name
        pkg_name = IMPORT_MAP.get(mod, mod).lower()
        
        # Check if package is in listed requirements
        found = False
        for req in listed_reqs:
            if req == pkg_name or req.replace('-', '_') == pkg_name.replace('-', '_'):
                found = True
                break
        
        if not found:
            # Maybe it's installed but not listed? Or mapped incorrectly?
            missing.add(f"{mod} (package: {pkg_name})")

    print("\n=== Analysis Result ===")
    if missing:
        print("POTENTIALLY MISSING PACKAGES:")
        for m in sorted(missing):
            print(f" - {m}")
    else:
        print("✅ All third-party imports seem to be covered in requirements.txt")

if __name__ == "__main__":
    main()
