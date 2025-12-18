"""
Create a clean distribution zip of Nexus LLM Analytics project
Excludes development files, node_modules, caches, etc.
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime

# Directories and files to EXCLUDE
EXCLUDE_DIRS = {
    'node_modules',
    '__pycache__',
    '.next',
    '.git',
    '.vscode',
    '.idea',
    'venv',
    'env',
    '.env',
    'dist',
    'build',
    '.pytest_cache',
    '.mypy_cache',
    '.tox',
    'htmlcov',
    '.coverage',
    'chroma_db',  # ChromaDB data (large, regenerated)
    'logs',  # Log files
}

EXCLUDE_FILES = {
    '.DS_Store',
    'Thumbs.db',
    '.gitignore',
    '.eslintcache',
    '.env.local',
    '.env.development',
    '.env.production',
    'package-lock.json',
    'yarn.lock',
    'pnpm-lock.yaml',
    '.python-version',
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '.coverage',
    'coverage.xml',
    '*.log',
    '*.db',
    '*.sqlite',
    '*.sqlite3',
}

# Directories to EXCLUDE by path pattern
EXCLUDE_PATTERNS = {
    'reports',  # Generated reports
    'data/uploads',  # User uploaded files (can be large)
    'data/audit',  # Audit logs
    'src/backend/chroma_db',  # Vector database
    'src/backend/logs',  # Backend logs
    'src/backend/reports',  # Generated reports
    'src/backend/data/uploads',  # Backend uploads
}

def should_exclude(path_str, root_dir):
    """Check if a path should be excluded"""
    path = Path(path_str)
    
    # Check if any parent directory is in exclude list
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    
    # Check filename
    if path.name in EXCLUDE_FILES:
        return True
    
    # Check file extensions
    for pattern in EXCLUDE_FILES:
        if '*' in pattern:
            ext = pattern.replace('*', '')
            if path.name.endswith(ext):
                return True
    
    # Check path patterns
    rel_path = os.path.relpath(path_str, root_dir)
    for pattern in EXCLUDE_PATTERNS:
        if pattern in rel_path.replace('\\', '/'):
            return True
    
    return False

def create_distribution_zip():
    """Create distribution zip file"""
    
    # Get project root
    root_dir = Path(__file__).parent.absolute()
    
    # Create zip filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f'nexus-llm-analytics-distribution_{timestamp}.zip'
    zip_path = root_dir / zip_filename
    
    print(f"🎯 Creating distribution package: {zip_filename}")
    print(f"📁 Root directory: {root_dir}")
    print(f"\n📦 Excluding:")
    print(f"  - node_modules (frontend dependencies)")
    print(f"  - __pycache__ (Python cache)")
    print(f"  - .next (Next.js build)")
    print(f"  - chroma_db (vector database - will be regenerated)")
    print(f"  - logs (log files)")
    print(f"  - reports (generated reports)")
    print(f"  - data/uploads (user files)")
    print(f"  - Development files (lock files, cache, etc.)")
    
    # Create zip file
    file_count = 0
    total_size = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # Walk through directory
        for root, dirs, files in os.walk(root_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), root_dir)]
            
            # Add files
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip the zip file itself
                if file_path == str(zip_path):
                    continue
                
                # Skip excluded files
                if should_exclude(file_path, root_dir):
                    continue
                
                # Add to zip
                arcname = os.path.relpath(file_path, root_dir)
                try:
                    zipf.write(file_path, arcname)
                    file_count += 1
                    total_size += os.path.getsize(file_path)
                    
                    # Print progress every 50 files
                    if file_count % 50 == 0:
                        print(f"  ✓ Added {file_count} files...")
                except Exception as e:
                    print(f"  ⚠️  Skipped {arcname}: {e}")
    
    # Get zip file size
    zip_size = os.path.getsize(zip_path)
    
    print(f"\n✅ Distribution package created successfully!")
    print(f"\n📊 Statistics:")
    print(f"  Files included: {file_count}")
    print(f"  Original size: {total_size / (1024*1024):.2f} MB")
    print(f"  Compressed size: {zip_size / (1024*1024):.2f} MB")
    print(f"  Compression ratio: {(1 - zip_size/total_size)*100:.1f}%")
    print(f"\n📦 Package location:")
    print(f"  {zip_path}")
    print(f"\n🚀 Your friend can:")
    print(f"  1. Extract the zip file")
    print(f"  2. Run: cd src/frontend && npm install")
    print(f"  3. Run: pip install -r requirements.txt")
    print(f"  4. Follow SETUP_INSTRUCTIONS.txt")
    
    return str(zip_path)

if __name__ == "__main__":
    try:
        zip_path = create_distribution_zip()
        print(f"\n✨ Done! Share {os.path.basename(zip_path)} with your friend.")
    except Exception as e:
        print(f"\n❌ Error creating zip: {e}")
        import traceback
        traceback.print_exc()
