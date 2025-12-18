#!/usr/bin/env python3
"""
System Health Check Script for Nexus LLM Analytics

This script checks if your system meets the requirements to run Nexus LLM Analytics,
including RAM availability, Python dependencies, and Ollama installation.
"""

import sys
import os
import subprocess
from pathlib import Path

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print the script header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🔍 Nexus LLM Analytics - System Health Check{Colors.END}")
    print("=" * 60)

def check_python_version():
    """Check if Python version is supported"""
    print(f"\n{Colors.BOLD}📋 Checking Python Version...{Colors.END}")
    version = sys.version_info
    
    if version >= (3, 8):
        print(f"{Colors.GREEN}✅ Python {version.major}.{version.minor}.{version.micro} - Supported{Colors.END}")
    else:
        print(f"{Colors.RED}❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+{Colors.END}")
        return False
    
    # Check if we're in a virtual environment
    return check_virtual_environment()

def check_virtual_environment():
    """Check if running in a virtual environment"""
    print(f"\n{Colors.BOLD}🐍 Checking Virtual Environment...{Colors.END}")
    
    # Check various indicators of virtual environment
    in_venv = (
        hasattr(sys, 'real_prefix') or  # virtualenv
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or  # venv
        'VIRTUAL_ENV' in os.environ  # environment variable
    )
    
    if in_venv:
        venv_path = os.environ.get('VIRTUAL_ENV', sys.prefix)
        print(f"{Colors.GREEN}✅ Virtual environment active: {venv_path}{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}❌ No virtual environment detected{Colors.END}")
        print(f"{Colors.YELLOW}📋 Please activate your virtual environment:{Colors.END}")
        
        # Check if env directory exists
        env_dir = Path("env")
        if env_dir.exists():
            print(f"   Windows: .\\env\\Scripts\\activate")
            print(f"   Linux/Mac: source env/bin/activate")
        else:
            print(f"   Create new: python -m venv env")
            print(f"   Then activate: .\\env\\Scripts\\activate (Windows)")
        
        return False

def check_memory():
    """Check available system memory"""
    print(f"\n{Colors.BOLD}💾 Checking System Memory...{Colors.END}")
    
    if not HAS_PSUTIL:
        print(f"{Colors.YELLOW}⚠️  psutil not installed - install with: pip install psutil{Colors.END}")
        print(f"{Colors.YELLOW}⚠️  Assuming sufficient memory for basic operation{Colors.END}")
        return True, "ollama/phi3:mini"
    
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    
    print(f"Total RAM: {total_gb:.1f} GB")
    print(f"Available RAM: {available_gb:.1f} GB")
    
    if available_gb >= 6:
        print(f"{Colors.GREEN}✅ Excellent! Can run Llama 3.1 8B model{Colors.END}")
        recommended_model = "ollama/llama3.1:8b"
    elif available_gb >= 4:
        print(f"{Colors.GREEN}✅ Good! Can run Phi-3 Mini model{Colors.END}")
        recommended_model = "ollama/phi3:mini"
    elif available_gb >= 2:
        print(f"{Colors.YELLOW}⚠️  Limited RAM. Consider Phi-3 Mini with optimizations{Colors.END}")
        recommended_model = "ollama/phi3:mini"
    else:
        print(f"{Colors.RED}❌ Insufficient RAM. Minimum 2GB required{Colors.END}")
        recommended_model = None
    
    return available_gb >= 2, recommended_model

def check_dependencies():
    """Check if required Python packages are installed"""
    print(f"\n{Colors.BOLD}📦 Checking Python Dependencies...{Colors.END}")
    
    required_packages = [
        'fastapi', 'crewai', 'pandas', 'plotly', 'chromadb', 
        'uvicorn', 'python-dotenv', 'pydantic'
    ]
    
    if not HAS_PKG_RESOURCES:
        print(f"{Colors.YELLOW}⚠️  pkg_resources not available - using pip list instead{Colors.END}")
        return check_dependencies_with_pip(required_packages)
    
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.require(package)
            installed_packages.append(package)
            print(f"{Colors.GREEN}✅ {package}{Colors.END}")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
            print(f"{Colors.RED}❌ {package} - Not installed{Colors.END}")
    
    if missing_packages:
        print(f"\n{Colors.YELLOW}📋 Install missing packages with:{Colors.END}")
        print(f"pip install {' '.join(missing_packages)}")
    
    return len(missing_packages) == 0

def check_dependencies_with_pip(required_packages):
    """Alternative dependency check using pip list"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"{Colors.RED}❌ Failed to run pip list{Colors.END}")
            return False
        
        installed = [line.split()[0].lower() for line in result.stdout.split('\n') 
                    if line and not line.startswith('-')]
        
        missing_packages = []
        for package in required_packages:
            package_name = package.lower().replace('-', '_')
            if package_name in installed or package.lower() in installed:
                print(f"{Colors.GREEN}✅ {package}{Colors.END}")
            else:
                missing_packages.append(package)
                print(f"{Colors.RED}❌ {package} - Not installed{Colors.END}")
        
        if missing_packages:
            print(f"\n{Colors.YELLOW}📋 Install missing packages with:{Colors.END}")
            print(f"pip install {' '.join(missing_packages)}")
        
        return len(missing_packages) == 0
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error checking dependencies: {e}{Colors.END}")
        return False

def check_ollama():
    """Check if Ollama is installed and running"""
    print(f"\n{Colors.BOLD}🤖 Checking Ollama Installation...{Colors.END}")
    
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ Ollama installed: {result.stdout.strip()}{Colors.END}")
            
            # Check if ollama is running
            try:
                result = subprocess.run(['ollama', 'list'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"{Colors.GREEN}✅ Ollama server is running{Colors.END}")
                    
                    # List installed models
                    if result.stdout.strip():
                        print("\nInstalled models:")
                        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                            if line.strip():
                                model_name = line.split()[0]
                                print(f"  • {model_name}")
                    else:
                        print(f"{Colors.YELLOW}⚠️  No models installed yet{Colors.END}")
                    
                    return True
                else:
                    print(f"{Colors.RED}❌ Ollama server not running. Start with: ollama serve{Colors.END}")
                    return False
            except subprocess.TimeoutExpired:
                print(f"{Colors.RED}❌ Ollama server not responding{Colors.END}")
                return False
        else:
            print(f"{Colors.RED}❌ Ollama not installed{Colors.END}")
            return False
    except FileNotFoundError:
        print(f"{Colors.RED}❌ Ollama not found. Install from: https://ollama.ai{Colors.END}")
        return False
    except subprocess.TimeoutExpired:
        print(f"{Colors.RED}❌ Ollama command timed out{Colors.END}")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    print(f"\n{Colors.BOLD}📁 Checking Project Structure...{Colors.END}")
    
    required_dirs = [
        'src/backend',
        'src/frontend', 
        'data',
        'scripts',
        'docs',
        'tests'
    ]
    
    required_files = [
        'requirements.txt',
        'pyproject.toml',
        'src/backend/main.py',
        'config/.env.example'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"{Colors.GREEN}✅ {dir_path}/{Colors.END}")
        else:
            print(f"{Colors.RED}❌ {dir_path}/ - Missing{Colors.END}")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"{Colors.GREEN}✅ {file_path}{Colors.END}")
        else:
            print(f"{Colors.RED}❌ {file_path} - Missing{Colors.END}")
            all_good = False
    
    return all_good

def print_recommendations(memory_ok, recommended_model, deps_ok, ollama_ok, structure_ok):
    """Print final recommendations"""
    print(f"\n{Colors.BOLD}🎯 Recommendations:{Colors.END}")
    print("=" * 60)
    
    if not deps_ok:
        print(f"{Colors.YELLOW}1. Install missing Python dependencies:{Colors.END}")
        print("   pip install -r requirements.txt")
    
    if not ollama_ok:
        print(f"{Colors.YELLOW}2. Install and start Ollama:{Colors.END}")
        print("   - Download from: https://ollama.ai")
        print("   - Run: ollama serve")
        
    if memory_ok and recommended_model:
        print(f"{Colors.YELLOW}3. Install recommended AI model:{Colors.END}")
        print(f"   ollama pull {recommended_model.replace('ollama/', '')}")
        print("   ollama pull nomic-embed-text")
    
    if not structure_ok:
        print(f"{Colors.YELLOW}4. Fix project structure issues listed above{Colors.END}")
    
    if all([memory_ok, deps_ok, ollama_ok, structure_ok]):
        print(f"{Colors.GREEN}🎉 All checks passed! You're ready to run Nexus LLM Analytics!{Colors.END}")
        print(f"\n{Colors.BOLD}Start the application:{Colors.END}")
        print("   python -m uvicorn src.backend.main:app --reload --host 127.0.0.1 --port 8000")
        print("   cd src/frontend && npm run dev")

def main():
    """Main health check function"""
    print_header()
    
    python_ok = check_python_version()
    memory_ok, recommended_model = check_memory()
    deps_ok = check_dependencies()
    ollama_ok = check_ollama()
    structure_ok = check_project_structure()
    
    print_recommendations(memory_ok, recommended_model, deps_ok, ollama_ok, structure_ok)
    
    # Summary
    checks_passed = sum([python_ok, memory_ok, deps_ok, ollama_ok, structure_ok])
    total_checks = 5
    
    print(f"\n{Colors.BOLD}📊 Health Check Summary: {checks_passed}/{total_checks} checks passed{Colors.END}")
    
    if checks_passed == total_checks:
        print(f"{Colors.GREEN}🟢 System is ready!{Colors.END}")
        sys.exit(0)
    else:
        print(f"{Colors.YELLOW}🟡 Some issues need attention{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()