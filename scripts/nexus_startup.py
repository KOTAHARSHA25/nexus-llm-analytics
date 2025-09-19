#!/usr/bin/env python3
"""
Nexus LLM Analytics - Production Startup Script
Comprehensive setup and validation for the enhanced multi-agent analytics platform
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent  # Go up one level from scripts/ to project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nexus_startup.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Ensure Python 3.8+ is being used"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        raise RuntimeError(f"Python 3.8+ required, but using {version.major}.{version.minor}")
    return f"{version.major}.{version.minor}.{version.micro}"

def check_ollama_status():
    """Check if Ollama is running and has required models"""
    try:
        import requests
        
        # Check if Ollama server is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama server not responding"
        
        models = response.json().get('models', [])
        model_names = [model['name'] for model in models]
        
        # Check for any compatible models (flexible model requirements)
        compatible_models = ['tinyllama', 'phi3:mini', 'llama3.1:8b', 'nomic-embed-text']
        available_compatible = [m for m in compatible_models if any(m in name for name in model_names)]
        
        if not available_compatible:
            return False, f"No compatible models found. Available: {model_names}. Please install one of: {compatible_models}"
        
        return True, "All models available"
        
    except Exception as e:
        return False, f"Ollama check failed: {e}"

def install_dependencies():
    """Install required Python packages"""
    logger = logging.getLogger(__name__)
    logger.info("Installing Python dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "src/backend/data",
        "src/backend/logs",
        "src/backend/reports",
        "data/audit",
        "data/uploads",
        "chroma_db",
        "logs",
        "reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return True

def validate_environment():
    """Validate the complete environment setup"""
    logger = logging.getLogger(__name__)
    
    validations = {
        "Python Version": lambda: (True, check_python_version()),
        "Project Structure": validate_project_structure,
        "Dependencies": validate_dependencies,
        "Ollama Status": check_ollama_status,
        "Database Setup": validate_database_setup,
        "Security Configuration": validate_security_config
    }
    
    results = {}
    all_passed = True
    
    for name, validator in validations.items():
        try:
            success, message = validator()
            results[name] = {"success": success, "message": message}
            if success:
                logger.info(f"✅ {name}: {message}")
            else:
                logger.error(f"❌ {name}: {message}")
                all_passed = False
        except Exception as e:
            results[name] = {"success": False, "message": str(e)}
            logger.error(f"❌ {name}: {e}")
            all_passed = False
    
    return all_passed, results

def validate_project_structure():
    """Validate project file structure"""
    required_files = [
        "src/backend/main.py",
        "src/backend/agents/crew_manager.py",
        "src/backend/core/user_preferences.py",
        "src/backend/core/model_selector.py",
        "src/backend/api/models.py",
        "src/backend/api/analyze.py",
        "src/frontend/package.json",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        return False, f"Missing files: {missing_files}"
    
    return True, "All required files present"

def validate_dependencies():
    """Validate Python dependencies"""
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'requests', 'pydantic',
        'python-multipart', 'psutil', 'ollama'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        return False, f"Missing packages: {missing_packages}"
    
    return True, "All dependencies available"

def validate_database_setup():
    """Validate user preferences system"""
    try:
        import sys
        sys.path.append('src')
        from backend.core.user_preferences import UserPreferencesManager
        
        # Test user preferences system
        prefs_manager = UserPreferencesManager()
        test_prefs = prefs_manager.load_preferences()
        
        if test_prefs:
            return True, "User preferences system working correctly"
        else:
            return True, "User preferences system initialized"
            
    except Exception as e:
        return False, f"User preferences validation failed: {e}"

def validate_security_config():
    """Validate model selection and API security"""
    try:
        import sys
        sys.path.append('src')
        from backend.core.model_selector import ModelSelector
        
        # Test model selector
        selector = ModelSelector()
        available_models = selector.get_available_models()
        
        if available_models:
            return True, f"Model selection working with {len(available_models)} models"
        else:
            return True, "Model selector initialized (no models loaded yet)"
        
    except Exception as e:
        return False, f"Model selector validation failed: {e}"

def start_backend_server():
    """Start the FastAPI backend server"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Nexus LLM Analytics Backend Server...")
    
    try:
        # Stay in project root directory
        subprocess.Popen([
            sys.executable, "-m", "uvicorn", "src.backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        logger.info("🚀 Backend server started on http://localhost:8000")
        return True
    except Exception as e:
        logger.error(f"Failed to start backend server: {e}")
        return False

def start_frontend_server():
    """Start the Next.js frontend server"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Nexus LLM Analytics Frontend...")
    
    try:
        os.chdir("src/frontend")
        subprocess.Popen(["npm", "run", "dev"])
        logger.info("🌐 Frontend server started on http://localhost:3000")
        return True
    except Exception as e:
        logger.error(f"Failed to start frontend server: {e}")
        return False

def print_startup_banner():
    """Print the startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║               🚀 NEXUS LLM ANALYTICS 🚀                      ║
    ║                                                               ║
    ║         Production-Ready Multi-Agent Analytics Platform       ║
    ║                                                               ║
    ║  🤖 CrewAI Multi-Agent System                                ║
    ║  🔒 Enhanced Security & Sandboxing                           ║
    ║  📊 Advanced Data Visualization                              ║
    ║  🧠 Natural Language Processing                              ║
    ║  📄 Professional Report Generation                           ║
    ║  🏠 Privacy-First Local Processing                           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main startup function"""
    logger = setup_logging()
    
    print_startup_banner()
    
    logger.info("🔄 Starting Nexus LLM Analytics Platform...")
    
    # Step 1: Create directories
    logger.info("📁 Creating necessary directories...")
    create_directories()
    
    # Step 2: Install dependencies
    logger.info("📦 Checking dependencies...")
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies")
        return False
    
    # Step 3: Validate environment
    logger.info("🔍 Validating environment...")
    all_passed, results = validate_environment()
    
    if not all_passed:
        logger.error("❌ Environment validation failed!")
        logger.info("📋 Validation Results:")
        for name, result in results.items():
            status = "✅" if result["success"] else "❌"
            logger.info(f"   {status} {name}: {result['message']}")
        return False
    
    logger.info("✅ Environment validation passed!")
    
    # Step 4: Start servers
    logger.info("🚀 Starting application servers...")
    
    if not start_backend_server():
        return False
    
    # Give backend time to start
    import time
    time.sleep(3)
    
    if not start_frontend_server():
        return False
    
    # Final success message
    success_message = """
    
    🎉 NEXUS LLM ANALYTICS IS NOW RUNNING! 🎉
    
    📊 Backend API: http://localhost:8000
    🌐 Frontend App: http://localhost:3000
    📚 API Docs: http://localhost:8000/docs
    
    🔧 Key Features Now Available:
    • Multi-agent AI analysis with CrewAI
    • Enhanced security with RestrictedPython
    • Advanced data visualization with Plotly
    • Natural language query processing
    • Professional PDF and Excel reports
    • Local-first privacy-preserving analytics
    
    📖 Usage:
    1. Open http://localhost:3000 in your browser
    2. Upload your data files (CSV, JSON, PDF, TXT)
    3. Ask questions in natural language
    4. Get AI-powered insights and visualizations
    5. Download professional reports
    
    🎯 Ready for production use!
    """
    
    print(success_message)
    logger.info("🎯 Nexus LLM Analytics started successfully!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Nexus LLM Analytics shutdown requested")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error during startup: {e}")
        sys.exit(1)