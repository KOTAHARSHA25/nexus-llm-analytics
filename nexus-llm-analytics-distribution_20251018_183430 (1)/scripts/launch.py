#!/usr/bin/env python3
"""
Nexus LLM Analytics - Application Launcher

This script handles the complete startup process for Nexus LLM Analytics,
including environment setup, dependency checks, and launching both backend and frontend.
"""

import os
import sys
import subprocess
import time
import signal
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class NexusLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_root = Path(__file__).parent.parent
        
    def print_banner(self):
        """Print the application banner"""
        banner = f"""
{Colors.BOLD}{Colors.BLUE}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🚀 NEXUS LLM ANALYTICS - Application Launcher 🚀         ║
║                                                              ║
║    Multi-Agent Data Analysis Assistant                       ║
║    Local-First • Privacy-Preserving • AI-Powered            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}
        """
        print(banner)

    def check_requirements(self):
        """Check if system meets requirements"""
        print(f"{Colors.BOLD}🔍 Checking system requirements...{Colors.END}")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print(f"{Colors.RED}❌ Python 3.8+ required. Current: {sys.version}{Colors.END}")
            return False
        
        # Check if requirements.txt exists
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print(f"{Colors.RED}❌ requirements.txt not found{Colors.END}")
            return False
        
        # Check if main backend file exists
        main_file = self.project_root / "src" / "backend" / "main.py"
        if not main_file.exists():
            print(f"{Colors.RED}❌ Backend main.py not found at {main_file}{Colors.END}")
            return False
        
        # Check if frontend package.json exists
        package_json = self.project_root / "src" / "frontend" / "package.json"
        if not package_json.exists():
            print(f"{Colors.RED}❌ Frontend package.json not found{Colors.END}")
            return False
        
        print(f"{Colors.GREEN}✅ System requirements check passed{Colors.END}")
        return True

    def check_ollama(self):
        """Check if Ollama is available and required models are installed"""
        print(f"{Colors.BOLD}🤖 Checking Ollama...{Colors.END}")
        
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ Ollama is installed{Colors.END}")
                
                # Check if server is running
                try:
                    result = subprocess.run(['ollama', 'list'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"{Colors.GREEN}✅ Ollama server is running{Colors.END}")
                        
                        # Check required models
                        models_output = result.stdout
                        return self.verify_required_models(models_output)
                    else:
                        print(f"{Colors.YELLOW}⚠️  Ollama server not running. Starting...{Colors.END}")
                        return self.start_ollama()
                except subprocess.TimeoutExpired:
                    print(f"{Colors.YELLOW}⚠️  Ollama server not responding. Starting...{Colors.END}")
                    return self.start_ollama()
            else:
                print(f"{Colors.RED}❌ Ollama not working properly{Colors.END}")
                return False
        except FileNotFoundError:
            print(f"{Colors.RED}❌ Ollama not installed. Please install from https://ollama.ai{Colors.END}")
            print(f"\n{Colors.BOLD}Installation instructions:{Colors.END}")
            print(f"  Windows: Download from https://ollama.ai/download")
            print(f"  macOS:   brew install ollama")
            print(f"  Linux:   curl https://ollama.ai/install.sh | sh")
            return False

    def verify_required_models(self, models_output):
        """Verify that required LLM models are installed"""
        print(f"{Colors.BOLD}🔍 Checking required models...{Colors.END}")
        
        required_models = {
            'llama3.1:8b': 'Primary LLM (Analysis, RAG, Visualization)',
            'phi3:mini': 'Review LLM (Quality Control)',
            'nomic-embed-text': 'Embeddings (ChromaDB/RAG)'
        }
        
        missing_models = []
        installed_models = []
        
        for model_name, purpose in required_models.items():
            # Check if model appears in the list (handle different naming formats)
            model_base = model_name.split(':')[0]
            if model_base in models_output.lower() or model_name in models_output.lower():
                print(f"{Colors.GREEN}  ✅ {model_name} - {purpose}{Colors.END}")
                installed_models.append(model_name)
            else:
                print(f"{Colors.RED}  ❌ {model_name} - {purpose} (MISSING){Colors.END}")
                missing_models.append(model_name)
        
        if missing_models:
            print(f"\n{Colors.YELLOW}⚠️  Missing required models for full functionality!{Colors.END}")
            print(f"\n{Colors.BOLD}To install missing models, run:{Colors.END}")
            for model in missing_models:
                print(f"  ollama pull {model}")
            print(f"\n{Colors.YELLOW}Note: The application will start but AI features will be limited.{Colors.END}")
            
            # Ask user if they want to continue
            try:
                response = input(f"\n{Colors.BOLD}Continue anyway? (y/n): {Colors.END}").lower()
                if response != 'y':
                    print(f"{Colors.YELLOW}Exiting... Please install models and try again.{Colors.END}")
                    return False
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Cancelled by user{Colors.END}")
                return False
        else:
            print(f"{Colors.GREEN}✅ All required models are installed!{Colors.END}")
        
        return True

    def start_ollama(self):
        """Start Ollama server in background"""
        try:
            print(f"{Colors.YELLOW}🔄 Starting Ollama server...{Colors.END}")
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Verify it's running
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ Ollama server started successfully{Colors.END}")
                
                # Check models after starting server
                return self.verify_required_models(result.stdout)
            else:
                print(f"{Colors.RED}❌ Failed to start Ollama server{Colors.END}")
                return False
        except Exception as e:
            print(f"{Colors.RED}❌ Error starting Ollama: {e}{Colors.END}")
            return False

    def setup_environment(self):
        """Setup environment variables"""
        print(f"{Colors.BOLD}⚙️  Setting up environment...{Colors.END}")
        
        env_file = self.project_root / ".env"
        env_example = self.project_root / "config" / ".env.example"
        
        if not env_file.exists() and env_example.exists():
            print(f"{Colors.YELLOW}📄 Creating .env from template...{Colors.END}")
            import shutil
            shutil.copy(env_example, env_file)
            print(f"{Colors.GREEN}✅ .env file created{Colors.END}")
        
        return True

    def install_dependencies(self, backend_only=False):
        """Install Python and Node dependencies"""
        print(f"{Colors.BOLD}📦 Installing dependencies...{Colors.END}")
        
        # Install Python dependencies
        try:
            print(f"{Colors.YELLOW}🐍 Installing Python dependencies...{Colors.END}")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                                  cwd=self.project_root, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ Python dependencies installed{Colors.END}")
            else:
                print(f"{Colors.RED}❌ Failed to install Python dependencies{Colors.END}")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"{Colors.RED}❌ Error installing Python dependencies: {e}{Colors.END}")
            return False
        
        if backend_only:
            return True
        
        # Install Node dependencies
        frontend_dir = self.project_root / "src" / "frontend"
        if frontend_dir.exists() and (frontend_dir / "package.json").exists():
            try:
                print(f"{Colors.YELLOW}📦 Installing Node.js dependencies...{Colors.END}")
                result = subprocess.run(['npm', 'install'], cwd=frontend_dir, 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"{Colors.GREEN}✅ Node.js dependencies installed{Colors.END}")
                else:
                    print(f"{Colors.RED}❌ Failed to install Node.js dependencies{Colors.END}")
                    print(result.stderr)
                    return False
            except Exception as e:
                print(f"{Colors.RED}❌ Error installing Node.js dependencies: {e}{Colors.END}")
                return False
        
        return True

    def start_backend(self):
        """Start the FastAPI backend server"""
        print(f"{Colors.BOLD}🔧 Starting backend server...{Colors.END}")
        
        try:
            cmd = [
                sys.executable, '-m', 'uvicorn', 
                'src.backend.main:app',
                '--reload',
                '--host', '127.0.0.1',
                '--port', '8000'
            ]
            
            self.backend_process = subprocess.Popen(
                cmd, cwd=self.project_root,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment and check if it started
            time.sleep(3)
            if self.backend_process.poll() is None:
                print(f"{Colors.GREEN}✅ Backend server started on http://127.0.0.1:8000{Colors.END}")
                return True
            else:
                stdout, stderr = self.backend_process.communicate()
                print(f"{Colors.RED}❌ Backend server failed to start{Colors.END}")
                print(f"Error: {stderr}")
                return False
                
        except Exception as e:
            print(f"{Colors.RED}❌ Error starting backend: {e}{Colors.END}")
            return False

    def start_frontend(self):
        """Start the Next.js frontend development server"""
        print(f"{Colors.BOLD}🎨 Starting frontend development server...{Colors.END}")
        
        frontend_dir = self.project_root / "src" / "frontend"
        
        if not frontend_dir.exists():
            print(f"{Colors.YELLOW}⚠️  Frontend directory not found, skipping...{Colors.END}")
            return True
        
        try:
            self.frontend_process = subprocess.Popen(
                ['npm', 'run', 'dev'],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment and check if it started
            time.sleep(5)
            if self.frontend_process.poll() is None:
                print(f"{Colors.GREEN}✅ Frontend server started on http://localhost:3000{Colors.END}")
                return True
            else:
                stdout, stderr = self.frontend_process.communicate()
                print(f"{Colors.RED}❌ Frontend server failed to start{Colors.END}")
                print(f"Error: {stderr}")
                return False
                
        except Exception as e:
            print(f"{Colors.RED}❌ Error starting frontend: {e}{Colors.END}")
            return False

    def cleanup(self):
        """Clean up running processes"""
        print(f"\n{Colors.YELLOW}🧹 Cleaning up...{Colors.END}")
        
        if self.backend_process and self.backend_process.poll() is None:
            print("Stopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process and self.frontend_process.poll() is None:
            print("Stopping frontend server...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        print(f"{Colors.GREEN}✅ Cleanup complete{Colors.END}")

    def run(self, backend_only=False, skip_deps=False, no_ollama_check=False):
        """Main run method"""
        self.print_banner()
        
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print(f"\n{Colors.YELLOW}📢 Received shutdown signal...{Colors.END}")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Run checks
            if not self.check_requirements():
                return False
            
            # Check Ollama unless explicitly skipped
            if not no_ollama_check:
                ollama_ok = self.check_ollama()
                if not ollama_ok:
                    print(f"\n{Colors.YELLOW}⚠️  Continuing without Ollama (limited functionality){Colors.END}")
                    print(f"{Colors.YELLOW}   • File uploads will work{Colors.END}")
                    print(f"{Colors.YELLOW}   • API routing will work{Colors.END}")
                    print(f"{Colors.RED}   • AI analysis will NOT work (requires Ollama + models){Colors.END}")
                    print(f"{Colors.RED}   • RAG queries will NOT work{Colors.END}")
                    print(f"{Colors.RED}   • Visualizations will NOT work{Colors.END}")
                    print(f"{Colors.RED}   • Reports will NOT work{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠️  Skipping Ollama check (--no-ollama-check flag){Colors.END}")
            
            if not self.setup_environment():
                return False
            
            if not skip_deps and not self.install_dependencies(backend_only):
                print(f"{Colors.RED}❌ Dependency installation failed{Colors.END}")
                return False
            
            # Start services
            if not self.start_backend():
                return False
            
            if not backend_only and not self.start_frontend():
                self.cleanup()
                return False
            
            # Print success message
            print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Nexus LLM Analytics is running!{Colors.END}")
            print(f"\n{Colors.BOLD}🌐 Access your application at:{Colors.END}")
            print(f"   • Backend API: http://127.0.0.1:8000")
            if not backend_only:
                print(f"   • Frontend UI: http://localhost:3000")
            print(f"   • API Docs: http://127.0.0.1:8000/docs")
            
            print(f"\n{Colors.BOLD}📋 Available endpoints:{Colors.END}")
            print(f"   • POST /analyze/ - Analyze data with natural language")
            print(f"   • POST /upload-documents/ - Upload data files")
            print(f"   • POST /generate-report/ - Generate reports")
            print(f"   • POST /visualize/generate - Create visualizations")
            
            print(f"\n{Colors.BOLD}⚡ Tech Stack Running:{Colors.END}")
            print(f"   • LLM: Llama 3.1 8B (Primary) + Phi-3-mini (Review)")
            print(f"   • Framework: CrewAI Multi-Agent System")
            print(f"   • Backend: FastAPI + Uvicorn")
            print(f"   • Frontend: Next.js 14 + React 18")
            print(f"   • Vector DB: ChromaDB")
            print(f"   • Data: Pandas + Polars")
            print(f"   • Viz: Plotly + Recharts")
            print(f"   • Reports: OpenPyXL + ReportLab")
            
            print(f"\n{Colors.BOLD}💡 Quick Start:{Colors.END}")
            print(f"   1. Upload a file (CSV, JSON, PDF, TXT)")
            print(f"   2. Ask natural language questions")
            print(f"   3. Get AI-powered analysis, charts, and reports")
            
            print(f"\n{Colors.BOLD}📌 Important Notes:{Colors.END}")
            print(f"   • Ollama must be running for AI features")
            print(f"   • All processing is 100% local (private)")
            print(f"   • Files stored in: data/samples/")
            print(f"   • Reports saved in: reports/")
            
            print(f"\n{Colors.YELLOW}Press Ctrl+C to stop all services{Colors.END}")
            
            # Keep running until interrupted
            try:
                while True:
                    # Check if processes are still running
                    if self.backend_process and self.backend_process.poll() is not None:
                        print(f"{Colors.RED}❌ Backend process stopped unexpectedly{Colors.END}")
                        break
                    
                    if not backend_only and self.frontend_process and self.frontend_process.poll() is not None:
                        print(f"{Colors.RED}❌ Frontend process stopped unexpectedly{Colors.END}")
                        break
                    
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
            return False
        finally:
            self.cleanup()
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Nexus LLM Analytics Launcher')
    parser.add_argument('--backend-only', action='store_true', 
                       help='Start only the backend server')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--no-ollama-check', action='store_true',
                       help='Skip Ollama verification (for testing without AI features)')
    
    args = parser.parse_args()
    
    launcher = NexusLauncher()
    success = launcher.run(
        backend_only=args.backend_only, 
        skip_deps=args.skip_deps,
        no_ollama_check=args.no_ollama_check
    )
    
    if success:
        print(f"\n{Colors.GREEN}👋 Thanks for using Nexus LLM Analytics!{Colors.END}")
    else:
        print(f"\n{Colors.RED}❌ Startup failed. Check the errors above.{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()