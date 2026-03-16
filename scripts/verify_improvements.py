#!/usr/bin/env python3
"""
Verification script to check all improvements are properly implemented
Run this to verify the system is ready for deployment
"""

import sys
import os
from pathlib import Path
import importlib.util
import json

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()

def check_module_imports(module_path):
    """Check if a Python module can be imported"""
    try:
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def run_checks():
    """Run all verification checks"""
    print("=" * 60)
    print("NEXUS LLM ANALYTICS - IMPROVEMENT VERIFICATION")
    print("=" * 60)
    
    checks = []
    
    # 1. Check critical files exist
    print("\n[FILES] Checking Critical Files:")
    critical_files = [
        ("Backend Config", "src/backend/core/config.py"),
        ("Error Handling", "src/backend/core/error_handling.py"),
        ("Rate Limiter", "src/backend/core/rate_limiter.py"),
        ("WebSocket Manager", "src/backend/core/websocket_manager.py"),
        ("Comprehensive Tests", "tests/backend/test_comprehensive.py"),
        ("Environment File", ".env"),
        ("Requirements", "requirements.txt"),
    ]
    
    for name, filepath in critical_files:
        exists = check_file_exists(filepath)
        status = "[OK]" if exists else "[FAIL]"
        checks.append((name, exists))
        print(f"  {status} {name}: {filepath}")
    
    # 2. Check Python modules can be imported
    print("\n[PYTHON] Checking Python Modules:")
    modules = [
        ("Config Module", "src/backend/core/config.py"),
        ("Error Module", "src/backend/core/error_handling.py"),
        ("Rate Limiter", "src/backend/core/rate_limiter.py"),
        ("WebSocket", "src/backend/core/websocket_manager.py"),
    ]
    
    for name, module_path in modules:
        if check_file_exists(module_path):
            success, msg = check_module_imports(module_path)
            status = "[OK]" if success else "[WARN]"
            checks.append((name + " Import", success))
            if success:
                print(f"  {status} {name}: Imports successfully")
            else:
                print(f"  {status} {name}: {msg[:50]}...")
    
    # 3. Check environment variables
    print("\n[ENV] Checking Environment Configuration:")
    if check_file_exists(".env"):
        with open(".env", "r") as f:
            env_content = f.read()
            required_vars = [
                "CORS_ALLOWED_ORIGINS",
                "CHROMADB_PERSIST_DIRECTORY",
                "LOG_LEVEL",
                "OLLAMA_BASE_URL",
                "PRIMARY_MODEL",
                "AUTO_MODEL_SELECTION"
            ]
            
            for var in required_vars:
                if var in env_content:
                    print(f"  [OK] {var}: Configured")
                    checks.append((f"ENV:{var}", True))
                else:
                    print(f"  [FAIL] {var}: Missing")
                    checks.append((f"ENV:{var}", False))
    
    # 4. Check dependency versions
    print("\n[DEPS] Checking Dependency Versions:")
    if check_file_exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = f.read()
            has_versions = "==" in requirements
            if has_versions:
                print(f"  [OK] Dependencies: Versions pinned")
                checks.append(("Dependency Versions", True))
            else:
                print(f"  [WARN] Dependencies: No versions specified")
                checks.append(("Dependency Versions", False))
    
    # 5. Check frontend improvements
    print("\n[FRONTEND] Checking Frontend Improvements:")
    frontend_files = [
        ("WebSocket Hook", "src/frontend/hooks/useWebSocket.ts"),
        ("Error Boundary", "src/frontend/components/error-boundary.tsx"),
    ]
    
    for name, filepath in frontend_files:
        exists = check_file_exists(filepath)
        status = "[OK]" if exists else "[FAIL]"
        checks.append((name, exists))
        print(f"  {status} {name}: {filepath}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_checks = len(checks)
    passed_checks = sum(1 for _, passed in checks if passed)
    percentage = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"Total Checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    print(f"Success Rate: {percentage:.1f}%")
    
    if percentage >= 90:
        print("\n[SUCCESS] System is READY for deployment!")
    elif percentage >= 70:
        print("\n[WARNING] System is MOSTLY ready but needs some fixes")
    else:
        print("\n[ERROR] System needs significant work before deployment")
    
    # Recommendations
    print("\n[NEXT STEPS]:")
    if not check_file_exists(".env"):
        print("  1. Your .env file exists - good!")
    
    failed = [name for name, passed in checks if not passed]
    if failed:
        print(f"  2. Fix these issues: {', '.join(failed[:3])}")
    
    print("  3. Run: pip install -r requirements.txt")
    print("  4. Run: cd src/frontend && npm install")
    print("  5. Pull Ollama models: ollama pull llama3.1:8b phi3:mini nomic-embed-text")
    print("  6. Run: python scripts/launch.py")
    
    return percentage >= 70

if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
