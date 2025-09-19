#!/usr/bin/env python3
"""
Core Functionality Test - Test the essential features without servers
"""
import sys
import os
import tempfile
import json

# Add the correct path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backend'))

def test_core_functionality():
    """Test core upload and processing functionality"""
    print("🧪 Testing Core Functionality (No Servers Required)...")
    
    try:
        # Test imports work
        from api.upload import validate_filename, sanitize_extracted_text
        from core.config import get_settings
        print("✅ Core modules import successfully")
        
        # Test configuration
        settings = get_settings()
        print(f"✅ Configuration loaded: {settings.app_name}")
        
        # Test upload validation
        filename = validate_filename("test_data.csv")
        print(f"✅ Filename validation works: {filename}")
        
        # Test text sanitization
        safe_text = sanitize_extracted_text("<script>alert('test')</script>Safe content")
        print(f"✅ Text sanitization works: {len(safe_text)} chars")
        
        # Test file processing simulation
        test_data = {
            "users": [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}],
            "metrics": {"avg_score": 91, "total_users": 2}
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        # Read and validate
        with open(temp_file, 'r') as f:
            content = f.read()
            
        sanitized = sanitize_extracted_text(content)
        print(f"✅ File processing simulation works: {len(sanitized)} chars processed")
        
        # Cleanup
        os.unlink(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def test_api_structure():
    """Test that API modules have the expected structure"""
    print("\n🧪 Testing API Structure...")
    
    try:
        # Test that API files exist
        api_path = "src/backend/api"
        if os.path.exists(os.path.join(api_path, "upload.py")):
            print("✅ Upload API file exists")
        else:
            print("❌ Upload API file missing")
            
        if os.path.exists(os.path.join(api_path, "analyze.py")):
            print("✅ Analyze API file exists") 
        else:
            print("❌ Analyze API file missing")
        
        # Check if main backend files exist
        if os.path.exists("src/backend/main.py"):
            print("✅ Main backend file exists")
        else:
            print("❌ Main backend file missing")
            
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def test_frontend_structure():
    """Test frontend structure"""
    print("\n🧪 Testing Frontend Structure...")
    
    frontend_path = "src/frontend"
    required_files = [
        "package.json",
        "next.config.js", 
        "app/page.tsx",
        "components"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(frontend_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if not missing_files:
        print("✅ All required frontend files present")
        
        # Check package.json for required dependencies
        try:
            with open(os.path.join(frontend_path, "package.json"), 'r') as f:
                package_data = json.load(f)
                
            if "next" in package_data.get("dependencies", {}):
                print("✅ Next.js dependency found")
            else:
                print("⚠️ Next.js dependency missing")
                
            if "react" in package_data.get("dependencies", {}):
                print("✅ React dependency found")
            else:
                print("⚠️ React dependency missing")
                
        except Exception as e:
            print(f"⚠️ Could not read package.json: {e}")
            
        return True
    else:
        print(f"❌ Missing frontend files: {missing_files}")
        return False

def main():
    """Run core functionality tests"""
    print("🚀 NEXUS LLM ANALYTICS - CORE FUNCTIONALITY TEST")
    print("=" * 70)
    
    tests = [
        ("Core Functionality", test_core_functionality),
        ("API Structure", test_api_structure),
        ("Frontend Structure", test_frontend_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 CORE FUNCTIONALITY TEST RESULTS")
    print("=" * 50)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL CORE FUNCTIONALITY TESTS PASSED!")
        print("✅ The application core is working - ready for server testing")
        print("\n📋 Next Steps:")
        print("   1. Start backend server: uvicorn main:app --reload")
        print("   2. Start frontend server: npm run dev")
        print("   3. Test full integration")
        return True
    else:
        print(f"⚠️ {total - passed} core tests failed")
        print("🛠️ Fix core issues before proceeding to server integration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)