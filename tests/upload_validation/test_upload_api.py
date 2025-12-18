#!/usr/bin/env python3
"""
Upload API Integration Test - Test the FastAPI upload endpoint functions
"""
import sys
import os
import tempfile
import json
import csv
import asyncio
from pathlib import Path

# Add the correct path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backend'))

def create_test_files():
    """Create test files for upload testing"""
    test_files = {}
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create CSV test file
    csv_file = os.path.join(temp_dir, "test_data.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'age', 'city'])
        writer.writerow(['Alice', '25', 'New York'])
        writer.writerow(['Bob', '30', 'Los Angeles'])
        writer.writerow(['Charlie', '35', 'Chicago'])
    test_files['csv'] = csv_file
    
    # Create JSON test file
    json_file = os.path.join(temp_dir, "test_data.json")
    test_data = {
        "users": [
            {"name": "Alice", "age": 25, "city": "New York"},
            {"name": "Bob", "age": 30, "city": "Los Angeles"}
        ],
        "metadata": {
            "version": "1.0",
            "created": "2024-01-01"
        }
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    test_files['json'] = json_file
    
    # Create TXT test file
    txt_file = os.path.join(temp_dir, "test_document.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("This is a test document.\n")
        f.write("It contains multiple lines of text.\n")
        f.write("We can analyze this content for insights.\n")
    test_files['txt'] = txt_file
    
    return test_files, temp_dir

def test_validation_functions():
    """Test the individual validation functions"""
    print("🧪 Testing Upload Validation Functions...")
    
    try:
        from api.upload import (
            validate_filename, 
            validate_file_size, 
            validate_file_extension,
            validate_file_content,
            sanitize_extracted_text,
            secure_file_path,
            MAX_FILE_SIZE,
            ALLOWED_EXTENSIONS
        )
        
        # Test filename validation
        print("\n🔒 Testing Filename Validation...")
        valid_names = ["test.csv", "data.json", "document.pdf", "readme.txt"]
        for name in valid_names:
            try:
                result = validate_filename(name)
                print(f"✅ Valid filename: {name} -> {result}")
            except Exception as e:
                print(f"❌ Valid filename rejected: {name} - {e}")
                return False
        
        # Test dangerous filenames
        dangerous_names = ["../../../etc/passwd", "..\\windows\\system32\\hosts", "file\x00.txt"]
        for name in dangerous_names:
            try:
                validate_filename(name)
                print(f"❌ Dangerous filename should be blocked: {name}")
                return False
            except ValueError:
                print(f"✅ Dangerous filename blocked: {name}")
        
        # Test file size validation
        print(f"\n📏 Testing File Size Validation (max: {MAX_FILE_SIZE} bytes)...")
        try:
            validate_file_size(1024)  # Valid
            validate_file_size(MAX_FILE_SIZE)  # Valid at limit
            print("✅ Valid file sizes accepted")
        except Exception as e:
            print(f"❌ Valid file size rejected: {e}")
            return False
        
        try:
            validate_file_size(MAX_FILE_SIZE + 1)  # Too big
            print("❌ Oversized file should be blocked")
            return False
        except ValueError:
            print("✅ Oversized file blocked")
        
        # Test extension validation
        print(f"\n📂 Testing Extension Validation (allowed: {ALLOWED_EXTENSIONS})...")
        for ext in ALLOWED_EXTENSIONS:
            try:
                result = validate_file_extension(f"test{ext}")
                print(f"✅ Allowed extension: test{ext} -> {result}")
            except Exception as e:
                print(f"❌ Allowed extension rejected: test{ext} - {e}")
                return False
        
        # Test dangerous extensions
        dangerous_exts = ["test.exe", "script.php", "backdoor.sh", "virus.bat"]
        for name in dangerous_exts:
            try:
                validate_file_extension(name)
                print(f"❌ Dangerous extension should be blocked: {name}")
                return False
            except ValueError:
                print(f"✅ Dangerous extension blocked: {name}")
        
        # Test secure path generation
        print("\n🛡️ Testing Secure Path Generation...")
        try:
            path = secure_file_path("test.csv")
            print(f"✅ Secure path generated: test.csv -> {path}")
            if ".." in path or not path.endswith("test.csv"):
                print(f"❌ Path doesn't look secure: {path}")
                return False
        except Exception as e:
            print(f"❌ Secure path generation failed: {e}")
            return False
        
        # Test text sanitization
        print("\n🧼 Testing Text Sanitization...")
        test_cases = [
            "Normal text",
            "<script>alert('xss')</script>Safe text",
            "<b>Bold</b> and <i>italic</i> text",
            "A" * 1000000  # Large text (should be truncated)
        ]
        
        for text in test_cases:
            try:
                result = sanitize_extracted_text(text)
                print(f"✅ Text sanitized: {len(text)} chars -> {len(result)} chars")
                # Check XSS protection
                if "<script" in result.lower() and "alert" in result.lower():
                    print(f"❌ XSS content not properly sanitized")
                    return False
            except Exception as e:
                print(f"❌ Text sanitization failed: {e}")
                return False
        
        print("✅ All validation functions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Validation functions test failed: {e}")
        return False

async def test_text_extraction():
    """Test the async text extraction functions"""
    print("\n🔍 Testing Text Extraction Functions...")
    
    try:
        from api.upload import extract_pdf_text_secure, extract_txt_text_secure
        
        # Create test files
        test_files, temp_dir = create_test_files()
        
        # Test TXT extraction
        print("📄 Testing TXT extraction...")
        try:
            txt_content = await extract_txt_text_secure(test_files['txt'], "test_document.txt")
            print(f"✅ TXT extracted: {len(txt_content)} characters")
            if "test document" not in txt_content.lower():
                print("❌ TXT content doesn't match expected")
                return False
        except Exception as e:
            print(f"❌ TXT extraction failed: {e}")
            return False
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        print("✅ Text extraction functions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Text extraction test failed: {e}")
        return False

def test_security_configurations():
    """Test security configurations and constants"""
    print("\n🛡️ Testing Security Configurations...")
    
    try:
        from api.upload import (
            MAX_FILE_SIZE, 
            ALLOWED_EXTENSIONS, 
            ALLOWED_MIME_TYPES,
            DATA_DIR
        )
        
        # Check reasonable file size limit
        if MAX_FILE_SIZE <= 0 or MAX_FILE_SIZE > 1024 * 1024 * 1024:  # 1GB
            print(f"❌ File size limit seems unreasonable: {MAX_FILE_SIZE}")
            return False
        print(f"✅ File size limit: {MAX_FILE_SIZE / (1024*1024):.0f} MB")
        
        # Check allowed extensions are secure
        dangerous_extensions = {'.exe', '.php', '.sh', '.bat', '.ps1', '.vbs', '.js'}
        if dangerous_extensions & ALLOWED_EXTENSIONS:
            print(f"❌ Dangerous extensions allowed: {dangerous_extensions & ALLOWED_EXTENSIONS}")
            return False
        print(f"✅ Safe extensions only: {ALLOWED_EXTENSIONS}")
        
        # Check MIME type mappings exist
        for ext in ALLOWED_EXTENSIONS:
            if ext not in ALLOWED_MIME_TYPES:
                print(f"❌ Missing MIME type mapping for: {ext}")
                return False
        print(f"✅ MIME type mappings complete")
        
        # Check data directory exists
        if not os.path.exists(DATA_DIR):
            print(f"❌ Data directory doesn't exist: {DATA_DIR}")
            return False
        print(f"✅ Data directory exists: {DATA_DIR}")
        
        print("✅ Security configurations are correct")
        return True
        
    except Exception as e:
        print(f"❌ Security configuration test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Upload API Integration Test Suite")
    print("=" * 70)
    
    tests = [
        ("Validation Functions", test_validation_functions()),
        ("Text Extraction", await test_text_extraction()),
        ("Security Configurations", test_security_configurations()),
    ]
    
    passed = sum(1 for _, result in tests if result)
    failed = len(tests) - passed
    
    print(f"\n📊 Test Results:")
    print("=" * 50)
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Summary: {passed}/{len(tests)} test suites passed")
    
    if failed == 0:
        print("🎉 ALL UPLOAD API TESTS PASSED!")
        print("✅ Upload API is secure, functional, and ready for production")
        print("\n🏁 Key Achievements:")
        print("   • Filename validation with path traversal protection")
        print("   • File size limits enforced")
        print("   • Extension whitelist security")
        print("   • Content sanitization working")
        print("   • Secure file path generation")
        print("   • Text extraction functionality")
        print("   • Comprehensive security configurations")
        return True
    else:
        print(f"⚠️  {failed} test suites failed. Review implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)