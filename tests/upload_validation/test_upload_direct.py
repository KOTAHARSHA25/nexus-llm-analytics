#!/usr/bin/env python3
"""
Direct Upload Test - Test upload.py functions directly
"""
import sys
import os

# Add the correct path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'backend'))

def test_imports():
    """Test that we can import the upload functions"""
    print("🧪 Testing Upload Module Imports...")
    
    try:
        from api.upload import (
            validate_filename, 
            validate_file_size, 
            validate_file_extension,
            validate_file_content,
            sanitize_extracted_text,
            secure_file_path
        )
        print("✅ Successfully imported all upload functions")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_filename_validation():
    """Test filename validation security"""
    print("\n🔒 Testing Filename Validation...")
    
    try:
        from api.upload import validate_filename
        
        # Test valid filenames
        valid_files = ["test.txt", "data.csv", "report.pdf", "config.json"]
        for filename in valid_files:
            try:
                result = validate_filename(filename)
                print(f"✅ Valid file accepted: {filename} -> {result}")
            except Exception as e:
                print(f"❌ Valid file rejected: {filename} - {e}")
                return False
        
        # Test dangerous filenames
        dangerous_files = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\hosts",
            "/etc/shadow",
            "\\windows\\system.ini",
            "file\x00.txt",
        ]
        
        for filename in dangerous_files:
            try:
                result = validate_filename(filename)
                print(f"❌ Dangerous file should have been blocked: {filename}")
                return False
            except ValueError as e:
                print(f"✅ Dangerous file blocked: {filename}")
        
        print("✅ Filename validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Filename validation test failed: {e}")
        return False

def test_file_size_validation():
    """Test file size limits"""
    print("\n📏 Testing File Size Validation...")
    
    try:
        from api.upload import validate_file_size, MAX_FILE_SIZE
        
        # Test valid sizes
        valid_sizes = [1024, 1024*1024, MAX_FILE_SIZE - 1, MAX_FILE_SIZE]
        for size in valid_sizes:
            try:
                validate_file_size(size)
                print(f"✅ Valid size accepted: {size} bytes")
            except Exception as e:
                print(f"❌ Valid size rejected: {size} bytes - {e}")
                return False
        
        # Test invalid sizes  
        invalid_sizes = [MAX_FILE_SIZE + 1, MAX_FILE_SIZE * 2]
        for size in invalid_sizes:
            try:
                validate_file_size(size)
                print(f"❌ Oversized file should have been blocked: {size} bytes")
                return False
            except ValueError as e:
                print(f"✅ Oversized file blocked: {size} bytes")
        
        print("✅ File size validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ File size validation test failed: {e}")
        return False

def test_extension_validation():
    """Test file extension validation"""
    print("\n📂 Testing File Extension Validation...")
    
    try:
        from api.upload import validate_file_extension
        
        # Test allowed extensions
        allowed_files = ["test.csv", "data.json", "document.pdf", "readme.txt"]
        for filename in allowed_files:
            try:
                ext = validate_file_extension(filename)
                print(f"✅ Allowed extension: {filename} -> {ext}")
            except Exception as e:
                print(f"❌ Allowed extension rejected: {filename} - {e}")
                return False
        
        # Test dangerous extensions
        dangerous_files = ["malware.exe", "script.php", "backdoor.sh", "virus.bat", "hack.py"]
        for filename in dangerous_files:
            try:
                validate_file_extension(filename)
                print(f"❌ Dangerous extension should have been blocked: {filename}")
                return False
            except ValueError as e:
                print(f"✅ Dangerous extension blocked: {filename}")
        
        print("✅ Extension validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Extension validation test failed: {e}")
        return False

def test_text_sanitization():
    """Test text sanitization"""
    print("\n🧼 Testing Text Sanitization...")
    
    try:
        from api.upload import sanitize_extracted_text
        
        # Test cases
        test_cases = [
            ("Normal text", "Normal text"),
            ("<script>alert('xss')</script>Safe text", "Safe text"),
            ("Text with <b>bold</b> and <i>italic</i>", "Text with bold and italic"),
            ("A" * 2000000, None),  # Test length limiting (2MB -> 1MB)
        ]
        
        for input_text, expected_behavior in test_cases:
            try:
                result = sanitize_extracted_text(input_text)
                
                if expected_behavior is None:
                    # Check length limiting
                    if len(result) <= 1024 * 1024:  # 1MB limit
                        print(f"✅ Text length limited correctly: {len(input_text)} -> {len(result)}")
                    else:
                        print(f"❌ Text length not limited: {len(result)}")
                        return False
                else:
                    # Check sanitization - HTML should be escaped, not executed
                    if "&lt;script" in result or ("<script" not in result and "alert" not in result):
                        print(f"✅ Text sanitized correctly: {input_text[:50]}...")
                    else:
                        print(f"❌ Dangerous content not properly escaped: {result}")
                        return False
            except Exception as e:
                print(f"❌ Sanitization failed for input: {e}")
                return False
        
        print("✅ Text sanitization working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Text sanitization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Upload.py Direct Function Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports()),
        ("Filename Validation", test_filename_validation()),
        ("File Size Validation", test_file_size_validation()),
        ("Extension Validation", test_extension_validation()),
        ("Text Sanitization", test_text_sanitization()),
    ]
    
    passed = sum(1 for _, result in tests if result)
    failed = len(tests) - passed
    
    print(f"\n📊 Test Results:")
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Summary: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("🎉 ALL UPLOAD FUNCTION TESTS PASSED!")
        print("✅ Upload.py security and validation functions are working correctly")
        return True
    else:
        print(f"⚠️  {failed} tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)