#!/usr/bin/env python3
"""
Upload Security Verification Test Suite

Verifies that all 6 critical vulnerabilities identified in UPLOAD_SECURITY_AUDIT.md
have been properly fixed in the updated upload.py file.
"""

import sys
import os
import tempfile
import asyncio
from io import BytesIO

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'backend'))

# Mock FastAPI UploadFile for testing 
class MockUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str = None):
        self.filename = filename
        self.content = BytesIO(content)
        self.size = len(content)
        self.content_type = content_type
        self._position = 0
    
    async def read(self, size: int = -1):
        if size == -1:
            return self.content.read()
        return self.content.read(size)

def test_path_traversal_protection():
    """Test that path traversal attacks are blocked"""
    print("🔒 Testing Path Traversal Protection...")
    
    from api.upload import validate_filename, secure_file_path
    
    dangerous_filenames = [
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "/etc/passwd", 
        "\\windows\\system32\\hosts",
        "test/../../secret.txt",
        "normal_file.txt/../../../etc/shadow"
    ]
    
    passed = 0
    for filename in dangerous_filenames:
        try:
            validated = validate_filename(filename)
            # Should either throw exception or return safe filename
            try:
                path = secure_file_path(validated)
                # If we get here, check that path is safe
                from api.upload import DATA_DIR
                if not os.path.realpath(path).startswith(os.path.realpath(DATA_DIR)):
                    print(f"❌ Path traversal not blocked: {filename} -> {path}")
                    return False
                else:
                    print(f"✅ Path traversal blocked: {filename}")
                    passed += 1
            except ValueError:
                print(f"✅ Path traversal blocked: {filename}")
                passed += 1
        except ValueError:
            print(f"✅ Path traversal blocked: {filename}")  
            passed += 1
    
    print(f"✅ Path Traversal Protection: {passed}/{len(dangerous_filenames)} attacks blocked")
    return passed == len(dangerous_filenames)

def test_file_size_limits():
    """Test that file size limits are enforced"""
    print("\n🔒 Testing File Size Limits...")
    
    from api.upload import validate_file_size, MAX_FILE_SIZE
    
    test_cases = [
        (MAX_FILE_SIZE - 1, True),   # Just under limit - should pass
        (MAX_FILE_SIZE, True),       # Exactly at limit - should pass
        (MAX_FILE_SIZE + 1, False),  # Over limit - should fail
        (MAX_FILE_SIZE * 2, False),  # Way over limit - should fail
    ]
    
    passed = 0
    for size, should_pass in test_cases:
        try:
            validate_file_size(size)
            if should_pass:
                print(f"✅ Size {size} correctly allowed")
                passed += 1
            else:
                print(f"❌ Size {size} should have been blocked!")
                return False
        except ValueError:
            if not should_pass:
                print(f"✅ Size {size} correctly blocked")
                passed += 1
            else:
                print(f"❌ Size {size} should have been allowed!")
                return False
    
    print(f"✅ File Size Limits: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)

def test_file_type_validation():
    """Test file extension and content validation"""
    print("\n🔒 Testing File Type Validation...")
    
    from api.upload import validate_file_extension
    
    # Test extension validation
    allowed_files = ['test.csv', 'data.json', 'document.pdf', 'readme.txt']
    dangerous_files = ['malware.exe', 'script.php', 'backdoor.sh', 'virus.bat']
    
    passed = 0
    
    for filename in allowed_files:
        try:
            ext = validate_file_extension(filename)
            print(f"✅ Allowed extension: {filename} -> {ext}")
            passed += 1
        except ValueError:
            print(f"❌ Valid file wrongly blocked: {filename}")
            return False
    
    for filename in dangerous_files:
        try:
            validate_file_extension(filename)
            print(f"❌ Dangerous file not blocked: {filename}")
            return False
        except ValueError:
            print(f"✅ Dangerous extension blocked: {filename}")
            passed += 1
    
    total_tests = len(allowed_files) + len(dangerous_files)
    print(f"✅ File Type Validation: {passed}/{total_tests} tests passed")
    return passed == total_tests

def test_input_sanitization():
    """Test text sanitization functionality"""
    print("\n🔒 Testing Input Sanitization...")
    
    from api.upload import sanitize_extracted_text
    
    test_cases = [
        # (input, description)
        ("<script>alert('xss')</script>Hello", "XSS script removal"),
        ("Normal text content", "Clean text unchanged"),
        ("<iframe src='evil.com'></iframe>Content", "Iframe removal"),
        ("A" * 2000000, "Text length limiting"),  # 2MB text
        ("<b>Bold</b> and <i>italic</i> text", "HTML tag removal"),
    ]
    
    passed = 0
    for test_input, description in test_cases:
        try:
            result = sanitize_extracted_text(test_input)
            
            # Check that result is safe
            dangerous_patterns = ['<script', '<iframe', 'javascript:', 'onload=']
            is_safe = not any(pattern in result.lower() for pattern in dangerous_patterns)
            
            # Check length limit
            max_length = 1024 * 1024  # 1MB
            length_ok = len(result) <= max_length
            
            if is_safe and length_ok:
                print(f"✅ Sanitization passed: {description}")
                passed += 1
            else:
                print(f"❌ Sanitization failed: {description}")
                if not is_safe:
                    print(f"   Dangerous content still present in: {result[:100]}...")
                if not length_ok:
                    print(f"   Length {len(result)} exceeds limit {max_length}")
                return False
                
        except Exception as e:
            print(f"❌ Sanitization error for {description}: {e}")
            return False
    
    print(f"✅ Input Sanitization: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)

async def test_secure_file_handling():
    """Test secure temporary file handling"""
    print("\n🔒 Testing Secure File Handling...")
    
    # Create test files
    test_files = [
        ("test.txt", b"Hello, world!", "text/plain"),
        ("test.csv", b"name,age\nJohn,30\nJane,25", "text/csv"),
        ("large.txt", b"A" * 1000, "text/plain"),  # 1KB file
    ]
    
    passed = 0
    for filename, content, content_type in test_files:
        try:
            mock_file = MockUploadFile(filename, content, content_type)
            
            # Import and test the upload function
            from api.upload import upload_document
            
            # The function should handle the file securely
            result = await upload_document(mock_file)
            
            if "error" not in result:
                print(f"✅ Secure handling: {filename}")
                passed += 1
                
                # Cleanup uploaded file
                if "filename" in result:
                    from api.upload import DATA_DIR
                    uploaded_path = os.path.join(DATA_DIR, result["filename"])
                    if os.path.exists(uploaded_path):
                        os.unlink(uploaded_path)
                    extracted_path = uploaded_path + '.extracted.txt'
                    if os.path.exists(extracted_path):
                        os.unlink(extracted_path)
            else:
                print(f"❌ Upload failed for valid file {filename}: {result['error']}")
        except Exception as e:
            print(f"❌ Secure handling failed for {filename}: {e}")
            return False
    
    print(f"✅ Secure File Handling: {passed}/{len(test_files)} tests passed")
    return passed == len(test_files)

def test_error_handling():
    """Test that error handling doesn't leak information"""
    print("\n🔒 Testing Secure Error Handling...")
    
    from api.upload import validate_filename
    
    # Test that errors don't expose internal details
    dangerous_inputs = [
        "",  # Empty filename
        None,  # None filename  
        "../../../etc/passwd",  # Path traversal
        "file\x00.txt",  # Null byte injection
    ]
    
    passed = 0
    for dangerous_input in dangerous_inputs:
        try:
            validate_filename(dangerous_input)
            print(f"❌ Should have failed for: {dangerous_input}")
        except Exception as e:
            error_msg = str(e)
            # Check that error doesn't contain sensitive paths or system info
            sensitive_patterns = ['/usr/', '/etc/', 'C:\\Windows\\', 'system32', 'root']
            
            contains_sensitive = any(pattern in error_msg for pattern in sensitive_patterns)
            if not contains_sensitive and len(error_msg) < 200:  # Reasonable error length
                print(f"✅ Safe error for: {repr(dangerous_input)}")
                passed += 1
            else:
                print(f"❌ Error message too revealing: {error_msg}")
                return False
    
    print(f"✅ Secure Error Handling: {passed}/{len(dangerous_inputs)} tests passed")
    return passed == len(dangerous_inputs)

async def main():
    """Run all upload security tests"""
    print("🔐 Upload Security Verification Test Suite")
    print("Testing fixes for all 6 critical vulnerabilities from UPLOAD_SECURITY_AUDIT.md")
    print("=" * 80)
    
    tests = [
        ("Path Traversal Protection", test_path_traversal_protection()),
        ("File Size Limits", test_file_size_limits()),
        ("File Type Validation", test_file_type_validation()),
        ("Input Sanitization", test_input_sanitization()),
        ("Secure File Handling", await test_secure_file_handling()),
        ("Secure Error Handling", test_error_handling()),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, result in tests:
        if result:
            passed += 1
            print(f"\n✅ {test_name}: PASSED")
        else:
            failed += 1
            print(f"\n❌ {test_name}: FAILED")
        print("-" * 50)
    
    print(f"\n🔐 Upload Security Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL UPLOAD SECURITY TESTS PASSED!")
        print("All 6 critical vulnerabilities from UPLOAD_SECURITY_AUDIT.md have been FIXED!")
        return True
    else:
        print(f"\n⚠️  {failed} SECURITY TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)