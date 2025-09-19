#!/usr/bin/env python3
"""
End-to-End Upload Test - Test complete upload workflow
"""
import sys
import os
import tempfile
import json
import csv
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
    
    # Create malicious test file (should be blocked)
    malicious_file = os.path.join(temp_dir, "malicious_script.php")
    with open(malicious_file, 'w', encoding='utf-8') as f:
        f.write("<?php echo 'This should be blocked'; ?>")
    test_files['malicious'] = malicious_file
    
    return test_files, temp_dir

def test_file_processing():
    """Test the complete file processing workflow"""
    print("🧪 Testing Complete File Processing Workflow...")
    
    try:
        from api.upload import process_uploaded_file, validate_filename
        
        # Create test files
        test_files, temp_dir = create_test_files()
        
        # Test CSV processing
        print("\n📊 Testing CSV File Processing...")
        try:
            csv_result = process_uploaded_file(
                test_files['csv'], 
                "test_data.csv",
                temp_dir
            )
            print(f"✅ CSV processed successfully:")
            print(f"   File ID: {csv_result['file_id']}")
            print(f"   Content preview: {csv_result['content'][:100]}...")
            print(f"   Metadata: {csv_result['metadata']}")
        except Exception as e:
            print(f"❌ CSV processing failed: {e}")
            return False
        
        # Test JSON processing
        print("\n📋 Testing JSON File Processing...")
        try:
            json_result = process_uploaded_file(
                test_files['json'], 
                "test_data.json",
                temp_dir
            )
            print(f"✅ JSON processed successfully:")
            print(f"   File ID: {json_result['file_id']}")
            print(f"   Content preview: {json_result['content'][:100]}...")
            print(f"   Metadata: {json_result['metadata']}")
        except Exception as e:
            print(f"❌ JSON processing failed: {e}")
            return False
        
        # Test TXT processing
        print("\n📄 Testing TXT File Processing...")
        try:
            txt_result = process_uploaded_file(
                test_files['txt'], 
                "test_document.txt",
                temp_dir
            )
            print(f"✅ TXT processed successfully:")
            print(f"   File ID: {txt_result['file_id']}")
            print(f"   Content: {txt_result['content']}")
            print(f"   Metadata: {txt_result['metadata']}")
        except Exception as e:
            print(f"❌ TXT processing failed: {e}")
            return False
        
        # Test malicious file blocking
        print("\n🛡️ Testing Malicious File Blocking...")
        try:
            malicious_result = process_uploaded_file(
                test_files['malicious'], 
                "malicious_script.php",
                temp_dir
            )
            print(f"❌ Malicious file should have been blocked!")
            return False
        except ValueError as e:
            print(f"✅ Malicious file correctly blocked: {e}")
        except Exception as e:
            print(f"❌ Unexpected error with malicious file: {e}")
            return False
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        print("✅ Complete file processing workflow working correctly")
        return True
        
    except Exception as e:
        print(f"❌ File processing workflow test failed: {e}")
        return False

def test_content_extraction():
    """Test content extraction from different file types"""
    print("\n🔍 Testing Content Extraction...")
    
    try:
        from api.upload import extract_text_content
        
        # Create test files
        test_files, temp_dir = create_test_files()
        
        # Test CSV content extraction
        csv_content = extract_text_content(test_files['csv'], '.csv')
        print(f"✅ CSV content extracted: {len(csv_content)} characters")
        assert 'Alice' in csv_content and 'Bob' in csv_content
        
        # Test JSON content extraction
        json_content = extract_text_content(test_files['json'], '.json')
        print(f"✅ JSON content extracted: {len(json_content)} characters")
        assert 'Alice' in json_content and 'users' in json_content
        
        # Test TXT content extraction
        txt_content = extract_text_content(test_files['txt'], '.txt')
        print(f"✅ TXT content extracted: {len(txt_content)} characters")
        assert 'test document' in txt_content
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        print("✅ Content extraction working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Content extraction test failed: {e}")
        return False

def test_error_handling():
    """Test error handling scenarios"""
    print("\n⚠️ Testing Error Handling...")
    
    try:
        from api.upload import process_uploaded_file
        
        # Test non-existent file
        try:
            result = process_uploaded_file("/nonexistent/file.csv", "test.csv", "/tmp")
            print("❌ Should have failed for non-existent file")
            return False
        except Exception as e:
            print(f"✅ Non-existent file correctly handled: {type(e).__name__}")
        
        # Test invalid extension
        temp_dir = tempfile.mkdtemp()
        bad_file = os.path.join(temp_dir, "test.exe")
        with open(bad_file, 'w') as f:
            f.write("malicious content")
        
        try:
            result = process_uploaded_file(bad_file, "test.exe", temp_dir)
            print("❌ Should have failed for .exe file")
            return False
        except ValueError as e:
            print(f"✅ Invalid extension correctly blocked: {e}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        print("✅ Error handling working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_security_measures():
    """Test security measures in upload processing"""
    print("\n🛡️ Testing Security Measures...")
    
    try:
        from api.upload import secure_file_path, sanitize_extracted_text
        
        # Test path traversal protection
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/shadow",
            "\\windows\\system.ini"
        ]
        
        for path in dangerous_paths:
            try:
                secure_path = secure_file_path(path, "/safe/dir")
                if "/safe/dir" not in secure_path or ".." in secure_path:
                    print(f"❌ Path traversal not prevented: {path} -> {secure_path}")
                    return False
                print(f"✅ Path traversal prevented: {path}")
            except Exception as e:
                print(f"✅ Dangerous path blocked: {path} - {e}")
        
        # Test XSS protection in text sanitization
        dangerous_texts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(\"xss\")'></iframe>"
        ]
        
        for text in dangerous_texts:
            sanitized = sanitize_extracted_text(text)
            if "<script" in sanitized.lower() and "alert" in sanitized.lower():
                print(f"❌ XSS content not properly sanitized: {text}")
                return False
            print(f"✅ XSS content sanitized: {text[:30]}...")
        
        print("✅ Security measures working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Security measures test failed: {e}")
        return False

def main():
    """Run all comprehensive tests"""
    print("🚀 Comprehensive Upload System Test Suite")
    print("=" * 80)
    
    tests = [
        ("File Processing Workflow", test_file_processing),
        ("Content Extraction", test_content_extraction),
        ("Error Handling", test_error_handling),
        ("Security Measures", test_security_measures),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    print(f"\n📊 Final Test Results:")
    print("=" * 50)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Summary: {passed}/{len(results)} test suites passed")
    
    if failed == 0:
        print("🎉 ALL UPLOAD SYSTEM TESTS PASSED!")
        print("✅ Upload system is secure, functional, and ready for production")
        return True
    else:
        print(f"⚠️  {failed} test suites failed. Review implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)