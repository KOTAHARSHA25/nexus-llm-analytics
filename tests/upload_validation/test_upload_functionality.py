#!/usr/bin/env python3
"""
Quick Upload Test - Test the upload.py functionality directly
"""

import sys
import os
import tempfile
import asyncio
from io import BytesIO

# Add src/backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'backend'))

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

async def test_upload_basic_functionality():
    """Test basic upload functionality with a simple text file"""
    print("🧪 Testing Basic Upload Functionality...")
    
    try:
        from api.upload import upload_document
        
        # Create a simple test file
        test_content = b"Hello, this is a test file for upload functionality!"
        test_file = MockUploadFile("test.txt", test_content, "text/plain")
        
        # Test the upload
        result = await upload_document(test_file)
        
        if "error" in result:
            print(f"❌ Upload failed: {result['error']}")
            return False
        else:
            print(f"✅ Upload succeeded: {result.get('message', 'Success')}")
            print(f"   File: {result.get('filename')}")
            print(f"   Size: {result.get('file_size')} bytes")
            
            # Cleanup uploaded file
            if result.get('filename'):
                from api.upload import DATA_DIR
                uploaded_path = os.path.join(DATA_DIR, result['filename'])
                if os.path.exists(uploaded_path):
                    os.unlink(uploaded_path)
                    print(f"   Cleaned up: {uploaded_path}")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_upload_security():
    """Test that security measures are working"""
    print("\n🔒 Testing Upload Security...")
    
    try:
        from api.upload import validate_filename, validate_file_size, validate_file_extension
        
        # Test filename validation
        try:
            result = validate_filename("../../../etc/passwd")
            print(f"❌ Path traversal should have been blocked, but got: {result}")
            return False
        except ValueError as e:
            print(f"✅ Path traversal blocked: {e}")
        
        # Test file size validation
        try:
            validate_file_size(200 * 1024 * 1024)  # 200MB - over limit
            print("❌ Large file should have been blocked")
            return False
        except ValueError as e:
            print(f"✅ Large file blocked: {e}")
        
        # Test file extension validation
        try:
            validate_file_extension("malware.exe")
            print("❌ Dangerous extension should have been blocked")
            return False
        except ValueError as e:
            print(f"✅ Dangerous extension blocked: {e}")
        
        print("✅ Security measures working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False

async def test_csv_upload():
    """Test CSV file upload"""
    print("\n📊 Testing CSV Upload...")
    
    try:
        from api.upload import upload_document
        
        # Create a test CSV
        csv_content = b"name,age,city\nJohn,30,New York\nJane,25,Los Angeles\nBob,35,Chicago"
        csv_file = MockUploadFile("test_data.csv", csv_content, "text/csv")
        
        result = await upload_document(csv_file)
        
        if "error" in result:
            print(f"❌ CSV upload failed: {result['error']}")
            return False
        else:
            print(f"✅ CSV upload succeeded: {result.get('filename')}")
            
            # Cleanup
            if result.get('filename'):
                from api.upload import DATA_DIR
                uploaded_path = os.path.join(DATA_DIR, result['filename'])
                if os.path.exists(uploaded_path):
                    os.unlink(uploaded_path)
            return True
            
    except Exception as e:
        print(f"❌ CSV test failed: {e}")
        return False

async def test_json_upload():
    """Test JSON file upload"""
    print("\n📄 Testing JSON Upload...")
    
    try:
        from api.upload import upload_document
        
        # Create a test JSON
        json_content = b'{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}], "total": 2}'
        json_file = MockUploadFile("test_data.json", json_content, "application/json")
        
        result = await upload_document(json_file)
        
        if "error" in result:
            print(f"❌ JSON upload failed: {result['error']}")
            return False
        else:
            print(f"✅ JSON upload succeeded: {result.get('filename')}")
            
            # Cleanup
            if result.get('filename'):
                from api.upload import DATA_DIR
                uploaded_path = os.path.join(DATA_DIR, result['filename'])
                if os.path.exists(uploaded_path):
                    os.unlink(uploaded_path)
            return True
            
    except Exception as e:
        print(f"❌ JSON test failed: {e}")
        return False

async def main():
    """Run all upload tests"""
    print("🧪 Upload.py Functionality Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Upload", await test_upload_basic_functionality()),
        ("Security Measures", await test_upload_security()),
        ("CSV Upload", await test_csv_upload()),
        ("JSON Upload", await test_json_upload()),
    ]
    
    passed = sum(1 for _, result in tests if result)
    failed = len(tests) - passed
    
    print(f"\n📊 Test Results:")
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Summary: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("🎉 ALL UPLOAD TESTS PASSED! Upload functionality is working correctly.")
        return True
    else:
        print(f"⚠️  {failed} tests failed. Check the upload implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)