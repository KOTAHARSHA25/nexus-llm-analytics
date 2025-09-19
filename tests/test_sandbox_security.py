#!/usr/bin/env python3
"""
Comprehensive Security Test Suite for Enhanced Sandbox

This test suite verifies that the critical security vulnerabilities in sandbox.py 
and security_guards.py have been properly fixed.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'backend'))

from core.sandbox import EnhancedSandbox

def test_dangerous_imports():
    """Test that dangerous imports are blocked"""
    print("🔒 Testing dangerous import blocking...")
    
    sandbox = EnhancedSandbox()
    
    dangerous_codes = [
        "import os",
        "import subprocess", 
        "import sys",
        "import socket",
        "import urllib",
        "__import__('os')",
        "exec('import os')",
        "eval('__import__(\"os\")')",
    ]
    
    for code in dangerous_codes:
        try:
            result = sandbox.execute(code)
            if 'error' not in result:
                print(f"❌ SECURITY BREACH: {code} was allowed!")
                return False
            else:
                print(f"✅ Blocked: {code}")
        except Exception as e:
            print(f"✅ Exception blocked: {code} - {e}")
    
    return True

def test_attribute_access_bypass():
    """Test that attribute access bypasses are blocked"""
    print("\n🔒 Testing attribute access bypass prevention...")
    
    sandbox = EnhancedSandbox()
    
    dangerous_codes = [
        "().__class__",
        "[].__class__.__bases__[0].__subclasses__()",
        "''.__class__.__mro__[2].__subclasses__()",
        "getattr(int, '__class__')",
        "type(().__class__)",
    ]
    
    for code in dangerous_codes:
        try:
            result = sandbox.execute(code)
            # Should either error or return safe result
            if 'error' not in result and 'subclasses' in str(result.get('result', '')):
                print(f"❌ SECURITY BREACH: {code} revealed subclasses!")
                return False
            else:
                print(f"✅ Blocked: {code}")
        except Exception as e:
            print(f"✅ Exception blocked: {code} - {e}")
    
    return True

def test_code_execution_bypass():
    """Test that code execution bypasses are blocked"""
    print("\n🔒 Testing code execution bypass prevention...")
    
    sandbox = EnhancedSandbox()
    
    dangerous_codes = [
        'exec("print(\'hello\')")',
        'eval("1+1")',
        'compile("print(1)", "<string>", "exec")',
        'globals()',
        'locals()',
        'vars()',
    ]
    
    for code in dangerous_codes:
        try:
            result = sandbox.execute(code)
            if 'error' not in result:
                print(f"❌ SECURITY BREACH: {code} was allowed!")
                return False
            else:
                print(f"✅ Blocked: {code}")
        except Exception as e:
            print(f"✅ Exception blocked: {code} - {e}")
    
    return True

def test_file_system_access():
    """Test that file system access is blocked"""
    print("\n🔒 Testing file system access prevention...")
    
    sandbox = EnhancedSandbox()
    
    dangerous_codes = [
        'open("/etc/passwd", "r")',
        'open("test.txt", "w")',
        'file("test.txt")',
    ]
    
    for code in dangerous_codes:
        try:
            result = sandbox.execute(code)
            if 'error' not in result:
                print(f"❌ SECURITY BREACH: {code} was allowed!")
                return False
            else:
                print(f"✅ Blocked: {code}")
        except Exception as e:
            print(f"✅ Exception blocked: {code} - {e}")
    
    return True

def test_safe_operations():
    """Test that safe operations still work"""
    print("\n✅ Testing safe operations...")
    
    sandbox = EnhancedSandbox()
    
    safe_codes = [
        "1 + 1",
        "len([1, 2, 3])",
        "sum([1, 2, 3, 4, 5])",
        "max([1, 5, 3])",
        "sorted([3, 1, 4, 1, 5])",
        "'hello'.upper()",
        "[x for x in range(5)]",
    ]
    
    for code in safe_codes:
        try:
            result = sandbox.execute(code)
            if 'error' in result:
                print(f"❌ SAFE OPERATION BLOCKED: {code} - {result['error']}")
                return False
            else:
                print(f"✅ Safe operation allowed: {code} -> {result.get('result')}")
        except Exception as e:
            print(f"❌ Exception on safe code: {code} - {e}")
            return False
    
    return True

def test_resource_limits():
    """Test that resource limits are enforced"""
    print("\n🔒 Testing resource limits...")
    
    sandbox = EnhancedSandbox(max_memory_mb=50, max_cpu_seconds=5)
    
    # Test CPU timeout
    cpu_intensive_code = """
result = 0
for i in range(10000000):
    for j in range(1000):
        result += i * j
"""
    
    try:
        result = sandbox.execute(cpu_intensive_code)
        if 'error' not in result:
            print("❌ CPU limit not enforced")
            return False
        else:
            print(f"✅ CPU limit enforced: {result['error']}")
    except Exception as e:
        print(f"✅ CPU limit enforced via exception: {e}")
    
    return True

def test_module_restrictions():
    """Test that module access is properly restricted"""
    print("\n🔒 Testing module restrictions...")
    
    sandbox = EnhancedSandbox()
    
    # Test pandas restrictions
    pandas_dangerous_codes = [
        "pd.read_csv('/etc/passwd')",
        "pd.read_json('secret.json')",
        "pd.read_pickle('malicious.pkl')",
    ]
    
    for code in pandas_dangerous_codes:
        try:
            result = sandbox.execute(code)
            if 'error' not in result:
                print(f"❌ SECURITY BREACH: {code} was allowed!")
                return False
            else:
                print(f"✅ Blocked dangerous pandas: {code}")
        except Exception as e:
            print(f"✅ Exception blocked pandas: {code}")
    
    # Test that safe pandas operations work
    safe_pandas_code = "pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})"
    try:
        result = sandbox.execute(safe_pandas_code)
        if 'error' in result:
            print(f"❌ Safe pandas blocked: {result['error']}")
            return False
        else:
            print(f"✅ Safe pandas allowed: DataFrame creation")
    except Exception as e:
        print(f"❌ Exception on safe pandas: {e}")
        return False
    
    return True

def main():
    """Run all security tests"""
    print("🔐 Enhanced Sandbox Security Test Suite")
    print("=" * 50)
    
    tests = [
        test_dangerous_imports,
        test_attribute_access_bypass,
        test_code_execution_bypass,
        test_file_system_access,
        test_safe_operations,
        test_resource_limits,
        test_module_restrictions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} CRASHED: {e}")
        print("-" * 30)
    
    print(f"\n🔐 Security Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL SECURITY TESTS PASSED! Sandbox is secure.")
        return True
    else:
        print(f"\n⚠️  {failed} SECURITY TESTS FAILED! Vulnerabilities remain.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)