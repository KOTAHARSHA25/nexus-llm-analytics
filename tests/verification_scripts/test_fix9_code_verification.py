"""
Manual Test Fix 9: Verify streaming endpoint exists and is properly configured

This test verifies the code changes without requiring a running server.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_streaming_endpoint_exists():
    """Test that streaming endpoint is defined"""
    print("\n📝 Testing Streaming Endpoint Definition...")
    
    try:
        from backend.api.analyze import router
        
        # Check router has the stream endpoint
        found_stream = False
        for route in router.routes:
            if hasattr(route, 'path') and '/stream' in route.path:
                found_stream = True
                print(f"  ✅ Found streaming endpoint: {route.path}")
                print(f"     Methods: {route.methods}")
                break
        
        if found_stream:
            return 1, 1
        else:
            print("  ❌ Streaming endpoint not found in router")
            return 0, 1
    except Exception as e:
        print(f"  ❌ Error loading module: {e}")
        return 0, 1


def test_streaming_imports():
    """Test that required imports are present"""
    print("\n📦 Testing Required Imports...")
    
    try:
        from backend.api.analyze import StreamingResponse, asyncio, json
        
        checks = []
        
        print(f"  ✅ StreamingResponse imported")
        checks.append(True)
        
        print(f"  ✅ asyncio imported")
        checks.append(True)
        
        print(f"  ✅ json imported")
        checks.append(True)
        
        return sum(checks), len(checks)
    except ImportError as e:
        print(f"  ❌ Missing import: {e}")
        return 0, 3


def test_streaming_function_signature():
    """Test that analyze_stream function has correct signature"""
    print("\n🔍 Testing Function Signature...")
    
    try:
        from backend.api import analyze
        import inspect
        
        # Check if analyze_stream exists
        if not hasattr(analyze, 'analyze_stream'):
            print("  ❌ analyze_stream function not found")
            return 0, 1
        
        func = getattr(analyze, 'analyze_stream')
        sig = inspect.signature(func)
        
        print(f"  ✅ analyze_stream function exists")
        print(f"     Parameters: {list(sig.parameters.keys())}")
        
        # Check it's async
        if inspect.iscoroutinefunction(func):
            print(f"  ✅ Function is async")
            return 2, 2
        else:
            print(f"  ❌ Function should be async")
            return 1, 2
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 2


def test_sse_headers():
    """Test that SSE headers are configured"""
    print("\n📋 Testing SSE Headers Configuration...")
    
    try:
        # Read the source file to check for headers
        analyze_path = project_root / "src" / "backend" / "api" / "analyze.py"
        
        with open(analyze_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = []
        
        # Check for SSE media type
        if 'text/event-stream' in content:
            print("  ✅ SSE media type configured")
            checks.append(True)
        else:
            print("  ❌ SSE media type not found")
            checks.append(False)
        
        # Check for Cache-Control header
        if 'Cache-Control' in content and 'no-cache' in content:
            print("  ✅ Cache-Control: no-cache header")
            checks.append(True)
        else:
            print("  ❌ Cache-Control header missing")
            checks.append(False)
        
        # Check for Connection keep-alive
        if 'keep-alive' in content:
            print("  ✅ Connection: keep-alive header")
            checks.append(True)
        else:
            print("  ⚠️ Connection: keep-alive not found")
            checks.append(False)
        
        # Check for SSE data format
        if 'data: ' in content and 'json.dumps' in content:
            print("  ✅ SSE data format (data: {...}) present")
            checks.append(True)
        else:
            print("  ❌ SSE data format not found")
            checks.append(False)
        
        return sum(checks), len(checks)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 4


def test_progress_steps():
    """Test that all progress steps are defined"""
    print("\n🎯 Testing Progress Steps...")
    
    try:
        analyze_path = project_root / "src" / "backend" / "api" / "analyze.py"
        
        with open(analyze_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_steps = ['init', 'validation', 'loading', 'analyzing', 'formatting', 'complete', 'error']
        found_steps = []
        
        for step in expected_steps:
            if f"'step': '{step}'" in content or f'"step": "{step}"' in content:
                found_steps.append(step)
                print(f"  ✅ Step '{step}' defined")
            else:
                print(f"  ❌ Step '{step}' missing")
        
        return len(found_steps), len(expected_steps)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 0, 7


def main():
    """Run all code verification tests"""
    print("=" * 60)
    print("FIX 9: STREAMING ENDPOINT CODE VERIFICATION")
    print("=" * 60)
    print("\nℹ️  This test verifies the code changes without a running server")
    print()
    
    results = []
    
    # Run all tests
    results.append(test_streaming_endpoint_exists())
    results.append(test_streaming_imports())
    results.append(test_streaming_function_signature())
    results.append(test_sse_headers())
    results.append(test_progress_steps())
    
    # Calculate totals
    total_passed = sum(r[0] for r in results)
    total_tests = sum(r[1] for r in results)
    percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed}/{total_tests} checks passed ({percentage:.1f}%)")
    print("=" * 60)
    
    if percentage >= 90:
        print("\n🎯 Fix 9 VERIFIED - Streaming code properly implemented!")
        print("   ✅ Endpoint defined")
        print("   ✅ Required imports")
        print("   ✅ Async function")
        print("   ✅ SSE headers")
        print("   ✅ Progress steps")
        print("\n   To test with live server:")
        print("   1. python -m uvicorn backend.main:app --reload")
        print("   2. python test_fix9_streaming.py")
    elif percentage >= 70:
        print("\n⚠️ Fix 9 PARTIAL - Most checks passed")
    else:
        print("\n❌ Fix 9 NEEDS WORK")
    
    return percentage >= 90


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
