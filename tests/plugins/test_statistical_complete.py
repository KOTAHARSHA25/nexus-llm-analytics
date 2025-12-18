"""
COMPLETE TEST SUITE - Statistical Analysis Agent
Master runner for Simple, Medium, and Advanced tests
"""

import subprocess
import sys
from pathlib import Path

print("="*80)
print("STATISTICAL AGENT - COMPLETE TEST SUITE")
print("="*80)
print("Running comprehensive tests across all complexity levels\n")

# Test files
test_dir = Path(__file__).parent
test_files = [
    ('SIMPLE', test_dir / 'test_statistical_simple.py'),
    ('MEDIUM', test_dir / 'test_statistical_medium.py'),
    ('ADVANCED', test_dir / 'test_statistical_advanced.py')
]

results = []

for level, test_file in test_files:
    print("\n" + "="*80)
    print(f"RUNNING {level} TESTS")
    print("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Print output
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check if successful
        success = result.returncode == 0 and "SUCCESS" in result.stdout
        results.append((level, success, result.returncode))
        
        if success:
            print(f"\n✅ {level} TESTS: PASSED")
        else:
            print(f"\n❌ {level} TESTS: FAILED (exit code: {result.returncode})")
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ {level} TESTS: TIMEOUT")
        results.append((level, False, -1))
    except Exception as e:
        print(f"\n❌ {level} TESTS: ERROR - {e}")
        results.append((level, False, -2))

# Final summary
print("\n" + "="*80)
print("FINAL TEST SUMMARY")
print("="*80)

total_passed = sum(1 for _, success, _ in results if success)
total_tests = len(results)

for level, success, code in results:
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {level} Tests (exit code: {code})")

print("\n" + "="*80)
print(f"OVERALL RESULT: {total_passed}/{total_tests} test suites passed")
print("="*80)

if total_passed == total_tests:
    print("\n🎉 ALL TESTS PASSED! Statistical Agent is production-ready!")
    sys.exit(0)
else:
    print(f"\n⚠️ {total_tests - total_passed} test suite(s) failed. Review output above.")
    sys.exit(1)
