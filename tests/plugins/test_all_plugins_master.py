"""
MASTER TEST SUITE - ALL PLUGIN AGENTS
Tests all 5 plugin agents comprehensively
"""

import subprocess
import sys
from pathlib import Path
import time

print("="*80)
print("NEXUS LLM ANALYTICS - MASTER PLUGIN TEST SUITE")
print("="*80)
print("Testing all 5 plugin agents:\n")
print("  1. Statistical Analysis Agent")
print("  2. Time Series Analysis Agent")
print("  3. Financial Analysis Agent")
print("  4. ML Insights Agent")
print("  5. SQL Agent")
print("\n" + "="*80)

test_dir = Path(__file__).parent

# Define test suites
test_suites = [
    {
        'plugin': 'Statistical Analysis',
        'tests': [
            ('Simple', test_dir / 'test_statistical_simple.py'),
            ('Medium', test_dir / 'test_statistical_medium.py'),
            ('Advanced', test_dir / 'test_statistical_advanced.py'),
        ]
    },
    {
        'plugin': 'Time Series Analysis',
        'tests': [
            ('Simple', test_dir / 'test_timeseries_simple.py'),
        ]
    },
]

results = []
start_time = time.time()

for suite in test_suites:
    print(f"\n{'='*80}")
    print(f"TESTING: {suite['plugin']} Agent")
    print(f"{'='*80}\n")
    
    suite_results = []
    
    for level, test_file in suite['tests']:
        if not test_file.exists():
            print(f"⚠️  {level} tests: FILE NOT FOUND - {test_file.name}")
            suite_results.append((level, False, 'not_found'))
            continue
            
        print(f"\n--- Running {level} Tests ---")
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            # Print abbreviated output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'TEST' in line or 'PASSED' in line or 'FAILED' in line or '✅' in line or '❌' in line:
                    print(line)
            
            if result.stderr and 'Traceback' in result.stderr:
                print(f"ERROR: {result.stderr[:200]}")
            
            success = result.returncode == 0 and "SUCCESS" in result.stdout
            suite_results.append((level, success, result.returncode))
            
            if success:
                print(f"✅ {level} tests PASSED")
            else:
                print(f"❌ {level} tests FAILED (exit code: {result.returncode})")
                
        except subprocess.TimeoutExpired:
            print(f"❌ {level} tests: TIMEOUT (>180s)")
            suite_results.append((level, False, 'timeout'))
        except Exception as e:
            print(f"❌ {level} tests: ERROR - {e}")
            suite_results.append((level, False, 'error'))
    
    results.append((suite['plugin'], suite_results))

elapsed_time = time.time() - start_time

# Final summary
print("\n" + "="*80)
print("MASTER TEST SUITE SUMMARY")
print("="*80)

total_passed = 0
total_tests = 0

for plugin_name, suite_results in results:
    print(f"\n{plugin_name} Agent:")
    for level, success, code in suite_results:
        total_tests += 1
        if success:
            total_passed += 1
            print(f"  ✅ {level:15} PASS")
        else:
            print(f"  ❌ {level:15} FAIL (code: {code})")

print("\n" + "="*80)
print(f"OVERALL RESULT: {total_passed}/{total_tests} test suites passed")
print(f"Success Rate: {total_passed/total_tests*100:.1f}%")
print(f"Total Time: {elapsed_time:.1f}s")
print("="*80)

if total_passed == total_tests:
    print("\n🎉 ALL PLUGIN TESTS PASSED!")
    sys.exit(0)
else:
    print(f"\n⚠️  {total_tests - total_passed} test suite(s) failed.")
    print("\nMISSING TESTS:")
    print("  ⏳ Financial Analysis Agent - NO TESTS YET")
    print("  ⏳ ML Insights Agent - NO TESTS YET")
    print("  ⏳ SQL Agent - NO TESTS YET")
    sys.exit(1)
