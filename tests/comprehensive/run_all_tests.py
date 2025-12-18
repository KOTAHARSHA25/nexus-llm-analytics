"""
MASTER TEST RUNNER
Runs all comprehensive tests in order: Unit → Integration → System
"""
import subprocess
import sys
from pathlib import Path
import time


def run_test_suite(test_file, description):
    """Run a test suite and return results"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print('='*70)
    
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    duration = time.time() - start_time
    
    print(f"\nCompleted in {duration:.2f}s")
    return result.returncode == 0


def main():
    """Run all comprehensive tests"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║         NEXUS LLM ANALYTICS - COMPREHENSIVE TEST SUITE             ║
║                                                                    ║
║  Testing: ALL Agents | ALL File Types | ALL Components            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    tests_dir = Path(__file__).parent
    
    # Test suite order
    test_suites = [
        # Phase 1: Unit Tests
        ("test_core_components.py", "UNIT TESTS: Core Components"),
        
        # Phase 2: Agent Tests
        ("test_all_agents.py", "UNIT TESTS: All 5 Plugin Agents"),
        
        # Phase 3: File Type Tests
        ("test_all_file_types.py", "INTEGRATION TESTS: All File Types (CSV/JSON/PDF/TXT)"),
        
        # Phase 4: API Integration Tests
        ("test_api_integration.py", "INTEGRATION TESTS: All API Endpoints"),
        
        # Phase 5: E2E Tests
        ("test_e2e_workflows.py", "SYSTEM TESTS: End-to-End Workflows"),
    ]
    
    results = []
    
    for test_file, description in test_suites:
        test_path = tests_dir / test_file
        if test_path.exists():
            success = run_test_suite(str(test_path), description)
            results.append((description, success))
        else:
            print(f"\n⚠️  {test_file} not found, skipping...")
            results.append((description, None))
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("  COMPREHENSIVE TEST RESULTS SUMMARY")
    print('='*70)
    
    for description, success in results:
        if success is None:
            status = "⊘ SKIPPED"
        elif success:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"{status:12} {description}")
    
    print('='*70)
    
    # Calculate overall success
    passed = sum(1 for _, s in results if s is True)
    failed = sum(1 for _, s in results if s is False)
    skipped = sum(1 for _, s in results if s is None)
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is fully operational.\n")
        return 0
    else:
        print(f"\n⚠️  {failed} test suite(s) failed. Review output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
