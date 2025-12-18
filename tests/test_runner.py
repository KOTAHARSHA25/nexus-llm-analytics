# Test Execution Guide and Documentation
# Comprehensive guide for running the complete test suite

"""
Nexus LLM Analytics - Comprehensive Test Suite
=============================================

This test suite provides complete coverage across 6 testing categories:
1. Unit Tests - Component functionality and reliability
2. Integration Tests - Component interactions and system behavior  
3. E2E Tests - Complete user workflows and scenarios
4. Performance Tests - Optimization validation and benchmarking
5. Property-Based/Fuzz Tests - Edge cases and robustness
6. Security Tests - Vulnerability assessment and protection

Test Structure:
--------------
tests/
├── conftest.py                           # Central test configuration and fixtures
├── pytest.ini                          # Pytest configuration and settings
├── unit/                               # Unit testing suite
│   ├── test_data_structures.py        # Data structure component tests
│   ├── test_llm_client.py             # LLM client component tests
│   └── test_file_io.py                # File I/O component tests
├── integration/                        # Integration testing suite
│   └── test_component_interactions.py  # Cross-component integration tests
├── e2e/                               # End-to-end testing suite
│   └── test_user_workflows.py         # Complete user workflow tests
├── performance/                        # Performance testing suite
│   └── test_benchmarks.py             # Performance benchmarks and validation
├── fuzz/                              # Property-based and fuzz testing
│   ├── test_property_based.py         # Hypothesis-based property testing
│   ├── test_boundary_validation.py    # Boundary condition and input validation
│   └── test_stress_robustness.py      # Stress testing and robustness
└── security/                          # Security testing suite
    ├── test_security_validation.py    # Security validation and input sanitization
    └── test_penetration_testing.py    # Penetration testing and vulnerability assessment

Running Tests:
-------------

1. Run All Tests:
   pytest

2. Run Specific Category:
   pytest tests/unit/                   # Unit tests only
   pytest tests/integration/            # Integration tests only
   pytest tests/e2e/                   # E2E tests only
   pytest tests/performance/            # Performance tests only
   pytest tests/fuzz/                  # Fuzz tests only
   pytest tests/security/              # Security tests only

3. Run with Coverage:
   pytest --cov=src --cov-report=html --cov-report=term

4. Run Performance Tests with Benchmarking:
   pytest tests/performance/ -v -s --durations=10

5. Run Security Tests with Vulnerability Assessment:
   pytest tests/security/ -v -s --tb=short

6. Run Fuzz Tests with Property-Based Testing:
   pytest tests/fuzz/ -v --hypothesis-show-statistics

7. Run Quick Smoke Tests:
   pytest -m "not slow" --maxfail=5

Test Markers:
------------
- @pytest.mark.unit          # Unit tests
- @pytest.mark.integration   # Integration tests
- @pytest.mark.e2e          # End-to-end tests
- @pytest.mark.performance  # Performance tests
- @pytest.mark.fuzz         # Fuzz/property-based tests
- @pytest.mark.security     # Security tests
- @pytest.mark.slow         # Slow-running tests
- @pytest.mark.async        # Async tests
- @pytest.mark.mock         # Tests using mocks

Environment Setup:
-----------------
1. Install test dependencies:
   pip install -r requirements.txt
   pip install pytest pytest-cov pytest-asyncio hypothesis selenium

2. Set environment variables:
   export NEXUS_TEST_ENV=true
   export NEXUS_LOG_LEVEL=DEBUG

3. Configure browser for E2E tests:
   # Chrome/Chromium required for Selenium tests
   # Download ChromeDriver and add to PATH

Dependencies:
------------
Core Testing:
- pytest>=7.0.0
- pytest-asyncio>=0.21.0
- pytest-cov>=4.0.0

Property-Based Testing:
- hypothesis>=6.0.0

E2E Testing:
- selenium>=4.0.0
- webdriver-manager>=3.8.0

Performance Testing:
- psutil>=5.9.0
- memory-profiler>=0.60.0

Security Testing:
- requests>=2.28.0
- PyJWT>=2.6.0

Test Configuration:
------------------
See pytest.ini for detailed configuration including:
- Coverage reporting settings
- Test markers and categories
- Output formatting options
- Parallel execution settings
- Tool integration (black, mypy, ruff, bandit)

Expected Results:
----------------
- Unit Tests: >95% pass rate, high code coverage
- Integration Tests: >90% pass rate, component interaction validation
- E2E Tests: >85% pass rate, complete workflow validation
- Performance Tests: Benchmark compliance, optimization validation
- Fuzz Tests: Robustness validation, edge case handling
- Security Tests: Vulnerability assessment, protection validation

Troubleshooting:
---------------
1. Import Errors:
   - Ensure src/ is in Python path
   - Check virtual environment activation

2. Selenium Errors:
   - Install ChromeDriver
   - Check browser compatibility
   - Verify display server (Linux)

3. Performance Test Failures:
   - Check system resources
   - Adjust benchmark thresholds
   - Run on dedicated test environment

4. Memory Issues:
   - Reduce test data sizes
   - Run tests in smaller batches
   - Check system memory availability

5. Security Test Issues:
   - Check file permissions
   - Verify network access
   - Review firewall settings

Continuous Integration:
----------------------
For CI/CD integration, use:
pytest --cov=src --cov-report=xml --junitxml=test-results.xml -m "not slow"

This provides coverage reports and test results in CI-friendly formats while
excluding slow-running tests for faster feedback loops.

Development Workflow:
-------------------
1. Write new features with corresponding unit tests
2. Run quick tests during development: pytest -x --ff
3. Run full test suite before commits: pytest
4. Run performance tests for optimization changes
5. Run security tests for security-related changes
6. Use coverage reports to identify untested code

Test Data Management:
-------------------
- Test fixtures are defined in conftest.py
- Temporary files are automatically cleaned up
- Mock data is generated using TestDataManager
- Large test datasets are created on-demand
- Sensitive test data is sanitized

Reporting:
---------
Test results include:
- Code coverage percentage and detailed reports
- Performance benchmarks and comparisons
- Security vulnerability assessments
- Property-based testing statistics
- Test execution times and resource usage

For detailed analysis, generate HTML coverage reports:
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
"""

import pytest
import os
import sys
from pathlib import Path

# Test runner utilities
class TestRunner:
    """Utility class for running specific test categories"""
    
    @staticmethod
    def run_unit_tests():
        """Run unit tests with coverage"""
        return pytest.main([
            'tests/unit/',
            '--cov=src',
            '--cov-report=term-missing',
            '-v'
        ])
    
    @staticmethod
    def run_integration_tests():
        """Run integration tests"""
        return pytest.main([
            'tests/integration/',
            '-v',
            '--tb=short'
        ])
    
    @staticmethod
    def run_e2e_tests():
        """Run E2E tests"""
        return pytest.main([
            'tests/e2e/',
            '-v',
            '--tb=short',
            '-s'  # Show output for browser tests
        ])
    
    @staticmethod
    def run_performance_tests():
        """Run performance tests with benchmarking"""
        return pytest.main([
            'tests/performance/',
            '-v',
            '-s',
            '--durations=10'
        ])
    
    @staticmethod
    def run_fuzz_tests():
        """Run fuzz and property-based tests"""
        return pytest.main([
            'tests/fuzz/',
            '-v',
            '--hypothesis-show-statistics',
            '--tb=short'
        ])
    
    @staticmethod
    def run_security_tests():
        """Run security tests"""
        return pytest.main([
            'tests/security/',
            '-v',
            '-s',
            '--tb=short'
        ])
    
    @staticmethod
    def run_smoke_tests():
        """Run quick smoke tests"""
        return pytest.main([
            '-m', 'not slow',
            '--maxfail=5',
            '-x',  # Stop on first failure
            '--ff'  # Failed first
        ])
    
    @staticmethod
    def run_full_suite():
        """Run complete test suite with coverage"""
        return pytest.main([
            '--cov=src',
            '--cov-report=html',
            '--cov-report=term',
            '--cov-report=xml',
            '--junitxml=test-results.xml',
            '-v'
        ])

def verify_test_environment():
    """Verify test environment is properly configured"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check required packages
    required_packages = [
        'pytest', 'pytest-asyncio', 'pytest-cov', 'hypothesis', 
        'selenium', 'psutil', 'requests'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            issues.append(f"Missing package: {package}")
    
    # Check src directory
    src_path = Path(__file__).parent.parent / 'src'
    if not src_path.exists():
        issues.append("src/ directory not found")
    
    # Check test directories
    test_dirs = ['unit', 'integration', 'e2e', 'performance', 'fuzz', 'security']
    tests_path = Path(__file__).parent
    
    for test_dir in test_dirs:
        if not (tests_path / test_dir).exists():
            issues.append(f"Test directory missing: {test_dir}")
    
    return issues

if __name__ == '__main__':
    """Main entry point for test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Nexus LLM Analytics Test Runner')
    parser.add_argument('category', nargs='?', choices=[
        'unit', 'integration', 'e2e', 'performance', 'fuzz', 'security', 'smoke', 'all'
    ], default='smoke', help='Test category to run')
    parser.add_argument('--verify', action='store_true', help='Verify test environment')
    
    args = parser.parse_args()
    
    if args.verify:
        print("Verifying test environment...")
        issues = verify_test_environment()
        if issues:
            print("Environment issues found:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        else:
            print("Test environment is properly configured.")
            sys.exit(0)
    
    runner = TestRunner()
    
    print(f"Running {args.category} tests...")
    
    if args.category == 'unit':
        exit_code = runner.run_unit_tests()
    elif args.category == 'integration':
        exit_code = runner.run_integration_tests()
    elif args.category == 'e2e':
        exit_code = runner.run_e2e_tests()
    elif args.category == 'performance':
        exit_code = runner.run_performance_tests()
    elif args.category == 'fuzz':
        exit_code = runner.run_fuzz_tests()
    elif args.category == 'security':
        exit_code = runner.run_security_tests()
    elif args.category == 'smoke':
        exit_code = runner.run_smoke_tests()
    elif args.category == 'all':
        exit_code = runner.run_full_suite()
    
    sys.exit(exit_code)