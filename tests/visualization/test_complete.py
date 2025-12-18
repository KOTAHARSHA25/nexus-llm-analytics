"""
Complete Visualization Test Suite (Phase 4.4)
Runs all visualization tests together
"""
import pytest
import sys

# Run all test modules
if __name__ == "__main__":
    exit_code = pytest.main([
        "tests/visualization/test_simple.py",
        "tests/visualization/test_medium.py",
        "tests/visualization/test_advanced.py",
        "-v",
        "--tb=short"
    ])
    sys.exit(exit_code)
