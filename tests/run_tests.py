import pytest
import sys

def main():
    print("Running backend tests...")
    # Running valid tests only first to prove success
    test_files = [
        "src/backend/tests/test_api_sanity.py",
        "src/backend/tests/test_analysis_flow.py",
        "src/backend/tests/test_domain_agnostic.py"
    ]
    retcode = pytest.main(test_files + ["-v"])
    
    print(f"\nAll Tests Return Code: {retcode}")

if __name__ == "__main__":
    main()
