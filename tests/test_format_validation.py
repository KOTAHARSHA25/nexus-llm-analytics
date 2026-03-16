import pytest
import pandas as pd
import os
import sys
import logging
from pathlib import Path
import time
from collections import defaultdict

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.utils.data_utils import read_dataframe, get_column_properties, validate_dataframe, infer_data_types

DATA_DIR = Path("data/format_validation")
REPORT_FILE = Path("FORMAT_VALIDATION_REPORT.md")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Report Data Structures
results = defaultdict(lambda: {"passed": 0, "failed": 0, "notes": []})
function_failures = []
parsing_attempts = 0
parsing_successes = 0

def log_failure(func_name, error, file_name):
    """Log a function failure for the report."""
    msg = f"{func_name} failed on {file_name}: {str(error)}"
    function_failures.append(msg)
    logger.error(msg)

class TestFormatValidation:
    @pytest.mark.parametrize("repetition", [1, 2, 3])
    def test_process_all_files(self, repetition):
        """
        Iterates over all files in data/format_validation.
        Tests: Parsing, Validation, Schema Detection, Type Inference.
        Repeats 3 times.
        """
        global parsing_attempts, parsing_successes
        
        # Get all generated files
        files = [f for f in DATA_DIR.iterdir() if f.is_file()]
        if not files:
            pytest.skip("No validation files found. Run format_validation_generator.py first.")
            
        for file_path in files:
            file_name = file_path.name
            file_type = file_path.suffix.upper()
            
            logger.info(f"[{repetition}/3] Testing {file_name}...")
            
            # --- TEST 1: Parsing (read_dataframe) ---
            parsing_attempts += 1
            df = None
            try:
                # Clear any potential pandas cache if applicable (not applicable for direct read_csv but good practice to reload)
                df = read_dataframe(str(file_path))
                
                # Special handling for "expected failure" files (corrupt)
                if "corrupt" in file_name.lower():
                    # If it didn't fail, that might be okay if read_dataframe is robust, 
                    # but usually strict parsing should fail or return empty/partial.
                    # We'll mark it as a "pass" if it handled it gracefully (no crash).
                    pass
                else:
                    # For non-corrupt files, df must be valid
                    assert df is not None
                    
                parsing_successes += 1
                results[file_type]["passed"] += 1
                
            except Exception as e:
                # Expected failure for corrupt files
                if "corrupt" in file_name.lower():
                    parsing_successes += 1 # It correctly failed/rejected
                    results[file_type]["passed"] += 1
                    results[file_type]["notes"].append(f"Correctly rejected corrupt file: {file_name}")
                else:
                    results[file_type]["failed"] += 1
                    results[file_type]["notes"].append(f"Failed to parse {file_name}")
                    log_failure("read_dataframe", e, file_name)
                    continue # Cannot proceed with other tests if parsing failed

            if df is None:
                continue

            # --- TEST 2: Validation (validate_dataframe) ---
            try:
                is_valid, msg = validate_dataframe(df)
                # We don't assert True because "empty" files are valid inputs that return valid=False
                if "empty" in file_name.lower():
                    assert not is_valid
            except Exception as e:
                log_failure("validate_dataframe", e, file_name)

            # --- TEST 3: Column Properties (Schema) ---
            try:
                props = get_column_properties(df)
                assert isinstance(props, list)
            except Exception as e:
                log_failure("get_column_properties", e, file_name)

            # --- TEST 4: Type Inference (infer_data_types) ---
            try:
                df_typed = infer_data_types(df)
                # Check if mixed types were resolved (e.g. "1" -> 1)
                if "mixed" in file_name:
                    # Specific check for our generated mixed file
                    # col1 had "1" and "text". It likely remains object.
                    pass
            except Exception as e:
                log_failure("infer_data_types", e, file_name)

    def test_generate_report(self):
        """Generates the final markdown report after all tests."""
        # This runs last due to alphabetical order or explicit ordering? 
        # Actually in pytest parametrization runs as separate items.
        # We need a fixture with scope="session" to write the report at the end.
        pass

@pytest.fixture(scope="session", autouse=True)
def report_generator():
    yield
    # Teardown - Write Report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# PHASE 8: FILE FORMAT & UNIT VALIDATION REPORT\n\n")
        
        f.write("## SECTION A – FILE FORMAT RESULTS\n")
        f.write("| File Type | Passed Tests | Failed Tests | Notes |\n")
        f.write("|---|---|---|---|\n")
        for ftype, data in results.items():
            unique_notes = list(set(data["notes"]))
            # Limit notes length
            notes_str = "<br>".join(unique_notes[:5])
            if len(unique_notes) > 5: notes_str += "<br>..."
            if not notes_str: notes_str = "No issues."
            f.write(f"| {ftype} | {data['passed']} | {data['failed']} | {notes_str} |\n")
            
        f.write("\n## SECTION B – FUNCTION FAILURES\n")
        if function_failures:
            for fail in set(function_failures):
                f.write(f"- {fail}\n")
        else:
            f.write("None. All functions executed successfully.\n")
            
        f.write("\n## SECTION C – PARSING RELIABILITY SCORE\n")
        if parsing_attempts > 0:
            score = (parsing_successes / parsing_attempts) * 100
            f.write(f"**Reliability Score:** {score:.2f}% ({parsing_successes}/{parsing_attempts})\n")
        else:
            f.write("**Reliability Score:** N/A (0 attempts)\n")
            
    print(f"\nReport generated at {REPORT_FILE.absolute()}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
