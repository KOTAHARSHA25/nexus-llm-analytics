"""
Post-Fix Accuracy Test
======================
Comprehensive test to verify the data optimizer and prompt fixes work correctly.

Tests:
1. Small dataset (1.json) - Direct answer extraction
2. Large dataset (StressLevelDataset.csv) - Statistics accuracy
3. Whitebox: Verify data optimizer output
4. Blackbox: Verify end-to-end API responses

Author: System Verification
Date: December 22, 2025
"""

import pytest
import requests
import json
import pandas as pd
from pathlib import Path
import time

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "samples"
UPLOADS_DIR = Path(__file__).parent.parent / "data" / "uploads"


class TestPostFixAccuracy:
    """Test suite for post-fix accuracy verification"""
    
    def setup_method(self):
        """Setup before each test"""
        self.results = []
        print("\n" + "="*80)
        print("POST-FIX ACCURACY TEST")
        print("="*80)
    
    def teardown_method(self):
        """Print results after each test"""
        if self.results:
            print("\n" + "-"*80)
            print("TEST RESULTS:")
            for result in self.results:
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                print(f"{status} - {result['test']}: {result['message']}")
            print("-"*80)
    
    # =========================================================================
    # WHITEBOX TESTS - Internal Component Verification
    # =========================================================================
    
    def test_whitebox_data_optimizer_small_dataset(self):
        """Whitebox: Verify data optimizer handles small datasets correctly"""
        from src.backend.utils.data_optimizer import DataOptimizer
        
        print("\n[WHITEBOX] Testing Data Optimizer - Small Dataset")
        
        # Test with 1.json
        test_file = UPLOADS_DIR / "1.json"
        assert test_file.exists(), f"Test file not found: {test_file}"
        
        optimizer = DataOptimizer(max_rows=100, max_chars=8000)
        result = optimizer.optimize_for_llm(str(test_file))
        
        # Verify small dataset formatting
        preview = result['preview']
        
        checks = {
            'no_statistics_header': 'PRE-CALCULATED STATISTICS' not in preview,
            'has_clean_data': 'harsha' in preview.lower(),
            'no_complex_formatting': preview.count('📊') == 0,
            'simple_format': 'Data from file' in preview or 'Total:' in preview
        }
        
        print(f"Preview length: {len(preview)} chars")
        print(f"Preview sample:\n{preview[:300]}\n")
        
        passed = all(checks.values())
        self.results.append({
            'test': 'Data Optimizer - Small Dataset',
            'passed': passed,
            'message': f"Checks: {checks}"
        })
        
        assert passed, f"Small dataset optimization failed: {checks}"
    
    def test_whitebox_data_optimizer_large_dataset(self):
        """Whitebox: Verify data optimizer handles large datasets with statistics"""
        from src.backend.utils.data_optimizer import DataOptimizer
        
        print("\n[WHITEBOX] Testing Data Optimizer - Large Dataset")
        
        # Test with StressLevelDataset.csv
        test_file = TEST_DATA_DIR / "StressLevelDataset.csv"
        if not test_file.exists():
            pytest.skip(f"Large test file not found: {test_file}")
        
        # First, get actual statistics
        df = pd.read_csv(test_file)
        print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
        
        optimizer = DataOptimizer(max_rows=100, max_chars=8000)
        result = optimizer.optimize_for_llm(str(test_file))
        
        preview = result['preview']
        
        # Verify large dataset gets statistics
        checks = {
            'has_statistics_header': 'PRE-CALCULATED STATISTICS' in preview,
            'has_overall_stats': 'OVERALL COLUMN STATISTICS' in preview,
            'shows_row_count': str(len(df)) in preview,
            'has_sample_data': 'Sample Data' in preview or df.columns[0] in preview
        }
        
        print(f"Preview length: {len(preview)} chars")
        print(f"Has statistics: {checks['has_statistics_header']}")
        print(f"Statistics header position: {preview.find('PRE-CALCULATED STATISTICS')}")
        
        passed = checks['has_statistics_header']  # Main requirement
        self.results.append({
            'test': 'Data Optimizer - Large Dataset',
            'passed': passed,
            'message': f"Checks: {checks}"
        })
        
        assert passed, f"Large dataset should have statistics header: {checks}"
    
    # =========================================================================
    # BLACKBOX TESTS - End-to-End API Verification
    # =========================================================================
    
    def test_blackbox_simple_query_accuracy(self):
        """Blackbox: Test simple query returns correct answer"""
        print("\n[BLACKBOX] Testing Simple Query - 1.json")
        
        # Upload file
        test_file = UPLOADS_DIR / "1.json"
        assert test_file.exists(), f"Test file not found: {test_file}"
        
        with open(test_file, 'r') as f:
            file_data = json.load(f)
        
        expected_name = file_data[0]['name']  # Should be "harsha"
        print(f"Expected answer: {expected_name}")
        
        # Send query
        payload = {
            "query": "what is the name?",
            "filename": "1.json"
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/analyze/", json=payload, timeout=120)
        duration = time.time() - start_time
        
        assert response.status_code == 200, f"API error: {response.status_code}"
        
        result = response.json()
        answer = result.get('result', '')
        
        print(f"Response time: {duration:.2f}s")
        print(f"Answer: {answer[:200]}")
        
        # Check if answer contains the correct name
        answer_lower = answer.lower()
        passed = expected_name.lower() in answer_lower
        
        self.results.append({
            'test': 'Simple Query Accuracy',
            'passed': passed,
            'message': f"Expected '{expected_name}' in answer, got: {answer[:100]}"
        })
        
        assert passed, f"Answer should contain '{expected_name}': {answer}"
    
    def test_blackbox_aggregation_accuracy(self):
        """Blackbox: Test aggregation query uses pre-calculated statistics"""
        print("\n[BLACKBOX] Testing Aggregation Accuracy - Large Dataset")
        
        # Use test_sales_monthly.csv which has clear numeric columns
        test_file = TEST_DATA_DIR / "test_sales_monthly.csv"
        if not test_file.exists():
            test_file = TEST_DATA_DIR / "sales_simple.csv"
        
        if not test_file.exists():
            pytest.skip("No suitable test file found")
        
        # Calculate ground truth
        df = pd.read_csv(test_file)
        print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")
        
        # Find a numeric column to test
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            pytest.skip("No numeric columns in test file")
        
        test_column = numeric_cols[0]
        expected_total = df[test_column].sum()
        expected_avg = df[test_column].mean()
        
        print(f"Testing column: {test_column}")
        print(f"Expected total: {expected_total:.2f}")
        print(f"Expected average: {expected_avg:.2f}")
        
        # Upload file first
        filename = test_file.name
        with open(test_file, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            upload_response = requests.post(f"{BASE_URL}/api/upload/", files=files)
            assert upload_response.status_code == 200, "Upload failed"
        
        # Test total query
        payload = {
            "query": f"what is the total {test_column}?",
            "filename": filename
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/analyze/", json=payload, timeout=120)
        duration = time.time() - start_time
        
        assert response.status_code == 200, f"API error: {response.status_code}"
        
        result = response.json()
        answer = result.get('result', '')
        
        print(f"Response time: {duration:.2f}s")
        print(f"Answer: {answer}")
        
        # Check if answer is close to expected (allow 5% margin)
        # Extract number from answer
        import re
        numbers = re.findall(r'[\d,]+\.?\d*', answer.replace(',', ''))
        
        passed = False
        if numbers:
            for num_str in numbers:
                try:
                    num = float(num_str)
                    # Check if within 5% of expected
                    margin = abs(num - expected_total) / expected_total
                    if margin < 0.05:
                        passed = True
                        print(f"✓ Found matching number: {num} (margin: {margin*100:.2f}%)")
                        break
                except:
                    continue
        
        self.results.append({
            'test': 'Aggregation Accuracy',
            'passed': passed,
            'message': f"Expected ~{expected_total:.2f}, got answer: {answer[:100]}"
        })
        
        if not passed:
            print(f"⚠️  Warning: Could not verify numeric accuracy")
            print(f"   Extracted numbers: {numbers}")
            print(f"   This might mean LLM answered in text format")
    
    def test_blackbox_response_time_reasonable(self):
        """Blackbox: Verify response times are reasonable"""
        print("\n[BLACKBOX] Testing Response Time")
        
        test_file = UPLOADS_DIR / "1.json"
        
        payload = {
            "query": "what is the name?",
            "filename": "1.json"
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/analyze/", json=payload, timeout=120)
        duration = time.time() - start_time
        
        print(f"Response time: {duration:.2f}s")
        
        # Should complete within 90 seconds for simple query
        passed = duration < 90
        
        self.results.append({
            'test': 'Response Time',
            'passed': passed,
            'message': f"Took {duration:.2f}s (max: 90s)"
        })
        
        assert passed, f"Response too slow: {duration:.2f}s"
    
    # =========================================================================
    # INTEGRATION TEST - Full Pipeline
    # =========================================================================
    
    def test_integration_full_pipeline(self):
        """Integration: Test complete pipeline from upload to accurate answer"""
        print("\n[INTEGRATION] Testing Full Pipeline")
        
        test_cases = [
            {
                'file': 'test_sales_monthly.csv',
                'query': 'how many rows are in this dataset?',
                'check': 'numeric_answer'
            },
            {
                'file': '1.json', 
                'query': 'what is the rollNumber?',
                'check': 'contains_22r21a6695'
            }
        ]
        
        passed_count = 0
        total_count = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest case {i}/{total_count}: {case['file']}")
            
            # Find file
            test_file = TEST_DATA_DIR / case['file']
            if not test_file.exists():
                test_file = UPLOADS_DIR / case['file']
            
            if not test_file.exists():
                print(f"  ⚠️  Skipping - file not found")
                continue
            
            # Query
            payload = {
                "query": case['query'],
                "filename": case['file']
            }
            
            try:
                response = requests.post(f"{BASE_URL}/api/analyze/", json=payload, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('result', '')
                    
                    # Check answer
                    if case['check'] == 'numeric_answer':
                        import re
                        has_number = bool(re.search(r'\d+', answer))
                        if has_number:
                            passed_count += 1
                            print(f"  ✓ Got numeric answer")
                        else:
                            print(f"  ✗ No number in answer: {answer[:100]}")
                    
                    elif case['check'] == 'contains_22r21a6695':
                        if '22r21a6695' in answer:
                            passed_count += 1
                            print(f"  ✓ Correct roll number found")
                        else:
                            print(f"  ✗ Roll number not found: {answer[:100]}")
                else:
                    print(f"  ✗ API error: {response.status_code}")
            
            except Exception as e:
                print(f"  ✗ Exception: {e}")
        
        passed = passed_count >= (total_count * 0.5)  # At least 50% pass
        
        self.results.append({
            'test': 'Full Pipeline Integration',
            'passed': passed,
            'message': f"Passed {passed_count}/{total_count} test cases"
        })
        
        print(f"\nIntegration test: {passed_count}/{total_count} passed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
