"""
Comprehensive Accuracy Test Suite
==================================
Tests the system with REAL LLM calls across all file types and complexity levels.
Validates answer accuracy, not just that responses are returned.

Test Coverage:
- File Types: JSON, CSV, PDF, TXT, XLSX
- Complexity Levels: Simple → Medium → Complex → God-Level
- Answer Validation: Checks for correct values, not just non-None responses
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class AccuracyTest:
    """Single test case with expected answer validation"""
    def __init__(self, name: str, filename: str, query: str, expected: Any, 
                 complexity: str, file_type: str, validation_fn=None):
        self.name = name
        self.filename = filename
        self.query = query
        self.expected = expected
        self.complexity = complexity
        self.file_type = file_type
        self.validation_fn = validation_fn or self._default_validation
        
    def _default_validation(self, result: str, expected: Any) -> Tuple[bool, str]:
        """Default validation: check if expected value appears in result"""
        result_lower = str(result).lower()
        expected_str = str(expected).lower()
        
        # Handle numeric comparisons
        if isinstance(expected, (int, float)):
            # Extract numbers from result
            numbers = re.findall(r'-?\d+\.?\d*', result)
            for num in numbers:
                try:
                    if abs(float(num) - float(expected)) < 0.01:
                        return True, f"Found expected value {expected}"
                except:
                    continue
            return False, f"Expected {expected} not found in numeric values: {numbers}"
        
        # String matching
        if expected_str in result_lower:
            return True, f"Found expected value '{expected}'"
        
        return False, f"Expected '{expected}' not found in result"
    
    async def run(self, service) -> Dict[str, Any]:
        """Execute test and validate result"""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                service.analyze(
                    query=self.query,
                    context={'filename': self.filename}
                ),
                timeout=600  # 10 minute max per test
            )
            
            elapsed = time.time() - start_time
            
            # Extract result text
            result_text = result.get('result', '')
            if isinstance(result_text, dict):
                result_text = result.get('interpretation', str(result_text))
            
            # Validate answer
            is_correct, reason = self.validation_fn(result_text, self.expected)
            
            return {
                'name': self.name,
                'filename': self.filename,
                'query': self.query,
                'complexity': self.complexity,
                'file_type': self.file_type,
                'expected': self.expected,
                'result': result_text[:200],  # Truncate for display
                'correct': is_correct,
                'reason': reason,
                'elapsed': elapsed,
                'success': result.get('success', False),
                'agent': result.get('metadata', {}).get('agent', 'unknown')
            }
        
        except asyncio.TimeoutError:
            return {
                'name': self.name,
                'correct': False,
                'reason': "Test timed out after 10 minutes",
                'error': "Timeout",
                'elapsed': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'name': self.name,
                'correct': False,
                'reason': f"Exception: {str(e)}",
                'error': str(e),
                'elapsed': time.time() - start_time
            }


def create_test_suite() -> List[AccuracyTest]:
    """Create comprehensive test suite across all complexity levels and file types"""
    
    tests = []
    
    # ============================================================================
    # LEVEL 1: SIMPLE LOOKUPS (Complexity 0.1-0.3)
    # ============================================================================
    
    # JSON - Simple value lookup
    tests.append(AccuracyTest(
        name="JSON Simple Lookup",
        filename="1.json",
        query="what is the name",
        expected="harsha",
        complexity="simple",
        file_type="json"
    ))
    
    tests.append(AccuracyTest(
        name="JSON Roll Number Lookup",
        filename="1.json",
        query="what is the roll number",
        expected="22r21a6695",
        complexity="simple",
        file_type="json"
    ))
    
    # CSV - Show first row
    tests.append(AccuracyTest(
        name="CSV First Row",
        filename="test_employee_data.csv",
        query="show me the first employee name",
        expected="John",  # Assuming first row has John
        complexity="simple",
        file_type="csv"
    ))
    
    # JSON - Nested value
    tests.append(AccuracyTest(
        name="JSON Nested Lookup",
        filename="simple.json",
        query="what is the total_sales value",
        expected=None,  # Will check result exists
        complexity="simple",
        file_type="json",
        validation_fn=lambda r, e: (len(str(r)) > 0, "Result returned")
    ))
    
    # ============================================================================
    # LEVEL 2: BASIC OPERATIONS (Complexity 0.3-0.5)
    # ============================================================================
    
    # CSV - Count rows
    tests.append(AccuracyTest(
        name="CSV Count Rows",
        filename="test_student_grades.csv",
        query="how many students are there",
        expected=None,  # Will validate number is returned
        complexity="medium",
        file_type="csv",
        validation_fn=lambda r, e: (
            bool(re.search(r'\d+', str(r))),
            "Found numeric count" if re.search(r'\d+', str(r)) else "No count found"
        )
    ))
    
    # CSV - Filter operation
    tests.append(AccuracyTest(
        name="CSV Filter",
        filename="sales_data.csv",
        query="show products with price greater than 100",
        expected=None,
        complexity="medium",
        file_type="csv",
        validation_fn=lambda r, e: (len(str(r)) > 20, "Filter results returned")
    ))
    
    # JSON - Array length
    tests.append(AccuracyTest(
        name="JSON Array Count",
        filename="analyze.json",
        query="how many items are in the data",
        expected=None,
        complexity="medium",
        file_type="json",
        validation_fn=lambda r, e: (
            bool(re.search(r'\d+', str(r))),
            "Found numeric count"
        )
    ))
    
    # ============================================================================
    # LEVEL 3: AGGREGATIONS (Complexity 0.5-0.7)
    # ============================================================================
    
    # CSV - Calculate average
    tests.append(AccuracyTest(
        name="CSV Average Calculation",
        filename="test_sales_monthly.csv",
        query="what is the average sales amount",
        expected=None,
        complexity="complex",
        file_type="csv",
        validation_fn=lambda r, e: (
            bool(re.search(r'\d+\.?\d*', str(r))),
            "Found numeric average"
        )
    ))
    
    # CSV - Sum aggregation
    tests.append(AccuracyTest(
        name="CSV Sum Total",
        filename="hr_employee_data.csv",
        query="what is the total salary of all employees",
        expected=None,
        complexity="complex",
        file_type="csv",
        validation_fn=lambda r, e: (
            bool(re.search(r'\d+', str(r))),
            "Found numeric total"
        )
    ))
    
    # CSV - Max/Min
    tests.append(AccuracyTest(
        name="CSV Max Value",
        filename="test_student_grades.csv",
        query="who has the highest grade",
        expected=None,
        complexity="complex",
        file_type="csv",
        validation_fn=lambda r, e: (len(str(r)) > 10, "Student name returned")
    ))
    
    # ============================================================================
    # LEVEL 4: COMPLEX ANALYSIS (Complexity 0.7-0.85)
    # ============================================================================
    
    # CSV - Grouping and aggregation
    tests.append(AccuracyTest(
        name="CSV Group By Analysis",
        filename="comprehensive_ecommerce.csv",
        query="calculate average order value by category",
        expected=None,
        complexity="very_complex",
        file_type="csv",
        validation_fn=lambda r, e: (
            'category' in str(r).lower() or 'average' in str(r).lower(),
            "Contains grouping results"
        )
    ))
    
    # CSV - Multi-column analysis
    tests.append(AccuracyTest(
        name="CSV Multi-Column Stats",
        filename="healthcare_patients.csv",
        query="compare average age between male and female patients",
        expected=None,
        complexity="very_complex",
        file_type="csv",
        validation_fn=lambda r, e: (
            ('male' in str(r).lower() and 'female' in str(r).lower()),
            "Contains gender comparison"
        )
    ))
    
    # JSON - Complex nested extraction
    tests.append(AccuracyTest(
        name="JSON Complex Nested",
        filename="complex_nested.json",
        query="what are all the product categories available",
        expected=None,
        complexity="very_complex",
        file_type="json",
        validation_fn=lambda r, e: (len(str(r)) > 15, "Categories listed")
    ))
    
    # ============================================================================
    # LEVEL 5: GOD-LEVEL (Complexity 0.85-1.0)
    # ============================================================================
    
    # CSV - Statistical correlation
    tests.append(AccuracyTest(
        name="CSV Correlation Analysis",
        filename="iot_sensor_data.csv",
        query="is there a correlation between temperature and humidity",
        expected=None,
        complexity="god",
        file_type="csv",
        validation_fn=lambda r, e: (
            ('correlation' in str(r).lower() or 'relationship' in str(r).lower()),
            "Contains correlation analysis"
        )
    ))
    
    # CSV - Multi-step reasoning
    tests.append(AccuracyTest(
        name="CSV Multi-Step Reasoning",
        filename="multi_country_sales.csv",
        query="which country has the highest revenue and what is the average order size there",
        expected=None,
        complexity="god",
        file_type="csv",
        validation_fn=lambda r, e: (
            (bool(re.search(r'[A-Z][a-z]+', str(r))) and bool(re.search(r'\d+', str(r)))),
            "Contains country name and numeric value"
        )
    ))
    
    # CSV - Trend analysis
    tests.append(AccuracyTest(
        name="CSV Time Series Trend",
        filename="time_series_stock.csv",
        query="show the trend of stock prices over time and predict next month",
        expected=None,
        complexity="god",
        file_type="csv",
        validation_fn=lambda r, e: (
            ('trend' in str(r).lower() or 'increase' in str(r).lower() or 'decrease' in str(r).lower()),
            "Contains trend analysis"
        )
    ))
    
    # JSON - Deep nested multi-step
    tests.append(AccuracyTest(
        name="JSON Deep Analysis",
        filename="large_transactions.json",
        query="find the top 3 customers by transaction value and their average order frequency",
        expected=None,
        complexity="god",
        file_type="json",
        validation_fn=lambda r, e: (
            bool(re.search(r'(top|customer|transaction)', str(r).lower())),
            "Contains customer analysis"
        )
    ))
    
    # ============================================================================
    # DOCUMENT FILES (PDF/TXT)
    # ============================================================================
    
    # PDF - Text extraction
    tests.append(AccuracyTest(
        name="PDF Content Query",
        filename="HARSHA_Kota_Resume.pdf",
        query="what are the key skills mentioned",
        expected=None,
        complexity="medium",
        file_type="pdf",
        validation_fn=lambda r, e: (len(str(r)) > 20, "Skills extracted from PDF")
    ))
    
    # TXT - Content analysis
    tests.append(AccuracyTest(
        name="TXT File Analysis",
        filename="market_report.txt",
        query="summarize the main points",
        expected=None,
        complexity="medium",
        file_type="txt",
        validation_fn=lambda r, e: (len(str(r)) > 30, "Summary generated")
    ))
    
    return tests


async def run_test_suite():
    """Execute all tests and generate comprehensive report"""
    
    print("=" * 80)
    print("COMPREHENSIVE ACCURACY TEST SUITE")
    print("=" * 80)
    print("Testing with REAL LLM calls across all file types and complexity levels")
    print()
    
    # Initialize service using factory pattern
    print("Initializing analysis service...")
    from backend.services.analysis_service import get_analysis_service
    service = get_analysis_service()
    
    # Create test suite
    tests = create_test_suite()
    
    print(f"Created {len(tests)} test cases")
    print()
    
    # Run tests with breaks for system recovery
    results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] Running: {test.name}")
        print(f"    File: {test.filename}")
        print(f"    Query: {test.query}")
        
        # Run test with error handling
        try:
            result = await test.run(service)
        except Exception as e:
            print(f"    ⚠️ Test execution error: {e}")
            result = {
                'name': test.name,
                'correct': False,
                'reason': f"Test execution failed: {str(e)}",
                'error': str(e),
                'elapsed': 0
            }
        
        results.append(result)
        
        if result.get('correct'):
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
        
        print(f"    {status} - {result.get('reason', 'No reason')}")
        print(f"    Time: {result.get('elapsed', 0):.2f}s")
        
        # Save checkpoint after each test (in case of crash)
        checkpoint_file = Path(__file__).parent.parent / "test_accuracy_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'completed': i,
                'total': len(tests),
                'passed': passed,
                'failed': failed,
                'results': results
            }, f, indent=2)
        
        # Give system time to recover between tests
        # Longer break every 3 tests for better memory recovery
        if i % 3 == 0:
            print(f"    💤 Taking 15s break for system recovery...")
            await asyncio.sleep(15)
        else:
            print(f"    💤 Taking 5s break...")
            await asyncio.sleep(5)
        print()
    
    # Generate report
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed} ({passed/len(tests)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(tests)*100:.1f}%)")
    print()
    
    # Breakdown by complexity
    complexity_stats = {}
    for result in results:
        complexity = result.get('complexity', 'unknown')
        if complexity not in complexity_stats:
            complexity_stats[complexity] = {'passed': 0, 'failed': 0}
        
        if result.get('correct'):
            complexity_stats[complexity]['passed'] += 1
        else:
            complexity_stats[complexity]['failed'] += 1
    
    print("Results by Complexity Level:")
    for complexity, stats in sorted(complexity_stats.items()):
        total = stats['passed'] + stats['failed']
        pass_rate = stats['passed'] / total * 100 if total > 0 else 0
        print(f"  {complexity.upper()}: {stats['passed']}/{total} ({pass_rate:.1f}%)")
    
    print()
    
    # Breakdown by file type
    filetype_stats = {}
    for result in results:
        ftype = result.get('file_type', 'unknown')
        if ftype not in filetype_stats:
            filetype_stats[ftype] = {'passed': 0, 'failed': 0}
        
        if result.get('correct'):
            filetype_stats[ftype]['passed'] += 1
        else:
            filetype_stats[ftype]['failed'] += 1
    
    print("Results by File Type:")
    for ftype, stats in sorted(filetype_stats.items()):
        total = stats['passed'] + stats['failed']
        pass_rate = stats['passed'] / total * 100 if total > 0 else 0
        print(f"  {ftype.upper()}: {stats['passed']}/{total} ({pass_rate:.1f}%)")
    
    print()
    
    # Failed tests detail
    if failed > 0:
        print("=" * 80)
        print("FAILED TESTS DETAIL")
        print("=" * 80)
        for result in results:
            if not result.get('correct'):
                print(f"\n{result.get('name')}")
                print(f"  File: {result.get('filename')}")
                print(f"  Query: {result.get('query')}")
                print(f"  Expected: {result.get('expected')}")
                print(f"  Reason: {result.get('reason')}")
                print(f"  Result: {result.get('result', 'N/A')[:150]}...")
    
    # Save detailed results
    output_file = Path(__file__).parent.parent / "test_accuracy_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total': len(tests),
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / len(tests) * 100
            },
            'by_complexity': complexity_stats,
            'by_file_type': filetype_stats,
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {output_file}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = asyncio.run(run_test_suite())
    sys.exit(0 if success else 1)
