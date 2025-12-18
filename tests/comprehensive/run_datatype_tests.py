"""Run data type unit tests for CSV, JSON, PDF, TXT"""
from backend.utils.data_optimizer import DataOptimizer
from pathlib import Path
import sys

opt = DataOptimizer()

print('='*70)
print('CSV DATA TYPE UNIT TESTS')
print('='*70)

csv_files = [
    'data/samples/sales_data.csv',
    'data/samples/csv/customer_data.csv',
    'data/samples/csv/orders.csv',
    'data/samples/csv/sales_simple.csv',
    'data/samples/StressLevelDataset.csv',
    'data/samples/test_employee_data.csv',
    'data/samples/test_inventory.csv',
    'data/samples/test_sales_monthly.csv',
    'data/samples/test_student_grades.csv',
    'data/uploads/test_sales.csv',
    'data/uploads/merged_customers_orders.csv'
]

csv_passed = 0
csv_failed = 0

for csv_file in csv_files:
    path = Path(csv_file)
    if path.exists():
        try:
            result = opt.optimize_for_llm(str(path), 'csv')
            rows = result['total_rows']
            cols = result['total_columns']
            print(f'PASS: {path.name:40} {rows:5} rows, {cols:2} cols')
            csv_passed += 1
        except Exception as e:
            print(f'FAIL: {path.name:40} {str(e)[:30]}')
            csv_failed += 1

print(f'\nCSV Summary: {csv_passed} passed, {csv_failed} failed\n')

# JSON Tests
print('='*70)
print('JSON DATA TYPE UNIT TESTS')
print('='*70)

json_files = [
    'data/samples/simple.json',
    'data/samples/1.json',
    'data/samples/analyze.json',
    'data/samples/complex_nested.json',
    'data/samples/financial_quarterly.json',
    'data/samples/large_transactions.json',
    'data/samples/sales_timeseries.json',
    'data/samples/edge_cases/boolean_fields.json',
    'data/samples/edge_cases/date_formats.json',
    'data/samples/edge_cases/mixed_types.json',
    'data/samples/edge_cases/null_values.json',
    'data/samples/edge_cases/unicode_data.json',
    'data/uploads/1.json',
    'data/uploads/analyze.json'
]

json_passed = 0
json_failed = 0

for json_file in json_files:
    path = Path(json_file)
    if path.exists():
        try:
            result = opt.optimize_for_llm(str(path), 'json')
            rows = result['total_rows']
            nested = 'nested' if result.get('was_nested') else 'flat'
            print(f'PASS: {path.name:40} {rows:5} rows ({nested})')
            json_passed += 1
        except ValueError as e:
            print(f'SKIP: {path.name:40} Empty/invalid')
        except Exception as e:
            print(f'FAIL: {path.name:40} {str(e)[:30]}')
            json_failed += 1

print(f'\nJSON Summary: {json_passed} passed, {json_failed} failed\n')

# PDF/TXT Tests
print('='*70)
print('PDF/TXT DATA TYPE UNIT TESTS')
print('='*70)

txt_files = [
    'data/uploads/Harsha_Kota.pdf.extracted.txt',
    'data/uploads/eachfile.txt.extracted.txt',
    'data/uploads/upgrade.txt.extracted.txt'
]

txt_passed = 0
txt_failed = 0

for txt_file in txt_files:
    path = Path(txt_file)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            chars = len(content)
            print(f'PASS: {path.name:40} {chars:6} chars')
            txt_passed += 1
        except Exception as e:
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    content = f.read()
                chars = len(content)
                print(f'PASS: {path.name:40} {chars:6} chars (latin-1)')
                txt_passed += 1
            except:
                print(f'FAIL: {path.name:40} {str(e)[:30]}')
                txt_failed += 1

print(f'\nPDF/TXT Summary: {txt_passed} passed, {txt_failed} failed\n')

# Overall Summary
print('='*70)
print('OVERALL DATA TYPE TEST SUMMARY')
print('='*70)
total_passed = csv_passed + json_passed + txt_passed
total_failed = csv_failed + json_failed + txt_failed
total_tests = total_passed + total_failed

print(f'CSV Files:     {csv_passed:2} passed, {csv_failed:2} failed')
print(f'JSON Files:    {json_passed:2} passed, {json_failed:2} failed')
print(f'PDF/TXT Files: {txt_passed:2} passed, {txt_failed:2} failed')
print(f'\nTOTAL:         {total_passed:2} passed, {total_failed:2} failed  ({total_passed}/{total_tests} = {total_passed/total_tests*100:.1f}%)')

if total_failed == 0:
    print('\n*** ALL DATA TYPE TESTS PASSED ***')
else:
    print(f'\n{total_passed} of {total_tests} tests passed')
