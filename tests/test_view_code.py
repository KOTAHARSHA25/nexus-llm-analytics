"""Test that generated Python code is visible in the output"""
import sys
sys.path.insert(0, 'src/backend')
import pandas as pd
from core.code_generator import get_code_generator

print('='*70)
print('TEST: VIEW GENERATED PYTHON CODE')
print('='*70)

# Load sample data
df = pd.read_csv('data/samples/sales_data.csv')
print(f'Data: {len(df)} rows, {len(df.columns)} cols\n')

# Get the code generator
gen = get_code_generator()

# Execute a query
query = "What is the total sales by region?"
print(f'Query: "{query}"\n')

result = gen.generate_and_execute(
    query=query,
    df=df,
    model='phi3:mini',
    max_retries=2,
    data_file='sales_data.csv',
    save_history=True
)

print('='*70)
print('EXECUTION RESULT')
print('='*70)

print(f'Success: {result.success}')
print(f'Execution ID: {result.execution_id}')
print(f'Model Used: {result.model_used}')
print(f'Attempts: {result.attempt_count}')
print(f'Execution Time: {result.execution_time_ms:.2f}ms')

print('\n' + '-'*70)
print('GENERATED CODE (Original LLM Output):')
print('-'*70)
print(result.generated_code or "(No generated code)")

print('\n' + '-'*70)
print('EXECUTED CODE (Cleaned):')
print('-'*70)
print(result.code or "(No executed code)")

print('\n' + '-'*70)
print('RESULT:')
print('-'*70)
print(result.result)

if result.retry_errors:
    print('\n' + '-'*70)
    print('RETRY ERRORS (if any):')
    print('-'*70)
    for err in result.retry_errors:
        print(f'  - {err}')

print('\n' + '='*70)
print('✅ GENERATED PYTHON CODE IS NOW VISIBLE!')
print('='*70)
